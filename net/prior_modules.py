import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftRegionMask(nn.Module):
    """
    Convert index_map (B,H,W) -> soft mask S (B,K,H',W').
    K is inferred per batch: max(index_map)+1.
    """

    def __init__(self, init_logit_scale: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.eps = eps

    def forward(self, index_map: torch.Tensor, target_hw) -> torch.Tensor:
        """
        index_map: (B,H,W) int64 with values in [0..K-1]
        target_hw: (H_out, W_out) to align with feature resolution
        returns: soft mask S (B,K,H_out,W_out)
        """
        b, h, w = index_map.shape
        k = int(index_map.max().item()) + 1
        one_hot = F.one_hot(index_map, num_classes=k).permute(0, 3, 1, 2).float()  # (B,K,H,W)
        if (h, w) != target_hw:
            # Match CIDNet downsample behavior (bilinear resize) to keep grids consistent.
            one_hot = F.interpolate(one_hot, size=target_hw, mode="bilinear", align_corners=True)
            one_hot = one_hot.clamp_min(0.0)
        # softmax(log(mask)) == mask / sum(mask); scale controls sharpness via mask**scale
        logits = torch.log(one_hot + self.eps) * self.logit_scale
        return F.softmax(logits, dim=1)


class BoundaryMap(nn.Module):
    """
    Convert index_map (B,H,W) into a soft boundary map A (B,1,H',W').
    Boundary pixels are those whose region id differs from a 4-neighbor.
    """

    def forward(self, index_map: torch.Tensor, target_hw) -> torch.Tensor:
        b, h, w = index_map.shape
        edge = torch.zeros((b, h, w), device=index_map.device, dtype=torch.bool)
        edge[:, :, 1:] |= index_map[:, :, 1:] != index_map[:, :, :-1]
        edge[:, :, :-1] |= index_map[:, :, 1:] != index_map[:, :, :-1]
        edge[:, 1:, :] |= index_map[:, 1:, :] != index_map[:, :-1, :]
        edge[:, :-1, :] |= index_map[:, 1:, :] != index_map[:, :-1, :]
        edge = edge.float().unsqueeze(1)  # (B,1,H,W)
        if (h, w) != target_hw:
            edge = F.interpolate(edge, size=target_hw, mode="bilinear", align_corners=True)
        return edge.clamp(0.0, 1.0)


class StructureGate(nn.Module):
    """
    A very stable prior injection: use a 1-channel structure/boundary map to gate features.
    Starts near identity (alpha small, projection zero-init).
    """

    def __init__(self, channels: int, init_alpha: float = -6.0):
        super().__init__()
        self.proj = nn.Conv2d(1, channels, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, feat: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        # boundary: (B,1,H,W)
        gate = torch.tanh(self.proj(boundary))
        a = torch.sigmoid(self.alpha)
        return feat * (1.0 + a * gate)


class RegionPooling(nn.Module):
    """
    Region pooling: aggregate feature F with soft mask S to region descriptors.
    """

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        feat: (B,C,H,W)
        mask: (B,K,H,W)
        returns: V (B,K,C)
        """
        b, c, h, w = feat.shape
        _, k, mh, mw = mask.shape
        if (h, w) != (mh, mw):
            raise ValueError(f"Feature size {(h,w)} and mask size {(mh,mw)} must match.")
        feat_flat = feat.view(b, c, -1)                # (B,C,HW)
        mask_flat = mask.view(b, k, -1)                # (B,K,HW)
        weights = mask_flat / (mask_flat.sum(-1, keepdim=True) + 1e-6)  # normalize per region
        # v_k = sum_{x,y} F * S_k
        v = torch.einsum("bkn,bcn->bkc", weights, feat_flat)  # (B,K,C)
        return v


class RegionPolicyMLP(nn.Module):
    """
    Map region descriptors to FiLM parameters gamma/beta.
    """

    def __init__(self, channels: int, hidden: int = None):
        super().__init__()
        hid = hidden or channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, channels * 2),
        )
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, region_vec: torch.Tensor):
        """
        region_vec: (B,K,C)
        returns gamma, beta: each (B,K,C)
        """
        b, k, c = region_vec.shape
        out = self.mlp(region_vec)
        out = out.view(b, k, 2, c)
        gamma = out[:, :, 0, :]
        beta = out[:, :, 1, :]
        if self.training:
            with torch.no_grad():
                self.last_gamma_mean = float(gamma.mean().detach().cpu().item())
                self.last_gamma_std = float(gamma.std(unbiased=False).detach().cpu().item())
                self.last_beta_mean = float(beta.mean().detach().cpu().item())
                self.last_beta_std = float(beta.std(unbiased=False).detach().cpu().item())
        return gamma, beta


class RegionCrossAttention(nn.Module):
    """
    SKF-like fusion using region tokens:
      - region tokens V (B,K,C) come from RegionPooling(feat, mask)
      - pixel queries come from feat
      - attention is over K regions (cheap, since K is small)

    This treats the label mask as routing (who should talk to which region token),
    rather than directly forcing multiplicative FiLM on every channel.
    """

    def __init__(
        self,
        channels: int,
        init_alpha: float = -2.197225,
        area_gate_power: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.k = nn.Linear(channels, channels, bias=True)
        self.v = nn.Linear(channels, channels, bias=True)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.mask_bias_scale = nn.Parameter(torch.tensor(0.5))
        self.area_gate_power = float(area_gate_power)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.eps = float(eps)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor, region_vec: torch.Tensor) -> torch.Tensor:
        """
        feat: (B,C,H,W)
        mask: (B,K,H,W), soft mask (sum over K = 1)
        region_vec: (B,K,C)
        """
        b, c, h, w = feat.shape
        _, k, mh, mw = mask.shape
        if (h, w) != (mh, mw):
            raise ValueError(f"Feature size {(h,w)} and mask size {(mh,mw)} must match.")

        # Optional: down-weight small regions (noisy tokens)
        v_tokens = region_vec
        if self.area_gate_power > 0.0:
            area = mask.sum(dim=(2, 3))  # (B,K) 每个区域“有多少像素”(软面积)
            p = area / float(h * w)    # 面积比例
            denom = p.max(dim=1, keepdim=True).values.clamp_min(1e-6)
            g = (p / denom).clamp_min(0.0).pow(self.area_gate_power)  # (B,K) in [0,1] 最大区域 g=1，小区域 g<1
            v_tokens = v_tokens * g.unsqueeze(-1)

        q = self.q(feat)  # (B,C,H,W)
        q = q.flatten(2).transpose(1, 2)  # (B,HW,C)
        k_tok = self.k(v_tokens)  # (B,K,C)
        v_tok = self.v(v_tokens)  # (B,K,C)

        # Attention over K regions per pixel: (B,HW,K)
        attn_logits = torch.einsum("bmc,bkc->bmk", q, k_tok) / (c ** 0.5)
        mask_flat = mask.flatten(2).transpose(1, 2)  # (B,HW,K)
        attn_logits = attn_logits + self.mask_bias_scale * torch.log(mask_flat + self.eps)
        attn = torch.softmax(attn_logits, dim=-1)

        context = torch.einsum("bmk,bkc->bmc", attn, v_tok)  # (B,HW,C)
        context = context.transpose(1, 2).view(b, c, h, w)
        delta = self.proj(context)

        a = torch.sigmoid(self.alpha)
        out = feat + a * delta

        if self.training:
            with torch.no_grad():
                self.last_a = float(a.detach().cpu().item())
                delta_f = delta.float()
                base_f = feat.float()
                delta_rms = torch.sqrt(torch.mean(delta_f * delta_f) + 1e-12)
                base_rms = torch.sqrt(torch.mean(base_f * base_f) + 1e-12)
                self.last_delta_ratio = float((delta_rms / base_rms).detach().cpu().item())

        return out


class RegionFiLM(nn.Module):
    """
    Apply region-wise FiLM modulation back to spatial feature map.
    """

    # NOTE: init_alpha controls how much the prior branch participates at the start:
    # sigmoid(-2)≈0.119, sigmoid(-1)≈0.269, sigmoid(-6)≈0.0025.
    def __init__(
        self,
        gamma_scale: float = 0.1,
        beta_scale: float = 0.1,
        init_alpha: float = -1.386294,
        area_gate_power: float = 0.5,
    ):
        super().__init__()
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        self.area_gate_power = float(area_gate_power)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, feat: torch.Tensor, mask: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        """
        feat: (B,C,H,W)
        mask: (B,K,H,W)
        gamma, beta: (B,K,C)
        returns: feat_mod (B,C,H,W)
        """
        b, c, h, w = feat.shape
        _, k, mh, mw = mask.shape
        if (h, w) != (mh, mw):
            raise ValueError(f"Feature size {(h,w)} and mask size {(mh,mw)} must match.")
        # constrain to near-identity at init for stable training
        gamma_dev = self.gamma_scale * torch.tanh(gamma)
        beta_used = self.beta_scale * torch.tanh(beta)

        # Down-weight small regions: they are noisier (estimated from fewer pixels).
        # We only scale the deviations (gamma_dev/beta_used), keeping the identity term intact.
        if self.area_gate_power > 0.0:
            area = mask.sum(dim=(2, 3))  # (B,K)
            p = area / float(h * w)      # (B,K), sum≈1
            denom = p.max(dim=1, keepdim=True).values.clamp_min(1e-6)
            g = (p / denom).clamp_min(0.0).pow(self.area_gate_power)  # (B,K) in [0,1], max=1
            g = g.unsqueeze(-1)  # (B,K,1)
            gamma_dev = gamma_dev * g
            beta_used = beta_used * g

        gamma_used = 1.0 + gamma_dev
        # mix region policies back to per-pixel gamma/beta
        # gamma_map/beta_map: (B,C,H,W) via einsum over K
        gamma_map = torch.einsum("bkc,bkhw->bchw", gamma_used, mask)
        beta_map = torch.einsum("bkc,bkhw->bchw", beta_used, mask)
        mod = gamma_map * feat + beta_map
        # gated residual: starts as identity when a≈0, constrained to [0,1]
        a = torch.sigmoid(self.alpha)
        out = feat + a * (mod - feat)

        if self.training:
            with torch.no_grad():
                gamma_dev = gamma_used - 1.0
                self.last_a = float(a.detach().cpu().item())
                self.last_gamma_dev_mean = float(gamma_dev.mean().detach().cpu().item())
                self.last_gamma_dev_std = float(gamma_dev.std(unbiased=False).detach().cpu().item())
                self.last_beta_used_mean = float(beta_used.mean().detach().cpu().item())
                self.last_beta_used_std = float(beta_used.std(unbiased=False).detach().cpu().item())

                delta = (mod - feat).float()
                base = feat.float()
                delta_rms = torch.sqrt(torch.mean(delta * delta) + 1e-12)
                base_rms = torch.sqrt(torch.mean(base * base) + 1e-12)
                self.last_delta_ratio = float((delta_rms / base_rms).detach().cpu().item())

        return out
