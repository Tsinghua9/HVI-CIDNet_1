import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import *
from net.waveformer_ops import Wave2D
try:
    from torchdiffeq import odeint_adjoint as _odeint
except Exception:
    try:
        from torchdiffeq import odeint as _odeint
    except Exception:
        _odeint = None

# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
  
  
# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim) # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = self.gdfn(self.norm(x))
        return x
    
class I_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x)) 
        return x


class PixelAttention(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        return self.sigmoid(self.conv(torch.cat([x, y], dim=1)))


class ODEFunc(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, t, x):
        return self.func(x)


class Beltrami(nn.Module):
    def __init__(self, dim, k=8):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)
        self.k = k

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Beltrami expects 3D tokens, got {x.dim()}D")
        b, n, c = x.shape
        feat_pos = self.fc(x)
        feat = feat_pos[:, :, :c]
        pos = feat_pos[:, :, c:]
        pos = F.normalize(pos, p=2, dim=-1)
        sim = pos @ pos.transpose(-1, -2)
        k = min(self.k, n)
        topksim, topkid = torch.topk(sim, k=k, dim=-1)
        topkid = topkid.flatten(1)
        topkfeat = torch.gather(feat, dim=1, index=topkid.unsqueeze(-1).expand(-1, -1, c))
        topkfeat = topkfeat.view(b, n, k, c)
        attn = topksim.softmax(dim=-1)
        return (attn.unsqueeze(-1) * topkfeat).sum(dim=-2)


class BeltramiODE(nn.Module):
    def __init__(self, dim, k=8, method="rk4", tol=1e-3):
        super().__init__()
        self.odefunc = ODEFunc(Beltrami(dim=dim, k=k))
        self.method = method
        self.tol = tol

    def forward(self, x):
        if _odeint is None:
            raise RuntimeError("torchdiffeq is required for ODE; please install torchdiffeq.")
        t = x.new_tensor([0.0, 1.0])
        out = _odeint(self.odefunc, x, t, method=self.method, rtol=self.tol, atol=self.tol)
        return out[-1]


class TokenODE(nn.Module):
    def __init__(self, ode):
        super().__init__()
        self.ode = ode

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.ode(x)
        return x.transpose(1, 2).view(b, c, h, w)


class WindowODE(nn.Module):
    def __init__(self, ode, window_size=8):
        super().__init__()
        self.ode = ode
        self.window_size = window_size

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        h_pad, w_pad = x.shape[-2:]
        x = x.view(b, c, h_pad // ws, ws, w_pad // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, ws * ws, c)
        x = self.ode(x)
        x = x.view(b, h_pad // ws, w_pad // ws, ws, ws, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, h_pad, w_pad)
        return x[:, :, :h, :w]


class MAFM(nn.Module):
    def __init__(self, dim, reduction=4, bias=False):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=bias),
            nn.Sigmoid(),
        )
        self.pa = PixelAttention(dim, bias=bias)
        self.phi = nn.Parameter(torch.ones(1))
        self.omega = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        f_init = x + y
        ca = self.ca(f_init)
        ca_feat = f_init * ca + f_init
        sa = self.sa(torch.cat([
            f_init.mean(dim=1, keepdim=True),
            f_init.max(dim=1, keepdim=True)[0],
        ], dim=1))
        sa_feat = f_init * sa + f_init
        wc = self.pa(ca_feat, f_init)
        ws = self.pa(sa_feat, f_init)
        phi = torch.sigmoid(self.phi)
        omega = torch.sigmoid(self.omega)
        w = phi * wc + omega * ws
        return f_init + w * x + (1.0 - w) * y


class MFEM(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.b1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.b2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=bias)
        self.b3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim, bias=bias)
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.act(self.fuse(out))


class CDEM(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion=2.0, bias=False, ode_cfg=None):
        super().__init__()
        self.attn = CAB(dim, num_heads, bias=bias)
        hidden = max(int(dim * ffn_expansion), dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=bias),
        )
        self.use_ode = bool(ode_cfg and ode_cfg.get("enabled"))
        self.ode = None
        if self.use_ode:
            ode_block = BeltramiODE(
                dim=dim,
                k=int(ode_cfg.get("k", 8)),
                method=str(ode_cfg.get("method", "rk4")),
                tol=float(ode_cfg.get("tol", 1e-3)),
            )
            if bool(ode_cfg.get("window", False)):
                self.ode = WindowODE(ode_block, window_size=int(ode_cfg.get("window_size", 8)))
            else:
                self.ode = TokenODE(ode_block)
        self.mfem = MFEM(dim, bias=bias)
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.lam = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.mu = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x, y):
        z = self.attn(x, y)
        if self.use_ode:
            z_hat = self.lam * self.ode(self.alpha * z + self.beta * y) + self.mu * z
        else:
            z_hat = self.lam * self.ffn(self.alpha * z + self.beta * y) + self.mu * z
        fused = x + z_hat
        return self.mfem(fused) + fused


class DIEMCross(nn.Module):
    def __init__(self, dim, num_heads, bias=False, ode_cfg=None):
        super().__init__()
        self.mafm1 = MAFM(dim, bias=bias)
        self.cdem = CDEM(dim, num_heads, bias=bias, ode_cfg=ode_cfg)
        self.mafm2 = MAFM(dim, bias=bias)

    def forward(self, x, y):
        fused = self.mafm1(x, y)
        enhanced = self.cdem(x, fused)
        return self.mafm2(enhanced, y)


class DIEMHV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, ode_cfg=None):
        super().__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = DIEMCross(dim, num_heads, bias=bias, ode_cfg=ode_cfg)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


class DIEMI_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, ode_cfg=None):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = DIEMCross(dim, num_heads, bias=bias, ode_cfg=ode_cfg)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x


class _WaveFormerCross(nn.Module):
    def __init__(self, dim, embed_res=8, bias=False):
        super().__init__()
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.wpo = Wave2D(res=embed_res, dim=dim, hidden_dim=dim)
        self.freq_embed = nn.Parameter(torch.zeros(1, dim, embed_res, embed_res))
        nn.init.normal_(self.freq_embed, std=0.02)

    def _get_freq_embed(self, h, w):
        freq = F.interpolate(self.freq_embed, size=(h, w), mode="bilinear", align_corners=False)
        return freq.squeeze(0).permute(1, 2, 0).contiguous()

    def forward(self, x, y):
        fuse = self.fuse(torch.cat([x, y], dim=1))
        freq = self._get_freq_embed(fuse.shape[2], fuse.shape[3])
        return self.wpo(fuse, freq)


class WaveFormerHV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, embed_res=8):
        super(WaveFormerHV_LCA, self).__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = _WaveFormerCross(dim, embed_res=embed_res, bias=bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


class WaveFormerI_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, embed_res=8):
        super(WaveFormerI_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = _WaveFormerCross(dim, embed_res=embed_res, bias=bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x


class IllumEstimator(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        hidden = max(dim // 2, 8)
        self.net = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=bias),
        )

    def forward(self, x, y):
        illum = self.net(torch.cat([x, y], dim=1))
        return torch.sigmoid(illum)


class RegionIllumAttention(nn.Module):
    supports_prior_ctx = True

    def __init__(self, dim, num_heads, bias=False, lambda_init=1.0, lambda_max=1.5, illum_gate_temp=1.0, eps=1e-6):
        super().__init__()
        self.num_heads = int(num_heads)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.lambda_p = nn.Parameter(torch.full((self.num_heads, 1, 1), float(lambda_init)))
        self.lambda_p_min = 0.0
        self.lambda_p_max = float(lambda_max)
        self.prior_scale = nn.Parameter(torch.tensor(0.1))
        self.illum_gate_temp = max(float(illum_gate_temp), 1e-3)
        self.eps = float(eps)
        self.prior_vec_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.illum_head = nn.Linear(1, self.num_heads, bias=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _compose_prior(self, y, prior_ctx, h, w):
        if prior_ctx is None:
            return y, None, None
        entries = prior_ctx.get("entries", None)
        if not entries:
            return y, None, None

        prior_feat = 0.0
        prior_token = 0.0
        weight_sum = 0.0
        kept = []
        b, c, _, _ = y.shape

        for entry in entries:
            S = entry.get("S", None)
            V = entry.get("V", None)
            weight = float(entry.get("weight", 1.0))
            if S is None or V is None:
                continue
            if S.shape[-2:] != (h, w):
                S = F.interpolate(S, size=(h, w), mode="bilinear", align_corners=True)
            S = S.clamp_min(0.0)
            token_map = torch.einsum("bkhw,bkc->bchw", S, V)
            prior_feat = prior_feat + weight * token_map
            prior_token = prior_token + weight * V.mean(dim=1)
            weight_sum += weight
            kept.append({"S": S, "V": V, "weight": weight})

        if weight_sum <= 0.0 or len(kept) == 0:
            return y, None, None

        prior_feat = prior_feat / weight_sum
        prior_token = prior_token / weight_sum
        y = y + torch.tanh(self.prior_scale) * prior_feat

        c_per_head = c // self.num_heads
        token_4d = prior_token.view(b, c, 1, 1)
        vec = self.prior_vec_proj(token_4d).view(b, self.num_heads, c_per_head)
        vec = F.normalize(vec, dim=-1)
        bias = torch.einsum("bhc,bhd->bhcd", vec, vec)
        return y, bias, kept

    def forward(self, x, y, illum_map=None, prior_ctx=None, return_aux=False):
        b, _, h, w = x.shape

        y_in, prior_bias, kept = self._compose_prior(y, prior_ctx, h, w)
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y_in))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        if prior_bias is not None:
            with torch.no_grad():
                self.lambda_p.clamp_(min=float(self.lambda_p_min), max=float(self.lambda_p_max))
            attn_logits = attn_logits + self.lambda_p.view(1, self.num_heads, 1, 1) * prior_bias

        if illum_map is not None:
            illum_token = F.adaptive_avg_pool2d(illum_map, output_size=1).flatten(1)
            illum_gate = torch.sigmoid(self.illum_head(illum_token) / self.illum_gate_temp).view(b, self.num_heads, 1, 1)
            attn_logits = attn_logits * (1.0 + illum_gate)

        attn = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        if not return_aux:
            return out

        aux = {}
        if kept:
            loss_sum = 0.0
            w_sum = 0.0
            for entry in kept:
                S = entry["S"]
                V = entry["V"]
                w_e = float(entry.get("weight", 1.0))
                denom = S.flatten(2).sum(dim=-1, keepdim=True).clamp_min(self.eps)
                V_pred = torch.einsum("bchw,bkhw->bkc", out, S) / denom
                loss_sum = loss_sum + w_e * F.smooth_l1_loss(V_pred, V, reduction="mean")
                w_sum += w_e
            if w_sum > 0.0:
                aux["prior_align"] = loss_sum / w_sum
        return out, aux


class _ExpertConv(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, expansion=2.0, bias=False):
        super().__init__()
        hidden = max(int(dim * expansion), dim)
        padding = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class Top2SparseMoE(nn.Module):
    supports_prior_ctx = True

    def __init__(
        self,
        dim,
        num_experts=3,
        router_temp=1.0,
        sparse_topk=2,
        bias=False,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.sparse_topk = max(1, int(sparse_topk))
        self.router_temp_base = max(float(router_temp), 1e-3)
        self.router_temp = self.router_temp_base
        self.router_warm_temp = max(1.5, self.router_temp_base)
        self.progress = 1.0

        expert_cfg = [(3, 1), (5, 1), (3, 3)]
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            ksz, dil = expert_cfg[i % len(expert_cfg)]
            self.experts.append(_ExpertConv(dim, kernel_size=ksz, dilation=dil, bias=bias))

        self.router = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, self.num_experts),
        )

    def set_progress(self, progress: float):
        p = float(max(0.0, min(1.0, progress)))
        self.progress = p
        if p < 0.2:
            ratio = p / 0.2
            self.router_temp = self.router_warm_temp - (self.router_warm_temp - self.router_temp_base) * ratio
        else:
            self.router_temp = self.router_temp_base

    def _prior_token(self, prior_ctx, feat_like):
        if prior_ctx is None:
            return torch.zeros_like(feat_like)
        entries = prior_ctx.get("entries", None)
        if not entries:
            return torch.zeros_like(feat_like)
        token = 0.0
        weight_sum = 0.0
        for entry in entries:
            V = entry.get("V", None)
            weight = float(entry.get("weight", 1.0))
            if V is None:
                continue
            token = token + weight * V.mean(dim=1)
            weight_sum += weight
        if weight_sum <= 0.0:
            return torch.zeros_like(feat_like)
        return token / weight_sum

    def forward(self, x, x_ref, y_ref, prior_ctx=None, return_aux=False):
        gap_z = x.mean(dim=(2, 3))
        gap_x = x_ref.mean(dim=(2, 3))
        gap_y = y_ref.mean(dim=(2, 3))
        gap_p = self._prior_token(prior_ctx, gap_z)
        router_in = torch.cat([gap_z, gap_x, gap_y, gap_p], dim=1)
        logits = self.router(router_in) / (self.router_temp + 1e-6)
        topk = min(self.sparse_topk, self.num_experts)
        if topk < self.num_experts:
            topv, topi = torch.topk(logits, k=topk, dim=-1)
            sparse_logits = torch.full_like(logits, -1e4)
            sparse_logits.scatter_(1, topi, topv)
            probs = F.softmax(sparse_logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        expert_outs = [expert(x) for expert in self.experts]
        expert_outs = torch.stack(expert_outs, dim=1)  # (B,E,C,H,W)
        mixed = (probs[:, :, None, None, None] * expert_outs).sum(dim=1)

        if not return_aux:
            return mixed

        eps = 1e-8
        mean_prob = probs.mean(dim=0)
        router_entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean()
        expert_usage_std = mean_prob.std(unbiased=False)
        return mixed, {
            "router_probs": probs,
            "router_entropy": router_entropy,
            "expert_usage_std": expert_usage_std,
        }


class IllumRefine(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )
        self.illum_proj = nn.Conv2d(1, dim, kernel_size=1, bias=True)

    def forward(self, x, illum_map):
        if illum_map.shape[-2:] != x.shape[-2:]:
            illum_map = F.interpolate(illum_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        illum_gate = torch.sigmoid(self.illum_proj(illum_map))
        return self.local(x) * illum_gate


class RIPMixer(nn.Module):
    supports_prior_ctx = True

    def __init__(
        self,
        dim,
        num_heads,
        num_experts=3,
        router_temp=1.0,
        sparse_topk=2,
        illum_gate_temp=1.0,
        bias=False,
    ):
        super().__init__()
        self.illum_estimator = IllumEstimator(dim, bias=bias)
        self.attn = RegionIllumAttention(
            dim,
            num_heads,
            bias=bias,
            illum_gate_temp=illum_gate_temp,
        )
        self.moe = Top2SparseMoE(
            dim,
            num_experts=num_experts,
            router_temp=router_temp,
            sparse_topk=sparse_topk,
            bias=bias,
        )
        self.illum_refine = IllumRefine(dim, bias=bias)
        self.g1 = nn.Parameter(torch.full((1, dim, 1, 1), 0.1))
        self.g2 = nn.Parameter(torch.full((1, dim, 1, 1), 0.1))
        self.g3 = nn.Parameter(torch.full((1, dim, 1, 1), 0.1))

    def set_progress(self, progress: float):
        self.moe.set_progress(progress)

    def forward(self, x, y, prior_ctx=None, return_aux=False):
        illum_map = self.illum_estimator(x, y)
        if return_aux:
            z_attn, attn_aux = self.attn(x, y, illum_map=illum_map, prior_ctx=prior_ctx, return_aux=True)
            z_moe, moe_aux = self.moe(x, x, y, prior_ctx=prior_ctx, return_aux=True)
        else:
            z_attn = self.attn(x, y, illum_map=illum_map, prior_ctx=prior_ctx, return_aux=False)
            z_moe = self.moe(x, x, y, prior_ctx=prior_ctx, return_aux=False)
            attn_aux = {}
            moe_aux = {}
        z_illu = self.illum_refine(x, illum_map)

        out = x + torch.tanh(self.g1) * z_attn + torch.tanh(self.g2) * z_moe + torch.tanh(self.g3) * z_illu

        if not return_aux:
            return out
        aux = {}
        if "prior_align" in attn_aux:
            aux["prior_align"] = attn_aux["prior_align"]
        for key in ("router_probs", "router_entropy", "expert_usage_std"):
            if key in moe_aux:
                aux[key] = moe_aux[key]
        return out, aux


class DIEMCrossV2(nn.Module):
    supports_prior_ctx = True

    def __init__(
        self,
        dim,
        num_heads,
        num_experts=3,
        router_temp=1.0,
        sparse_topk=2,
        illum_gate_temp=1.0,
        bias=False,
    ):
        super().__init__()
        self.mixer = RIPMixer(
            dim=dim,
            num_heads=num_heads,
            num_experts=num_experts,
            router_temp=router_temp,
            sparse_topk=sparse_topk,
            illum_gate_temp=illum_gate_temp,
            bias=bias,
        )

    def set_progress(self, progress: float):
        self.mixer.set_progress(progress)

    def forward(self, x, y, prior_ctx=None, return_aux=False):
        return self.mixer(x, y, prior_ctx=prior_ctx, return_aux=return_aux)


class DIEMHV_LCA_V2(nn.Module):
    supports_prior_ctx = True

    def __init__(
        self,
        dim,
        num_heads,
        num_experts=3,
        router_temp=1.0,
        sparse_topk=2,
        illum_gate_temp=1.0,
        bias=False,
    ):
        super().__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = DIEMCrossV2(
            dim,
            num_heads,
            num_experts=num_experts,
            router_temp=router_temp,
            sparse_topk=sparse_topk,
            illum_gate_temp=illum_gate_temp,
            bias=bias,
        )

    def set_progress(self, progress: float):
        self.ffn.set_progress(progress)

    def forward(self, x, y, prior_ctx=None, return_aux=False):
        if return_aux:
            delta, aux = self.ffn(self.norm(x), self.norm(y), prior_ctx=prior_ctx, return_aux=True)
        else:
            delta = self.ffn(self.norm(x), self.norm(y), prior_ctx=prior_ctx, return_aux=False)
            aux = None
        x = x + delta
        x = self.gdfn(self.norm(x))
        if return_aux:
            return x, aux
        return x


class DIEMI_LCA_V2(nn.Module):
    supports_prior_ctx = True

    def __init__(
        self,
        dim,
        num_heads,
        num_experts=3,
        router_temp=1.0,
        sparse_topk=2,
        illum_gate_temp=1.0,
        bias=False,
    ):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = DIEMCrossV2(
            dim,
            num_heads,
            num_experts=num_experts,
            router_temp=router_temp,
            sparse_topk=sparse_topk,
            illum_gate_temp=illum_gate_temp,
            bias=bias,
        )

    def set_progress(self, progress: float):
        self.ffn.set_progress(progress)

    def forward(self, x, y, prior_ctx=None, return_aux=False):
        if return_aux:
            delta, aux = self.ffn(self.norm(x), self.norm(y), prior_ctx=prior_ctx, return_aux=True)
        else:
            delta = self.ffn(self.norm(x), self.norm(y), prior_ctx=prior_ctx, return_aux=False)
            aux = None
        x = x + delta
        x = x + self.gdfn(self.norm(x))
        if return_aux:
            return x, aux
        return x
