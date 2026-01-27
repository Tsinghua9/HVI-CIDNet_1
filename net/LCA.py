import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import *
from net.waveformer_ops import Wave2D

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
    def __init__(self, dim, num_heads, ffn_expansion=2.0, bias=False):
        super().__init__()
        self.attn = CAB(dim, num_heads, bias=bias)
        hidden = max(int(dim * ffn_expansion), dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=bias),
        )
        self.mfem = MFEM(dim, bias=bias)
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.lam = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.mu = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x, y):
        z = self.attn(x, y)
        z_hat = self.lam * self.ffn(self.alpha * z + self.beta * y) + self.mu * z
        fused = x + z_hat
        return self.mfem(fused) + fused


class DIEMCross(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.mafm1 = MAFM(dim, bias=bias)
        self.cdem = CDEM(dim, num_heads, bias=bias)
        self.mafm2 = MAFM(dim, bias=bias)

    def forward(self, x, y):
        fused = self.mafm1(x, y)
        enhanced = self.cdem(x, fused)
        return self.mafm2(enhanced, y)


class DIEMHV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = DIEMCross(dim, num_heads, bias=bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


class DIEMI_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = DIEMCross(dim, num_heads, bias=bias)

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
