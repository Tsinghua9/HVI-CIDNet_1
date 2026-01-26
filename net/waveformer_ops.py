from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _has_fft_dct():
    return hasattr(torch.fft, "dct") and hasattr(torch.fft, "idct")


try:
    import torch_dct  # type: ignore
except Exception:
    torch_dct = None


def dct2d(x: torch.Tensor) -> torch.Tensor:
    if _has_fft_dct():
        x = torch.fft.dct(x, type=2, dim=-2, norm="ortho")
        x = torch.fft.dct(x, type=2, dim=-1, norm="ortho")
        return x
    if torch_dct is None or not hasattr(torch_dct, "dct_2d"):
        raise ImportError("WaveFormer WPO requires torch.fft.dct or torch_dct.dct_2d.")
    return torch_dct.dct_2d(x, norm="ortho")


def idct2d(x: torch.Tensor) -> torch.Tensor:
    if _has_fft_dct():
        x = torch.fft.idct(x, type=2, dim=-2, norm="ortho")
        x = torch.fft.idct(x, type=2, dim=-1, norm="ortho")
        return x
    if torch_dct is None or not hasattr(torch_dct, "idct_2d"):
        raise ImportError("WaveFormer WPO requires torch.fft.idct or torch_dct.idct_2d.")
    return torch_dct.idct_2d(x, norm="ortho")


class Wave2D(nn.Module):
    """
    Wave equation operator from WaveFormer (WPO).
    """
    def __init__(self, infer_mode: bool = False, res: int = 14, dim: int = 96, hidden_dim: int = 96):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        self.c = nn.Parameter(torch.ones(1) * 1)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        z = z.permute(0, 3, 1, 2).contiguous()

        x_u0 = dct2d(x)
        x_v0 = dct2d(x)

        if freq_embed is not None:
            t = self.to_k(freq_embed.unsqueeze(0).expand(b, -1, -1, -1).contiguous())
        else:
            t = torch.zeros((b, h, w, c), device=x.device, dtype=x.dtype)
        cos_term = torch.cos(self.c * t).permute(0, 3, 1, 2).contiguous()
        sin_term = torch.sin(self.c * t).permute(0, 3, 1, 2).contiguous() / self.c

        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        final_term = wave_term + velocity_term

        x_final = idct2d(final_term)
        x = self.out_norm(x_final.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x * F.silu(z)
        x = self.out_linear(x.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
