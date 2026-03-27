import argparse
import time
from typing import Optional, Tuple

import torch

from net.CIDNet import CIDNet


def build_model(args: argparse.Namespace, device: torch.device) -> CIDNet:
    # Defaults follow your LOLv1 training setup.
    model = CIDNet(
        use_wtconv_i=True,
        use_dwconv_hv=False,
        fe_type="dual_gate",
        lca_type="diem",
        max_regions=args.max_regions,
        pre_lca_film=True,
        pre_lca_film_scale=0.1,
        pre_lca_film_bias=0.1,
        pre_lca_film_alpha=-2.197225,
        attn_alpha1_init=-2.197225,
        attn_alpha2_init=-3.891,
        attn_mask_bias_scale1_init=1.0,
        attn_mask_bias_scale2_init=0.5,
        attn_mask_bias_scale1_max=-1.0,
        attn_mask_bias_scale2_max=-1.0,
    ).to(device)
    model.eval()
    return model


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_inference_time(
    model: torch.nn.Module,
    x: torch.Tensor,
    index_map: Optional[torch.Tensor],
    prior_mode: str,
    warmup: int,
    repeat: int,
) -> float:
    # Average latency per image (seconds).
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            if index_map is None:
                _ = model(x)
            else:
                _ = model(x, index_map=index_map, prior_mode=prior_mode)

    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max(1, repeat)):
            if index_map is None:
                _ = model(x)
            else:
                _ = model(x, index_map=index_map, prior_mode=prior_mode)
    if x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / float(max(1, repeat))


def profile_flops(
    model: torch.nn.Module,
    x: torch.Tensor,
    index_map: Optional[torch.Tensor],
    prior_mode: str,
) -> Optional[Tuple[float, float]]:
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None

    if index_map is None:
        macs, params = profile(model, inputs=(x,), verbose=False)
    else:
        # Wrap to pass prior_mode as a fixed keyword argument.
        class _Wrapper(torch.nn.Module):
            def __init__(self, net: torch.nn.Module, mode: str):
                super().__init__()
                self.net = net
                self.mode = mode

            def forward(self, x_in: torch.Tensor, map_in: torch.Tensor) -> torch.Tensor:
                return self.net(x_in, index_map=map_in, prior_mode=self.mode)

        wrapped = _Wrapper(model, prior_mode).to(x.device).eval()
        macs, params = profile(wrapped, inputs=(x, index_map), verbose=False)

    # Keep the historical convention from the old script:
    # FLOPs ~= MACs, and report in G.
    return macs / (2**30), params / (2**20)


def main() -> None:
    parser = argparse.ArgumentParser("CIDNet params / FLOPs / latency test")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--with_prior", action="store_true", help="Include index_map + prior_mode path in forward/profile")
    parser.add_argument("--prior_mode", type=str, default="attn", choices=["attn", "film", "gate", "glib"])
    parser.add_argument("--max_regions", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, fallback to CPU")

    model = build_model(args, device)
    x = torch.rand(args.batch, 3, args.h, args.w, device=device)
    index_map = None
    if args.with_prior:
        index_map = torch.randint(
            low=0,
            high=args.max_regions,
            size=(args.batch, args.h, args.w),
            device=device,
            dtype=torch.long,
        )

    total, trainable = count_params(model)
    print("=== CIDNet Stats ===")
    print(f"device: {device}")
    print(f"input : (B={args.batch}, C=3, H={args.h}, W={args.w})")
    print(f"prior : {'on' if args.with_prior else 'off'} (mode={args.prior_mode})")
    print(f"params_total    : {total} ({total / 1e6:.4f} M)")
    print(f"params_trainable: {trainable} ({trainable / 1e6:.4f} M)")

    lat = run_inference_time(
        model=model,
        x=x,
        index_map=index_map,
        prior_mode=args.prior_mode,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(f"latency_avg: {lat * 1000.0:.3f} ms / image")

    flops_info = profile_flops(model, x, index_map, args.prior_mode)
    if flops_info is None:
        print("FLOPs: skipped (thop not installed)")
    else:
        flops_g, params_m_legacy = flops_info
        print(f"FLOPs: {flops_g:.4f} G")
        print(f"(thop params, legacy 2^20 unit): {params_m_legacy:.4f} M")


if __name__ == "__main__":
    main()
