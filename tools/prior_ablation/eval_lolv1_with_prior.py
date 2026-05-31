#!/usr/bin/env python3
"""Evaluate a SGD-Net checkpoint on LOLv1 with an optional region-index prior.

This is intended for the rebuttal prior-source ablation. Unlike the legacy
`eval.py`, this script can pass an index_map during inference and computes
PSNR/SSIM/LPIPS-Alex directly with the same metric functions used by measure.py.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

from measure import calculate_psnr, calculate_ssim
from net.CIDNet import CIDNet

try:
    import lpips
except Exception:  # pragma: no cover
    lpips = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def build_model(args: argparse.Namespace) -> CIDNet:
    model = CIDNet(
        fe_type=args.fe_type,
        use_mwfe=args.use_mwfe,
        lca_type=args.lca_type,
        use_cbc=args.use_cbc,
        use_wtconv_i=args.use_wtconv_i,
        use_dwconv_hv=args.use_dwconv_hv,
        pre_lca_film=args.pre_lca_film,
        pre_lca_film_scale=args.pre_lca_film_scale,
        pre_lca_film_bias=args.pre_lca_film_bias,
        pre_lca_film_alpha=args.pre_lca_film_alpha,
        pre_lca_film_branches=args.pre_lca_film_branches,
        pre_lca_film_layers=args.pre_lca_film_layers,
        pre_lca_film_depth_decay=args.pre_lca_film_depth_decay,
        attn_alpha1_init=args.attn_alpha1_init,
        attn_alpha2_init=args.attn_alpha2_init,
        attn_mask_bias_scale1_init=args.attn_mask_bias_scale1_init,
        attn_mask_bias_scale2_init=args.attn_mask_bias_scale2_init,
        attn_mask_bias_scale1_max=args.attn_mask_bias_scale1_max,
        attn_mask_bias_scale2_max=args.attn_mask_bias_scale2_max,
        max_regions=args.max_regions,
    ).cuda().eval()
    # These clamp attributes are not part of state_dict; set them explicitly.
    if hasattr(model, "region_attn"):
        model.region_attn.mask_bias_scale_max = None if args.attn_mask_bias_scale1_max < 0 else args.attn_mask_bias_scale1_max
    if hasattr(model, "region_attn2"):
        model.region_attn2.mask_bias_scale_max = None if args.attn_mask_bias_scale2_max < 0 else args.attn_mask_bias_scale2_max
    return model


def remap_index_map(mask_np: np.ndarray, max_regions: int) -> np.ndarray:
    uniq = np.unique(mask_np)
    remapped = np.searchsorted(uniq, mask_np).astype(np.int64)
    if max_regions > 0:
        k = int(remapped.max()) + 1 if remapped.size else 0
        if k > max_regions:
            counts = np.bincount(remapped.reshape(-1), minlength=k)
            keep_n = max(max_regions - 1, 1)
            keep_ids = np.argsort(counts)[::-1][:keep_n]
            mapping = np.zeros((k,), dtype=np.int64)
            for new_id, old_id in enumerate(keep_ids, start=1):
                mapping[int(old_id)] = int(new_id)
            remapped = mapping[remapped]
    return remapped.astype(np.int64)


def resolve_label(label_dir: Path, image_name: str) -> Path:
    base = Path(image_name).stem
    candidates = [
        label_dir / image_name,
        label_dir / f"{base}_labels.png",
        label_dir / f"{base}_labels.jpg",
        label_dir / f"{base}.png",
        label_dir / f"{base}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find prior label for {image_name} in {label_dir}")


def load_index_map(label_dir: Path, image_name: str, hw: tuple[int, int], max_regions: int) -> torch.Tensor:
    h, w = hw
    p = resolve_label(label_dir, image_name)
    lab = Image.open(p).convert("L")
    if lab.size != (w, h):
        lab = lab.resize((w, h), Image.NEAREST)
    arr = remap_index_map(np.array(lab), max_regions=max_regions)
    return torch.from_numpy(arr).long().unsqueeze(0).cuda(non_blocking=True)


def apply_gt_mean(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    mean_restored = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY).mean()
    mean_target = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).mean()
    return np.clip(pred * (mean_target / (mean_restored + 1e-8)), 0, 255)


def append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", required=True, help="Name written to CSV, e.g. sam2/slic/grid/random/no_prior.")
    ap.add_argument("--low_dir", default="datasets/LOLdataset/eval15/low")
    ap.add_argument("--high_dir", default="datasets/LOLdataset/eval15/high")
    ap.add_argument("--label_dir", default=None, help="Required unless --no_prior is set.")
    ap.add_argument("--no_prior", action="store_true")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--summary_csv", default="output/rebuttal_prior_ablation_lolv1/summary.csv")
    ap.add_argument("--max_regions", type=int, default=16)
    ap.add_argument("--prior_mode", default="attn", choices=["gate", "film", "attn", "glib"])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--use_gt_mean", action="store_true")
    ap.add_argument("--compute_lpips", dest="compute_lpips", action="store_true", default=True)
    ap.add_argument("--no_lpips", dest="compute_lpips", action="store_false")

    # Submitted model defaults.
    ap.add_argument("--fe_type", default="dual_gate", choices=["legacy", "dual_gate"])
    ap.add_argument("--use_mwfe", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    ap.add_argument("--lca_type", default="diem", choices=["cab", "diem", "waveformer"])
    ap.add_argument("--use_cbc", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    ap.add_argument("--use_wtconv_i", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    ap.add_argument("--use_dwconv_hv", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    ap.add_argument("--pre_lca_film", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    ap.add_argument("--pre_lca_film_scale", type=float, default=0.1)
    ap.add_argument("--pre_lca_film_bias", type=float, default=0.1)
    ap.add_argument("--pre_lca_film_alpha", type=float, default=-2.197225)
    ap.add_argument("--pre_lca_film_branches", default="i", choices=["i", "hv", "both"])
    ap.add_argument("--pre_lca_film_layers", default="12", choices=["12", "all"])
    ap.add_argument("--pre_lca_film_depth_decay", type=float, default=0.7)
    ap.add_argument("--attn_alpha1_init", type=float, default=-2.197225)
    ap.add_argument("--attn_alpha2_init", type=float, default=-3.891)
    ap.add_argument("--attn_mask_bias_scale1_init", type=float, default=1.0)
    ap.add_argument("--attn_mask_bias_scale2_init", type=float, default=0.5)
    ap.add_argument("--attn_mask_bias_scale1_max", type=float, default=-1.0)
    ap.add_argument("--attn_mask_bias_scale2_max", type=float, default=-1.0)
    args = ap.parse_args()

    if not args.no_prior and not args.label_dir:
        raise ValueError("--label_dir is required unless --no_prior is set")
    if args.compute_lpips and lpips is None:
        raise RuntimeError("lpips package unavailable; install lpips or set --compute_lpips False by editing the script")

    torch.backends.cudnn.benchmark = True
    model = build_model(args)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.trans.gated = True  # LOLv1 convention used by eval.py

    low_files = sorted([p for p in Path(args.low_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS])
    high_dir = Path(args.high_dir)
    label_dir = Path(args.label_dir) if args.label_dir else None
    output_dir = Path(args.output_dir or f"output/rebuttal_prior_ablation_lolv1/{args.variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    lpips_fn = lpips.LPIPS(net="alex").cuda().eval() if args.compute_lpips else None
    to_tensor = transforms.ToTensor()
    psnrs, ssims, lpipss = [], [], []
    with torch.inference_mode():
        for low_path in tqdm(low_files, desc=args.variant):
            name = low_path.name
            gt_path = high_dir / name
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing GT image: {gt_path}")
            low_img = Image.open(low_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            w, h = low_img.size
            x = to_tensor(low_img).unsqueeze(0).cuda(non_blocking=True)
            if args.no_prior:
                y = model(x ** args.gamma)
            else:
                idx = load_index_map(label_dir, name, (h, w), args.max_regions)
                y = model(x ** args.gamma, index_map=idx, prior_mode=args.prior_mode)
            y = torch.clamp(y, 0, 1)
            pred = (y.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            gt = np.array(gt_img)
            if pred.shape[:2] != gt.shape[:2]:
                pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BICUBIC))
            Image.fromarray(pred).save(output_dir / name)
            pred_eval = apply_gt_mean(pred.astype(np.float32), gt.astype(np.float32)) if args.use_gt_mean else pred
            psnrs.append(float(calculate_psnr(pred_eval, gt)))
            ssims.append(float(calculate_ssim(pred_eval, gt)))
            if lpips_fn is not None:
                ex_p0 = lpips.im2tensor(pred_eval).cuda()
                ex_ref = lpips.im2tensor(gt).cuda()
                lpipss.append(float(lpips_fn.forward(ex_ref, ex_p0).item()))

    row = {
        "variant": args.variant,
        "weights": args.weights,
        "label_dir": "NONE" if args.no_prior else str(label_dir),
        "max_regions": args.max_regions,
        "use_gt_mean": int(args.use_gt_mean),
        "PSNR": f"{np.mean(psnrs):.4f}",
        "SSIM": f"{np.mean(ssims):.4f}",
        "LPIPS": "" if not lpipss else f"{np.mean(lpipss):.4f}",
    }
    print("variant,PSNR,SSIM,LPIPS")
    print(f"{row['variant']},{row['PSNR']},{row['SSIM']},{row['LPIPS']}")
    append_summary(Path(args.summary_csv), row)
    print(f"[summary] appended to {args.summary_csv}")


if __name__ == "__main__":
    main()
