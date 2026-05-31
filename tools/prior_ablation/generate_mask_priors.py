#!/usr/bin/env python3
"""Generate non-SAM mask priors for SGD-Net rebuttal ablations.

The output label files follow the dataset loader convention: <stem>_labels.png.
Each label is an integer region-index map; data/LOLdataset.py will remap it
again to the configured max_regions during training.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def stable_seed(name: str, base_seed: int) -> int:
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) + int(base_seed)) % (2**32 - 1)


def remap_to_contiguous(label: np.ndarray, max_regions: int) -> np.ndarray:
    label = np.asarray(label)
    uniq = np.unique(label)
    remapped = np.searchsorted(uniq, label).astype(np.int64)
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
    return remapped


def make_grid(h: int, w: int, k: int) -> np.ndarray:
    rows = max(1, int(np.floor(np.sqrt(k))))
    cols = max(1, int(np.ceil(k / rows)))
    label = np.zeros((h, w), dtype=np.int64)
    idx = 0
    for r in range(rows):
        y0 = int(round(r * h / rows))
        y1 = int(round((r + 1) * h / rows))
        for c in range(cols):
            if idx >= k:
                break
            x0 = int(round(c * w / cols))
            x1 = int(round((c + 1) * w / cols))
            label[y0:y1, x0:x1] = idx
            idx += 1
    return label


def make_random_voronoi(h: int, w: int, k: int, seed: int) -> np.ndarray:
    """Random but spatially coherent masks; avoids high-frequency pixel noise."""
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, h, size=k)
    xs = rng.integers(0, w, size=k)
    yy, xx = np.indices((h, w))
    # h,w are small in LOL; this dense distance computation is fine.
    dist = (yy[..., None] - ys[None, None, :]) ** 2 + (xx[..., None] - xs[None, None, :]) ** 2
    return dist.argmin(axis=-1).astype(np.int64)


def make_random_pixel(h: int, w: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, size=(h, w), dtype=np.int64)


def make_slic(rgb: np.ndarray, k: int, compactness: float, sigma: float) -> np.ndarray:
    try:
        from skimage.segmentation import slic
    except Exception as exc:  # pragma: no cover - depends on server env
        raise RuntimeError(
            "SLIC prior requires scikit-image. Install it in the training env, e.g. "
            "`pip install scikit-image`, or skip the slic variant."
        ) from exc
    img = rgb.astype(np.float32) / 255.0
    seg = slic(
        img,
        n_segments=k,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
    )
    return seg.astype(np.int64)


def save_label(label: np.ndarray, out_path: Path, max_regions: int) -> None:
    label = remap_to_contiguous(label, max_regions=max_regions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if max_regions <= 255:
        arr = label.astype(np.uint8)
    else:
        arr = label.astype(np.uint16)
    Image.fromarray(arr).save(out_path)


def iter_images(low_dir: Path) -> Iterable[Path]:
    return sorted([p for p in low_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--low_dir", required=True, help="Directory containing low-light images.")
    ap.add_argument("--out_root", required=True, help="Output root; variant subfolders are created inside it.")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["slic", "grid", "random"],
        choices=["slic", "grid", "random", "random_pixel", "single"],
    )
    ap.add_argument("--max_regions", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--slic_compactness", type=float, default=10.0)
    ap.add_argument("--slic_sigma", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    low_dir = Path(args.low_dir)
    out_root = Path(args.out_root)
    files = list(iter_images(low_dir))
    if not files:
        raise RuntimeError(f"No images found in {low_dir}")
    print(f"[generate] n={len(files)}, variants={args.variants}, K={args.max_regions}, out={out_root}")

    for p in tqdm(files, desc="images"):
        rgb = np.array(Image.open(p).convert("RGB"))
        h, w = rgb.shape[:2]
        for variant in args.variants:
            out_path = out_root / variant / f"{p.stem}_labels.png"
            if out_path.exists() and not args.overwrite:
                continue
            seed = stable_seed(p.name + variant, args.seed)
            if variant == "grid":
                label = make_grid(h, w, args.max_regions)
            elif variant == "random":
                label = make_random_voronoi(h, w, args.max_regions, seed)
            elif variant == "random_pixel":
                label = make_random_pixel(h, w, args.max_regions, seed)
            elif variant == "single":
                label = np.zeros((h, w), dtype=np.int64)
            elif variant == "slic":
                label = make_slic(rgb, args.max_regions, args.slic_compactness, args.slic_sigma)
            else:
                raise ValueError(variant)
            save_label(label, out_path, max_regions=args.max_regions)

    print("[done]")


if __name__ == "__main__":
    main()
