#!/usr/bin/env python3
"""Generate SAM1 label_uint8 maps for a folder of images.

This provides a recognized alternative segmentation foundation model prior
for the ACM MM rebuttal: SAM2 vs SAM1 vs random/no-prior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance


def enhance(rgb: np.ndarray, brightness: float, contrast: float, sharpness: float) -> np.ndarray:
    img = Image.fromarray(rgb)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return np.array(img)


def make_label_map(masks, h: int, w: int, max_labels: int = 255) -> np.ndarray:
    label_map = np.zeros((h, w), dtype=np.uint16)
    sorted_masks = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
    idx = 1
    for ann in sorted_masks:
        seg = ann["segmentation"].astype(bool)
        if not seg.any():
            continue
        label_map[(label_map == 0) & seg] = idx
        idx += 1
        if idx >= max_labels:
            break
    uniq = np.unique(label_map)
    return np.searchsorted(uniq, label_map).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="Output root; label_uint8/ will be created.")
    ap.add_argument("--sam1_root", default="/home/zqh/code/HVI-CIDNet/segment-anything")
    ap.add_argument("--checkpoint", default="/home/zqh/code/HVI-CIDNet/weights/sam_vit_h_4b8939.pth")
    ap.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--points_per_side", type=int, default=32)
    ap.add_argument("--pred_iou_thresh", type=float, default=0.88)
    ap.add_argument("--stability_score_thresh", type=float, default=0.90)
    ap.add_argument("--crop_n_layers", type=int, default=1)
    ap.add_argument("--min_mask_region_area", type=int, default=80)
    ap.add_argument("--brighten", action="store_true")
    ap.add_argument("--brightness", type=float, default=1.3)
    ap.add_argument("--contrast", type=float, default=1.3)
    ap.add_argument("--sharpness", type=float, default=1.1)
    args = ap.parse_args()

    sam1_root = Path(args.sam1_root)
    if str(sam1_root) not in sys.path:
        sys.path.insert(0, str(sam1_root))
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except Exception as exc:
        raise RuntimeError(
            f"Cannot import SAM1 from {sam1_root}. Set --sam1_root to a segment-anything repo, "
            "or install segment-anything."
        ) from exc

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    label_dir = out_dir / "label_uint8"
    vis_dir = out_dir / "label_vis"
    meta_dir = out_dir / "meta_json"
    for d in [label_dir, vis_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sam1] root={sam1_root} checkpoint={args.checkpoint} device={device}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device=device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=args.min_mask_region_area,
    )

    files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    with torch.inference_mode():
        for p in files:
            rgb = np.array(Image.open(p).convert("RGB"))
            h, w = rgb.shape[:2]
            seg_rgb = enhance(rgb, args.brightness, args.contrast, args.sharpness) if args.brighten else rgb
            masks = generator.generate(seg_rgb)
            lab = make_label_map(masks, h, w)
            Image.fromarray(lab).save(label_dir / f"{p.stem}_labels.png")
            vis = (lab.astype(np.float32) / max(1, int(lab.max())) * 255).astype(np.uint8)
            Image.fromarray(vis).save(vis_dir / f"{p.stem}_labels_vis.png")
            meta = {"image": p.name, "num_masks": len(masks), "num_labels": int(lab.max() + 1), "brighten": bool(args.brighten)}
            (meta_dir / f"{p.stem}.json").write_text(json.dumps(meta, indent=2))
            print("[done]", p.name, meta)


if __name__ == "__main__":
    main()
