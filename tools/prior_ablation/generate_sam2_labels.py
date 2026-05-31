#!/usr/bin/env python3
"""Generate SAM2 label_uint8 maps for a folder of images.

This is optional. Use it only if the server does not already have SAM2 label maps.
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
    ap.add_argument("--sam2_root", default="/home/zqh/code/sam2")
    ap.add_argument("--checkpoint", default="/home/zqh/code/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    ap.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--points_per_side", type=int, default=48)
    ap.add_argument("--points_per_batch", type=int, default=96)
    ap.add_argument("--pred_iou_thresh", type=float, default=0.65)
    ap.add_argument("--stability_score_thresh", type=float, default=0.88)
    ap.add_argument("--crop_n_layers", type=int, default=1)
    ap.add_argument("--brighten", action="store_true")
    ap.add_argument("--brightness", type=float, default=1.3)
    ap.add_argument("--contrast", type=float, default=1.3)
    ap.add_argument("--sharpness", type=float, default=1.1)
    args = ap.parse_args()

    sam2_root = Path(args.sam2_root)
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    label_dir = out_dir / "label_uint8"
    vis_dir = out_dir / "label_vis"
    meta_dir = out_dir / "meta_json"
    for d in [label_dir, vis_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[sam2] root={sam2_root} device={device}")
    sam2 = build_sam2(args.model_cfg, args.checkpoint, device=device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=0.7,
        crop_n_layers=args.crop_n_layers,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=80,
        use_m2m=True,
    )

    files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
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
