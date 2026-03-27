#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

from PIL import Image, ImageColor, ImageDraw
import numpy as np


def _parse_point(s: str) -> Tuple[int, int]:
    try:
        x_str, y_str = s.split(",")
        return int(x_str), int(y_str)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --point '{s}', expected x,y"
        ) from exc


def _parse_roi(s: str) -> Tuple[int, int, int, int]:
    try:
        x_str, y_str, w_str, h_str = s.split(",")
        x, y, w, h = int(x_str), int(y_str), int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError("w/h must be > 0")
        return x, y, w, h
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --roi '{s}', expected x,y,w,h"
        ) from exc


def _roi_from_point(
    point: Tuple[int, int], box_w: int, box_h: int, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    cx, cy = point
    x0 = cx - box_w // 2
    y0 = cy - box_h // 2
    x0 = max(0, min(x0, img_w - box_w))
    y0 = max(0, min(y0, img_h - box_h))
    x1 = min(img_w, x0 + box_w)
    y1 = min(img_h, y0 + box_h)
    return x0, y0, x1, y1


def _auto_box_size(
    img_w: int,
    img_h: int,
    ratio: float,
    style: str,
    min_box: int,
    n_regions: int,
) -> Tuple[int, int]:
    # Use a slightly smaller ratio when multiple regions are requested.
    effective_ratio = ratio if n_regions <= 1 else ratio * 0.85
    if style == "square":
        rw, rh = effective_ratio, effective_ratio
    elif style == "tall":
        rw, rh = effective_ratio * 0.7, effective_ratio * 1.45
    else:  # wide
        rw, rh = effective_ratio * 1.45, effective_ratio * 0.7

    box_w = max(min_box, int(round(img_w * rw)))
    box_h = max(min_box, int(round(img_h * rh)))
    box_w = min(box_w, max(1, img_w - 1))
    box_h = min(box_h, max(1, img_h - 1))
    return box_w, box_h


def _auto_style_ratio_from_point(
    gray: np.ndarray,
    point: Tuple[int, int],
    base_ratio: float,
) -> Tuple[str, float]:
    h, w = gray.shape
    cx, cy = point
    r = int(max(24, min(min(w, h) * 0.12, 180)))
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    patch = gray[y0:y1, x0:x1]
    if patch.size < 25:
        return "square", base_ratio

    gx = np.zeros_like(patch, dtype=np.float32)
    gy = np.zeros_like(patch, dtype=np.float32)
    gx[:, 1:] = np.abs(patch[:, 1:] - patch[:, :-1])
    gy[1:, :] = np.abs(patch[1:, :] - patch[:-1, :])
    mag = gx + gy

    thr = np.percentile(mag, 75)
    ys, xs = np.where(mag >= thr)
    if xs.size < 20:
        return "square", base_ratio

    var_x = float(np.var(xs)) + 1e-6
    var_y = float(np.var(ys)) + 1e-6
    aspect = (var_x / var_y) ** 0.5
    if aspect > 1.35:
        style = "wide"
    elif aspect < 0.74:
        style = "tall"
    else:
        style = "square"

    edge_density = float(xs.size) / float(patch.size)
    if edge_density > 0.33:
        ratio = base_ratio * 0.85
    elif edge_density < 0.16:
        ratio = base_ratio * 1.15
    else:
        ratio = base_ratio
    ratio = max(0.08, min(0.30, ratio))
    return style, ratio


def _clamp_roi(
    roi: Tuple[int, int, int, int], img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    x0 = max(0, min(x, img_w - 1))
    y0 = max(0, min(y, img_h - 1))
    x1 = max(x0 + 1, min(img_w, x0 + w))
    y1 = max(y0 + 1, min(img_h, y0 + h))
    return x0, y0, x1, y1


def _derive_output_path(input_path: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(input_path))
    if not ext:
        ext = ".png"
    return os.path.join(".", f"{stem}_zoom{ext}")


def _build_rois(
    points: List[Tuple[int, int]],
    rois: List[Tuple[int, int, int, int]],
    box_w: int,
    box_h: int,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for p in points:
        out.append(_roi_from_point(p, box_w, box_h, img_w, img_h))
    for r in rois:
        out.append(_clamp_roi(r, img_w, img_h))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a figure with highlighted ROI and zoomed insets."
    )
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output image path (default: ./<input_stem>_zoom.<ext>)",
    )
    parser.add_argument(
        "--point",
        "-p",
        action="append",
        type=_parse_point,
        default=[],
        help="Point center as x,y (repeatable)",
    )
    parser.add_argument(
        "--roi",
        action="append",
        type=_parse_roi,
        default=[],
        help="ROI as x,y,w,h (repeatable)",
    )
    parser.add_argument(
        "--box-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("W", "H"),
        help="Crop size for each --point (default: auto by image size)",
    )
    parser.add_argument(
        "--auto-box-ratio",
        type=float,
        default=0.18,
        help="Auto box ratio w.r.t image size (default: 0.18)",
    )
    parser.add_argument(
        "--auto-box-style",
        choices=["auto", "square", "tall", "wide"],
        default="auto",
        help="Auto box shape style (default: auto)",
    )
    parser.add_argument(
        "--auto-box-min",
        type=int,
        default=24,
        help="Minimum auto box size in pixels (default: 24)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=3.0,
        help="Zoom scale for insets (default: 3.0)",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=2,
        help="Maximum regions to draw from provided points/rois (default: 2)",
    )
    parser.add_argument(
        "--layout",
        choices=["bottom", "right"],
        default="bottom",
        help="Inset layout (default: bottom)",
    )
    parser.add_argument(
        "--bottom-full-width",
        type=int,
        default=1,
        choices=[0, 1],
        help="For single bottom inset, stretch inset to image width (default: 1)",
    )
    parser.add_argument("--gap", type=int, default=0, help="Gap size (default: 0)")
    parser.add_argument("--padding", type=int, default=0, help="Canvas padding (default: 0)")
    parser.add_argument(
        "--color",
        default="#ffcc00",
        help="Box and connector color (default: #ffcc00)",
    )
    parser.add_argument("--thickness", type=int, default=4, help="Line thickness (default: 4)")
    parser.add_argument(
        "--draw-line",
        type=int,
        default=0,
        choices=[0, 1],
        help="Draw connector lines from ROI to inset (default: 0)",
    )

    args = parser.parse_args()

    if not args.point and not args.roi:
        raise ValueError("Provide at least one region via --point or --roi.")
    if args.box_size is not None and (args.box_size[0] <= 0 or args.box_size[1] <= 0):
        raise ValueError("--box-size values must be > 0")
    if args.zoom <= 0:
        raise ValueError("--zoom must be > 0")
    if args.auto_box_ratio <= 0:
        raise ValueError("--auto-box-ratio must be > 0")
    if args.auto_box_min <= 0:
        raise ValueError("--auto-box-min must be > 0")

    img = Image.open(args.input).convert("RGB")
    img_w, img_h = img.size
    n_req = len(args.point) + len(args.roi)
    if args.box_size is None:
        box_w, box_h = _auto_box_size(
            img_w,
            img_h,
            args.auto_box_ratio,
            args.auto_box_style,
            args.auto_box_min,
            n_req,
        )
    else:
        box_w, box_h = args.box_size

    rois: List[Tuple[int, int, int, int]] = []
    if args.point:
        if args.auto_box_style == "auto":
            gray = np.asarray(img.convert("L"), dtype=np.float32)
            for p in args.point:
                style, ratio = _auto_style_ratio_from_point(gray, p, args.auto_box_ratio)
                bw, bh = _auto_box_size(
                    img_w, img_h, ratio, style, args.auto_box_min, n_req
                )
                rois.append(_roi_from_point(p, bw, bh, img_w, img_h))
        else:
            rois.extend(_build_rois(args.point, [], box_w, box_h, img_w, img_h))
    if args.roi:
        rois.extend(_build_rois([], args.roi, box_w, box_h, img_w, img_h))

    if args.max_regions > 0:
        rois = rois[: args.max_regions]
    if not rois:
        raise ValueError("No valid region to draw.")

    insets = []
    for (x0, y0, x1, y1) in rois:
        crop = img.crop((x0, y0, x1, y1))
        zoom_w = max(1, int((x1 - x0) * args.zoom))
        zoom_h = max(1, int((y1 - y0) * args.zoom))
        inset = crop.resize((zoom_w, zoom_h), Image.Resampling.BICUBIC)
        insets.append(inset)

    if args.layout == "bottom" and len(insets) == 1 and args.bottom_full_width == 1:
        iw, ih = insets[0].size
        target_w = img_w
        target_h = max(1, int(round(ih * (target_w / iw))))
        insets[0] = insets[0].resize((target_w, target_h), Image.Resampling.BICUBIC)

    pad = args.padding
    gap = args.gap
    color = ImageColor.getrgb(args.color)
    t = args.thickness

    if args.layout == "bottom":
        inset_total_w = sum(im.size[0] for im in insets) + gap * (len(insets) - 1)
        inset_max_h = max(im.size[1] for im in insets)
        canvas_w = max(img_w, inset_total_w) + 2 * pad
        canvas_h = img_h + inset_max_h + gap + 2 * pad
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        img_x = (canvas_w - img_w) // 2
        img_y = pad
        canvas.paste(img, (img_x, img_y))

        inset_x = (canvas_w - inset_total_w) // 2
        inset_y = img_y + img_h + gap
        inset_boxes = []
        for inset in insets:
            iw, ih = inset.size
            canvas.paste(inset, (inset_x, inset_y))
            inset_boxes.append((inset_x, inset_y, inset_x + iw, inset_y + ih))
            inset_x += iw + gap
    else:
        inset_max_w = max(im.size[0] for im in insets)
        inset_total_h = sum(im.size[1] for im in insets) + gap * (len(insets) - 1)
        canvas_w = img_w + inset_max_w + gap + 2 * pad
        canvas_h = max(img_h, inset_total_h) + 2 * pad
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        img_x = pad
        img_y = (canvas_h - img_h) // 2
        canvas.paste(img, (img_x, img_y))

        inset_x = img_x + img_w + gap
        inset_y = (canvas_h - inset_total_h) // 2
        inset_boxes = []
        for inset in insets:
            iw, ih = inset.size
            canvas.paste(inset, (inset_x, inset_y))
            inset_boxes.append((inset_x, inset_y, inset_x + iw, inset_y + ih))
            inset_y += ih + gap

    draw = ImageDraw.Draw(canvas)

    for idx, (x0, y0, x1, y1) in enumerate(rois):
        rx0, ry0, rx1, ry1 = x0 + img_x, y0 + img_y, x1 + img_x, y1 + img_y
        draw.rectangle((rx0, ry0, rx1, ry1), outline=color, width=t)
        bx0, by0, bx1, by1 = inset_boxes[idx]
        draw.rectangle((bx0, by0, bx1, by1), outline=color, width=t)

        src_cx = (rx0 + rx1) // 2
        src_cy = (ry0 + ry1) // 2
        if args.layout == "bottom":
            dst_cx = (bx0 + bx1) // 2
            dst_cy = by0
        else:
            dst_cx = bx0
            dst_cy = (by0 + by1) // 2
        if args.draw_line == 1:
            draw.line((src_cx, src_cy, dst_cx, dst_cy), fill=color, width=max(1, t - 1))

    out_path = args.output or _derive_output_path(args.input)
    canvas.save(out_path)
    print(f"Saved: {out_path}")
    print(f"Box size used: {box_w}x{box_h}")


if __name__ == "__main__":
    main()
