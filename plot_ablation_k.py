#!/usr/bin/env python3
"""Dual Y-axis ablation plot for K vs. PSNR/SSIM."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Data
K_LABELS = ["4", "8", "16", "32", "64"]
PSNR = [27.686, 27.876, 28.464, 27.855, 27.685]
SSIM = [0.8694, 0.8712, 0.8850, 0.8724, 0.8697]

# Colors
COLOR_PSNR = "#8A0F0F"  # deep crimson
COLOR_SSIM = "#0B3C5D"  # navy blue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dual-axis ablation chart for semantic region count K.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("/home/zqh/code/HVI-CIDNet_1/output/ablation_k/ablation_k.pdf"),
        help="Output PDF path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    x = np.arange(len(K_LABELS))

    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax2 = ax1.twinx()  # SSIM

    # PSNR
    line_psnr, = ax1.plot(
        x, PSNR,
        color=COLOR_PSNR, linestyle="-", marker="o",
        markersize=8, linewidth=2.5, label="PSNR",
    )

    # SSIM
    line_ssim, = ax2.plot(
        x, SSIM,
        color=COLOR_SSIM, linestyle="--", marker="s",
        markersize=8, linewidth=2.5, label="SSIM",
    )

    # X-axis
    ax1.set_xlabel("Number of Semantic Regions, K", fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(K_LABELS, fontsize=14)

    # Y-axes
    ax1.set_ylabel("PSNR (dB)", fontsize=16, color=COLOR_PSNR)
    ax2.set_ylabel("SSIM", fontsize=16, color=COLOR_SSIM)

    ax1.tick_params(axis="y", labelsize=14, colors=COLOR_PSNR)
    ax2.tick_params(axis="y", labelsize=14, colors=COLOR_SSIM)
    ax1.tick_params(axis="x", labelsize=14)

    ax1.spines["left"].set_color(COLOR_PSNR)
    ax2.spines["right"].set_color(COLOR_SSIM)
    # Grid on PSNR axis
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Legend (upper right)
    legend = ax1.legend(
        [line_psnr, line_ssim],
        ["PSNR", "SSIM"],
        loc="upper right",
        fontsize=14,
        frameon=True,
    )
    legend.get_frame().set_linewidth(0.8)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches="tight")
    print(f"Saved figure to: {args.save_path.resolve()}")


if __name__ == "__main__":
    main()
