"""
Quick dataloader sanity check for lol_v1 with region priors.

Usage (example):
python3 check_dataloader.py \
  --train_root ./datasets/LOLdataset/our485 \
  --label_dir /home/zqh/code/sam2/notebooks/runs/automatic_mask_lolv1_final_packpkl_plus_labels_plus_npz/label_uint8 \
  --crop_size 256

Prints tensor shapes, dtype, min/max/unique count of the index_map, and sample file names.
"""

import argparse
import sys
import torch
from torch.utils.data import DataLoader

# Prevent data.options/LOLdataset argparse from seeing our script args
_argv_backup = sys.argv
sys.argv = [sys.argv[0]]
from data.data import get_lol_training_set
sys.argv = _argv_backup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", type=str, required=True, help="root containing low/high folders")
    p.add_argument("--label_dir", type=str, required=True, help="folder of label pngs")
    p.add_argument("--crop_size", type=int, default=256, help="crop size used in training")
    p.add_argument("--batch_size", type=int, default=1, help="batch size for the check")
    p.add_argument("--num_batches", type=int, default=1, help="number of batches to inspect")
    return p.parse_args()


def main():
    args = parse_args()
    ds = get_lol_training_set(
        args.train_root,
        size=args.crop_size,
        label_dir=args.label_dir,
        use_prior=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    it = iter(loader)
    for i in range(args.num_batches):
        try:
            im1, im2, index_map, file1, file2 = next(it)
        except StopIteration:
            break
        # Shapes: im1/im2 (B,3,H,W), index_map (B,H,W)
        print(f"[batch {i}] low shape:", tuple(im1.shape), "high shape:", tuple(im2.shape), "index_map shape:", tuple(index_map.shape))
        print(f"[batch {i}] index_map dtype:", index_map.dtype, "min:", index_map.min().item(), "max:", index_map.max().item(),
              "unique K:", len(torch.unique(index_map)))
        print(f"[batch {i}] file names low/high:", file1, file2)


if __name__ == "__main__":
    main()
