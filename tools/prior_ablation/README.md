# LOLv1 Prior-source Ablation for ACM MM Rebuttal

This folder contains scripts for the reviewer concern:

> Semantic injection may rely on SAM/SAM2 segmentation accuracy; evaluate other segmentation/random mask priors.

## Recommended experiment

Run a **same-budget retraining ablation** on LOLv1 with the same architecture/loss/schedule/K and change only the region-index source:

| Variant | Meaning | Rebuttal role |
|---|---|---|
| `sam2` | submitted SAM2 region labels | target/source prior |
| `slic` | SLIC superpixels | alternative non-foundation segmentation prior |
| `grid` | regular grid regions | non-semantic structured prior |
| `random` | fixed random Voronoi regions | random mask prior |
| `no_prior` | no region index map | whether the semantic-prior path helps |

If time is limited, run `EPOCHS=600` for all rows and report it as a same-budget rebuttal ablation. If time allows, set `EPOCHS=1500` to match the submitted training budget.

## Server command

```bash
cd /path/to/HVI-CIDNet_1
conda activate CIDNet

# If SAM2 eval labels are not present, set RUN_SAM2_EVAL_GEN=1.
# If SAM2 train labels are also missing, set RUN_SAM2_TRAIN_GEN=1.
EPOCHS=600 \
VARIANTS="sam2 slic grid random no_prior" \
SAM2_TRAIN_LABEL_DIR=/home/zqh/code/sam2/notebooks/runs/automatic_mask_lolv1_final_packpkl_plus_labels_plus_npz/label_uint8 \
RUN_SAM2_EVAL_GEN=1 \
bash tools/prior_ablation/run_lolv1_prior_ablation.sh
```

Results are appended to:

```text
output/rebuttal_prior_ablation_lolv1/summary.csv
```

The evaluation script passes the corresponding `index_map` at inference time. This is important because the legacy `eval.py` does not pass region priors.

## Quick dry run

```bash
DRY_RUN=1 VARIANTS="slic grid random no_prior" bash tools/prior_ablation/run_lolv1_prior_ablation.sh
```

## Notes for writing rebuttal

- The cleanest claim is: *SAM2 performs best; SLIC/grid/random/no-prior are lower, so the gain is not from arbitrary masks.*
- If random/grid are close to SAM2, still phrase carefully: *the gated residual design is robust to imperfect priors, while meaningful region priors remain preferable.*
- Do not mix a full 1500-epoch SAM2 checkpoint with 600-epoch alternatives in the same main table unless clearly labeled.
