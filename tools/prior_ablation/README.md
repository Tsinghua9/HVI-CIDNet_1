# LOLv1 Prior-source Ablation for ACM MM Rebuttal

Reviewer concern: semantic injection may rely on SAM/SAM2 mask accuracy. The cleanest rebuttal experiment is to keep **architecture, loss, schedule, and K fixed**, and change only the region-index-map source.

## Recommended minimal table

| Variant | Meaning | Why included |
|---|---|---|
| `sam2` | Submitted SAM2 labels | main method / expected best prior |
| `sam1` | SAM1 labels | recognized alternative foundation segmentation model |
| `random` | fixed random Voronoi masks | directly answers random-mask concern |
| `no_prior` | no region prior | shows whether region injection helps |

`SLIC` and `grid` are still supported by the scripts, but they are fallback/extra controls. For the one-page rebuttal, `SAM2/SAM1/Random/No prior` is enough and easier to explain.

## Server command for your AutoDL paths

Assumed paths:

```text
Dataset root:      /root/autodl-tmp/LOLdataset
SAM2 train labels: /root/autodl-tmp/lolv1_label_uint8
```

Run full-budget ablation if time allows:

```bash
cd /path/to/HVI-CIDNet_1
git checkout ablation && git pull origin ablation
conda activate CIDNet

EPOCHS=1500 \
SNAPSHOTS=100 \
VARIANTS="sam2 sam1 random no_prior" \
LOL_ROOT=/root/autodl-tmp/LOLdataset \
SAM2_TRAIN_LABEL_DIR=/root/autodl-tmp/lolv1_label_uint8 \
RUN_SAM2_EVAL_GEN=1 \
RUN_SAM1_TRAIN_GEN=1 \
RUN_SAM1_EVAL_GEN=1 \
SAM2_ROOT=/root/autodl-tmp/sam2 \
SAM1_ROOT=/root/autodl-tmp/segment-anything \
SAM1_CKPT=/root/autodl-tmp/sam_vit_h_4b8939.pth \
bash tools/prior_ablation/run_lolv1_prior_ablation.sh
```

If the server does not have SAM1 code/checkpoint, either copy/install SAM1, or replace `sam1` with `slic` as a weaker classical segmentation-prior fallback:

```bash
VARIANTS="sam2 slic random no_prior" ... bash tools/prior_ablation/run_lolv1_prior_ablation.sh
```

If time is tight, use the submitted SAM2 checkpoint and train only alternatives:

```bash
SAM2_EXISTING_CKPT=/path/to/epoch_1300.pth \
VARIANTS="sam2 sam1 random no_prior" \
... bash tools/prior_ablation/run_lolv1_prior_ablation.sh
```

Results are appended to:

```text
output/rebuttal_prior_ablation_lolv1/summary.csv
```

The evaluation script passes the corresponding `index_map` during inference. The legacy `eval.py` does not do this, so use `tools/prior_ablation/eval_lolv1_with_prior.py` for this ablation.

## Training parameters

The wrapper uses the LOLv1 best-run settings from `best_metrics_lol_v1_2026-02-23_14-39-20.md`: AdamW, lr 2e-4, batch 8, crop 256, HVI/L1/D/E/P = 1.0/1.0/0.5/50/0.01, GTMean-RGB on, CCL weight 0.5, `fe_type=dual_gate`, `lca_type=diem`, `pre_lca_film=True`, `max_regions=16`, and the same attention initialization/clamp settings.
