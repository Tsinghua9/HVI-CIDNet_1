#!/usr/bin/env bash
# Rebuttal ablation for the reviewer concern:
# "Does semantic injection rely on SAM/SAM2 mask quality?"
#
# Recommended quick run on a server:
#   conda activate CIDNet
#   EPOCHS=600 VARIANTS="sam2 slic grid random no_prior" \
#     bash tools/prior_ablation/run_lolv1_prior_ablation.sh
#
# For a full same-budget paper-quality run, set EPOCHS=1500.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

# If CONDA_ENV_NAME is set, use `conda run`; otherwise assume the env is already activated.
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  PY=(conda run -n "$CONDA_ENV_NAME" python)
else
  PY=("${PYTHON_BIN:-python}")
fi

MAX_REGIONS=${MAX_REGIONS:-16}
EPOCHS=${EPOCHS:-600}
SNAPSHOTS=${SNAPSHOTS:-100}
if (( EPOCHS % SNAPSHOTS != 0 )); then
  echo "[warn] EPOCHS=$EPOCHS is not divisible by SNAPSHOTS=$SNAPSHOTS; setting SNAPSHOTS=$EPOCHS so final ckpt is saved."
  SNAPSHOTS=$EPOCHS
fi
BATCH_SIZE=${BATCH_SIZE:-8}
THREADS=${THREADS:-16}
SEED=${SEED:-42}
VARIANTS=${VARIANTS:-"sam2 slic grid random no_prior"}
DRY_RUN=${DRY_RUN:-0}
FORCE_TRAIN=${FORCE_TRAIN:-0}
FORCE_LABELS=${FORCE_LABELS:-0}
RUN_SAM2_TRAIN_GEN=${RUN_SAM2_TRAIN_GEN:-0}
RUN_SAM2_EVAL_GEN=${RUN_SAM2_EVAL_GEN:-0}

TRAIN_LOW_DIR=${TRAIN_LOW_DIR:-datasets/LOLdataset/our485/low}
EVAL_LOW_DIR=${EVAL_LOW_DIR:-datasets/LOLdataset/eval15/low}
EVAL_HIGH_DIR=${EVAL_HIGH_DIR:-datasets/LOLdataset/eval15/high}

LABEL_ROOT=${LABEL_ROOT:-output/rebuttal_prior_ablation_lolv1/labels}
SAVE_ROOT=${SAVE_ROOT:-weights/rebuttal_prior_ablation_lolv1}
RESULT_ROOT=${RESULT_ROOT:-output/rebuttal_prior_ablation_lolv1}
SUMMARY_CSV=${SUMMARY_CSV:-$RESULT_ROOT/summary.csv}

# Existing SAM2 train labels from the submitted experiment. Override on the server if needed.
SAM2_TRAIN_LABEL_DIR=${SAM2_TRAIN_LABEL_DIR:-/home/zqh/code/sam2/notebooks/runs/automatic_mask_lolv1_final_packpkl_plus_labels_plus_npz/label_uint8}
# Eval SAM2 labels are generated if missing and RUN_SAM2_EVAL_GEN=1.
SAM2_EVAL_LABEL_DIR=${SAM2_EVAL_LABEL_DIR:-$LABEL_ROOT/eval/sam2/label_uint8}
SAM2_ROOT=${SAM2_ROOT:-/home/zqh/code/sam2}
SAM2_CKPT=${SAM2_CKPT:-$SAM2_ROOT/checkpoints/sam2.1_hiera_base_plus.pt}

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

contains_variant() {
  local needle="$1"
  for v in $VARIANTS; do
    [[ "$v" == "$needle" ]] && return 0
  done
  return 1
}

mkdir -p "$LABEL_ROOT" "$SAVE_ROOT" "$RESULT_ROOT"
rm -f "$SUMMARY_CSV"

# Generate non-SAM priors. This is fast and deterministic.
NON_SAM_VARIANTS=()
for v in $VARIANTS; do
  case "$v" in
    slic|grid|random|random_pixel|single) NON_SAM_VARIANTS+=("$v") ;;
  esac
done
if (( ${#NON_SAM_VARIANTS[@]} > 0 )); then
  OVERWRITE_FLAG=()
  [[ "$FORCE_LABELS" == "1" ]] && OVERWRITE_FLAG=(--overwrite)
  run_cmd "${PY[@]}" tools/prior_ablation/generate_mask_priors.py \
    --low_dir "$TRAIN_LOW_DIR" --out_root "$LABEL_ROOT/train" \
    --variants "${NON_SAM_VARIANTS[@]}" --max_regions "$MAX_REGIONS" --seed 2026 "${OVERWRITE_FLAG[@]}"
  run_cmd "${PY[@]}" tools/prior_ablation/generate_mask_priors.py \
    --low_dir "$EVAL_LOW_DIR" --out_root "$LABEL_ROOT/eval" \
    --variants "${NON_SAM_VARIANTS[@]}" --max_regions "$MAX_REGIONS" --seed 2026 "${OVERWRITE_FLAG[@]}"
fi

# Optionally generate SAM2 labels if they are not already available on the server.
if contains_variant sam2; then
  if [[ ! -d "$SAM2_TRAIN_LABEL_DIR" ]]; then
    if [[ "$RUN_SAM2_TRAIN_GEN" == "1" ]]; then
      run_cmd "${PY[@]}" tools/prior_ablation/generate_sam2_labels.py \
        --img_dir "$TRAIN_LOW_DIR" --out_dir "$LABEL_ROOT/train/sam2" \
        --sam2_root "$SAM2_ROOT" --checkpoint "$SAM2_CKPT" --brighten
      SAM2_TRAIN_LABEL_DIR="$LABEL_ROOT/train/sam2/label_uint8"
    else
      echo "[error] SAM2_TRAIN_LABEL_DIR not found: $SAM2_TRAIN_LABEL_DIR"
      echo "        Set SAM2_TRAIN_LABEL_DIR=... or RUN_SAM2_TRAIN_GEN=1."
      exit 2
    fi
  fi
  if [[ ! -d "$SAM2_EVAL_LABEL_DIR" ]]; then
    if [[ "$RUN_SAM2_EVAL_GEN" == "1" ]]; then
      run_cmd "${PY[@]}" tools/prior_ablation/generate_sam2_labels.py \
        --img_dir "$EVAL_LOW_DIR" --out_dir "$LABEL_ROOT/eval/sam2" \
        --sam2_root "$SAM2_ROOT" --checkpoint "$SAM2_CKPT" --brighten
      SAM2_EVAL_LABEL_DIR="$LABEL_ROOT/eval/sam2/label_uint8"
    else
      echo "[error] SAM2_EVAL_LABEL_DIR not found: $SAM2_EVAL_LABEL_DIR"
      echo "        Set SAM2_EVAL_LABEL_DIR=... or RUN_SAM2_EVAL_GEN=1."
      exit 2
    fi
  fi
fi

common_train_args=(
  --dataset lol_v1
  --nEpochs "$EPOCHS"
  --snapshots "$SNAPSHOTS"
  --lr 0.0002
  --optim adamw
  --weight_decay 1e-4
  --batchSize "$BATCH_SIZE"
  --cropSize 256
  --threads "$THREADS"
  --seed "$SEED"
  --prior_mode attn
  --pre_lca_film True
  --pre_lca_film_scale 0.1
  --pre_lca_film_bias 0.1
  --pre_lca_film_alpha -2.197225
  --pre_lca_film_branches i
  --pre_lca_film_layers 12
  --pre_lca_film_depth_decay 0.7
  --lca_type diem
  --fe_type dual_gate
  --use_wtconv_i True
  --use_dwconv_hv False
  --use_mwfe True
  --use_cbc True
  --loss_ccl True
  --ccl_weight 0.5
  --loss_gtmean_rgb True
  --loss_gtmean_hvi False
  --gtmean_sigma 0.1
  --max_regions "$MAX_REGIONS"
  --attn_alpha1_init -2.197225
  --attn_alpha2_init -3.891
  --attn_mask_bias_scale1_init 1.0
  --attn_mask_bias_scale2_init 0.5
  --attn_mask_bias_scale1_max -1.0
  --attn_mask_bias_scale2_max -1.0
  --save_dir "$SAVE_ROOT"
)

train_one() {
  local variant="$1"
  local train_label_dir=""
  local eval_label_dir=""
  local use_prior=True
  case "$variant" in
    sam2) train_label_dir="$SAM2_TRAIN_LABEL_DIR"; eval_label_dir="$SAM2_EVAL_LABEL_DIR" ;;
    slic|grid|random|random_pixel|single) train_label_dir="$LABEL_ROOT/train/$variant"; eval_label_dir="$LABEL_ROOT/eval/$variant" ;;
    no_prior) use_prior=False ;;
    *) echo "[error] unknown variant: $variant"; exit 2 ;;
  esac

  local ckpt="$SAVE_ROOT/$variant/epoch_${EPOCHS}.pth"
  if [[ -f "$ckpt" && "$FORCE_TRAIN" != "1" ]]; then
    echo "[skip train] $variant checkpoint exists: $ckpt"
  else
    local train_args=("${common_train_args[@]}" --run_name "$variant" --val_folder "$RESULT_ROOT/train_val/$variant/")
    if [[ "$use_prior" == "True" ]]; then
      train_args+=(--use_region_prior True --prior_label_dir "$train_label_dir")
    else
      train_args+=(--use_region_prior False)
    fi
    run_cmd "${PY[@]}" train.py "${train_args[@]}"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[error] expected checkpoint not found: $ckpt"
    exit 3
  fi

  local eval_args=(
    tools/prior_ablation/eval_lolv1_with_prior.py
    --weights "$ckpt"
    --variant "$variant"
    --low_dir "$EVAL_LOW_DIR"
    --high_dir "$EVAL_HIGH_DIR"
    --output_dir "$RESULT_ROOT/eval_outputs/$variant"
    --summary_csv "$SUMMARY_CSV"
    --max_regions "$MAX_REGIONS"
    --prior_mode attn
    --use_gt_mean
  )
  if [[ "$use_prior" == "True" ]]; then
    eval_args+=(--label_dir "$eval_label_dir")
  else
    eval_args+=(--no_prior)
  fi
  run_cmd "${PY[@]}" "${eval_args[@]}"
}

for variant in $VARIANTS; do
  echo "==================== $variant ===================="
  train_one "$variant"
done

echo "[done] summary: $SUMMARY_CSV"
if [[ -f "$SUMMARY_CSV" ]]; then
  cat "$SUMMARY_CSV"
fi
