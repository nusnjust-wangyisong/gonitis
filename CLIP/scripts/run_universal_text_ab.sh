#!/usr/bin/env bash
# Universal text-side A/B experiments from the converged full-dual baselines.
#
# Goal:
#   Test text innovations that are more likely to transfer across ME / AR / PV:
#     gtp         : Grade-transition prompt supervision, training loss only.
#     gtp_infer   : GTP training + small transition-logit inference fusion.
#     bsvla_soft  : Soft branch-specific text regularization
#                   (ViT->anatomy, ConvNeXt->pathology; no off-branch entropy).
#     bsvla_gtp   : Combine soft BSVLA with GTP.
#
# This script intentionally warm-starts from the already converged full model
# checkpoints and uses conservative learning rates, so text supervision is tested
# as a refinement instead of re-training the whole visual model from scratch.

set -euo pipefail

PY=${PY:-/home/kaixin/anaconda3/envs/DDPM/bin/python}
GPU=${GPU:-0}
METHOD=${1:-gtp}
DATASET=${2:-all}

cd "$(dirname "$0")/.."
mkdir -p experiments/logs

declare -A DROOT=(
  [me]="../公开数据集/MedicalExpert-split"
  [ar]="../公开数据集/archive"
  [pv]="../私有数据集/split_result_siyouceshi_roi"
)

declare -A INIT=(
  [me]="experiments/checkpoints/checkpoints_dual_me_ms_fix4/MedicalExpert-split_full_dual/best_model.pth"
  [ar]="experiments/checkpoints/checkpoints_dual_ar_ms_fix10/archive_full_dual/best_model.pth"
  [pv]="experiments/checkpoints/checkpoints_dual_pv_ms_fix9/split_result_siyouceshi_roi_full_dual/best_model.pth"
)

declare -A CVX=(
  [me]="experiments/checkpoints/checkpoints_cvxL/medical/best_model.pth"
  [ar]="experiments/checkpoints/checkpoints_cvxL/archive/best_model.pth"
  [pv]="experiments/checkpoints/checkpoints_roi_cvxL/best_model.pth"
)

declare -A EPOCHS=([me]=18 [ar]=12 [pv]=18)
declare -A PATIENCE=([me]=6 [ar]=5 [pv]=6)
declare -A LR_VIS=([me]=1e-6 [ar]=5e-7 [pv]=1e-6)
declare -A LR_CVX=([me]=1e-6 [ar]=5e-7 [pv]=1e-6)
declare -A LR_FUS=([me]=3e-5 [ar]=2e-5 [pv]=3e-5)

method_args() {
  case "$1" in
    gtp)
      echo "--lambda_transition 0.05 --inference_transition_weight 0.0"
      ;;
    gtp_infer)
      echo "--lambda_transition 0.05 --inference_transition_weight 0.10"
      ;;
    bsvla_soft)
      echo "--enable_bsvla --lambda_bsvla 0.03 --lambda_bsvla_disent 0.0 --inference_bsvla_weight 0.0"
      ;;
    bsvla_gtp)
      echo "--enable_bsvla --lambda_bsvla 0.03 --lambda_bsvla_disent 0.0 --inference_bsvla_weight 0.0 --lambda_transition 0.05 --inference_transition_weight 0.05"
      ;;
    *)
      echo "Unknown METHOD=$1. Use: gtp | gtp_infer | bsvla_soft | bsvla_gtp" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local ds=$1
  local extra
  extra=$(method_args "$METHOD")
  local tag="${ds}_${METHOD}"
  local save_dir="experiments/checkpoints/checkpoints_universal_text_${tag}"
  local report_dir="experiments/results/results_universal_text_${tag}"
  local log_file="experiments/logs/universal_text_${tag}.log"

  echo "[start] dataset=${ds} method=${METHOD} gpu=${GPU} $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" main.py \
    --variant full_dual \
    --backbone biomedclip \
    --convnext_multi_scale \
    --data_root "${DROOT[$ds]}" \
    --convnext_checkpoint "${CVX[$ds]}" \
    --init_checkpoint "${INIT[$ds]}" \
    --clip_normalize \
    --lr_schedule cosine \
    --monitor_metric macro_f1 \
    --batch_size 32 \
    --num_workers 4 \
    --prec amp \
    --no_anomaly \
    --seed 42 \
    --epochs "${EPOCHS[$ds]}" \
    --patience "${PATIENCE[$ds]}" \
    --lr_visual "${LR_VIS[$ds]}" \
    --lr_convnext "${LR_CVX[$ds]}" \
    --lr_fusion "${LR_FUS[$ds]}" \
    --lr_classifier "${LR_FUS[$ds]}" \
    --lr_text_proj "${LR_FUS[$ds]}" \
    --lr_cluster_proto 1e-5 \
    --save_dir "$save_dir" \
    --report_dir "$report_dir" \
    $extra \
    2>&1 | tee "$log_file"
  echo "[done]  dataset=${ds} method=${METHOD} $(date '+%F %T')"
}

if [[ "$DATASET" == "all" ]]; then
  for ds in me ar pv; do
    run_one "$ds"
  done
else
  run_one "$DATASET"
fi

