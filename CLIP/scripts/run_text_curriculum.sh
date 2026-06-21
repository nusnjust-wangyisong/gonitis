#!/usr/bin/env bash
# Two-stage text curriculum for the full_dual multi-scale model.
#
# Stage 1 (text):   initialize multi-scale model from the single-scale dual branch
#                   checkpoint, add weak BSVLA + transition text supervision.
# Stage 2 (refine): initialize from stage 1 best model, disable text losses, and
#                   refine only with the main visual KL objective.
#
# Rationale: text supervision shapes the early multi-scale representation, but
# does not keep pulling the final classifier after the visual boundary is learned.

set -euo pipefail

PY=${PY:-/home/kaixin/anaconda3/envs/DDPM/bin/python}
GPU=${GPU:-3}
DATASET=${1:-me}
TEXT_LAMBDA=${TEXT_LAMBDA:-0.005}
TRANS_LAMBDA=${TRANS_LAMBDA:-0.005}
TAG_SUFFIX=${TAG_SUFFIX:-lt005}

cd "$(dirname "$0")/.."
mkdir -p experiments/logs

declare -A DROOT=(
  [me]="../公开数据集/MedicalExpert-split"
  [ar]="../公开数据集/archive"
  [pv]="../私有数据集/split_result_siyouceshi_roi"
)

declare -A SINGLE=(
  [me]="experiments/checkpoints/checkpoints_dual_me_v3/MedicalExpert-split_full_dual/best_model.pth"
  [ar]="experiments/checkpoints/checkpoints_dual_ar_v2/archive_full_dual/best_model.pth"
  [pv]="experiments/checkpoints/checkpoints_dual_pv_v3/split_result_siyouceshi_roi_full_dual/best_model.pth"
)

declare -A CVX=(
  [me]="experiments/checkpoints/checkpoints_cvxL/medical/best_model.pth"
  [ar]="experiments/checkpoints/checkpoints_cvxL/archive/best_model.pth"
  [pv]="experiments/checkpoints/checkpoints_roi_cvxL/best_model.pth"
)

declare -A STAGE1_EPOCHS=([me]=12 [ar]=8 [pv]=12)
declare -A STAGE2_EPOCHS=([me]=18 [ar]=12 [pv]=18)
declare -A STAGE1_PATIENCE=([me]=5 [ar]=4 [pv]=5)
declare -A STAGE2_PATIENCE=([me]=6 [ar]=5 [pv]=6)

declare -A LR_VIS_1=([me]=3e-6 [ar]=1e-6 [pv]=3e-6)
declare -A LR_CVX_1=([me]=3e-6 [ar]=1e-6 [pv]=3e-6)
declare -A LR_FUS_1=([me]=1e-4 [ar]=8e-5 [pv]=1e-4)

declare -A LR_VIS_2=([me]=1e-6 [ar]=5e-7 [pv]=1e-6)
declare -A LR_CVX_2=([me]=1e-6 [ar]=5e-7 [pv]=1e-6)
declare -A LR_FUS_2=([me]=3e-5 [ar]=2e-5 [pv]=3e-5)

run_stage1() {
  local ds=$1
  echo "[stage1:text] dataset=${ds} gpu=${GPU} $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" main.py \
    --variant full_dual \
    --backbone biomedclip \
    --convnext_multi_scale \
    --data_root "${DROOT[$ds]}" \
    --convnext_checkpoint "${CVX[$ds]}" \
    --init_ms_from_single "${SINGLE[$ds]}" \
    --clip_normalize \
    --monitor_metric macro_f1 \
    --lr_schedule cosine \
    --batch_size 32 \
    --num_workers 4 \
    --prec amp \
    --no_anomaly \
    --seed 42 \
    --epochs "${STAGE1_EPOCHS[$ds]}" \
    --patience "${STAGE1_PATIENCE[$ds]}" \
    --lr_visual "${LR_VIS_1[$ds]}" \
    --lr_convnext "${LR_CVX_1[$ds]}" \
    --lr_fusion "${LR_FUS_1[$ds]}" \
    --lr_classifier "${LR_FUS_1[$ds]}" \
    --lr_text_proj "${LR_FUS_1[$ds]}" \
    --lr_cluster_proto 1e-5 \
    --enable_bsvla \
    --lambda_bsvla "$TEXT_LAMBDA" \
    --lambda_bsvla_disent 0.0 \
    --inference_bsvla_weight 0.0 \
    --lambda_transition "$TRANS_LAMBDA" \
    --inference_transition_weight 0.0 \
    --save_dir "experiments/checkpoints/checkpoints_text_curriculum_${ds}_${TAG_SUFFIX}_stage1" \
    --report_dir "experiments/results/results_text_curriculum_${ds}_${TAG_SUFFIX}_stage1" \
    2>&1 | tee "experiments/logs/text_curriculum_${ds}_${TAG_SUFFIX}_stage1.log"
}

run_stage2() {
  local ds=$1
  local init="experiments/checkpoints/checkpoints_text_curriculum_${ds}_${TAG_SUFFIX}_stage1/"
  case "$ds" in
    me) init="${init}MedicalExpert-split_full_dual/best_model.pth" ;;
    ar) init="${init}archive_full_dual/best_model.pth" ;;
    pv) init="${init}split_result_siyouceshi_roi_full_dual/best_model.pth" ;;
  esac
  echo "[stage2:refine] dataset=${ds} init=${init} gpu=${GPU} $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" main.py \
    --variant full_dual \
    --backbone biomedclip \
    --convnext_multi_scale \
    --data_root "${DROOT[$ds]}" \
    --convnext_checkpoint "${CVX[$ds]}" \
    --init_checkpoint "$init" \
    --clip_normalize \
    --monitor_metric macro_f1 \
    --lr_schedule cosine \
    --batch_size 32 \
    --num_workers 4 \
    --prec amp \
    --no_anomaly \
    --seed 42 \
    --epochs "${STAGE2_EPOCHS[$ds]}" \
    --patience "${STAGE2_PATIENCE[$ds]}" \
    --lr_visual "${LR_VIS_2[$ds]}" \
    --lr_convnext "${LR_CVX_2[$ds]}" \
    --lr_fusion "${LR_FUS_2[$ds]}" \
    --lr_classifier "${LR_FUS_2[$ds]}" \
    --lr_text_proj "${LR_FUS_2[$ds]}" \
    --lr_cluster_proto 1e-5 \
    --save_dir "experiments/checkpoints/checkpoints_text_curriculum_${ds}_${TAG_SUFFIX}_stage2" \
    --report_dir "experiments/results/results_text_curriculum_${ds}_${TAG_SUFFIX}_stage2" \
    2>&1 | tee "experiments/logs/text_curriculum_${ds}_${TAG_SUFFIX}_stage2.log"
}

run_stage1 "$DATASET"
run_stage2 "$DATASET"
