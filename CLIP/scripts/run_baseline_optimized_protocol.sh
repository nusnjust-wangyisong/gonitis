#!/usr/bin/env bash
set -euo pipefail

# Optimized Baseline Protocol:
# Each baseline receives the same validation-tuning budget. We keep the existing
# base run as recipe r0 and train two additional recipes:
#   r1: lower backbone LR + higher head LR
#   r2: lower backbone LR + higher head LR + mild class-weighted CE
# Final reporting is selected by validation Macro F1, never by test metrics.

PYTHON="${PYTHON:-/home/kaixin/anaconda3/envs/DDPM/bin/python}"
DATASETS="${DATASETS:-me ar pv}"
MODELS="${MODELS:-resnet mobilenet_v3 vit densenet pvt_v2 convnext_v2 efficientnet spatialmamba}"
EPOCHS="${EPOCHS:-60}"
AR_EPOCHS="${AR_EPOCHS:-40}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-experiments/results/baseline_compare}"
SAVE_ROOT="${SAVE_ROOT:-experiments/checkpoints/baseline_compare}"

run_one() {
  local dataset="$1"
  local model="$2"
  local tag="$3"
  local lr="$4"
  local head_lr="$5"
  local use_cw="$6"
  local epochs="$7"

  local out_dir="${OUT_ROOT}/${dataset}_${model}_seed${SEED}_${tag}"
  if [[ -f "${out_dir}/test_metrics.json" && -f "${out_dir}/val_metrics.json" ]]; then
    echo "[skip] ${dataset}/${model}/${tag} exists"
    return
  fi

  echo "==== optimized dataset=${dataset} model=${model} tag=${tag} lr=${lr} head_lr=${head_lr} cw=${use_cw} epochs=${epochs} ===="
  local args=(
    scripts/train_baseline_backbone.py
    --dataset "${dataset}"
    --model "${model}"
    --epochs "${epochs}"
    --patience "${PATIENCE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --lr "${lr}"
    --head_lr "${head_lr}"
    --run_tag "${tag}"
    --out_root "${OUT_ROOT}"
    --save_root "${SAVE_ROOT}"
  )
  if [[ "${use_cw}" == "1" ]]; then
    args+=(--use_class_weights --class_weight_power 0.5)
  fi
  "${PYTHON}" "${args[@]}"
}

for dataset in ${DATASETS}; do
  ds_epochs="${EPOCHS}"
  if [[ "${dataset}" == "ar" ]]; then
    ds_epochs="${AR_EPOCHS}"
  fi
  for model in ${MODELS}; do
    run_one "${dataset}" "${model}" "r1_low_lr" "5e-5" "1e-3" "0" "${ds_epochs}"
    run_one "${dataset}" "${model}" "r2_cw_low_lr" "2e-5" "1e-3" "1" "${ds_epochs}"
  done
done

"${PYTHON}" scripts/summarize_baseline_optimized_protocol.py \
  --result_root "${OUT_ROOT}" \
  --out_csv "${OUT_ROOT}/optimized_comparison_summary.csv" \
  --out_md "${OUT_ROOT}/optimized_comparison_summary.md" \
  --out_ref_csv "${OUT_ROOT}/optimized_comparison_reference_style.csv" \
  --out_ref_md "${OUT_ROOT}/optimized_comparison_reference_style.md"
