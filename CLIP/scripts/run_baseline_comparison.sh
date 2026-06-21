#!/usr/bin/env bash
set -euo pipefail

# Paper comparison baselines:
# ResNet, MobileNetV3, ViT, DenseNet, PVTv2, ConvNeXt-V2, EfficientNet,
# official Spatial-Mamba-T baseline.
#
# Usage examples:
#   bash scripts/run_baseline_comparison.sh
#   DATASETS="me" MODELS="mobilenet_v3 vit" EPOCHS=3 bash scripts/run_baseline_comparison.sh
#   PYTHON=/home/kaixin/anaconda3/envs/DDPM/bin/python bash scripts/run_baseline_comparison.sh

PYTHON="${PYTHON:-/home/kaixin/anaconda3/envs/DDPM/bin/python}"
DATASETS="${DATASETS:-me ar pv}"
MODELS="${MODELS:-resnet mobilenet_v3 vit densenet pvt_v2 convnext_v2 efficientnet spatialmamba}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-experiments/results/baseline_compare}"
SAVE_ROOT="${SAVE_ROOT:-experiments/checkpoints/baseline_compare}"

for dataset in ${DATASETS}; do
  case "${dataset}" in
    ar)
      ds_epochs="${AR_EPOCHS:-40}"
      ;;
    me)
      ds_epochs="${ME_EPOCHS:-${EPOCHS}}"
      ;;
    pv)
      ds_epochs="${PV_EPOCHS:-${EPOCHS}}"
      ;;
    *)
      ds_epochs="${EPOCHS}"
      ;;
  esac

  for model in ${MODELS}; do
    echo "==== dataset=${dataset} model=${model} epochs=${ds_epochs} ===="
    "${PYTHON}" scripts/train_baseline_backbone.py \
      --dataset "${dataset}" \
      --model "${model}" \
      --epochs "${ds_epochs}" \
      --patience "${PATIENCE}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --seed "${SEED}" \
      --out_root "${OUT_ROOT}" \
      --save_root "${SAVE_ROOT}"
  done
done

"${PYTHON}" scripts/summarize_baseline_comparison.py \
  --result_root "${OUT_ROOT}" \
  --out_csv "${OUT_ROOT}/comparison_summary.csv" \
  --out_md "${OUT_ROOT}/comparison_summary.md"
