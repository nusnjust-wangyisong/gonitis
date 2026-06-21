#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" scripts/calibrated_eval.py \
  --data_root ../公开数据集/MedicalExpert-split \
  --checkpoint experiments/checkpoints/checkpoints_final/MedicalExpert-split_full/best_model.pth \
  --output_dir experiments/results/results_calibrated/MedicalExpert-split_full \
  --batch_size 128 \
  --num_workers 4 \
  --hflip

"${PYTHON_BIN}" scripts/calibrated_eval.py \
  --data_root ../公开数据集/archive \
  --checkpoint experiments/checkpoints/checkpoints_final/archive_full/best_model.pth \
  --output_dir experiments/results/results_calibrated/archive_full \
  --batch_size 256 \
  --num_workers 4 \
  --hflip

"${PYTHON_BIN}" scripts/calibrated_eval.py \
  --data_root ../私有数据集/split_result_siyouceshi \
  --checkpoint experiments/checkpoints/checkpoints_final/split_result_siyouceshi_full/best_model.pth \
  --output_dir experiments/results/results_calibrated/split_result_siyouceshi_full \
  --batch_size 128 \
  --num_workers 4 \
  --hflip

"${PYTHON_BIN}" scripts/summarize_results.py --results-dir experiments/results/results_calibrated
