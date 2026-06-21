#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
export TOKENIZERS_PARALLELISM=false

COMMON_ARGS=(
  --epochs 30
  --patience 8
  --batch_size 64
  --num_workers 4
  --prec amp
  --save_dir experiments/checkpoints/checkpoints_final
  --report_dir experiments/results/results_final
  --no_anomaly
  --lr_visual 1e-5
  --lr_fusion 1e-4
  --lr_classifier 1e-4
  --lr_text_proj 5e-5
  --lr_cluster_proto 1e-4
)

for variant in exp1 exp2 exp3 exp4 full; do
  "${PYTHON_BIN}" main.py \
    --data_root ../公开数据集/MedicalExpert-split \
    --variant "${variant}" \
    "${COMMON_ARGS[@]}"
done

"${PYTHON_BIN}" main.py --data_root ../私有数据集/split_result_siyouceshi --variant full "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" main.py \
  --data_root ../公开数据集/archive \
  --variant full \
  --epochs 20 \
  --patience 6 \
  --batch_size 128 \
  --num_workers 4 \
  --prec amp \
  --save_dir experiments/checkpoints/checkpoints_final \
  --report_dir experiments/results/results_final \
  --no_anomaly \
  --lr_visual 1e-5 \
  --lr_fusion 1e-4 \
  --lr_classifier 1e-4 \
  --lr_text_proj 5e-5 \
  --lr_cluster_proto 1e-4

"${PYTHON_BIN}" scripts/summarize_results.py --results-dir experiments/results/results_final
