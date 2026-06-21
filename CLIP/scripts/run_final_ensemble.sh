#!/usr/bin/env bash
# 复现三个数据集的最终最佳结果（异构集成：BiomedCLIP + ConvNeXt-V2）。
# 前置：已训好以下 checkpoint / 概率（由 run_biomedclip_experiments.sh 与
# train_convnextv2.py 产出）。运行前请确认对应目录存在。
set -euo pipefail
# 运行前先激活项目环境，或用 PYTHON_BIN 指定解释器
PYTHON_BIN="${PYTHON_BIN:-python}"
export TOKENIZERS_PARALLELISM=false
GPU="${GPU:-0}"

echo "==================== Medical (目标 88.2) ===================="
# 最佳 0.8661：4 模型等权（BiomedCLIP bm224 + ConvNeXtV2-base + base(seed1) + large）
# 注：等权优于 val 校准（Medical val 仅 217 张，校准会过拟合）
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" scripts/ensemble_logits.py \
  --data_root ../公开数据集/MedicalExpert-split --hflip \
  --member experiments/checkpoints/checkpoints_biomedclip_bm224/MedicalExpert-split_full/best_model.pth:biomedclip:224 \
  --member npy:experiments/results/results_cvx/medical \
  --member npy:experiments/results/results_cvx_s1/medical \
  --member npy:experiments/results/results_cvxL/medical

echo "==================== archive (目标 73.5) ===================="
# 最佳 0.7295：5 模型等权（bm224+bm384+cvxBase+cvxBase_seed1+cvxLarge）
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" scripts/ensemble_logits.py \
  --data_root ../公开数据集/archive --hflip \
  --member experiments/checkpoints/checkpoints_biomedclip_bm224/archive_full/best_model.pth:biomedclip:224 \
  --member experiments/checkpoints/checkpoints_biomedclip_bm384/archive_full/best_model.pth:biomedclip:384 \
  --member npy:experiments/results/results_cvx/archive \
  --member npy:experiments/results/results_cvx_s1/archive \
  --member npy:experiments/results/results_cvxL/archive

echo "==================== private ===================="
# 最佳 0.6208：BiomedCLIP bm384 + DISAG-ONS（见 evidence_stacking_eval.py）
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" scripts/evidence_stacking_eval.py \
  --data_root ../私有数据集/split_result_siyouceshi --img_size 384 --backbone biomedclip \
  --checkpoint experiments/checkpoints/checkpoints_biomedclip_bm384/split_result_siyouceshi_full/best_model.pth \
  --output_dir experiments/results/results_bm_evidence/private_dsa_disag_ons \
  --hflip --train_splits train --feature_mode logits_prob_boundary_disagree \
  --calibrator logreg_balanced --c_value 1.0 --proto_weight 0.05 --cluster_weight 0.4 \
  --temperature 0.8 --auto_gate_quantile --auto_ordinal_smoothing --gate_threshold_source train --num_workers 0
