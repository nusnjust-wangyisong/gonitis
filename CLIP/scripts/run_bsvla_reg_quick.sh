#!/bin/bash
# BSVLA 训练正则化快速验证
# 从已收敛全模型微调，仅加文本监督损失，推理时 inference_bsvla_weight=0 不用文本
# 核心问题：文本对齐梯度回流到骨干，能否改善全模型在 ME/AR 上的性能？

cd "$(dirname "$0")/.."

# ─────────────────────────────────────────────
# ME 快速验证（15 epoch，超低骨干 LR，不破坏已收敛权重）
# ─────────────────────────────────────────────
echo "=== [ME] BSVLA 训练正则化验证 ==="
python main.py \
  --variant full_dual \
  --backbone biomedclip \
  --convnext_multi_scale \
  --data_root ../公开数据集/MedicalExpert-split \
  --convnext_checkpoint experiments/checkpoints/checkpoints_cvxL/medical/best_model.pth \
  --init_checkpoint experiments/checkpoints/checkpoints_dual_me_ms_fix4/MedicalExpert-split_full_dual/best_model.pth \
  --enable_bsvla \
  --lambda_bsvla 0.1 \
  --lambda_bsvla_disent 0.05 \
  --inference_bsvla_weight 0 \
  --lr_visual 3e-6 \
  --lr_convnext 3e-6 \
  --lr_fusion 3e-4 \
  --lr_schedule cosine \
  --batch_size 32 \
  --epochs 15 \
  --output_dir experiments/checkpoints/checkpoints_dual_me_bsvla_reg \
  2>&1 | tee experiments/logs/me_bsvla_reg_quick.log

echo ""
echo "=== [AR] BSVLA 训练正则化验证 ==="
python main.py \
  --variant full_dual \
  --backbone biomedclip \
  --convnext_multi_scale \
  --data_root ../公开数据集/archive \
  --convnext_checkpoint experiments/checkpoints/checkpoints_cvxL/archive/best_model.pth \
  --init_checkpoint experiments/checkpoints/checkpoints_dual_ar_ms_fix10/archive_full_dual/best_model.pth \
  --enable_bsvla \
  --lambda_bsvla 0.1 \
  --lambda_bsvla_disent 0.05 \
  --inference_bsvla_weight 0 \
  --lr_visual 1e-6 \
  --lr_convnext 1e-6 \
  --lr_fusion 3e-4 \
  --lr_schedule cosine \
  --batch_size 32 \
  --epochs 15 \
  --output_dir experiments/checkpoints/checkpoints_dual_ar_bsvla_reg \
  2>&1 | tee experiments/logs/ar_bsvla_reg_quick.log
