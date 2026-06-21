#!/usr/bin/env bash
# BiomedCLIP 主干实验：在不改动 OCM-CLIP 其余结构的前提下，仅替换视觉塔
#   ① bm224 : OpenAI CLIP 视觉塔 -> BiomedCLIP 视觉塔（领域预训练），224 分辨率
#   ② bm384 : 在 ① 基础上把输入分辨率提升到 384（位置编码插值），强化 KL1-3 细微特征
# 结果写入独立目录，不覆盖现有 openai_clip 的 experiments/results/results_final/experiments/checkpoints/checkpoints_final，便于对比。
set -euo pipefail

# 运行前先激活项目环境（见 README：torch>=2.3 + cuda + transformers + timm + open_clip），
# 或用 PYTHON_BIN 指定解释器，例如：PYTHON_BIN=/path/to/envs/koa/bin/python bash scripts/run_biomedclip_experiments.sh
PYTHON_BIN="${PYTHON_BIN:-python}"
export TOKENIZERS_PARALLELISM=false

REPORT_DIR="${REPORT_DIR:-experiments/results/results_biomedclip}"
SAVE_DIR="${SAVE_DIR:-experiments/checkpoints/checkpoints_biomedclip}"

run_one () {
  local data_root="$1"; local img_size="$2"; local bs="$3"
  local epochs="$4"; local patience="$5"; local tag="$6"
  echo "=================================================================="
  echo ">> BiomedCLIP | $data_root | img=$img_size bs=$bs | tag=$tag"
  echo "=================================================================="
  "${PYTHON_BIN}" main.py \
    --data_root "$data_root" \
    --variant full \
    --backbone biomedclip \
    --img_size "$img_size" \
    --epochs "$epochs" \
    --patience "$patience" \
    --batch_size "$bs" \
    --num_workers 4 \
    --prec amp \
    --save_dir "${SAVE_DIR}_${tag}" \
    --report_dir "${REPORT_DIR}_${tag}" \
    --no_anomaly \
    --lr_visual 3e-5 \
    --lr_fusion 1e-4 \
    --lr_classifier 1e-4 \
    --lr_text_proj 5e-5 \
    --lr_cluster_proto 1e-4
}

# 注：BiomedCLIP 全新视觉塔收敛更慢，lr_visual 用 3e-5（默认 1e-5 会欠拟合），
#     epoch 数也相应增大；MedicalExpert 实测 30 epoch 仍在爬升，故给到 45。
# ---- ① 仅换主干，224 分辨率 ----
run_one ../公开数据集/MedicalExpert-split      224 64  45 12 bm224
run_one ../私有数据集/split_result_siyouceshi  224 64  45 12 bm224
run_one ../公开数据集/archive                  224 128 30 8  bm224

# ---- ② 换主干 + 384 分辨率（384 显存约为 224 的 ~3x，batch 相应调小） ----
run_one ../公开数据集/MedicalExpert-split      384 24  45 12 bm384
run_one ../私有数据集/split_result_siyouceshi  384 24  45 12 bm384
run_one ../公开数据集/archive                  384 48  30 8  bm384

echo "==== 汇总 ===="
"${PYTHON_BIN}" scripts/summarize_results.py --results-dir "${REPORT_DIR}_bm224" || true
"${PYTHON_BIN}" scripts/summarize_results.py --results-dir "${REPORT_DIR}_bm384" || true
