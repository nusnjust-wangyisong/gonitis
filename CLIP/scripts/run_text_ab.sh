#!/usr/bin/env bash
# 受控 A/B：文本侧创新点对照实验
# 每个数据集从单尺度 warm-start 出发，完全相同 recipe + seed，跑 3 组：
#   off   = 复现基线（无文本创新）
#   otacp = 文本锚定聚类原型（--text_anchor_cluster）
#   otdd  = 序数文本分布蒸馏（--lambda_text_distill 0.1）
# 每张 GPU 固定一条串行链，避免显存冲突。
set -u
export TOKENIZERS_PARALLELISM=false
PY=/home/kaixin/anaconda3/envs/DDPM/bin/python
cd "$(dirname "$0")/.."

COMMON="--variant full_dual --convnext_multi_scale --epochs 30 --patience 8 \
  --batch_size 64 --num_workers 4 --prec amp --no_anomaly --seed 42 \
  --lr_visual 1e-5 --lr_fusion 1e-4 --lr_classifier 1e-4 \
  --lr_text_proj 5e-5 --lr_cluster_proto 1e-4"

declare -A DROOT=(
  [me]="../公开数据集/MedicalExpert-split"
  [ar]="../公开数据集/archive"
  [pv]="../私有数据集/split_result_siyouceshi_roi"
)
declare -A WARM=(
  [me]="experiments/checkpoints/checkpoints_dual_me_ms/MedicalExpert-split_full_dual/best_model.pth"
  [ar]="experiments/checkpoints/checkpoints_dual_ar_ms/archive_full_dual/best_model.pth"
  [pv]="experiments/checkpoints/checkpoints_dual_pv_ms/split_result_siyouceshi_roi_full_dual/best_model.pth"
)
declare -A EXTRA=(
  [off]=""
  [otacp]="--text_anchor_cluster"
  [otdd]="--lambda_text_distill 0.1 --text_distill_sigma 0.8"
)

run_one() {  # $1=gpu $2=dataset $3=method
  local gpu=$1 ds=$2 m=$3 tag="${2}_${3}"
  echo "[launch gpu$gpu] $tag @ $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py \
    --data_root "${DROOT[$ds]}" \
    --init_ms_from_single "${WARM[$ds]}" \
    $COMMON ${EXTRA[$m]} \
    --save_dir "experiments/checkpoints/checkpoints_textab_${tag}" \
    --report_dir "experiments/results/results_textab_${tag}" \
    > "/tmp/textab_${tag}.log" 2>&1
  echo "[done   gpu$gpu] $tag (exit $?) @ $(date +%H:%M:%S)"
}

# 每张 GPU 一条串行链（AR 最慢，三组分散到 gpu0/1/2 并行先跑）
chain0() { run_one 0 ar off;   run_one 0 me off;   }
chain1() { run_one 1 ar otacp; run_one 1 me otacp; }
chain2() { run_one 2 ar otdd;  run_one 2 me otdd;  }
chain3() { run_one 3 pv off;   run_one 3 pv otacp; run_one 3 pv otdd; }

chain0 & chain1 & chain2 & chain3 &
wait
echo "ALL DONE @ $(date +%H:%M:%S)"
