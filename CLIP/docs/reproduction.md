# 复现说明

环境、数据、运行与验证。方法与消融见 [`report.md`](report.md)。

## 0. 推荐配置

- 操作系统：Linux（在 Ubuntu 上验证）
- Python：3.8–3.10
- PyTorch：>= 2.3，搭配 CUDA 12.x
- GPU：显存 >= 24 GB 可跑 224 分辨率；384 建议 >= 32 GB（或减小 batch size）
- 磁盘：模型权重与中间产物合计约 10–20 GB
- 网络：首次运行联网下载 BiomedCLIP 与 ConvNeXt-V2 预训练权重（约 1.5 GB）；离线见 §3、§6

参考耗时（单张 L40）：单个数据集训练 5–30 分钟；集成评估 1–3 分钟。

## 1. 环境准备

主路径（conda）：

```bash
conda create -n koa python=3.10 -y
conda activate koa
pip install -r requirements.txt
```

备选（venv + pip）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

包名对应：`open_clip` → `open_clip_torch`，`cv2` → `opencv-python`，`PIL` → `pillow`，`sklearn` → `scikit-learn`。
`conda activate` 报错时先 `conda init bash` 再重开终端，或直接用 venv。

## 2. 环境自检

```bash
python scripts/check_runtime.py
```

列出各依赖版本、CUDA 是否可用、GPU 数量。出现 MISSING 按提示补装。

## 3. 数据准备

三个数据集均按 `train/val/test/0..4`（每个 KL 等级一个子目录）组织，默认放在项目上一级：

```
<上级目录>/
  公开数据集/
    MedicalExpert-split/{train,val,test}/{0,1,2,3,4}/
    archive/{train,val,test}/{0,1,2,3,4}/
  私有数据集/
    split_result_siyouceshi/{train,val,test}/{0,1,2,3,4}/
CLIP/
  PubMedBERT/        # 已随项目提供的本地文本编码器权重
```

放到别处时用脚本的 `--data_root` 指定。校验：确认每个 split 下 `0..4` 子目录存在且非空。
预训练权重：BiomedCLIP / ConvNeXt-V2 首次运行自动下载并缓存到 `~/.cache`；PubMedBERT 已本地提供，无需联网。

## 4. 运行

进入 `CLIP/` 目录，激活环境后：

第一步，训练 BiomedCLIP 成员（bm224 / bm384）：

```bash
bash scripts/run_biomedclip_experiments.sh
```

要点：视觉层学习率 `--lr_visual 3e-5`。分辨率统一用 `--img_size 224`——已裁的 MedicalExpert/archive 本就用 224；private 经下面的 ROI 前端裁到单膝后也用 224（未加 ROI 的原始 private 才需要 384，见 report.md §2.6/§3.7）。

第二步，训练 ConvNeXt-V2 成员（以 archive base 为例；多 seed 改 `--seed` 与输出目录名，large 把 `--model` 改为 `convnextv2_large.fcmae_ft_in22k_in1k`）：

```bash
python scripts/train_convnextv2.py \
  --data_root ../公开数据集/archive --model convnextv2_base.fcmae_ft_in22k_in1k \
  --img_size 224 --epochs 40 --patience 10 --batch_size 64 --lr 5e-5 \
  --save_dir experiments/checkpoints/checkpoints_cvx/archive --out_dir experiments/results/results_cvx/archive
```

训练结束会在 `out_dir` 保存 `test_probs.npy` / `val_probs.npy` 及标签，供集成读取。

第三步，生成最终集成结果：

```bash
bash scripts/run_final_ensemble.sh
```

集成脚本也可单独调用，例如 archive 的 5 模型等权：

```bash
python scripts/ensemble_logits.py --data_root ../公开数据集/archive --hflip \
  --member experiments/checkpoints/checkpoints_biomedclip_bm224/archive_full/best_model.pth:biomedclip:224 \
  --member experiments/checkpoints/checkpoints_biomedclip_bm384/archive_full/best_model.pth:biomedclip:384 \
  --member npy:experiments/results/results_cvx/archive \
  --member npy:experiments/results/results_cvx_s1/archive \
  --member npy:experiments/results/results_cvxL/archive
```

`--member` 格式为 `checkpoint:backbone:img_size` 或 `npy:<目录>`。加 `--calibrate` 会在验证集搜权重（通常等权更稳）。
显存不足时减小 `--batch_size`；多卡用 `CUDA_VISIBLE_DEVICES=<id>` 指定。

复现早期 KL1–3 配置（archive，Logit Adjustment）：

```bash
python main.py --data_root ../公开数据集/archive --variant full --backbone biomedclip \
  --img_size 224 --epochs 30 --batch_size 128 --lr_visual 3e-5 --logit_adjust_tau 0.5 \
  --save_dir experiments/checkpoints/checkpoints_la/archive --report_dir experiments/results/results_la
```

复现 private 的最终方案（ROI 前端 + 与公开数据集相同的异构集成，report.md §2.6/§3.7/§4.3）：

```bash
# 1) 生成 ROI 裁剪版数据集（免训练，确定性；落盘到 ../私有数据集/split_result_siyouceshi_roi）
python -c "import glob,os;from PIL import Image;from scripts.roi_crop import crop_knee_roi; \
S='../私有数据集/split_result_siyouceshi';D=S+'_roi'; \
[ (os.makedirs(f'{D}/{sp}/{c}',exist_ok=True), \
   [crop_knee_roi(Image.open(f)).convert('L').save(f'{D}/{sp}/{c}/'+os.path.basename(f)) \
    for f in glob.glob(f'{S}/{sp}/{c}/*.png')]) \
  for sp in ['train','val','test'] for c in '01234']"
# 也可直接 python scripts/roi_crop.py 先在 /tmp 看裁剪效果可视化

# 2) 用同一管线在裁剪版上训练（结构不改，仅 --data_root 指向 _roi，统一 224）
ROI=../私有数据集/split_result_siyouceshi_roi
python main.py --data_root $ROI --variant full --backbone biomedclip --img_size 224 \
  --epochs 45 --patience 12 --batch_size 64 --prec amp --lr_visual 3e-5 \
  --save_dir experiments/checkpoints/checkpoints_roi_bm224 --report_dir experiments/results/results_roi_bm224 --no_anomaly
python scripts/train_convnextv2.py --data_root $ROI --img_size 224 --epochs 30 --patience 8 \
  --model convnextv2_base.fcmae_ft_in22k_in1k --hflip \
  --save_dir experiments/checkpoints/checkpoints_roi_cvx --out_dir experiments/results/results_cvx_roi/private
python scripts/train_convnextv2.py --data_root $ROI --img_size 224 --epochs 30 --patience 8 \
  --model convnextv2_large.fcmae_ft_in22k_in1k --hflip --batch_size 32 \
  --save_dir experiments/checkpoints/checkpoints_roi_cvxL --out_dir experiments/results/results_cvx_roi_L/private

# 3) 验证集选配置、测试集只报一次（枚举等权子集，按 val acc 选）
python scripts/private_unified_ensemble.py --data_root $ROI --member_set roi_full
```

val 选出 BiomedCLIP@224 + ConvNeXt-V2-base + large 等权，test accuracy ≈ 0.67（report.md §4.3）。

生成 Grad-CAM 关节区注意力热力图（两种主干都支持，每行一个类、左原图右叠加）：

```bash
# ConvNeXt-V2（最后一个卷积 stage 上做标准 Grad-CAM）
python scripts/gradcam_eval.py --backbone convnext \
  --checkpoint experiments/checkpoints/checkpoints_roi_cvx/best_model.pth --img_size 224 \
  --data_root ../私有数据集/split_result_siyouceshi_roi --out_dir figures/gradcam/cvx_roi
# BiomedCLIP（最后一个 Transformer block 的 norm1 上做 token 级 Grad-CAM）
python scripts/gradcam_eval.py --backbone biomedclip \
  --checkpoint experiments/checkpoints/checkpoints_roi_bm224/split_result_siyouceshi_roi_full/best_model.pth --img_size 224 \
  --data_root ../私有数据集/split_result_siyouceshi_roi --out_dir figures/gradcam/bm_roi
```

把 `--data_root` 换成未裁的 `split_result_siyouceshi` 并用对应未裁 checkpoint，可得"未裁 vs ROI"注意力对比：裁剪后关注收敛到关节线/骨赘，不再蹭到无关骨干。`--num_per_class` 调每类张数，`--target true` 用真实类（默认用预测类）反传。

## 5. 结果校验

- 单模型训练结束打印并保存 `<report_dir>/<dataset>_full/test_metrics.json`（accuracy / macro_f1 / mae / 每类指标 / 混淆矩阵）。
- 集成脚本在终端打印每个成员与集成后的 accuracy / Macro F1 / MAE / 每类召回。
- 参考量级：archive 异构集成约 0.73、MedicalExpert 约 0.866、private（ROI + 异构集成）约 0.67（见 report.md 第 4 节）。0.5 个百分点以内差异属正常波动，不必追求逐位一致。

