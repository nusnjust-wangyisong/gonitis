# 膝关节 KL 五分类（Knee OA KL Grading）

基于 CLIP / PubMedBERT 的膝关节 X-ray Kellgren-Lawrence 0–4 五分类。在原 OCM-CLIP 方案上，用**同一条管线**（领域预训练主干 BiomedCLIP@224 + ConvNeXt-V2 异构集成）覆盖三个数据集；对唯一"整图未裁"的 private 在管线最前面加一步免训练的膝关节 ROI 前端，把输入对齐成与另两者相同的"单膝已裁"，使三者统一到同一方法、同一分辨率。三个数据集成绩：MedicalExpert 0.866、archive 0.730、private 0.670。

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/report.md`](docs/report.md) | 方法、各创新点消融、最终效果 |
| [`docs/reproduction.md`](docs/reproduction.md) | 环境、数据、运行、验证、常见问题 |

基础 OCM-CLIP 方案的早期文档保留在 `docs/archive/`。

## 目录速览

| 路径 | 内容 |
| --- | --- |
| `main.py` | OCM-CLIP 训练与评估入口 |
| `clip/`、`data/` | 模型与数据管线 |
| `scripts/` | 训练、集成、校准、复现脚本 |
| `PubMedBERT/` | 本地文本编码器权重 |
| `docs/` | 报告与说明 |
| `experiments/results/results_*`、`experiments/checkpoints/checkpoints_*` | 各阶段结果与权重（体积大，默认不入库，按文档重新生成） |

## 快速开始

```bash
pip install -r requirements.txt
python scripts/check_runtime.py              # 环境自检
bash scripts/run_biomedclip_experiments.sh   # 训练 BiomedCLIP 成员
bash scripts/run_final_ensemble.sh           # 生成最终集成结果
```

数据放置、分步说明见 [`docs/reproduction.md`](docs/reproduction.md)。

## 主要脚本

- `scripts/run_biomedclip_experiments.sh`：训练 BiomedCLIP 主干的 bm224 / bm384 模型。
- `scripts/train_convnextv2.py`：训练 ConvNeXt-V2 异构集成成员。
- `scripts/ensemble_logits.py`：多成员软概率集成（含 val 权重校准）。
- `scripts/run_final_ensemble.sh`：复现三个数据集的最终最好结果。
- `scripts/calibrated_eval.py`、`scripts/evidence_stacking_eval.py`：VC-MLC 与 DISAG-ONS 后处理校准。
- `scripts/summarize_results.py`：汇总 `experiments/results/results_*/*/test_metrics.json`。
