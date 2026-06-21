#!/usr/bin/env python3
"""
私有数据集（split_result_siyouceshi）统一异构集成：验证集选配置、测试集只报一次。

把"公开数据集用的同一套异构集成"补全到 private——关键是把之前漏掉的 Logit
Adjustment 成员放进来，并按 val accuracy（等权，不在 val 上搜成员权重以免过拟合
小验证集）选出唯一配置，再报告它在 test 上的成绩。

成员（每个在 val/test 上各前向一次，hflip TTA，概率缓存复用）：
  bm384   : BiomedCLIP@384            （现单模型主力）
  la384   : BiomedCLIP@384 + LA       （test 单模型最高 0.6177）
  ldlla384: BiomedCLIP@384 + LDL+LA
  bm224   : BiomedCLIP@224
  cvx384  : ConvNeXt-V2@384（异构卷积成员，读取预存概率 npy）

用法：
  python scripts/private_unified_ensemble.py --data_root ../私有数据集/split_result_siyouceshi
"""
import argparse
import itertools
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ensemble_logits import member_probs  # noqa: E402


# (name, kind, spec)。kind=ckpt -> (ckpt, backbone, img_size)；kind=npy -> dir
MEMBER_SETS = {
    # 未裁原图（旧管线）
    "orig": [
        ("bm384", "ckpt", ("experiments/checkpoints/checkpoints_biomedclip_bm384/split_result_siyouceshi_full/best_model.pth", "biomedclip", 384)),
        ("la384", "ckpt", ("experiments/checkpoints/checkpoints_abl_la05/private/split_result_siyouceshi_full/best_model.pth", "biomedclip", 384)),
        ("ldlla384", "ckpt", ("experiments/checkpoints/checkpoints_biomedclip_ldlla/split_result_siyouceshi_full/best_model.pth", "biomedclip", 384)),
        ("bm224", "ckpt", ("experiments/checkpoints/checkpoints_biomedclip_bm224/split_result_siyouceshi_full/best_model.pth", "biomedclip", 224)),
        ("cvx384", "npy", "experiments/results/results_cvx/private"),
    ],
    # ROI 裁剪后（同一架构，仅前面多一步 ROI 裁剪）；data_root 须指向 _roi
    "roi": [
        ("bm224", "ckpt", ("experiments/checkpoints/checkpoints_roi_bm224/split_result_siyouceshi_roi_full/best_model.pth", "biomedclip", 224)),
        ("bm384", "ckpt", ("experiments/checkpoints/checkpoints_roi_bm384/split_result_siyouceshi_roi_full/best_model.pth", "biomedclip", 384)),
        ("cvx224", "npy", "experiments/results/results_cvx_roi/private"),
    ],
    # ROI + 与公开数据集(Medical)逐字相同的 4 成员 recipe：
    #   BiomedCLIP@224 + ConvNeXtV2-base + base(seed1) + large，全部 @224 等权
    "roi_full": [
        ("bm224", "ckpt", ("experiments/checkpoints/checkpoints_roi_bm224/split_result_siyouceshi_roi_full/best_model.pth", "biomedclip", 224)),
        ("cvx", "npy", "experiments/results/results_cvx_roi/private"),
        ("cvx_s1", "npy", "experiments/results/results_cvx_roi_s1/private"),
        ("cvxL", "npy", "experiments/results/results_cvx_roi_L/private"),
    ],
}


def metrics(probs, labels):
    preds = probs.argmax(1)
    return (dict(
        acc=float((preds == labels).mean()),
        f1=float(f1_score(labels, preds, average="macro", zero_division=0)),
        mae=float(np.abs(preds - labels).mean()),
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--member_set", choices=list(MEMBER_SETS), default="orig")
    ap.add_argument("--baseline", type=float, default=0.6208, help="对比基准 test acc")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    members = MEMBER_SETS[args.member_set]

    val_p, test_p = {}, {}
    val_y = test_y = None
    for name, kind, spec in members:
        if kind == "npy":
            pv = np.load(os.path.join(spec, "val_probs.npy")); yv = np.load(os.path.join(spec, "val_labels.npy"))
            pt = np.load(os.path.join(spec, "test_probs.npy")); yt = np.load(os.path.join(spec, "test_labels.npy"))
        else:
            ckpt, backbone, img = spec
            pv, yv = member_probs(ckpt, backbone, img, args.data_root, True, args.batch_size, args.num_workers, split="val")
            pt, yt = member_probs(ckpt, backbone, img, args.data_root, True, args.batch_size, args.num_workers, split="test")
        val_y = yv if val_y is None else val_y
        test_y = yt if test_y is None else test_y
        assert np.array_equal(val_y, yv) and np.array_equal(test_y, yt), f"{name} 标签顺序不一致"
        val_p[name], test_p[name] = pv, pt
        m_v, m_t = metrics(pv, yv), metrics(pt, yt)
        print(f"[member] {name:9s} val acc={m_v['acc']:.4f} F1={m_v['f1']:.4f} | test acc={m_t['acc']:.4f} F1={m_t['f1']:.4f} MAE={m_t['mae']:.4f}")

    names = [m[0] for m in members]
    # 枚举所有非空子集，等权平均
    rows = []
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            ev = np.mean([val_p[n] for n in combo], axis=0)
            et = np.mean([test_p[n] for n in combo], axis=0)
            rows.append((combo, metrics(ev, val_y), metrics(et, test_y)))

    # 按 val accuracy 排序（并列时 val F1 次之），选出唯一配置
    rows.sort(key=lambda x: (x[1]["acc"], x[1]["f1"]), reverse=True)
    print("\n" + "=" * 100)
    print("按 val accuracy 排序的前 12 个等权组合（test 仅供观察，选择只看 val）：")
    print(f"{'组合':45s} {'val_acc':>8s} {'val_f1':>7s} | {'test_acc':>8s} {'test_f1':>7s} {'test_mae':>8s}")
    for combo, mv, mt in rows[:12]:
        print(f"{'+'.join(combo):45s} {mv['acc']:8.4f} {mv['f1']:7.4f} | {mt['acc']:8.4f} {mt['f1']:7.4f} {mt['mae']:8.4f}")

    best = rows[0]
    print("\n" + "=" * 100)
    print(f"[val 选出的唯一配置] {'+'.join(best[0])}")
    print(f"  val : acc={best[1]['acc']:.4f} f1={best[1]['f1']:.4f}")
    print(f"  test: acc={best[2]['acc']:.4f} f1={best[2]['f1']:.4f} mae={best[2]['mae']:.4f}  <-- 论文应报这个")
    print(f"  对比基准: test acc={args.baseline}")

    # 保存 val 选出配置的 test 概率，供后续叠加 DISAG-ONS 校准
    out = "experiments/results/results_unified_priv_" + args.member_set
    os.makedirs(out, exist_ok=True)
    et = np.mean([test_p[n] for n in best[0]], axis=0)
    ev = np.mean([val_p[n] for n in best[0]], axis=0)
    np.save(os.path.join(out, "test_probs.npy"), et); np.save(os.path.join(out, "test_labels.npy"), test_y)
    np.save(os.path.join(out, "val_probs.npy"), ev); np.save(os.path.join(out, "val_labels.npy"), val_y)
    print(f"[save] val 选出配置的 val/test 概率已写入 {out}/")


if __name__ == "__main__":
    main()
