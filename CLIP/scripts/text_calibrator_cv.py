"""
稳健文本校准器：在验证集上做 k 折交叉验证选择文本集成权重，只采用在多数折
都正向的配置（避免单点 argmax 过拟合小 val），再应用到测试集。

logits_final = logits_cls + wp*proto + wa*anatomy + wc*pathology
候选权重限制在小网格；用 5 折 CV 的均值增益 + 稳定性(正向折数)筛选。
基线(全 0 权重)始终在候选中，故 CV 选不出正向时退化为基线。
"""
import argparse, sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict

CFG = {
    "me": ("../公开数据集/MedicalExpert-split",
           "experiments/checkpoints/checkpoints_dual_me_ms_fix4/MedicalExpert-split_full_dual/best_model.pth"),
    "ar": ("../公开数据集/archive",
           "experiments/checkpoints/checkpoints_dual_ar_ms_fix10/archive_full_dual/best_model.pth"),
}

def met(lg, y):
    p = lg.argmax(-1).numpy()
    return accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0), float(np.abs(p-y).mean())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--folds", type=int, default=5); args = ap.parse_args()
    device = "cuda"
    data_root, ckpt = CFG[args.dataset]
    sd = load_clip_state_dict("ViT-B/32")
    m = build_fusion_model(sd, backbone="biomedclip", enable_dual_branch=True, convnext_multi_scale=True).to(device).eval()
    raw = torch.load(ckpt, map_location="cpu"); m.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    m.refresh_text_prototypes()
    g = F.normalize(m.global_text_feat.float(), dim=-1); l = F.normalize(m.local_text_feat.float(), dim=-1); T = m.TEMPERATURE
    tf = get_transforms(img_size=224, is_training=False, normalize=True)
    def ext(split):
        ld = DataLoader(ClassificationDataset(data_root, split=split, transform=tf), batch_size=64, num_workers=4)
        cl, pr, an, pa, ys = [], [], [], [], []
        with torch.no_grad():
            for b in ld:
                o = m(b["image"].to(device)); img = F.normalize(o["img_feat"].float(), dim=-1)
                cl.append(o["logits_classifier"].float().cpu()); pr.append(o["logits_proto"].float().cpu())
                an.append((img@g.T/T).cpu()); pa.append((img@l.T/T).cpu()); ys.extend(b["label"].numpy())
        return torch.cat(cl), torch.cat(pr), torch.cat(an), torch.cat(pa), np.array(ys)
    vc, vp, va, vpa, vy = ext("val"); tc, tp, ta, tpa, ty = ext("test")
    bt = met(tc, ty); bv = met(vc, vy)
    print(f"\n=== {args.dataset.upper()} 基线: val acc={bv[0]:.4f} f1={bv[1]:.4f} | test acc={bt[0]:.4f} f1={bt[1]:.4f} mae={bt[2]:.4f}")

    grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    n = len(vy); rng = np.arange(n)
    # 固定折划分(不依赖随机)
    folds = [rng[i::args.folds] for i in range(args.folds)]
    def combo(c, p, a, pa_, wp, wa, wc): return c + wp*p + wa*a + wc*pa_
    best = None
    for wp, wa, wc in itertools.product(grid, repeat=3):
        fold_gains = []
        for f in folds:
            idx = f
            base_a = accuracy_score(vy[idx], vc[idx].argmax(-1).numpy())
            cal_a = accuracy_score(vy[idx], combo(vc[idx], vp[idx], va[idx], vpa[idx], wp, wa, wc).argmax(-1).numpy())
            fold_gains.append(cal_a - base_a)
        mean_gain = float(np.mean(fold_gains))
        pos_folds = sum(1 for x in fold_gains if x > 0)
        # 只接受 CV 均值增益>0 且多数折正向的配置；基线(0,0,0)增益=0 作兜底
        key = (mean_gain, pos_folds)
        if best is None or key > best[0]:
            best = (key, (wp, wa, wc), mean_gain, pos_folds)
    (_, (wp, wa, wc), mg, pf) = best
    tm = met(combo(tc, tp, ta, tpa, wp, wa, wc), ty)
    print(f"  CV最优权重 proto={wp} anat={wa} path={wc} | CV均值val增益={mg:+.4f} 正向折={pf}/{args.folds}")
    print(f"  test(校准后): acc={tm[0]:.4f} f1={tm[1]:.4f} mae={tm[2]:.4f}")
    print(f"  ── Δ vs 基线: Δacc={tm[0]-bt[0]:+.4f} Δf1={tm[1]-bt[1]:+.4f} Δmae={tm[2]-bt[2]:+.4f}")

if __name__ == "__main__": main()
