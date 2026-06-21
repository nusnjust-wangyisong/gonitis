"""
TPLC：文本原型 Logit 校准（训练无关的文本侧推理增强）。

在已收敛基线模型上，推理时融合：
    logits_final = logits_cls + wa*logits_proto + wb*logits_anatomy + wc*logits_pathology
其中 proto=主KL文本原型，anatomy/pathology=分支特异性文本原型（与 post-fusion
img_feat 对齐）。权重 (wa,wb,wc) 在【验证集】上网格搜索最优，再应用到【测试集】。
w=0 退化为基线，故验证集上保证不劣于基线；只检验增益能否迁移到测试集。

用法：
  python scripts/tplc_eval.py --dataset me|ar|pv
"""
import argparse, sys, os, json, itertools
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
    "pv": ("../私有数据集/split_result_siyouceshi_roi",
           "experiments/checkpoints/checkpoints_dual_pv_ms_fix9/split_result_siyouceshi_roi_full_dual/best_model.pth"),
}


def extract(model, loader, device):
    """返回 logits_cls, logits_proto, logits_anatomy, logits_pathology, labels。"""
    g = F.normalize(model.global_text_feat.float(), dim=-1)   # anatomy [5,512]
    l = F.normalize(model.local_text_feat.float(),  dim=-1)   # pathology [5,512]
    T = model.TEMPERATURE
    cls, proto, anat, path, ys = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["image"].to(device))
            img = F.normalize(out["img_feat"].float(), dim=-1)  # post-fusion
            cls.append(out["logits_classifier"].float().cpu())
            proto.append(out["logits_proto"].float().cpu())
            anat.append((img @ g.T / T).cpu())
            path.append((img @ l.T / T).cpu())
            ys.extend(b["label"].numpy())
    return (torch.cat(cls), torch.cat(proto), torch.cat(anat),
            torch.cat(path), np.array(ys))


def metrics(logits, y):
    p = logits.argmax(-1).numpy()
    return (accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0),
            float(np.abs(p - y).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root, ckpt = CFG[args.dataset]

    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(sd, backbone="biomedclip", enable_dual_branch=True,
                               convnext_multi_scale=True).to(device).eval()
    raw = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    model.refresh_text_prototypes()

    tf = get_transforms(img_size=224, is_training=False, normalize=True)
    loaders = {s: DataLoader(ClassificationDataset(data_root, split=s, transform=tf),
                             batch_size=64, shuffle=False, num_workers=4)
               for s in ["val", "test"]}
    V = extract(model, loaders["val"], device)
    Te = extract(model, loaders["test"], device)
    vc, vp, va, vpa, vy = V
    tc, tp, ta, tpa, ty = Te

    base_v = metrics(vc, vy); base_t = metrics(tc, ty)
    print(f"\n=== {args.dataset.upper()} 基线 ===")
    print(f"  val : acc={base_v[0]:.4f} f1={base_v[1]:.4f} mae={base_v[2]:.4f}")
    print(f"  test: acc={base_t[0]:.4f} f1={base_t[1]:.4f} mae={base_t[2]:.4f}")

    # 在 val 上网格搜索 (wa,wb,wc)，目标：先最大化 acc，平手时最小化 mae
    grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    best = None
    for wa, wb, wc in itertools.product(grid, repeat=3):
        lg = vc + wa*vp + wb*va + wc*vpa
        acc, f1, mae = metrics(lg, vy)
        key = (acc, f1, -mae)
        if best is None or key > best[0]:
            best = (key, (wa, wb, wc), (acc, f1, mae))
    (_, (wa, wb, wc), valm) = best
    print(f"\n  val最优权重: proto={wa} anat={wb} path={wc}")
    print(f"  val(调优后): acc={valm[0]:.4f} f1={valm[1]:.4f} mae={valm[2]:.4f}  "
          f"(Δacc={valm[0]-base_v[0]:+.4f})")

    # 应用到 test
    lt = tc + wa*tp + wb*ta + wc*tpa
    tm = metrics(lt, ty)
    print(f"  test(应用后): acc={tm[0]:.4f} f1={tm[1]:.4f} mae={tm[2]:.4f}")
    print(f"  ── test 增益: Δacc={tm[0]-base_t[0]:+.4f}  Δf1={tm[1]-base_t[1]:+.4f}  "
          f"Δmae={tm[2]-base_t[2]:+.4f}")

    # 也报告单源 proto 的 1-DOF 结果（最稳健、最不易过拟合 val）
    best1 = None
    for w in grid:
        acc, f1, mae = metrics(vc + w*vp, vy)
        key = (acc, f1, -mae)
        if best1 is None or key > best1[0]:
            best1 = (key, w)
    w1 = best1[1]
    t1 = metrics(tc + w1*tp, ty)
    print(f"\n  [单源proto 1-DOF] val最优w={w1}  →  test: acc={t1[0]:.4f} f1={t1[1]:.4f} mae={t1[2]:.4f}  "
          f"(Δacc={t1[0]-base_t[0]:+.4f} Δf1={t1[1]-base_t[1]:+.4f} Δmae={t1[2]-base_t[2]:+.4f})")


if __name__ == "__main__":
    main()
