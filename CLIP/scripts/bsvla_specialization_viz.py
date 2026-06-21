"""
BSVLA 语义专化可视化：证明双分支语义分工 + 跨分支解耦。

冻结基线，训练两个文本投影头（对比 λ_disent=0 与 >0），在测试集上计算
"分支 × 文本域" 的 KL 分级准确率 2×2 矩阵：
    ViT→解剖文本、ViT→病理文本、ConvNeXt→解剖文本、ConvNeXt→病理文本
预期：对角线（ViT-解剖、ConvNeXt-病理）高，非对角（off-branch）随解耦增强
趋向随机基线（最大类占比）。输出热图 PNG。

用法：
  python scripts/bsvla_specialization_viz.py --dataset pv --init <ckpt> --out figures/bsvla_spec.png
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from clip.model import build_fusion_model, branch_neg_entropy, BranchTextAlignHead
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict

CFG = {
    "me": "../公开数据集/MedicalExpert-split",
    "ar": "../公开数据集/archive",
    "pv": "../私有数据集/split_result_siyouceshi_roi",
}


def train_heads(model, tr, device, anat, path, T, epochs, lam_disent):
    """新建并训练两个文本头（冻结骨干），返回 (vit_head, cvx_head)。"""
    emb = anat.shape[1]
    vh = BranchTextAlignHead(emb, 0.1).to(device)
    ch = BranchTextAlignHead(emb, 0.1).to(device)
    opt = torch.optim.AdamW(list(vh.parameters()) + list(ch.parameters()), lr=1e-3, weight_decay=0.01)
    ce = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        vh.train(); ch.train()
        for b in tr:
            imgs = b["image"].to(device); y = b["label"].long().to(device)
            with torch.no_grad():
                vit = model.encode_image(imgs); cvx = model.visual_local(imgs)
            zv = F.normalize(vh(vit), dim=-1); zc = F.normalize(ch(cvx), dim=-1)
            loss = ce(zv@anat.T/T, y) + ce(zc@path.T/T, y)
            if lam_disent > 0:
                loss = loss + lam_disent * (branch_neg_entropy(zv@path.T/T) + branch_neg_entropy(zc@anat.T/T))
            opt.zero_grad(); loss.backward(); opt.step()
    return vh.eval(), ch.eval()


def spec_matrix(model, vh, ch, loader, device, anat, path, T):
    """返回 2×2 准确率矩阵 [[ViT-anat, ViT-path],[Cvx-anat, Cvx-path]] 及标签分布。"""
    va, vp, ca, cp, ys = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            imgs = b["image"].to(device)
            vit = model.encode_image(imgs); cvx = model.visual_local(imgs)
            zv = F.normalize(vh(vit), dim=-1); zc = F.normalize(ch(cvx), dim=-1)
            va.append((zv@anat.T/T).argmax(-1).cpu()); vp.append((zv@path.T/T).argmax(-1).cpu())
            ca.append((zc@anat.T/T).argmax(-1).cpu()); cp.append((zc@path.T/T).argmax(-1).cpu())
            ys.extend(b["label"].numpy())
    ys = np.array(ys)
    g = lambda pred: accuracy_score(ys, torch.cat(pred).numpy())
    chance = np.bincount(ys, minlength=5).max() / len(ys)
    return np.array([[g(va), g(vp)], [g(ca), g(cp)]]), chance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--init", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="figures/bsvla_specialization.png")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(42); np.random.seed(42)
    data_root = CFG[args.dataset]
    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(sd, backbone="biomedclip", enable_dual_branch=True,
                               convnext_multi_scale=True).to(device).eval()
    raw = torch.load(args.init, map_location="cpu")
    model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    model.refresh_text_prototypes()
    for p in model.parameters(): p.requires_grad = False
    anat = F.normalize(model.global_text_feat.float(), dim=-1)
    path = F.normalize(model.local_text_feat.float(), dim=-1); T = model.TEMPERATURE
    tf_tr = get_transforms(img_size=224, is_training=True, normalize=True)
    tf_ev = get_transforms(img_size=224, is_training=False, normalize=True)
    tr = DataLoader(ClassificationDataset(data_root, split="train", transform=tf_tr),
                    batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    te = DataLoader(ClassificationDataset(data_root, split="test", transform=tf_ev),
                    batch_size=64, num_workers=4)

    mats = {}
    chance = None
    for lam in [0.0, 0.2]:
        vh, ch = train_heads(model, tr, device, anat, path, T, args.epochs, lam)
        M, chance = spec_matrix(model, vh, ch, te, device, anat, path, T)
        mats[lam] = M
        print(f"λ_disent={lam}: ViT[anat={M[0,0]:.3f} path={M[0,1]:.3f}] "
              f"Cvx[anat={M[1,0]:.3f} path={M[1,1]:.3f}]  (随机基线={chance:.3f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, lam in zip(axes, [0.0, 0.2]):
        M = mats[lam]
        im = ax.imshow(M, cmap="YlOrRd", vmin=chance, vmax=1.0)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Anatomy text", "Pathology text"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["ViT branch", "ConvNeXt branch"])
        ax.set_title(f"KL accuracy by branch×text  (λ_disent={lam})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        color="black", fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"BSVLA semantic specialization ({args.dataset.upper()}, chance={chance:.3f})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
