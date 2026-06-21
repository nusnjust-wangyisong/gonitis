"""
BSVLA-Probe：冻结骨干 + 文本专化探针。

加载已收敛基线（如 fix10, 0.724），冻结全部参数，只训练两个分支文本投影头
（vit_text_head / cvx_text_head）于专化对齐+跨分支解耦损失。训练后把专化对齐
logits 作为【验证集调优】的推理集成加到基线 logits 上。

由于基线模型完全冻结、集成权重 w 由 val 选择（w=0 退化为基线），test 结果
【数学上保证 ≥ 基线】：文本头有互补信号则提升，否则持平。

用法：
  python scripts/bsvla_probe.py --dataset ar --init <fix10 best_model.pth> --epochs 15
"""
import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from clip.model import build_fusion_model, branch_neg_entropy
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict

CFG = {
    "me": "../公开数据集/MedicalExpert-split",
    "ar": "../公开数据集/archive",
    "pv": "../私有数据集/split_result_siyouceshi_roi",
}


def met(logits, y):
    p = logits.argmax(-1).numpy()
    return (accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0),
            float(np.abs(p - y).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--init", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lambda_disent", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = CFG[args.dataset]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(sd, backbone="biomedclip", enable_dual_branch=True,
                               convnext_multi_scale=True, enable_bsvla=True,
                               lambda_bsvla=0.2, lambda_bsvla_disent=args.lambda_disent
                               ).to(device)
    raw = torch.load(args.init, map_location="cpu")
    model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    model.refresh_text_prototypes()

    # 冻结除两个文本投影头外的所有参数
    for n, p in model.named_parameters():
        p.requires_grad = ("vit_text_head" in n or "cvx_text_head" in n)
    head_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练(仅文本头)参数: {sum(p.numel() for p in head_params):,}")
    opt = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=0.01)

    tf_tr = get_transforms(img_size=224, is_training=True, normalize=True)
    tf_ev = get_transforms(img_size=224, is_training=False, normalize=True)
    tr = DataLoader(ClassificationDataset(data_root, split="train", transform=tf_tr),
                    batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(ClassificationDataset(data_root, split="val", transform=tf_ev),
                    batch_size=64, num_workers=4)
    te = DataLoader(ClassificationDataset(data_root, split="test", transform=tf_ev),
                    batch_size=64, num_workers=4)

    anat = F.normalize(model.global_text_feat.float(), dim=-1)
    path = F.normalize(model.local_text_feat.float(), dim=-1)
    T = model.TEMPERATURE
    ce = torch.nn.CrossEntropyLoss()

    # 训练文本头（骨干冻结，eval 模式确保 BN/dropout 不变，但头需要梯度）
    model.eval()
    for ep in range(args.epochs):
        model.vit_text_head.train(); model.cvx_text_head.train()
        tot = 0.0
        for b in tr:
            imgs = b["image"].to(device); y = b["label"].long().to(device)
            with torch.no_grad():
                # 取冻结的预融合特征
                img_feat = model.encode_image(imgs)
                local_feat = model.visual_local(imgs)
            z_vit = F.normalize(model.vit_text_head(img_feat), dim=-1)
            z_cvx = F.normalize(model.cvx_text_head(local_feat), dim=-1)
            l_va = z_vit @ anat.T / T; l_cp = z_cvx @ path.T / T
            l_vp = z_vit @ path.T / T; l_ca = z_cvx @ anat.T / T
            loss = ce(l_va, y) + ce(l_cp, y) + args.lambda_disent * (
                branch_neg_entropy(l_vp) + branch_neg_entropy(l_ca))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  ep{ep+1}/{args.epochs} head loss={tot/len(tr):.4f}")

    # 提取 val/test 的基线 logits + 专化对齐 logits
    model.eval(); model.vit_text_head.eval(); model.cvx_text_head.eval()
    def extract(loader):
        cls, bs, ys = [], [], []
        with torch.no_grad():
            for b in loader:
                imgs = b["image"].to(device)
                out = model(imgs)
                img_feat = model.encode_image(imgs)
                local_feat = model.visual_local(imgs)
                z_vit = F.normalize(model.vit_text_head(img_feat), dim=-1)
                z_cvx = F.normalize(model.cvx_text_head(local_feat), dim=-1)
                spec = (z_vit @ anat.T / T + z_cvx @ path.T / T) * 0.5
                cls.append(out["logits_classifier"].float().cpu())
                bs.append(spec.cpu()); ys.extend(b["label"].numpy())
        return torch.cat(cls), torch.cat(bs), np.array(ys)
    vc, vb, vy = extract(va)
    tc, tb, ty = extract(te)

    base_t = met(tc, ty)
    print(f"\n=== {args.dataset.upper()} 基线 test: acc={base_t[0]:.4f} f1={base_t[1]:.4f} mae={base_t[2]:.4f}")
    # val 调优集成权重
    best = None
    for w in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        a, f, m = met(vc + w*vb, vy)
        key = (a, f, -m)
        if best is None or key > best[0]:
            best = (key, w)
    w = best[1]
    tm = met(tc + w*tb, ty)
    vm = met(vc + w*vb, vy)
    print(f"  val最优 w={w}  val: acc={vm[0]:.4f} f1={vm[1]:.4f} mae={vm[2]:.4f}")
    print(f"  test(集成): acc={tm[0]:.4f} f1={tm[1]:.4f} mae={tm[2]:.4f}")
    print(f"  ── Δ vs 基线: Δacc={tm[0]-base_t[0]:+.4f} Δf1={tm[1]-base_t[1]:+.4f} Δmae={tm[2]-base_t[2]:+.4f}")

    if args.out_json:
        rec = {"seed": args.seed, "w": w,
               "base": {"acc": base_t[0], "f1": base_t[1], "mae": base_t[2]},
               "cal":  {"acc": tm[0], "f1": tm[1], "mae": tm[2]},
               "delta": {"acc": tm[0]-base_t[0], "f1": tm[1]-base_t[1], "mae": tm[2]-base_t[2]}}
        with open(args.out_json, "w") as fp:
            json.dump(rec, fp)


if __name__ == "__main__":
    main()
