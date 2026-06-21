"""
诊断：ME/AR 上文本专化 logits 到底有没有"可榨取的信号"。

对每个数据集，加载基线 + 训练好的 BSVLA 探针头（这里现训现用），提取：
  base = logits_classifier（基线）
  spec = 专化对齐 logits（ViT→解剖 + ConvNeXt→病理）/2
然后比较三档：
  1) baseline       : argmax(base)
  2) val最优全局w   : base + w*spec, w 由 val 选（迁移性现实值）
  3) oracle全局w    : base + w*spec, w 由 test 直接选（信号上界）
  4) oracle逐类偏置 : base + diag 校准（test 上界，看逐类还有多少空间）
若 oracle 也提不动 → 纯饱和；若 oracle 能提但 val 提不动 → 校准器设计问题。
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from clip.model import build_fusion_model, branch_neg_entropy
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
    ap.add_argument("--epochs", type=int, default=12); args = ap.parse_args()
    device = "cuda"
    data_root, ckpt = CFG[args.dataset]
    sd = load_clip_state_dict("ViT-B/32")
    m = build_fusion_model(sd, backbone="biomedclip", enable_dual_branch=True, convnext_multi_scale=True,
                           enable_bsvla=True, lambda_bsvla=0.2, lambda_bsvla_disent=0.05).to(device)
    raw = torch.load(ckpt, map_location="cpu"); m.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    m.refresh_text_prototypes()
    for n,p in m.named_parameters(): p.requires_grad = ("vit_text_head" in n or "cvx_text_head" in n)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
    anat = F.normalize(m.global_text_feat.float(),dim=-1); path=F.normalize(m.local_text_feat.float(),dim=-1); T=m.TEMPERATURE
    ce = torch.nn.CrossEntropyLoss()
    tf_tr=get_transforms(img_size=224,is_training=True,normalize=True); tf_ev=get_transforms(img_size=224,is_training=False,normalize=True)
    tr=DataLoader(ClassificationDataset(data_root,split="train",transform=tf_tr),batch_size=64,shuffle=True,num_workers=4,drop_last=True)
    m.eval()
    for ep in range(args.epochs):
        m.vit_text_head.train(); m.cvx_text_head.train()
        for b in tr:
            imgs=b["image"].to(device); y=b["label"].long().to(device)
            with torch.no_grad():
                vit=m.encode_image(imgs); cvx=m.visual_local(imgs)
            zv=F.normalize(m.vit_text_head(vit),dim=-1); zc=F.normalize(m.cvx_text_head(cvx),dim=-1)
            loss=ce(zv@anat.T/T,y)+ce(zc@path.T/T,y)+0.05*(branch_neg_entropy(zv@path.T/T)+branch_neg_entropy(zc@anat.T/T))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    def ext(split):
        ld=DataLoader(ClassificationDataset(data_root,split=split,transform=tf_ev),batch_size=64,num_workers=4)
        cl,sp,ys=[],[],[]
        with torch.no_grad():
            for b in ld:
                imgs=b["image"].to(device); o=m(imgs)
                vit=m.encode_image(imgs); cvx=m.visual_local(imgs)
                zv=F.normalize(m.vit_text_head(vit),dim=-1); zc=F.normalize(m.cvx_text_head(cvx),dim=-1)
                cl.append(o["logits_classifier"].float().cpu()); sp.append(((zv@anat.T/T+zc@path.T/T)*0.5).cpu()); ys.extend(b["label"].numpy())
        return torch.cat(cl),torch.cat(sp),np.array(ys)
    vc,vs,vy=ext("val"); tc,ts,ty=ext("test")
    grid=[0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.0,1.5,2.0,3.0]
    bt=met(tc,ty)
    print(f"\n=== {args.dataset.upper()} base test: acc={bt[0]:.4f} f1={bt[1]:.4f} mae={bt[2]:.4f}")
    # val最优全局
    bw=max(grid,key=lambda w:met(vc+w*vs,vy)[:2]); tvw=met(tc+bw*ts,ty)
    print(f"  val最优全局 w={bw}: test acc={tvw[0]:.4f} f1={tvw[1]:.4f}  (Δacc={tvw[0]-bt[0]:+.4f})")
    # oracle全局
    ow=max(grid,key=lambda w:met(tc+w*ts,ty)[0]); tow=met(tc+ow*ts,ty)
    print(f"  oracle全局 w={ow}: test acc={tow[0]:.4f} f1={tow[1]:.4f}  (Δacc={tow[0]-bt[0]:+.4f})  ← 全局信号上界")
    # oracle 逐类温度+权重(在test上贪心调每类的spec加权)——看逐类是否有更多空间
    best=tc.clone(); cur=met(best,ty)[0]
    perclass_w=[0.0]*5
    for _ in range(3):
        for c in range(5):
            for w in grid:
                cand=tc.clone(); cand[:,c]=tc[:,c]+w*ts[:,c]
                # 累加已选的其它类
                for cc in range(5):
                    if cc!=c: cand[:,cc]=tc[:,cc]+perclass_w[cc]*ts[:,cc]
                a=met(cand,ty)[0]
                if a>cur: cur=a; perclass_w[c]=w
    cand=tc.clone()
    for cc in range(5): cand[:,cc]=tc[:,cc]+perclass_w[cc]*ts[:,cc]
    toc=met(cand,ty)
    print(f"  oracle逐类 w={perclass_w}: test acc={toc[0]:.4f} f1={toc[1]:.4f}  (Δacc={toc[0]-bt[0]:+.4f})  ← 逐类信号上界")

    # —— 诚实可迁移校准器：val 调优、test 报告 ——
    def boost(logits, w, classes):
        c=logits.clone()
        for ci in classes: c[:,ci]=logits[:,ci]+w*ts[:,ci] if logits is tc else logits[:,ci]+w*vs[:,ci]
        return c
    # (a) 仅 KL1 加权，val 选 w
    bw1=max(grid,key=lambda w:met(boost(vc,w,[1]),vy)[:2]); t1=met(boost(tc,bw1,[1]),ty)
    print(f"  [val调优·仅KL1] w={bw1}: test acc={t1[0]:.4f} f1={t1[1]:.4f}  (Δacc={t1[0]-bt[0]:+.4f} Δf1={t1[1]-bt[1]:+.4f})")
    # (b) 早期带 KL0/1/2 共享一个 w，val 选
    bw012=max(grid,key=lambda w:met(boost(vc,w,[0,1,2]),vy)[:2]); t012=met(boost(tc,bw012,[0,1,2]),ty)
    print(f"  [val调优·KL012] w={bw012}: test acc={t012[0]:.4f} f1={t012[1]:.4f}  (Δacc={t012[0]-bt[0]:+.4f} Δf1={t012[1]-bt[1]:+.4f})")

if __name__=="__main__": main()
