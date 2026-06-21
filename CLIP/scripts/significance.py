#!/usr/bin/env python3
"""Significance of Ours vs strongest baseline, from saved predictions.
- bootstrap 95% CI for Ours (acc / macro-F1 / QWK / MAE), 10k resamples
- McNemar paired test Ours vs strongest fine-tuned baseline (by test acc)"""
import os, sys, json, csv
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.stats import chi2

BC = "experiments/results/baseline_compare"
NC = 5
RNG = np.random.default_rng(0)
B = 10000


def read_pred(path, label_col="label", pred_col="pred"):
    y, p = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            y.append(int(float(r[label_col]))); p.append(int(float(r[pred_col])))
    return np.array(y), np.array(p)


def mets(y, p):
    return (accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0),
            cohen_kappa_score(y, p, weights="quadratic", labels=list(range(NC))),
            np.abs(p - y).mean())


def strongest_baseline(ds):
    sel = json.load(open(os.path.join(BC, "optimized_selection.json")))
    best, best_acc = None, -1
    for k, v in sel.items():
        if k.startswith(ds + "/") and not k.endswith("/ours"):
            a = v["test"]["accuracy"]
            if a > best_acc:
                best_acc, best = a, (k.split("/")[1], v["run_dir"])
    return best


def mcnemar(y, p1, p2):
    c1 = (p1 == y); c2 = (p2 == y)
    b = int(np.sum(c1 & ~c2))   # ours right, base wrong
    c = int(np.sum(~c1 & c2))   # ours wrong, base right
    if b + c == 0:
        return b, c, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(chi2.sf(stat, 1))


def main():
    out = {}
    for ds, key in [("me", "me"), ("ar", "ar"), ("pv", "pv")]:
        yo, po = read_pred(os.path.join(BC, f"ours_{ds}_predictions.csv"))
        name, run_dir = strongest_baseline(ds)
        yb, pb = read_pred(os.path.join(run_dir, "test_predictions.csv"),
                           label_col="label", pred_col="pred")
        assert np.array_equal(yo, yb), f"{ds}: label order mismatch with baseline"
        n = len(yo)
        # bootstrap CI for Ours
        acc, f1, qwk, mae = mets(yo, po)
        boot = {"acc": [], "f1": [], "qwk": [], "mae": []}
        for _ in range(B):
            idx = RNG.integers(0, n, n)
            a, f, q, m = mets(yo[idx], po[idx])
            boot["acc"].append(a); boot["f1"].append(f); boot["qwk"].append(q); boot["mae"].append(m)
        ci = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
              for k, v in boot.items()}
        b, c, p = mcnemar(yo, po, pb)
        out[ds] = {"n": n, "ours": {"acc": round(acc, 4), "f1": round(f1, 4),
                                    "qwk": round(qwk, 4), "mae": round(mae, 4)},
                   "ours_ci95": ci, "vs_baseline": name,
                   "mcnemar_b_ours>base": b, "mcnemar_c_base>ours": c, "mcnemar_p": round(p, 5)}
        print(f"\n=== {ds} (n={n}) vs {name} ===")
        print(f"  Ours acc={acc:.4f} CI{ci['acc']}  f1={f1:.4f} CI{ci['f1']}  "
              f"qwk={qwk:.4f} CI{ci['qwk']}  mae={mae:.4f} CI{ci['mae']}")
        print(f"  McNemar: ours-only-correct={b}, base-only-correct={c}, p={p:.5f}")
    json.dump(out, open(os.path.join(BC, "significance.json"), "w"), indent=2)
    print("\nsaved", os.path.join(BC, "significance.json"))


if __name__ == "__main__":
    main()
