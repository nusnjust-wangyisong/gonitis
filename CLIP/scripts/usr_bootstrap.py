#!/usr/bin/env python3
"""Paired bootstrap test for UMOR vs RSOMR using the saved per-sample predictions.

Reports, per dataset, the observed metric deltas (USR - RSOMR) and their 95%
bootstrap CIs + one-sided p-values over 10k resamples. Honest by construction:
where the guard abstained (usr_pred == rsomr_pred) the delta is exactly 0.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

USR = Path("experiments/results/usr")
NC = 5
B = 10000
RNG = np.random.default_rng(0)


def load(ds):
    y, r, u = [], [], []
    with open(USR / f"{ds}_per_sample.csv") as f:
        for row in csv.DictReader(f):
            y.append(int(row["label"])); r.append(int(row["rsomr_pred"])); u.append(int(row["usr_pred"]))
    return np.array(y), np.array(r), np.array(u)


def metrics(y, p):
    return {
        "acc": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(y, p, weights="quadratic", labels=list(range(NC))),
        "mae": np.abs(p - y).mean(),
    }


def boot(ds):
    y, r, u = load(ds)
    n = len(y)
    obs = {k: metrics(y, u)[k] - metrics(y, r)[k] for k in ["acc", "macro_f1", "qwk", "mae"]}
    deltas = {k: np.empty(B) for k in obs}
    for b in range(B):
        idx = RNG.integers(0, n, n)
        yb, rb, ub = y[idx], r[idx], u[idx]
        mu, mr = metrics(yb, ub), metrics(yb, rb)
        for k in obs:
            deltas[k][b] = mu[k] - mr[k]
    out = {"dataset": ds, "n": int(n), "abstained": bool(np.array_equal(r, u))}
    for k in obs:
        d = deltas[k]
        better = d < 0 if k == "mae" else d > 0  # mae: lower is better
        out[k] = {
            "delta": round(float(obs[k]), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_improve": round(float(np.mean(~better)), 4),  # one-sided: P(no improvement)
        }
    return out


def main():
    datasets = sys.argv[1:] or ["ar", "pv", "me"]
    res = {}
    for ds in datasets:
        res[ds] = boot(ds)
        r = res[ds]
        print(f"\n=== {ds} (n={r['n']}, abstained={r['abstained']}) ===")
        for k in ["acc", "macro_f1", "qwk", "mae"]:
            print(f"  {k:9s} Δ={r[k]['delta']:+.4f}  CI95={r[k]['ci95']}  p={r[k]['p_improve']}")
    (USR / "bootstrap.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nsaved: {USR / 'bootstrap.json'}")


if __name__ == "__main__":
    main()
