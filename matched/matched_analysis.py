#!/usr/bin/env python3
"""Step 2 of the extension: compare all twelve runs at the SAME operating point.

The paper compared runs at their own default rule (argmax), so runs flagged
anywhere from 221 to 428 comments, and it had to remove that difference
statistically (partial correlation). With saved probabilities we can remove it
directly: make every run flag exactly the same number of comments, then compare.

Harmful score for each comment = p_hate + p_offensive (same "harmful" mapping as
the paper). A run "flags K" = its K comments with the highest harmful score.

Questions this answers, each with the paper's number for comparison:
  A. Ranking quality with no threshold at all: AUROC and average precision.
  B. Does the macro-F1 vs recall trade-off survive at matched K?
     (paper: rho = -0.74 raw, -0.17 semi-partial, -0.43 full partial)
  C. At matched K, do runs still flag different comments?  (paper: containment 0.720)
  D. At matched K, coverage of the 1,334 harmful comments: none / some / all 12,
     union, best single run, majority vote.  (paper: 54.5 / 39.4 / 6.1 %)
  E. Recall at a fixed false-positive rate (1%, 2%, 5%).

Note: thresholds for K and for fixed-FPR are chosen on the evaluation set itself.
That is fine for COMPARING runs at an equal operating point (every run gets the
same treatment), but these are not deployable thresholds.

    cd extension
    python matched_analysis.py

Reads extension/probs/*.csv and data/ledger12.csv. Writes extension/results/.
"""
from __future__ import annotations

import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROBS = os.path.join(HERE, "probs")
OUT = os.path.join(HERE, "results")
ARMS = ("main", "soft_uw1", "soft_off", "no_contrastive")
RUNS = [(a, s) for a in ARMS for s in (0, 1, 2)]
K_LIST = (221, 315, 428)      # smallest, near-median and largest flag count in the paper


def auroc(y, s):
    r = rankdata(s); pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def avg_precision(y, s):
    o = np.argsort(-s, kind="mergesort"); yy = y[o]
    tp = np.cumsum(yy); prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def top_k(s, k):
    """indices of the k highest scores; ties broken by row order (stable)."""
    return np.argsort(-s, kind="mergesort")[:k]


def recall_at_fpr(y, s, fpr):
    neg = np.sort(s[y == 0])[::-1]
    n_fp = int(np.floor(fpr * len(neg)))
    thr = neg[n_fp] if n_fp < len(neg) else -np.inf    # flag strictly above the (n_fp+1)-th benign score
    return float((s[y == 1] > thr).mean())


def fleiss(M):
    R = M.shape[1]; n1 = M.sum(1); n0 = R - n1
    Pi = (n1 * (n1 - 1) + n0 * (n0 - 1)) / (R * (R - 1))
    p1 = n1.sum() / (M.shape[0] * R); pe = p1 ** 2 + (1 - p1) ** 2
    return (Pi.mean() - pe) / (1 - pe)


def main() -> int:
    missing = [f"{a}_{s}" for a, s in RUNS if not os.path.exists(os.path.join(PROBS, f"{a}_{s}.csv"))]
    if missing:
        sys.exit(f"missing probability files in {PROBS}: {missing}\nRun save_probabilities.py first.")
    D = {k: pd.read_csv(os.path.join(PROBS, f"{k[0]}_{k[1]}.csv")) for k in RUNS}
    y = D[RUNS[0]]["gold"].to_numpy().astype(int)
    for k, d in D.items():
        assert (d["gold"].to_numpy() == y).all(), f"gold order differs: {k}"
    S = {k: (d["p_hate"] + d["p_offensive"]).to_numpy() for k, d in D.items()}
    A = {k: d[["p_hate", "p_offensive", "p_normal"]].to_numpy().argmax(1) for k, d in D.items()}
    led = pd.read_csv(os.path.join(REPO, "data", "ledger12.csv"))
    F1 = {(r.arm, int(r.seed)): r.test_macro_f1 for r in led.itertuples()}
    pos = y == 1; npos = int(pos.sum())
    os.makedirs(OUT, exist_ok=True)
    print(f"n={len(y)}  harmful={npos}  benign={len(y) - npos}\n")

    # sanity: argmax reproduces the paper's recall
    rows = []
    for k in RUNS:
        flag = A[k] != 2
        row = {"run": f"{k[0]}_{k[1]}", "macro_f1": F1[k],
               "argmax_flags": int(flag.sum()), "argmax_recall": round(float(flag[pos].mean()), 4),
               "auroc": round(auroc(y, S[k]), 4), "avg_precision": round(avg_precision(y, S[k]), 4)}
        for K in K_LIST:
            row[f"recall@{K}"] = round(float(pos[top_k(S[k], K)].sum() / npos), 4)
        for f in (0.01, 0.02, 0.05):
            row[f"recall@fpr{int(f * 100)}%"] = round(recall_at_fpr(y, S[k], f), 4)
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "per_run.csv"), index=False)
    print("Per run (argmax columns should equal the paper's runs12.csv):")
    print(T.to_string(index=False), "\n")

    # A + B + E: does the trade-off survive once the operating point is equal?
    x = T["macro_f1"].to_numpy()
    print("Spearman(macro-F1, metric)   [paper, argmax recall: rho = -0.739, p = 0.006]")
    summ = []
    for col in ["argmax_recall", "auroc", "avg_precision"] + [f"recall@{K}" for K in K_LIST] \
               + ["recall@fpr1%", "recall@fpr2%", "recall@fpr5%"]:
        sp = spearmanr(x, T[col])
        summ.append({"metric": col, "rho": round(sp.statistic, 3), "p": round(sp.pvalue, 4),
                     "min": T[col].min(), "max": T[col].max()})
        print(f"  {col:16s} rho = {sp.statistic:+.3f}  p = {sp.pvalue:.4f}   "
              f"range {T[col].min():.3f}-{T[col].max():.3f}")
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "tradeoff_matched.csv"), index=False)

    # C + D: item-level agreement and coverage at each matched K
    print("\nItem-level at matched flag count  [paper, argmax: containment 0.720, "
          "harmful none/some/all = 54.5/39.4/6.1 %, union .455, best .266, majority .158]")
    cov_rows = []
    for K in K_LIST:
        M = np.zeros((len(y), len(RUNS)), int)
        for j, k in enumerate(RUNS):
            M[top_k(S[k], K), j] = 1
        sets = [set(np.flatnonzero(M[:, j])) for j in range(len(RUNS))]
        shared = [len(a & b) / K for a, b in itertools.combinations(sets, 2)]  # = containment = overlap share at equal K
        V = M.sum(1); Vp = V[pos]
        best = max(M[pos, j].mean() for j in range(len(RUNS)))
        maj = V >= 7
        r = {"K": K, "overlap_mean": round(np.mean(shared), 3), "overlap_min": round(min(shared), 3),
             "overlap_max": round(max(shared), 3), "fleiss": round(fleiss(M), 3),
             "harm_none_%": round((Vp == 0).mean() * 100, 1),
             "harm_some_%": round(((Vp > 0) & (Vp < 12)).mean() * 100, 1),
             "harm_all_%": round((Vp == 12).mean() * 100, 1),
             "union": round((Vp >= 1).mean(), 3), "best_single": round(best, 3),
             "majority_recall": round(maj[pos].mean(), 3), "majority_fpr": round(maj[~pos].mean(), 3)}
        cov_rows.append(r)
        print(f"  K={K}: overlap {r['overlap_mean']} [{r['overlap_min']}-{r['overlap_max']}]  "
              f"Fleiss {r['fleiss']}  harmful none/some/all = {r['harm_none_%']}/{r['harm_some_%']}/"
              f"{r['harm_all_%']} %  union {r['union']}  best {r['best_single']}  "
              f"majority {r['majority_recall']} (FPR {r['majority_fpr']})")
    pd.DataFrame(cov_rows).to_csv(os.path.join(OUT, "coverage_matched.csv"), index=False)

    # within-arm version at the middle K, so seed and recipe stay separable
    K = K_LIST[1]
    print(f"\nWithin each recipe at K={K} (seed varies only):")
    for arm in ARMS:
        ks = [(arm, s) for s in (0, 1, 2)]
        sets = [set(top_k(S[k], K)) for k in ks]
        ov = np.mean([len(a & b) / K for a, b in itertools.combinations(sets, 2)])
        Mw = np.zeros((len(y), 3), int)
        for j, k in enumerate(ks):
            Mw[top_k(S[k], K), j] = 1
        Vp = Mw.sum(1)[pos]
        print(f"  {arm:16s} overlap {ov:.3f}  all-3 {(Vp == 3).mean() * 100:4.1f}%  union {(Vp >= 1).mean():.3f}")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
