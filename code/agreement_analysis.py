#!/usr/bin/env python3
"""Item-level agreement analysis over the twelve runs.

Inputs: second_corpus_<arm>_<seed>.csv for all 12 runs, and ledger12.csv.
Each prediction file holds comment_text, _gold_bin, pred_idx, pred_bin for the
same 2,834 rows in the same order.

Reproduces every number in Sections IV-B, IV-C and IV-D of the manuscript.
"""
import itertools, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, linregress

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
_PRED = os.path.join(_HERE, "..", "predictions")

RUNS = [(a, s) for a in ("main", "soft_uw1", "soft_off", "no_contrastive")
        for s in (0, 1, 2)]
P = {k: pd.read_csv(os.path.join(_PRED, f"second_corpus_{k[0]}_{k[1]}.csv")) for k in RUNS}
base = P[RUNS[0]]
for k, d in P.items():
    assert (d.row_id.values == base.row_id.values).all(), f"row order differs: {k}"
g = base._gold_bin.values.astype(int)
Y = {k: d.pred_bin.values.astype(int) for k, d in P.items()}
N, npos, nneg = len(g), int((g == 1).sum()), int((g == 0).sum())
F1 = {(r.arm, int(r.seed)): r.test_macro_f1
      for _, r in pd.read_csv(os.path.join(_DATA, "ledger12.csv")).iterrows()}

rec = lambda p: ((g == 1) & (p == 1)).sum() / npos
fpr = lambda p: ((g == 0) & (p == 1)).sum() / nneg
prc = lambda p: ((g == 1) & (p == 1)).sum() / max((p == 1).sum(), 1)

print(f"n={N}  harmful={npos}  benign={nneg}\n")

# --- IV-B: the operating-point confound -------------------------------------
x = np.array([F1[k] for k in RUNS])
r = np.array([rec(Y[k]) for k in RUNS])
nf = np.array([(Y[k] == 1).sum() for k in RUNS], float)
pr = np.array([prc(Y[k]) for k in RUNS])
lr = linregress(nf, r)
resid = r - (lr.intercept + lr.slope * nf)
print("IV-B  operating-point confound")
print(f"  recall ~ flag count        R^2 = {lr.rvalue**2:.4f}  p = {lr.pvalue:.1e}")
print(f"  flag counts                {int(nf.min())}-{int(nf.max())}")
for lab, v in (("recall", r), ("flag count", nf), ("precision", pr), ("recall | flag count", resid)):
    sp = spearmanr(x, v)
    print(f"  Spearman(macro-F1, {lab:19s}) = {sp.statistic:+.3f}  p = {sp.pvalue:.4f}")

# --- IV-C: containment and agreement ----------------------------------------
cont, jac, chance, cohen = [], [], [], []
for a, b in itertools.combinations(RUNS, 2):
    A = set(np.flatnonzero(Y[a] == 1)); B = set(np.flatnonzero(Y[b] == 1))
    small, big = (A, B) if len(A) <= len(B) else (B, A)
    cont.append(len(small & big) / len(small))
    jac.append(len(A & B) / len(A | B))
    exp = len(A) * len(B) / N                      # E|A&B| under independence
    chance.append(exp / (len(A) + len(B) - exp))
    po = (Y[a] == Y[b]).mean()
    pe = (Y[a] == 1).mean() * (Y[b] == 1).mean() + (Y[a] == 0).mean() * (Y[b] == 0).mean()
    cohen.append((po - pe) / (1 - pe))
M = np.stack([Y[k] for k in RUNS], 1); n1 = M.sum(1); n0 = len(RUNS) - n1
Pi = (n1 * (n1 - 1) + n0 * (n0 - 1)) / (len(RUNS) * (len(RUNS) - 1))
p1 = n1.sum() / (N * len(RUNS)); pe = p1 ** 2 + (1 - p1) ** 2
print("\nIV-C  containment and agreement over 66 run pairs")
print(f"  containment      mean {np.mean(cont):.3f}  [{min(cont):.3f}, {max(cont):.3f}]")
print(f"  Jaccard observed mean {np.mean(jac):.3f}   chance {np.mean(chance):.3f}")
print(f"  Cohen kappa      mean {np.mean(cohen):.3f}  [{min(cohen):.3f}, {max(cohen):.3f}]")
print(f"  Fleiss kappa (12 runs) {(Pi.mean()-pe)/(1-pe):.3f}   predicted-positive rate {p1:.3f}")

# --- IV-D: coverage of the harmful class ------------------------------------
V = M.sum(1); pos = g == 1
print("\nIV-D  coverage")
print(f"  flagged by >=1 run {int((V >= 1).sum())}   by all 12 {int((V == 12).sum())}")
print(f"  exactly one run    {int((V == 1).sum())}   of which gold-harmful {g[V == 1].mean():.2f}")
mv = (V >= 7).astype(int)
print(f"  intersection recall {(V[pos] == 12).mean():.3f}")
print(f"  majority (>=7)      {rec(mv):.3f}  FPR {fpr(mv):.3f}")
print(f"  best single run     {r.max():.3f}")
print(f"  union               {(V[pos] >= 1).mean():.3f}")
print(f"  harmful: all-12 {(V[pos]==12).mean()*100:.1f}%  "
      f"some {((V[pos]>=1)&(V[pos]<12)).mean()*100:.1f}%  none {(V[pos]==0).mean()*100:.1f}%")


# --- within-arm: seed variation only ------------
print("\nIV-C  within each configuration (seed varies, recipe fixed)")
print(f"  {'arm':16s} {'Fleiss':7s} {'contain':8s} {'Jaccard':8s} {'all-3':7s} {'union':6s}")
for arm in ("main", "soft_uw1", "soft_off", "no_contrastive"):
    ks = [(arm, s) for s in (0, 1, 2)]
    cn, jc = [], []
    for a, b in itertools.combinations(ks, 2):
        A = set(np.flatnonzero(Y[a] == 1)); B = set(np.flatnonzero(Y[b] == 1))
        sm, bg = (A, B) if len(A) <= len(B) else (B, A)
        cn.append(len(sm & bg) / len(sm)); jc.append(len(A & B) / len(A | B))
    Mw = np.stack([Y[k] for k in ks], 1); R = 3
    a1 = Mw.sum(1); a0 = R - a1
    Piw = (a1 * (a1 - 1) + a0 * (a0 - 1)) / (R * (R - 1))
    q1 = a1.sum() / (N * R); pew = q1 ** 2 + (1 - q1) ** 2
    Vw = Mw.sum(1); p = g == 1
    print(f"  {arm:16s} {(Piw.mean()-pew)/(1-pew):.3f}   {np.mean(cn):.3f}    "
          f"{np.mean(jc):.3f}    {(Vw[p]==3).mean()*100:4.1f}%   {(Vw[p]>=1).mean():.3f}")
