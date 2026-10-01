#!/usr/bin/env python3
"""Test the twelve saved models on new datasets (no retraining).

Three steps, run in order:

  python external_eval.py build
      Reads the downloaded datasets in this folder and makes one balanced
      test set per dataset in subsets/ (up to 1,500 harmful + 1,500 normal,
      seed 42). These files contain comment text: keep them private.

  python external_eval.py score --ckpt-dir PATH_TO_12_SAFETENSORS
      Runs all twelve models on every subset and saves per-comment
      probabilities in probs/<dataset>/<run>.csv (row numbers and numbers
      only, no text). Skips runs that are already saved, so it can be
      stopped and restarted.

  python external_eval.py analyze
      Computes, for every dataset, the same statistics as the paper and the
      matched-flag-count check. Prints a table and writes results/.

Datasets (harmful vs normal):
  wiki_person    Wikipedia talk pages. Harmful = most annotators marked an
                 attack on the person being replied to. Normal = most
                 annotators marked no attack. Test split only.
  cad_person     Reddit (CAD v1.1). PersonDirectedAbuse vs Neutral.
  cad_identity   Reddit (CAD v1.1). IdentityDirectedAbuse vs Neutral.
  stormfront     Stormfront forum (de Gibert 2018). hate vs noHate.
  dynahate       Dynamically Generated Hate (Vidgen 2021 v2.1). hate vs
                 nothate, test split only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "subsets")
FAMILY = "bert"   # set by --family; "roberta" reads/writes probs_roberta/ and results_roberta/
PROBS = os.path.join(HERE, "probs")
OUT = os.path.join(HERE, "results")
SEED = 42
PER_CLASS = 1500
ARMS = ("main", "soft_uw1", "soft_off", "no_contrastive")
RUNS = [f"{a}_{s}" for a in ARMS for s in (0, 1, 2)]
CANON = {"main": "main", "softoff": "soft_off", "softuw1": "soft_uw1",
         "nocontrastive": "no_contrastive"}
DATASETS = ["wiki_person", "cad_person", "cad_identity", "stormfront", "dynahate"]
SCORE_SETS = ["youtube"] + DATASETS   # youtube is scored only if subsets/youtube.csv exists
KIND = {"wiki_person": "person", "cad_person": "person", "cad_identity": "group",
        "stormfront": "group", "dynahate": "group", "youtube": "person"}


# ------------------------------------------------------------------ build
def balanced(df: pd.DataFrame) -> pd.DataFrame:
    """Equal numbers of harmful and normal, at most PER_CLASS each."""
    df = df[df["text"].astype(str).str.strip() != ""]
    n = min(PER_CLASS, int((df.gold == 1).sum()), int((df.gold == 0).sum()))
    parts = [df[df.gold == g].sample(n, random_state=SEED) for g in (1, 0)]
    out = pd.concat(parts).sort_values("source_id").reset_index(drop=True)
    out.insert(0, "row_id", range(len(out)))
    return out


def load_wiki() -> pd.DataFrame:
    d = os.path.join(HERE, "wikipedia_personal_attacks")
    a = pd.read_csv(os.path.join(d, "attack_annotated_comments.tsv"), sep="\t")
    b = pd.read_csv(os.path.join(d, "attack_annotations.tsv"), sep="\t")
    m = b.groupby("rev_id")[["attack", "recipient_attack"]].mean()
    a = a.join(m, on="rev_id")
    a = a[a.split == "test"].copy()
    a["gold"] = np.where(a.recipient_attack > 0.5, 1, np.where(a.attack < 0.5, 0, -1))
    a = a[a.gold >= 0]
    txt = (a.comment.str.replace("NEWLINE_TOKEN", " ", regex=False)
           .str.replace("TAB_TOKEN", " ", regex=False)
           .str.replace(r"\s+", " ", regex=True).str.strip())
    return pd.DataFrame({"source_id": a.rev_id.astype(str), "text": txt, "gold": a.gold})


def load_cad(which: str) -> pd.DataFrame:
    parts = [pd.read_csv(os.path.join(HERE, "reddit_cad", f"cad_v1_1_{s}.tsv"), sep="\t",
                         quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
             for s in ("train", "dev", "test")]
    c = pd.concat(parts)
    lab = c["labels"].str.strip()
    gold = np.where(lab == which, 1, np.where(lab == "Neutral", 0, -1))
    c = c.assign(gold=gold)[gold >= 0]
    txt = c.text.str.replace("[linebreak]", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.DataFrame({"source_id": c.id.astype(str), "text": txt, "gold": c.gold})


def load_stormfront() -> pd.DataFrame:
    d = os.path.join(HERE, "stromfront")
    m = pd.read_csv(os.path.join(d, "annotations_metadata.csv"))
    m = m[m.label.isin(["hate", "noHate"])]
    texts = []
    for fid in m.file_id:
        with open(os.path.join(d, "all_files", f"{fid}.txt"), encoding="utf-8", errors="replace") as f:
            texts.append(" ".join(f.read().split()))
    return pd.DataFrame({"source_id": m.file_id.astype(str), "text": texts,
                         "gold": (m.label == "hate").astype(int).to_numpy()})


def load_dynahate() -> pd.DataFrame:
    p = glob.glob(os.path.join(HERE, "vidgen_2021", "*.csv"))[0]
    d = pd.read_csv(p)
    d = d[d.split == "test"]
    return pd.DataFrame({"source_id": d["acl.id"].astype(str), "text": d.text.astype(str),
                         "gold": (d.label == "hate").astype(int).to_numpy()})


def build() -> None:
    os.makedirs(SUB, exist_ok=True)
    loaders = {"wiki_person": load_wiki, "cad_person": lambda: load_cad("PersonDirectedAbuse"),
               "cad_identity": lambda: load_cad("IdentityDirectedAbuse"),
               "stormfront": load_stormfront, "dynahate": load_dynahate}
    for name, fn in loaders.items():
        df = balanced(fn())
        df.to_csv(os.path.join(SUB, f"{name}.csv"), index=False)
        print(f"  {name:13s} {len(df):5d} comments ({int(df.gold.sum())} harmful)")


# ------------------------------------------------------------------ score
def score(ckpt_dir: str, base: str) -> None:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(base)
    cfg = AutoConfig.from_pretrained(base, num_labels=3)
    subsets = {n: pd.read_csv(os.path.join(SUB, f"{n}.csv")) for n in SCORE_SETS
               if os.path.exists(os.path.join(SUB, f"{n}.csv"))}
    if not subsets:
        sys.exit("no subsets found - run 'python external_eval.py build' first")

    for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.lower().startswith("roberta_"):
            stem = stem[len("roberta_"):]
        m = re.match(r"^(.*?)[_-](\d+)$", stem)
        arm = CANON.get(re.sub(r"[_\-\s]", "", m.group(1)).lower()) if m else None
        if arm is None:
            print(f"  {stem}: name not recognised, skipped"); continue
        run = f"{arm}_{m.group(2)}"
        todo = [n for n in subsets if not os.path.exists(os.path.join(PROBS, n, f"{run}.csv"))]
        if not todo:
            print(f"  {run}: already done"); continue
        sd = {k.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias"): v
              for k, v in load_file(path).items()}
        model = AutoModelForSequenceClassification.from_config(cfg)
        missing, _ = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  {run}: {len(missing)} missing keys - skipped"); continue
        model.eval().to(device)
        for n in todo:
            df = subsets[n]; texts = df.text.astype(str).tolist(); out = []
            with torch.no_grad():
                for i in range(0, len(texts), 64):
                    enc = tok(texts[i:i + 64], return_tensors="pt", truncation=True,
                              max_length=128, padding=True).to(device)
                    out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
                    print(f"  {run} {n}: {min(i + 64, len(texts))}/{len(texts)}", end="\r", flush=True)
            p = np.concatenate(out)
            os.makedirs(os.path.join(PROBS, n), exist_ok=True)
            pd.DataFrame({"row_id": df.row_id, "gold": df.gold, "p_hate": p[:, 0],
                          "p_offensive": p[:, 1], "p_normal": p[:, 2]}).to_csv(
                os.path.join(PROBS, n, f"{run}.csv"), index=False, float_format="%.6f")
            print(f"  {run} {n}: done{' ' * 20}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ analyze
def analyze_one(name: str, F1: dict) -> dict | None:
    from scipy.stats import spearmanr
    files = {r: os.path.join(PROBS, name, f"{r}.csv") for r in RUNS}
    if not all(os.path.exists(f) for f in files.values()):
        return None
    D = {r: pd.read_csv(f) for r, f in files.items()}
    y = D[RUNS[0]].gold.to_numpy().astype(int); pos = y == 1
    S = {r: (d.p_hate + d.p_offensive).to_numpy() for r, d in D.items()}
    FL = {r: d[["p_hate", "p_offensive", "p_normal"]].to_numpy().argmax(1) != 2 for r, d in D.items()}
    f1 = np.array([F1[r] for r in RUNS])
    flags = np.array([FL[r].sum() for r in RUNS])
    rec = np.array([FL[r][pos].mean() for r in RUNS])
    fpr = np.array([FL[r][~pos].mean() for r in RUNS])
    b, a = np.polyfit(flags, rec, 1)
    resid = rec - (a + b * flags)
    r2 = 1 - (resid ** 2).sum() / ((rec - rec.mean()) ** 2).sum()
    rho = spearmanr(f1, rec); rho_part = spearmanr(f1, resid)
    cont = [len(np.flatnonzero(FL[p] & FL[q])) / max(1, min(FL[p].sum(), FL[q].sum()))
            for p, q in itertools.combinations(RUNS, 2)]
    V = np.sum([FL[r] for r in RUNS], axis=0); Vp = V[pos]
    maj = V >= 7
    K = int(np.median(flags))
    topk = {r: set(np.argsort(-S[r], kind="mergesort")[:K]) for r in RUNS}
    rec_k = np.array([len([i for i in topk[r] if pos[i]]) / pos.sum() for r in RUNS])
    ov_k = np.mean([len(topk[p] & topk[q]) / K for p, q in itertools.combinations(RUNS, 2)])
    Mk = np.zeros((len(y), 12), int)
    for j, r in enumerate(RUNS):
        Mk[list(topk[r]), j] = 1
    Vk = Mk.sum(1)[pos]
    rho_k = spearmanr(f1, rec_k)
    return {
        "dataset": name, "kind": KIND.get(name, "?"), "n": len(y), "harmful": int(pos.sum()),
        "flags_min": int(flags.min()), "flags_max": int(flags.max()),
        "recall_min": round(rec.min(), 3), "recall_max": round(rec.max(), 3),
        "fpr_min": round(fpr.min(), 3), "fpr_max": round(fpr.max(), 3),
        "rho_f1_recall": round(rho.statistic, 2), "p": round(rho.pvalue, 3),
        "R2_flags_recall": round(r2, 3),
        "rho_after_flags": round(rho_part.statistic, 2), "p_after": round(rho_part.pvalue, 3),
        "containment": round(float(np.mean(cont)), 3),
        "all_%": round((Vp == 12).mean() * 100, 1), "some_%": round(((Vp > 0) & (Vp < 12)).mean() * 100, 1),
        "none_%": round((Vp == 0).mean() * 100, 1),
        "majority_recall": round(maj[pos].mean(), 3), "best_recall": round(rec.max(), 3),
        "union_recall": round((Vp >= 1).mean(), 3),
        "K": K, "recall_at_K_min": round(rec_k.min(), 3), "recall_at_K_max": round(rec_k.max(), 3),
        "rho_at_K": round(rho_k.statistic, 2), "p_at_K": round(rho_k.pvalue, 3),
        "overlap_at_K": round(ov_k, 3),
        "some_at_K_%": round(((Vk > 0) & (Vk < 12)).mean() * 100, 1),
    }


def analyze() -> None:
    led_path = os.path.join(HERE, "ledger12.csv" if FAMILY == "bert" else f"ledger_{FAMILY}.csv")
    if not os.path.exists(led_path):  # repo layout: external/ next to data/
        led_path = os.path.join(os.path.dirname(HERE), "data", os.path.basename(led_path))
    led = pd.read_csv(led_path)
    F1 = {f"{r.arm}_{int(r.seed)}": r.test_macro_f1 for r in led.itertuples()}
    names = ["youtube"] + DATASETS
    rows = [r for r in (analyze_one(n, F1) for n in names) if r]
    if not rows:
        sys.exit("no complete probability sets found - run 'score' first")
    T = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    T.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    for block in (["dataset", "kind", "n", "harmful", "flags_min", "flags_max", "recall_min", "recall_max",
                   "fpr_min", "fpr_max"],
                  ["dataset", "rho_f1_recall", "p", "R2_flags_recall", "rho_after_flags", "p_after"],
                  ["dataset", "containment", "all_%", "some_%", "none_%", "majority_recall", "best_recall",
                   "union_recall"],
                  ["dataset", "K", "recall_at_K_min", "recall_at_K_max", "rho_at_K", "p_at_K",
                   "overlap_at_K", "some_at_K_%"]):
        print(T[block].to_string(index=False), "\n")
    print("wrote", os.path.join(OUT, "summary.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "score", "analyze"])
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--base", default=None)
    ap.add_argument("--family", default="bert", choices=["bert", "roberta"])
    args = ap.parse_args()
    FAMILY = args.family
    if FAMILY != "bert":
        PROBS = os.path.join(HERE, f"probs_{FAMILY}")
        OUT = os.path.join(HERE, f"results_{FAMILY}")
    if args.base is None:
        args.base = "roberta-base" if FAMILY == "roberta" else "bert-base-uncased"
    if args.step == "build":
        build()
    elif args.step == "score":
        if not args.ckpt_dir:
            sys.exit("score needs --ckpt-dir")
        score(args.ckpt_dir, args.base)
    else:
        analyze()
