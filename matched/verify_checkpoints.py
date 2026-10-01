#!/usr/bin/env python3
"""Check that every checkpoint is the model it claims to be.

Hashing the files only finds copies that are byte-for-byte identical. A file
downloaded from the wrong notebook has its own unique hash and still passes.
This script checks behaviour instead: it loads each .safetensors file, runs the
second-corpus evaluation through it, and compares the recall and false-positive
rate it gets against what data/runs12.csv says that run scored.

A file that reproduces its recorded numbers is the right model, correctly
labelled. A file that does not is a duplicate, a download from the wrong
notebook, or a different architecture.

    python verify_checkpoints.py ^
        --ckpt-dir C:\\Users\\alina\\Downloads\\model_safetensors ^
        --corpus C:\\path\\to\\comments.csv

--corpus is the RAW Muminovic file (Kaggle: alinashifa/muminovic-dataset,
comments.csv). This script rebuilds the 2,834-row evaluation subset from it the
same way the notebook's SECOND-CORPUS EVALUATION cell did, and stops with a
clear message if the rebuild does not match predictions/*.csv exactly.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

LABELS = ["hate", "offensive", "normal"]
FLAG_CLASSES = {0, 1}

# these mirror the notebook cell exactly - do not "tidy" them
TEXT_COLS = ["comment_text", "text", "comment", "Text", "content"]
LABEL_COLS = ["human_label", "label", "Label", "cyberbullying", "is_bullying",
              "class", "Class", "cyberbullying_type", "CB_Label"]
MAX_ROWS = 3000
SAMPLE_SEED = 42

POSITIVE = {"1", "true", "yes", "bully", "bullying", "cyberbullying", "toxic",
            "offensive", "hate", "hateful", "abusive", "harassment"}
NEGATIVE = {"0", "false", "no", "not_bullying", "not bullying", "non_bullying",
            "non-bullying", "normal", "none", "not cyberbullying", "clean", "neutral"}


def to_binary(v) -> float:
    s = str(v).strip().lower()
    if s in POSITIVE:
        return 1.0
    if s in NEGATIVE:
        return 0.0
    try:
        return float(float(s) > 0)
    except ValueError:
        return np.nan


def build_eval_subset(path: str) -> tuple[list[str], np.ndarray]:
    """Rebuild the exact rows the predictions were generated from."""
    df = pd.read_csv(path)
    text_col = next((c for c in TEXT_COLS if c in df.columns), None)
    label_col = next((c for c in LABEL_COLS if c in df.columns), None)
    if text_col is None or label_col is None:
        sys.exit(f"could not find a text/label column in {list(df.columns)}.\n"
                 f"Expected one of {TEXT_COLS} and one of {LABEL_COLS}.")
    print(f"corpus {os.path.basename(path)}: {len(df)} rows, "
          f"text='{text_col}', label='{label_col}'")

    df["_gold_bin"] = df[label_col].map(to_binary)
    df = df.dropna(subset=["_gold_bin", text_col]).reset_index(drop=True)
    df["_gold_bin"] = df["_gold_bin"].astype(int)

    # keep a copy: some pandas versions consume the grouping column in .apply
    df["_gold_keep"] = df["_gold_bin"]

    if MAX_ROWS and len(df) > MAX_ROWS:
        df = df.groupby("_gold_bin", group_keys=False).apply(
            lambda g: g.sample(min(len(g), MAX_ROWS // 2), random_state=SAMPLE_SEED))
        print(f"  sampled down to {len(df)} rows")

    return df[text_col].astype(str).tolist(), df["_gold_keep"].to_numpy()


def norm(arm: str) -> str:
    """soft_off, softoff, soft-off all collapse to the same key."""
    return re.sub(r"[_\-\s]", "", arm).lower()


def parse_name(path: str) -> tuple[str, str]:
    """-> (display name, lookup key). 'softoff_1.safetensors' -> ('softoff_1', 'softoff_1')."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(.*?)[_-](\d+)$", stem)
    if not m:
        return stem, norm(stem)
    return stem, f"{norm(m.group(1))}_{m.group(2)}"


def remap_legacy_layernorm(sd: dict) -> dict:
    """Some checkpoints were saved with the old TF-BERT LayerNorm names
    (gamma/beta) instead of the modern ones (weight/bias). Without this,
    every LayerNorm silently fails to load under strict=False and stays
    randomly initialised - the model still "loads" but predicts near-random.
    """
    return {
        k.replace("LayerNorm.gamma", "LayerNorm.weight")
         .replace("LayerNorm.beta", "LayerNorm.bias"): v
        for k, v in sd.items()
    }


@torch.no_grad()
def score(model, tok, texts: list[str], device, batch: int = 64) -> np.ndarray:
    preds = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                  max_length=128, padding=True).to(device)
        preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
        print(f"    {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return np.concatenate(preds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="folder holding the .safetensors files")
    ap.add_argument("--corpus", required=True,
                    help="the raw Muminovic comments.csv")
    ap.add_argument("--base", default="bert-base-uncased",
                    help="HF repo id or local folder with config.json + tokenizer")
    ap.add_argument("--repo", default="..",
                    help="repo root holding data/ and predictions/ (default: parent folder)")
    args = ap.parse_args()

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    runs = pd.read_csv(os.path.join(args.repo, "data", "runs12.csv"))
    expected = {f"{norm(r.arm)}_{r.seed}": r for r in runs.itertuples()}

    pred_files = sorted(glob.glob(os.path.join(args.repo, "predictions", "*.csv")))
    if not pred_files:
        sys.exit(f"no predictions/*.csv under {args.repo}")
    ref = pd.read_csv(pred_files[0])
    recorded_gold = ref["_gold_bin"].to_numpy()

    texts, gold = build_eval_subset(args.corpus)

    if len(gold) != len(recorded_gold) or not np.array_equal(gold, recorded_gold):
        print("\nThe rebuilt subset does not match predictions/.", file=sys.stderr)
        print(f"  rebuilt   : {len(gold)} rows, {int((gold == 1).sum())} harmful",
              file=sys.stderr)
        print(f"  recorded  : {len(recorded_gold)} rows, "
              f"{int((recorded_gold == 1).sum())} harmful", file=sys.stderr)
        print("\nThis is almost always the wrong comments.csv, or a pandas version\n"
              "that samples differently. Nothing below would be meaningful, so stopping.",
              file=sys.stderr)
        return 1
    print("  rebuilt subset matches predictions/ exactly\n")

    harmful = gold == 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    cfg = AutoConfig.from_pretrained(args.base, num_labels=len(LABELS))
    print(f"device {device}, {len(texts)} comments, {int(harmful.sum())} harmful\n")

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.safetensors")))
    if not ckpts:
        sys.exit(f"no .safetensors files in {args.ckpt_dir}")

    rows, seen_flags = [], {}
    for path in ckpts:
        name, key = parse_name(path)
        exp = expected.get(key)
        print(f"  {name} ...")

        model = AutoModelForSequenceClassification.from_config(cfg)
        sd = remap_legacy_layernorm(load_file(path))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"    ! {len(missing)} missing keys after remap: {missing[:5]}"
                  f"{' ...' if len(missing) > 5 else ''}")
        model.eval().to(device)

        pred_idx = score(model, tok, texts, device)
        flag = np.isin(pred_idx, list(FLAG_CLASSES))
        n_flag = int(flag.sum())
        recall = float((flag & harmful).sum() / harmful.sum())
        fpr = float((flag & ~harmful).sum() / (~harmful).sum())

        if exp is None:
            verdict = "NO RECORD in runs12.csv"
        elif abs(recall - exp.recall) < 0.005 and abs(fpr - exp.fpr) < 0.005:
            verdict = "ok"
        else:
            verdict = "MISMATCH"

        seen_flags.setdefault(n_flag, []).append(name)
        rows.append({
            "run": name,
            "flags": n_flag,
            "recall": round(recall, 4),
            "exp_recall": round(float(exp.recall), 4) if exp is not None else None,
            "fpr": round(fpr, 4),
            "exp_fpr": round(float(exp.fpr), 4) if exp is not None else None,
            "missing_keys": len(missing),
            "verdict": verdict,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))

    dupes = {k: v for k, v in seen_flags.items() if len(v) > 1}
    bad = out[out.verdict != "ok"]

    print()
    for k, v in dupes.items():
        print(f"  identical behaviour ({k} flags): {', '.join(v)}")
    if len(bad):
        print(f"\n  {len(bad)} checkpoint(s) do not reproduce their recorded numbers:")
        for r in bad.itertuples():
            print(f"    {r.run}: recall {r.recall} vs {r.exp_recall}, "
                  f"fpr {r.fpr} vs {r.exp_fpr}")
        print("\n  Re-download those from their own Kaggle notebook output.")
        return 1

    print("  every checkpoint reproduces its recorded numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
