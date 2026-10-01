#!/usr/bin/env python3
"""Step 1 of the matched-operating-point extension: save probabilities.

The paper stored only each run's final label per comment, so runs could only be
compared at their own default decision rule. This script re-runs the twelve
verified checkpoints on the same 2,834 YouTube comments and saves the full
three-class logits and probabilities. No retraining.

It reuses the exact loading and subset-rebuild code from verify_checkpoints.py,
and checks every run twice before saving anything:
  1. the rebuilt comment subset matches predictions/*.csv row for row
  2. the argmax of the new probabilities reproduces that run's recorded labels

Run from inside the extension/ folder:

    python save_probabilities.py --ckpt-dir C:\\Users\\alina\\Downloads\\model_safetensors --corpus C:\\FULL\\PATH\\TO\\comments.csv

Output: extension/probs/<arm>_<seed>.csv, one file per run, columns
    row_id, gold, logit_hate, logit_offensive, logit_normal,
    p_hate, p_offensive, p_normal
The probabilities are plain softmax (temperature 1), so the argmax is exactly
the label the paper used.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from verify_checkpoints import (LABELS, build_eval_subset, norm, parse_name,  # noqa: E402
                                remap_legacy_layernorm)

CANON = {"main": "main", "softoff": "soft_off", "softuw1": "soft_uw1",
         "nocontrastive": "no_contrastive"}


@torch.no_grad()
def logits_for(model, tok, texts, device, batch=64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", truncation=True,
                  max_length=128, padding=True).to(device)
        out.append(model(**enc).logits.float().cpu().numpy())
        print(f"    {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="folder with the 12 .safetensors files")
    ap.add_argument("--corpus", required=True, help="the RAW Muminovic comments.csv")
    ap.add_argument("--base", default="bert-base-uncased",
                    help="HF repo id or local folder with config.json + tokenizer")
    ap.add_argument("--out", default=os.path.join(HERE, "probs"))
    args = ap.parse_args()

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    texts, gold = build_eval_subset(args.corpus)
    ref_files = sorted(glob.glob(os.path.join(REPO, "predictions", "second_corpus_*.csv")))
    if not ref_files:
        sys.exit(f"no predictions/second_corpus_*.csv under {REPO}")
    ref0 = pd.read_csv(ref_files[0])
    if len(gold) != len(ref0) or not np.array_equal(gold, ref0["_gold_bin"].to_numpy()):
        sys.exit("The rebuilt comment subset does not match predictions/. Wrong comments.csv? Stopping.")
    row_id = ref0["row_id"].to_numpy()
    print("  rebuilt subset matches predictions/ exactly\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    cfg = AutoConfig.from_pretrained(args.base, num_labels=len(LABELS))
    os.makedirs(args.out, exist_ok=True)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.safetensors")))
    if not ckpts:
        sys.exit(f"no .safetensors files in {args.ckpt_dir}")

    problems = 0
    for path in ckpts:
        name, key = parse_name(path)
        arm_key, seed = key.rsplit("_", 1)
        arm = CANON.get(arm_key)
        if arm is None:
            print(f"  {name}: unknown arm name, skipped"); problems += 1; continue
        ref_path = os.path.join(REPO, "predictions", f"second_corpus_{arm}_{seed}.csv")
        if not os.path.exists(ref_path):
            print(f"  {name}: no recorded predictions at {ref_path}, skipped"); problems += 1; continue
        recorded = pd.read_csv(ref_path)["pred_idx"].to_numpy()

        print(f"  {arm}_{seed} ...")
        model = AutoModelForSequenceClassification.from_config(cfg)
        missing, _ = model.load_state_dict(remap_legacy_layernorm(load_file(path)), strict=False)
        if missing:
            print(f"    ! {len(missing)} missing keys - not saving this run"); problems += 1; continue
        model.eval().to(device)

        lg = logits_for(model, tok, texts, device)
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        agree = float((lg.argmax(1) == recorded).mean())
        print(f"    argmax agrees with recorded labels on {agree:.2%} of comments")
        if agree < 0.995:
            print("    ! below 99.5% - this is not the recorded model, not saving"); problems += 1; continue

        pd.DataFrame({
            "row_id": row_id, "gold": gold,
            "logit_hate": lg[:, 0], "logit_offensive": lg[:, 1], "logit_normal": lg[:, 2],
            "p_hate": p[:, 0], "p_offensive": p[:, 1], "p_normal": p[:, 2],
        }).to_csv(os.path.join(args.out, f"{arm}_{seed}.csv"), index=False, float_format="%.6f")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = len(glob.glob(os.path.join(args.out, "*.csv")))
    print(f"\n  saved {n} of 12 runs to {args.out}")
    if problems:
        print(f"  {problems} run(s) had a problem - read the messages above")
    return 0 if n == 12 and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
