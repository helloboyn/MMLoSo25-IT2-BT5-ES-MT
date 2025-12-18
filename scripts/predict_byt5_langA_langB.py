#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained ByT5 (langA -> langB) model on the test split.

Inputs:
  --model_dir : local folder with saved model (e.g., runs/byt5_A2B)
  --data_dir  : dataset saved via preprocess script (contains 'test' with columns src,tgt)

Outputs (in --out_dir):
  - test.pred.txt     : model predictions (langB)
  - test.ref.txt      : references (langB)
  - test.src.txt      : sources (langA)
  - metrics.json      : {"bleu":..., "chrf++":..., "cer":...}
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets import load_from_disk
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sacrebleu


def cer(hyp: str, ref: str) -> float:
    """Character Error Rate."""
    a, b = hyp, ref
    n, m = len(a), len(b)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            prev, dp[j] = dp[j], min(
                dp[j] + 1,        # deletion
                dp[j - 1] + 1,    # insertion
                prev + cost       # substitution
            )
    return dp[m] / max(1, m)


def batched(iterable, n):
    """Yield lists of size n from iterable."""
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Local model folder (contains config.json, tokenizer, weights)")
    ap.add_argument("--data_dir", required=True, help="Folder from preprocess (load_from_disk)")
    ap.add_argument("--out_dir", required=True, help="Where to save predictions + metrics")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_source_length", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--lowercase", action="store_true", help="Lowercase src before generation (optional for roman-only)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_from_disk(args.data_dir)
    test = ds["test"]
    sources = list(test["src"])
    refs = list(test["tgt"])

    if args.lowercase:
        sources = [s.lower() for s in sources]

    # Load local tokenizer/model (no Hub)
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, local_files_only=True)
    mdl = T5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    mdl.eval()

    preds = []
    for batch in tqdm(batched(sources, args.batch_size), total=(len(sources) + args.batch_size - 1)//args.batch_size, desc="Generating"):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_source_length).to(device)
        with torch.no_grad():
            out = mdl.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                num_beams=max(1, args.num_beams),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))

    # Trim/normalize whitespace
    preds = [p.strip() for p in preds]
    refs  = [r.strip() for r in refs]
    srcs  = [s.strip() for s in sources]

    # Save outputs
    (out_dir / "test.pred.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")
    (out_dir / "test.ref.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")
    (out_dir / "test.src.txt").write_text("\n".join(srcs) + "\n", encoding="utf-8")

    # Metrics
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    cer_vals = [cer(h, r) for h, r in zip(preds, refs)]
    metrics = {"bleu": bleu, "chrf++": chrf, "cer": float(np.mean(cer_vals))}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("✅ Done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
