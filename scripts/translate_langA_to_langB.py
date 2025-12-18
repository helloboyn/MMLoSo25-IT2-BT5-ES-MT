#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-quality translation of mono.langA.txt -> mono.langB.txt
- Preserves alignment: writes exactly one output line per input line (including blanks)
- Strong decoding defaults for quality
"""

import argparse
from pathlib import Path
from typing import List
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from tqdm import tqdm


def read_all_lines_keep_blanks(path: str) -> List[str]:
    # Keep blanks: strip only trailing newlines, do NOT drop empty lines
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n\r") for ln in f]


def write_lines(lines: List[str], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def batched(items: List[str], bs: int):
    for i in range(0, len(items), bs):
        yield items[i:i+bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Local A->B model dir, e.g. runs/byt5_A2B")
    ap.add_argument("--input", required=True, help="Path to mono.langA.txt")
    ap.add_argument("--output", required=True, help="Path to mono.langB.txt (created)")
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--max_source_length", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    # High-quality decoding settings
    ap.add_argument("--num_beams", type=int, default=8)
    ap.add_argument("--length_penalty", type=float, default=1.05)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)

    # How to handle blank inputs: "empty" -> output empty; "copy" -> copy source
    ap.add_argument("--blank_policy", choices=["empty", "copy"], default="empty")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase inputs (optional for roman text).")
    args = ap.parse_args()

    # Load model/tokenizer locally
    print(f"🔹 Loading model from {args.model_dir} ...")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Read ALL lines, keep blanks
    src_all = read_all_lines_keep_blanks(args.input)
    print(f"🔹 Loaded {len(src_all)} lines from {args.input}")

    # Build a list of indices for non-blank lines, so we only send those to the model
    nonblank_idx = []
    nonblank_src = []
    for i, s in enumerate(src_all):
        if s.strip() == "":
            continue
        nonblank_idx.append(i)
        nonblank_src.append(s.lower() if args.lowercase else s)

    # Translate only non-blank lines
    preds_nonblank: List[str] = []
    total_batches = (len(nonblank_src) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(batched(nonblank_src, args.batch_size), total=total_batches, desc="Translating (A→B)"):
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_source_length
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                early_stopping=True,
            )
        preds_nonblank.extend(tok.batch_decode(out, skip_special_tokens=True))

    # Reconstruct full-length output, inserting blanks or copies where needed
    out_all = [""] * len(src_all)
    # fill predictions for non-blank rows
    for j, i in enumerate(nonblank_idx):
        out_all[i] = preds_nonblank[j].strip()
    # handle blank rows
    if args.blank_policy == "copy":
        for i, s in enumerate(src_all):
            if s.strip() == "":
                out_all[i] = s  # keep blank (copy is also blank)
    else:  # "empty" -> already empty strings
        pass

    # Sanity: preserve exact length
    assert len(out_all) == len(src_all), "Output length mismatch — this should never happen."

    write_lines(out_all, args.output)
    print("✅ Translation complete!")
    print(f"   Input : {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Total : {len(out_all)} lines (alignment preserved).")


if __name__ == "__main__":
    main()
