#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ByT5 for langA → langB (roman/Latin script).
Compatible with older Transformers (no evaluation_strategy).
"""
import os, argparse, numpy as np
from typing import List
from datasets import load_from_disk
from transformers import (
    T5ForConditionalGeneration, AutoTokenizer,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import sacrebleu


def cer(hyp: str, ref: str) -> float:
    a, b = hyp, ref
    n, m = len(a), len(b)
    if m == 0: return 0.0 if n == 0 else 1.0
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            prev, dp[j] = dp[j], min(dp[j]+1, dp[j-1]+1, prev+cost)
    return dp[m]/max(1,m)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--model_name", default="google/byt5-small")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--max_source_length", type=int, default=256)
parser.add_argument("--max_target_length", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--warmup_ratio", type=float, default=0.03)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lowercase", action="store_true")
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# --- Load dataset ---
raw = load_from_disk(args.data_dir)

def maybe_lower(batch):
    if not args.lowercase: return batch
    return {"src":[s.lower() for s in batch["src"]],
            "tgt":[t.lower() for t in batch["tgt"]]}

train = raw["train"].map(maybe_lower, batched=True)
val   = raw["validation"].map(maybe_lower, batched=True)

# --- Tokenizer & model ---
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
except Exception:
    from transformers import ByT5Tokenizer
    tokenizer = ByT5Tokenizer.from_pretrained(args.model_name)

model = T5ForConditionalGeneration.from_pretrained(
    args.model_name,
    use_safetensors=True
)

# --- Tokenization ---
def tokenize_fn(batch):
    x = tokenizer(batch["src"], truncation=True, max_length=args.max_source_length)
    y = tokenizer(batch["tgt"], truncation=True, max_length=args.max_target_length)
    x["labels"] = y["input_ids"]
    return x

train_tok = train.map(tokenize_fn, batched=True, remove_columns=train.column_names)
val_tok   = val.map(tokenize_fn,   batched=True, remove_columns=val.column_names)

collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- Metrics ---
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
    preds = [p.strip() for p in preds]
    refs = [r.strip() for r in refs]
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    cer_vals = [cer(h, r) for h, r in zip(preds, refs)]
    return {"bleu": bleu, "chrf++": chrf, "cer": float(np.mean(cer_vals))}

# --- Training arguments (old-version safe) ---
args_hf = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    warmup_ratio=args.warmup_ratio,
    save_total_limit=2,
    prediction_loss_only=False,
    logging_steps=200,
    fp16=args.fp16,
    seed=args.seed,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args_hf,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("🚀 Starting training...")
trainer.train()
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"✅ Training complete. Model saved to: {args.output_dir}")
