#!/usr/bin/env python3
"""
Preprocess parallel data for ByT5 (langA -> langB), both roman script.

Inputs (data/):
  - train.langA.txt, train.langB.txt
  - val.langA.txt,   val.langB.txt
  - test.langA.txt,  test.langB.txt

Output:
  - data/byt5_langA_langB/ (HF dataset with columns: src, tgt)
"""
import os, unicodedata, json
from datasets import Dataset, DatasetDict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR  = os.path.join(DATA_DIR, "byt5_langA_langB")

def read_parallel(a_path, b_path):
    with open(a_path, "r", encoding="utf-8") as fa, open(b_path, "r", encoding="utf-8") as fb:
        A = [l.rstrip("\n\r") for l in fa]
        B = [l.rstrip("\n\r") for l in fb]
    assert len(A) == len(B), f"Length mismatch: {a_path} vs {b_path}"
    return A, B

def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return " ".join(s.split())

def build(split: str):
    A, B = read_parallel(
        os.path.join(DATA_DIR, f"{split}.langA.txt"),
        os.path.join(DATA_DIR, f"{split}.langB.txt"),
    )
    A = [norm(x) for x in A]
    B = [norm(x) for x in B]
    return Dataset.from_dict({"src": A, "tgt": B})

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = DatasetDict({
        "train": build("train"),
        "validation": build("val"),
        "test": build("test"),
    })
    ds.save_to_disk(OUT_DIR)
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"src_lang":"langA","tgt_lang":"langB","norm":"NFC"}, f, ensure_ascii=False, indent=2)
    print("✅ Saved:", OUT_DIR)

if __name__ == "__main__":
    main()
