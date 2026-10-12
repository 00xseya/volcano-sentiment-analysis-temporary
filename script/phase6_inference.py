"""
Phase 7 — Inferensi sentimen ke SELURUH komentar yang TIDAK dianotasi.

Cara pakai (dari ROOT proyek):
    python script/phase6_inference.py --model tfidf
    python script/phase6_inference.py --model indobert

Komentar yang sudah masuk sampel anotasi (train/test) dikeluarkan berdasarkan
`source_row`, sehingga hasil inferensi murni data "baru" bagi model, lalu
gold label manusia digabung kembali sebagai sumber paling akurat (kolom
`label_source` = gold / model).

Output: data/processed/comments_with_sentiment.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_model import predict  # noqa: E402

RAW = Path("data/data_raw.csv")
FALLBACK_RAW = [Path("data/raw/comments_raw.csv")]
LABELED = [Path("data/data_train_labeled.csv"), Path("data/data_test_labeled.csv")]
OUT = Path("data/processed/comments_with_sentiment.csv")

def resolve_raw() -> Path:
    for p in [RAW] + FALLBACK_RAW:
        if p.exists():
            return p
    return RAW


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tfidf", "indobert"], default="tfidf")
    model = ap.parse_args().model

    raw = pd.read_csv(resolve_raw()).reset_index(names="source_row")
    gold = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in LABELED if p.exists()])
    gold_map = gold.set_index("source_row")["gold_label"].to_dict()

    unl = raw[~raw["source_row"].isin(gold_map)].copy()
    print(f"Total {len(raw)} | sudah dianotasi {len(gold_map)} | diprediksi {len(unl)} ({model})")

    if len(unl) > 0:
        unl["sentiment"] = predict(unl["comment_text"].fillna("").astype(str).tolist(), model=model)
    else:
        unl["sentiment"] = pd.Series(dtype=str)
    unl["label_source"] = f"model:{model}"

    lab = raw[raw["source_row"].isin(gold_map)].copy()
    lab["sentiment"] = lab["source_row"].map(gold_map)
    lab["label_source"] = "gold"

    out = pd.concat([lab, unl]).sort_values("source_row")
    
    # Protect against CSV injection
    def _escape_cell(value):
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(_escape_cell)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[OK] Disimpan -> {OUT}\n")

    print("Distribusi sentimen per gunung (%):")
    if "volcano_name" in out.columns:
        print((pd.crosstab(out["volcano_name"], out["sentiment"], normalize="index") * 100).round(1))
    print("\nTotal:", out["sentiment"].value_counts().to_dict())


if __name__ == "__main__":
    main()
