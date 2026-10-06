"""
Phase 3/4: Sentiment Analysis (3-class)
Labels: 0=neutral, 1=positive, 2=negative
"""
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline

IN  = Path("data/processed/comments_clean.csv")
OUT = Path("data/sentiment")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)
df["comment_text"] = df["comment_text"].fillna("").astype(str)
df["clean_light"]  = df["clean_light"].fillna("").astype(str)

print(f"Loaded {len(df)} comments")

# ---------- Load pretrained Indonesian sentiment model ----------
# w11wo/indonesian-roberta-base-sentiment-classifier
# Trained on SmSA + other Indonesian corpora; labels: positive/neutral/negative
MODEL = "w11wo/indonesian-roberta-base-sentiment-classifier"

print(f"\nLoading model: {MODEL}")
clf = pipeline(
    "sentiment-analysis",
    model=MODEL,
    tokenizer=MODEL,
    truncation=True,
    max_length=128,
    device=-1,      # CPU. Set to 0 if you have a GPU.
)

# ---------- Run inference ----------
# Use clean_light for sentiment (preserves negations like 'tidak')
# We fall back to comment_text if clean_light is empty
texts = [t if t.strip() else r for t, r in
         zip(df["clean_light"], df["comment_text"])]

print("Running inference...")
results = clf(texts, batch_size=16)

# ---------- Map model labels -> your 0/1/2 scheme ----------
# Model returns: 'positive', 'neutral', 'negative' (lowercase)
LABEL_MAP = {
    "positive": 1,
    "neutral":  0,
    "negative": 2,
}
# Some checkpoints use LABEL_0/LABEL_1/LABEL_2. Handle both.
FALLBACK_MAP = {
    "LABEL_0": 2,   # convention in many smSA models
    "LABEL_1": 0,
    "LABEL_2": 1,
}

def to_int(label):
    l = label.lower()
    if l in LABEL_MAP:
        return LABEL_MAP[l]
    if label in FALLBACK_MAP:
        return FALLBACK_MAP[label]
    return 0    # default neutral

df["sentiment_label"] = [to_int(r["label"]) for r in results]
df["sentiment_score"] = [float(r["score"]) for r in results]
df["sentiment_raw"]   = [r["label"] for r in results]

# ---------- Sanity checks ----------
print("\n--- Distribution ---")
print(df["sentiment_label"].value_counts().sort_index().to_string())
print("\n--- By volcano ---")
print(pd.crosstab(df["volcano_name"], df["sentiment_label"]).to_string())
print("\n--- By channel ---")
print(pd.crosstab(df["channel_name"], df["sentiment_label"]).to_string())

# ---------- Save ----------
cols = ["video_id", "video_title", "channel_name", "volcano_name",
        "eruption_year", "comment_text", "clean_light",
        "sentiment_label", "sentiment_score", "sentiment_raw", "like_count"]

df[cols].to_csv(OUT / "comments_sentiment.csv", index=False)
print(f"\nSaved -> {OUT / 'comments_sentiment.csv'}")

# ---------- Video-level aggregation (for framing) ----------
agg = df.groupby(["video_id", "video_title", "channel_name",
                  "volcano_name", "eruption_year"]).agg(
    n_comments        = ("sentiment_label", "size"),
    pct_positive      = ("sentiment_label", lambda x: (x == 1).mean()),
    pct_neutral       = ("sentiment_label", lambda x: (x == 0).mean()),
    pct_negative      = ("sentiment_label", lambda x: (x == 2).mean()),
    dominant_sentiment= ("sentiment_label",
                         lambda x: x.value_counts().idxmax()),
).reset_index()

agg.to_csv(OUT / "video_sentiment.csv", index=False)
print(f"Saved -> {OUT / 'video_sentiment.csv'}")

# ---------- Volcano-level aggregation ----------
vol = df.groupby("volcano_name").agg(
    n_comments   = ("sentiment_label", "size"),
    pct_positive = ("sentiment_label", lambda x: (x == 1).mean()),
    pct_neutral  = ("sentiment_label", lambda x: (x == 0).mean()),
    pct_negative = ("sentiment_label", lambda x: (x == 2).mean()),
).reset_index().sort_values("n_comments", ascending=False)

vol.to_csv(OUT / "volcano_sentiment.csv", index=False)
print(f"Saved -> {OUT / 'volcano_sentiment.csv'}")
print("\n--- Volcano-level sentiment ---")
print(vol.to_string(index=False))