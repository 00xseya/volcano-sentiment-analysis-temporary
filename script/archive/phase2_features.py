"""
Phase 2: Feature Engineering
Extracts TF, IDF, TF-IDF, BoW, N-grams, and handcrafted features.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import save_npz, diags
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer,
)

IN  = Path("data/processed/comments_clean.csv")
OUT = Path("data/features")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 0. LOAD
# ============================================================
df = pd.read_csv(IN)
print(f"Loaded {len(df)} comments")

for col in ["comment_text", "clean_light", "clean_heavy"]:
    df[col] = df[col].fillna("").astype(str)

# Ensure video_id exists
if "video_id" not in df.columns:
    raise SystemExit("comment_text / video_id missing from input.")

# ============================================================
# 1. HANDCRAFTED FEATURES
# ============================================================
print("\n[1] Handcrafted features (word/char counts, avg word length)")

def word_count(s):  return s.str.split().str.len().fillna(0).astype(int)
def char_count(s):  return s.str.replace(r"\s", "", regex=True).str.len().fillna(0).astype(int)

hand = pd.DataFrame({
    "video_id":      df["video_id"],
    "volcano_name":  df.get("volcano_name"),
    "eruption_year": df.get("eruption_year"),
    "channel_name":  df.get("channel_name"),
    "like_count":    df.get("like_count", 0),
})

hand["word_count_raw"]    = word_count(df["comment_text"])
hand["word_count_light"]  = word_count(df["clean_light"])
hand["word_count_heavy"]  = word_count(df["clean_heavy"])

hand["char_count_raw"]    = char_count(df["comment_text"])
hand["char_count_light"]  = char_count(df["clean_light"])
hand["char_count_heavy"]  = char_count(df["clean_heavy"])

hand["avg_word_len_raw"]   = hand["char_count_raw"]   / hand["word_count_raw"].clip(lower=1)
hand["avg_word_len_light"] = hand["char_count_light"] / hand["word_count_light"].clip(lower=1)
hand["avg_word_len_heavy"] = hand["char_count_heavy"] / hand["word_count_heavy"].clip(lower=1)

hand.to_csv(OUT / "handcrafted_features.csv", index=False)
print(f"    -> {hand.shape[0]} rows x {hand.shape[1]} cols")

# ============================================================
# 2. BAG OF WORDS (BoW) + TERM FREQUENCY (TF)
# ============================================================
print("\n[2] Bag of Words + Term Frequency (heavy)")

bow_vec = CountVectorizer(
    lowercase=False,
    token_pattern=r"\S+",     # tokens already split by Phase 1
    min_df=2,
    max_df=0.9,
)
X_bow = bow_vec.fit_transform(df["clean_heavy"])

# Term Frequency = same counts, but we also store a length-normalized version
row_sums = np.asarray(X_bow.sum(axis=1)).flatten()
row_sums[row_sums == 0] = 1
X_tf_norm = (diags(1.0 / row_sums) @ X_bow).astype(np.float32)

save_npz(OUT / "bow_matrix.npz",     X_bow)
save_npz(OUT / "tf_raw.npz",         X_bow)
save_npz(OUT / "tf_normalized.npz",  X_tf_norm)

with open(OUT / "bow_vocab.json", "w", encoding="utf-8") as f:
    json.dump(bow_vec.vocabulary_, f, ensure_ascii=False)

print(f"    BoW/TF shape: {X_bow.shape}  vocab: {len(bow_vec.vocabulary_)}")

# ============================================================
# 3. INVERSE DOCUMENT FREQUENCY (IDF)
# ============================================================
print("\n[3] Inverse Document Frequency (IDF)")

idf_vec = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"\S+",
    min_df=2,
    max_df=0.9,
    use_idf=True,
    norm=None,          # keep raw IDF for inspection
    smooth_idf=True,
)
idf_vec.fit(df["clean_heavy"])
idf_df = pd.DataFrame({
    "term": idf_vec.get_feature_names_out(),
    "idf":  idf_vec.idf_,
}).sort_values("idf", ascending=False)
idf_df.to_csv(OUT / "idf_values.csv", index=False)

print(f"    {len(idf_df)} terms")
print(f"    Highest IDF (rarest) : {idf_df.head(3)['term'].tolist()}")
print(f"    Lowest  IDF (common) : {idf_df.tail(3)['term'].tolist()}")

# ============================================================
# 4. TF-IDF  (heavy, unigram)
# ============================================================
print("\n[4] TF-IDF (heavy, unigram)")

tfidf_vec = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"\S+",
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,      # log(1+tf)
    norm="l2",
)
X_tfidf = tfidf_vec.fit_transform(df["clean_heavy"])
save_npz(OUT / "tfidf_heavy_unigram.npz", X_tfidf)
with open(OUT / "tfidf_heavy_unigram_vocab.json", "w", encoding="utf-8") as f:
    json.dump(tfidf_vec.vocabulary_, f, ensure_ascii=False)
print(f"    shape: {X_tfidf.shape}")

# ============================================================
# 5. TF-IDF  (light, unigram) — for sentiment
# ============================================================
print("\n[5] TF-IDF (light, unigram) — sentiment-preserving")

tfidf_light = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"\S+",
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    norm="l2",
)
X_tfidf_light = tfidf_light.fit_transform(df["clean_light"])
save_npz(OUT / "tfidf_light_unigram.npz", X_tfidf_light)
with open(OUT / "tfidf_light_unigram_vocab.json", "w", encoding="utf-8") as f:
    json.dump(tfidf_light.vocabulary_, f, ensure_ascii=False)
print(f"    shape: {X_tfidf_light.shape}")

# ============================================================
# 6. TF-IDF N-GRAMS  (1,2) and (1,3)
# ============================================================
print("\n[6] TF-IDF with N-grams")

for ng in [(1, 2), (1, 3)]:
    tag = f"{ng[0]}{ng[1]}"
    vec = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"\S+",
        min_df=2,
        max_df=0.9,
        ngram_range=ng,
        sublinear_tf=True,
        norm="l2",
    )
    X = vec.fit_transform(df["clean_heavy"])
    save_npz(OUT / f"tfidf_heavy_ngram_{tag}.npz", X)
    with open(OUT / f"tfidf_heavy_ngram_{tag}_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vec.vocabulary_, f, ensure_ascii=False)
    print(f"    ngram={ng}  shape: {X.shape}")

# ============================================================
# 7. N-GRAM FREQUENCY ANALYSIS
# ============================================================
print("\n[7] N-gram frequency analysis (top bigrams / trigrams)")

for n, name in [(2, "bigrams"), (3, "trigrams")]:
    vec = CountVectorizer(
        lowercase=False,
        token_pattern=r"\S+",
        min_df=2,
        max_df=0.9,
        ngram_range=(n, n),
    )
    X = vec.fit_transform(df["clean_heavy"])
    freqs = np.asarray(X.sum(axis=0)).flatten()
    top = (pd.DataFrame({"ngram": vec.get_feature_names_out(), "frequency": freqs})
             .sort_values("frequency", ascending=False)
             .head(50))
    top.to_csv(OUT / f"top_{name}.csv", index=False)
    print(f"    top 5 {name}: {top['ngram'].head(5).tolist()}")

# ============================================================
# 8. SAVE ALIGNED META INDEX
# ============================================================
meta = df[["video_id", "video_title", "channel_name", "channel_type",
           "volcano_name", "eruption_year", "comment_text",
           "clean_light", "clean_heavy"]].reset_index(drop=True)
meta.to_csv(OUT / "feature_index.csv", index=False)

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "=" * 62)
print("PHASE 2 SUMMARY")
print("=" * 62)
print(f"Documents                 : {len(df)}")
print(f"BoW vocabulary (heavy)    : {len(bow_vec.vocabulary_)}")
print(f"TF-IDF heavy vocab        : {len(tfidf_vec.vocabulary_)}")
print(f"TF-IDF light vocab        : {len(tfidf_light.vocabulary_)}")
print(f"Output directory          : {OUT.resolve()}")
print("\nHandcrafted feature stats:")
print(hand[["word_count_raw", "word_count_heavy",
            "char_count_raw", "avg_word_len_heavy"]].describe().round(2).to_string())
print("\nFiles written:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name:<45} ({p.stat().st_size/1024:>7.1f} KB)")