"""
Lexicon-based 3-class sentiment for Indonesian volcano comments.
0=neutral, 1=positive, 2=negative.
"""
import re
import pandas as pd
from pathlib import Path

IN  = Path("data/processed/comments_clean.csv")
OUT = Path("data/sentiment")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)
df["clean_light"]  = df["clean_light"].fillna("").astype(str)
df["comment_text"] = df["comment_text"].fillna("").astype(str)

# ---------- Lexicons ----------
POSITIVE = {
    "semoga","moga","selamat","sehat","sabar","tabah","kuat","baik",
    "bagus","keren","salut","terimakasih","makasih","doa","berkah",
    "lindung","lindungi","dilindungi","sayang","cinta","kasih",
    "alhamdulillah","subhanallah","masyaallah","amin","aamiin",
    "aman","tenang","pulih","membaik","reda","henti","bantu","bantuan",
    "relawan","evakuasi","selamatkan","syukur","sukses","tabah",
    "kuatkan","kesabaran","ketabahan","peduli","dukung","bangkit",
}
NEGATIVE = {
    "ngeri","serem","takut","panik","bahaya","parah","hancur","rusak",
    "korban","meninggal","tewas","mati","wafat","luka","bakar","azab",
    "murka","kiamat","bencana","musibah","sedih","pilu","kasian",
    "kasihan","prihatin","duka","berduka","innalillahi","innalilahi",
    "gagal","salah","bodoh","tolol","goblok","anjir","cok","jir",
    "kacau","gila","mengerikan","mencekam","dahsyat","ganas",
    "petaka","sial","rugi","merugi","keliru","kelalaian","lalai",
}
INTENSIFIERS = {"sangat","banget","bgt","sekali","amat","terlalu","paling"}

def label(text):
    t = " " + text.lower() + " "
    toks = re.findall(r"[a-z]+", t)
    if not toks:
        return 0, 0.0, "neutral"

    pos = sum(1 for w in toks if w in POSITIVE)
    neg = sum(1 for w in toks if w in NEGATIVE)
    intens = sum(1 for w in toks if w in INTENSIFIERS)
    pos += intens if pos > 0 else 0
    neg += intens if neg > 0 else 0

    total = pos + neg
    if total == 0:
        return 0, 0.0, "neutral"
    if pos > neg:
        return 1, pos / total, "positive"
    if neg > pos:
        return 2, neg / total, "negative"
    return 0, 0.5, "neutral"

labels, scores, raw = [], [], []
for t in df["clean_light"].where(df["clean_light"].str.strip() != "",
                                 df["comment_text"]):
    l, s, r = label(t)
    labels.append(l); scores.append(s); raw.append(r)

df["sentiment_label"] = labels
df["sentiment_score"] = scores
df["sentiment_raw"]   = raw

# ---------- Report ----------
print("--- Distribution ---")
print(df["sentiment_label"].value_counts().sort_index().to_string())
print("\n--- By volcano ---")
print(pd.crosstab(df["volcano_name"], df["sentiment_label"]).to_string())

# ---------- Save ----------
cols = ["video_id","video_title","channel_name","volcano_name",
        "eruption_year","comment_text","clean_light",
        "sentiment_label","sentiment_score","sentiment_raw","like_count"]
df[cols].to_csv(OUT / "comments_sentiment_lexicon.csv", index=False)

agg = df.groupby(["video_id","video_title","channel_name",
                  "volcano_name","eruption_year"]).agg(
    n_comments        = ("sentiment_label","size"),
    pct_positive      = ("sentiment_label", lambda x: (x==1).mean()),
    pct_neutral       = ("sentiment_label", lambda x: (x==0).mean()),
    pct_negative      = ("sentiment_label", lambda x: (x==2).mean()),
    dominant_sentiment= ("sentiment_label",
                         lambda x: x.value_counts().idxmax()),
).reset_index()
agg.to_csv(OUT / "video_sentiment_lexicon.csv", index=False)

vol = df.groupby("volcano_name").agg(
    n_comments   = ("sentiment_label","size"),
    pct_positive = ("sentiment_label", lambda x: (x==1).mean()),
    pct_neutral  = ("sentiment_label", lambda x: (x==0).mean()),
    pct_negative = ("sentiment_label", lambda x: (x==2).mean()),
).reset_index().sort_values("n_comments", ascending=False)
vol.to_csv(OUT / "volcano_sentiment_lexicon.csv", index=False)

print(f"\nSaved -> {OUT}")
print("\n--- Volcano-level ---")
print(vol.to_string(index=False))