"""
Phase 1: Full text preprocessing pipeline for Indonesian YouTube comments.
Steps: lowercasing, contraction expansion, punctuation/emoji removal,
       tokenization, stopword removal, slang replacement, spelling correction,
       stemming, rare word removal.
"""
import re
import unicodedata
from pathlib import Path
import pandas as pd
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from collections import Counter

# ---------- CONFIG ----------
IN  = Path("data/raw/comments_raw.csv")
OUT = Path("data/processed/comments_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

SLANG_FILE = Path("data/raw/slang_dict.txt")

# ---------- LOAD ----------
df = pd.read_csv(IN)
df["source_row"] = df.index
print(f"Loaded {len(df)} comments")

# ---------- 1. Load slang dictionary ----------
slang = {}
if SLANG_FILE.exists():
    for line in SLANG_FILE.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            slang[parts[0]] = parts[1]
        elif len(parts) == 1:
            slang[parts[0]] = parts[0]
print(f"Loaded {len(slang)} slang mappings")

# ---------- 2. Stopwords + stemmer ----------
stopwords = set(StopWordRemoverFactory().get_stop_words())
stemmer = StemmerFactory().create_stemmer()

# Additional domain stopwords specific to Indonesian volcano news
extra_stops = {
    "yg", "dg", "dgn", "utk", "tdk", "sdh", "blm", "jg", "dr",
    "nya", "si", "sih", "dong", "kok", "lah", "pun", "deh", "ya",
    "ini", "itu", "ada", "adalah", "akan", "juga", "sudah", "belum",
    "masih", "bisa", "dapat", "harus", "mau", "ingin", "saja", "aja",
    "sangat", "banget", "sekali", "paling", "lebih", "kurang",
}
stopwords |= extra_stops

# ---------- 3. Helper functions ----------
URL_RE    = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE= re.compile(r"@\w+")
HASHTAG_RE= re.compile(r"#\w+")
HTML_RE   = re.compile(r"<[^>]+>")
NUMBER_RE = re.compile(r"\d+")
SPACE_RE  = re.compile(r"\s+")
NONALPHA  = re.compile(r"[^a-z\s]")

def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

def normalize_unicode(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def expand_slang(tokens):
    out = []
    for t in tokens:
        if t in slang:
            rep = slang[t]
            if rep:
                out.extend(rep.split())
        else:
            out.append(t)
    return out

def full_preprocess(text, aggressive=True):
    """Returns (clean_text, tokens)"""
    if pd.isna(text) or not str(text).strip():
        return "", []

    t = str(text)

    # a) Remove URLs, mentions, hashtags, HTML
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = HTML_RE.sub(" ", t)

    # b) Remove emojis
    t = remove_emojis(t)

    # c) Lowercase
    t = t.lower()

    # d) Unicode normalization (handles odd accented chars)
    t = normalize_unicode(t)

    # e) Remove numbers
    t = NUMBER_RE.sub(" ", t)

    # f) Tokenize on whitespace (before punctuation strip so we keep words whole)
    tokens = SPACE_RE.split(t.strip())

    # g) Remove punctuation from each token
    tokens = [NONALPHA.sub("", tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok]

    # h) Slang expansion
    tokens = expand_slang(tokens)

    # i) Drop single-character tokens (noise)
    tokens = [tok for tok in tokens if len(tok) > 1]

    if not aggressive:
        # Light mode: stop here. Used for sentiment tasks where we
        # want to preserve as much signal as possible.
        return " ".join(tokens), tokens

    # j) Stopword removal
    tokens = [tok for tok in tokens if tok not in stopwords]

    # k) Stemming
    tokens = [stemmer.stem(tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok and tok not in stopwords]

    return " ".join(tokens), tokens

# ---------- 4. Apply ----------
print("Preprocessing (light + heavy)...")

light_texts, light_tokens = [], []
heavy_texts, heavy_tokens = [], []

for i, text in enumerate(df["comment_text"]):
    lt, ltok = full_preprocess(text, aggressive=False)
    ht, htok = full_preprocess(text, aggressive=True)
    light_texts.append(lt)
    light_tokens.append(ltok)
    heavy_texts.append(ht)
    heavy_tokens.append(htok)
    if (i + 1) % 50 == 0:
        print(f"  ...{i+1}/{len(df)}")

df["clean_light"]  = light_texts   # for sentiment
df["clean_heavy"]  = heavy_texts   # for topic modelling / clustering
df["tokens_light"] = light_tokens
df["tokens_heavy"] = heavy_tokens
df["token_count_light"] = [len(t) for t in light_tokens]
df["token_count_heavy"] = [len(t) for t in heavy_tokens]

# ---------- 5. Rare word removal (heavy only) ----------
all_tokens = [t for toks in heavy_tokens for t in toks]
counts = Counter(all_tokens)
RARE_THRESHOLD = 2
keep_vocab = {w for w, c in counts.items() if c >= RARE_THRESHOLD}
print(f"Vocabulary before rare filter: {len(counts)}")
print(f"Vocabulary after  rare filter: {len(keep_vocab)}")

def filter_rare(tokens):
    return [t for t in tokens if t in keep_vocab]

df["tokens_heavy"] = df["tokens_heavy"].apply(filter_rare)
df["clean_heavy"]  = df["tokens_heavy"].apply(lambda t: " ".join(t))
df["token_count_heavy"] = df["tokens_heavy"].apply(len)

# ---------- 6. Common word removal note ----------
# Common-word removal is done via max_df in the vectorizer (Phase 2).
# We don't drop them here, because TF-IDF uses IDF weighting to handle them.

# ---------- 7. Save ----------
# Convert token list to string form for CSV storage
df["tokens_light"] = df["tokens_light"].apply(lambda t: "|".join(t))
df["tokens_heavy"] = df["tokens_heavy"].apply(lambda t: "|".join(t))

cols = ["source_row"] + [c for c in df.columns if c != "source_row"]
df = df[cols]

df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\nSaved -> {OUT}")

# ---------- 8. Summary ----------
print("\n" + "=" * 60)
print("PHASE 1 SUMMARY")
print("=" * 60)
print(f"Comments processed:      {len(df)}")
print(f"Avg tokens (light):      {df['token_count_light'].mean():.1f}")
print(f"Avg tokens (heavy):      {df['token_count_heavy'].mean():.1f}")
print(f"Comments left empty (heavy): {(df['token_count_heavy'] == 0).sum()}")

# Sample output
print("\n--- SAMPLE (first 3 comments) ---")
for i in range(min(3, len(df))):
    print(f"\n[{i+1}] ORIGINAL: {df['comment_text'].iloc[i][:120]}")
    print(f"    LIGHT  : {df['clean_light'].iloc[i][:120]}")
    print(f"    HEAVY  : {df['clean_heavy'].iloc[i][:120]}")