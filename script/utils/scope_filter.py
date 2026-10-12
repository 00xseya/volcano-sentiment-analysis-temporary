import pandas as pd
from datetime import datetime

df = pd.read_csv("data/raw/videos_metadata_raw.csv")

# --- 1. Parse upload_date into datetime ---
def parse_date(d):
    if pd.isna(d):
        return pd.NaT
    d = str(d).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return pd.NaT

df["upload_date_parsed"] = df["upload_date"].apply(parse_date)
df["upload_year"] = df["upload_date_parsed"].dt.year

# --- 2. Temporal filter: 2020-2025 ---
before = len(df)
df = df[(df["upload_year"] >= 2020) & (df["upload_year"] <= 2025)]
print(f"Temporal filter: removed {before - len(df)} out-of-scope videos")

# --- 3. Volcano tagging from title ---
VOLCANO_KEYWORDS = {
    "Semeru": ["semeru"],
    "Lewotobi": ["lewotobi"],
    "Marapi": ["marapi"],  # careful: 'marapi' vs 'merapi'
    "Merapi": ["merapi"],  # check after Marapi
    "Anak Krakatau": ["anak krakatau", "krakatau"],
    "Sinabung": ["sinabung"],
    "Raung": ["raung"],
    "Ili Lewotolok": ["ile lewotolok", "ili lewotolok", "lewotolok"],
    "Ruang": ["ruang"],
    "Ibu": ["gunung ibu", "ibu halmahera"],
    "Kerinci": ["kerinci"],
}

def tag_volcano(title):
    if pd.isna(title):
        return None
    t = title.lower()
    # Check "marapi" and "merapi" carefully - Marapi is in West Sumatra,
    # Merapi is in Central Java/Yogyakarta
    for name, kws in VOLCANO_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return name
    return None

df["volcano_name"] = df["scraped_title"].fillna(df["raw_title"]).apply(tag_volcano)

# --- 4. Flag untagged videos for manual review ---
untagged = df[df["volcano_name"].isna()]
print(f"\n{len(untagged)} videos could not be auto-tagged:")
print(untagged[["video_id", "scraped_title", "channel_name"]].to_string())

# --- 5. Flag possible non-news content ---
NEWS_CHANNEL_HINTS = [
    "kompas", "metro", "tvone", "tv one", "cnn", "detik", "tribun",
    "suara", "kumparan", "sindo", "inews", "liputan", "beritasatu",
    "kabar", "trans7", "trans tv", "antv", "rcti", "mnc", "garuda",
]
def is_news_channel(name):
    if pd.isna(name):
        return False
    n = name.lower()
    return any(h in n for h in NEWS_CHANNEL_HINTS)

df["likely_news"] = df["channel_name"].apply(is_news_channel)

# --- 6. Save ---
df.to_csv("data/raw/videos_filtered.csv", index=False, encoding="utf-8")
print(f"\nSaved {len(df)} videos -> data/raw/videos_filtered.csv")