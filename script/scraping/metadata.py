import pandas as pd

df = pd.read_csv("data/raw/videos_filtered.csv")

# --- Sanity checks ---
print("Columns:", df.columns.tolist())
print("Rows before filter:", len(df))

# --- Normalize date column ---
if "upload_date_parsed" in df.columns:
    df["upload_date_parsed"] = pd.to_datetime(df["upload_date_parsed"], errors="coerce")
else:
    raise KeyError("Missing column: upload_date_parsed")

# --- Normalize boolean column ---
if "likely_news" in df.columns:
    df["likely_news"] = df["likely_news"].astype(str).str.lower().isin(["true", "1", "yes"])
else:
    raise KeyError("Missing column: likely_news")

# --- Build final table ---
final = pd.DataFrame({
    "video_id":       df["video_id"],
    "url":            df["url"],
    "video_title":    df["scraped_title"],
    "channel_name":   df["channel_name"],
    "channel_type":   df["channel_name"].apply(
        lambda c: "regional" if c and any(
            x in c.lower() for x in ["jatim", "surabaya", "malang", "sumut",
                                      "medan", "ntt", "kupang", "flores",
                                      "bali", "banyuwangi", "lumajang",
                                      "yogyakarta", "jogja", "sumbar",
                                      "padang", "bukittinggi"]
        ) else "national"
    ),
    "upload_date":    df["upload_date_parsed"].dt.strftime("%Y-%m-%d"),
    "volcano_name":   df["volcano_name"],
    "eruption_year":  df["upload_year"],
    "duration_sec":   df["duration_sec"],
    "view_count":     df["view_count"],
    "comment_count":  df["comment_count"],
    "likely_news":    df["likely_news"],
})

# --- Filter ---
before = len(final)
final = final[final["volcano_name"].notna()]
final = final[final["likely_news"] == True]
final = final.reset_index(drop=True)

print(f"Filtered: {before} → {len(final)}")

# --- Save ---
final.to_csv("data/raw/videos_metadata.csv", index=False, encoding="utf-8")
print(f"Final corpus: {len(final)} videos")
print(final["volcano_name"].value_counts().to_string())