import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/videos_parsed.csv")

before = len(df)
# Keep first occurrence of each video_id
df = df.drop_duplicates(subset="video_id", keep="first").reset_index(drop=True)
after = len(df)

print(f"Before: {before}, After: {after}, Removed: {before - after}")

df.to_csv("data/raw/videos_unique.csv", index=False, encoding="utf-8")
print(f"Saved {after} unique videos to data/raw/videos_unique.csv")