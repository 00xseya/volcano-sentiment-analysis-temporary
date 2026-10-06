import re
import pandas as pd
from pathlib import Path

INPUT = Path("data/raw/video_urls_working.txt")
OUTPUT = Path("data/raw/videos_parsed.csv")

# Match YouTube video IDs (11 chars) from watch?v= or youtu.be/
URL_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)

rows = []
with INPUT.open(encoding="utf-8") as f:
    for line_no, raw in enumerate(f, start=1):
        line = raw.strip()
        if not line:
            continue  # skip blank lines

        match = URL_PATTERN.search(line)
        if not match:
            print(f"[SKIP line {line_no}] no URL found: {line[:80]}")
            continue

        video_id = match.group(1)
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Title = everything before the URL, stripping trailing " - "
        title = line[:match.start()].rstrip(" -").strip()

        rows.append({
            "video_id": video_id,
            "url": url,
            "raw_title": title,
            "source_line": line_no,
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False, encoding="utf-8")
print(f"\nParsed {len(df)} entries -> {OUTPUT}")
print(f"Unique video_id count: {df['video_id'].nunique()}")
print(f"Duplicates: {len(df) - df['video_id'].nunique()}")