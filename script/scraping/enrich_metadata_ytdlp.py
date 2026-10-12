"""
enrich_metadata_ytdlp.py
--------------------------
Enriches video metadata using yt-dlp.
- Skips videos that have already been successfully processed.
- Handles individual video errors so one failure doesn't stop the batch.
- Provides clear, real-time progress and a summary report at the end.
"""
import pandas as pd
import yt_dlp
from pathlib import Path
import time

# ---------- CONFIG ----------
IN = Path("data/raw/videos_unique.csv")
OUT = Path("data/raw/videos_metadata_raw.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# yt-dlp options: quiet mode, no download, no playlists
YDL_OPTS = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': False,
    'no_warnings': True,
    'ignoreerrors': True, # Don't crash the entire process on one video error
    'retries': 3,
    'fragment_retries': 3,
    'socket_timeout': 30,
}

# ---------- LOAD INPUT & SKIP ALREADY PROCESSED ----------
df_input = pd.read_csv(IN)

# Check for existing progress to resume from
if OUT.exists() and OUT.stat().st_size > 0:
    print(f"Found existing progress at {OUT}. Resuming...")
    df_existing = pd.read_csv(OUT)
    processed_ids = set(df_existing['video_id'].astype(str))
    # Filter out videos that were already processed
    df_to_process = df_input[~df_input['video_id'].astype(str).isin(processed_ids)].copy()
    print(f"  - {len(processed_ids)} videos already processed.")
    print(f"  - {len(df_to_process)} videos remaining.")
else:
    df_existing = pd.DataFrame()
    df_to_process = df_input.copy()
    print(f"Starting fresh. Processing {len(df_to_process)} videos...")

if df_to_process.empty:
    print("All videos have already been processed. Nothing to do.")
    exit()

# ---------- PROCESS VIDEOS ----------
new_records = []
total = len(df_to_process)
success_count = 0
fail_count = 0

with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
    for i, row in df_to_process.iterrows():
        vid = row["video_id"]
        url = row["url"]
        print(f"[{i+1}/{total}] Processing: {vid}")

        try:
            # Extract metadata without downloading
            info = ydl.extract_info(url, download=False)

            # Check if extraction was successful
            if info is None:
                raise ValueError("yt-dlp returned no info (video likely unavailable)")

            new_records.append({
                "video_id": vid,
                "url": url,
                "raw_title": row["raw_title"],
                "scraped_title": info.get("title"),
                "channel_name": info.get("uploader") or info.get("channel"),
                "upload_date": info.get("upload_date"), # Format: YYYYMMDD
                "duration_sec": info.get("duration"),
                "view_count": info.get("view_count"),
                "comment_count": info.get("comment_count"),
                "error": None,
            })
            print(f"    OK: {info.get('title', 'N/A')[:70]}")
            success_count += 1

        except Exception as e:
            # Record the failure but continue with the next video
            new_records.append({
                "video_id": vid,
                "url": url,
                "raw_title": row["raw_title"],
                "scraped_title": None,
                "channel_name": None,
                "upload_date": None,
                "duration_sec": None,
                "view_count": None,
                "comment_count": None,
                "error": str(e),
            })
            print(f"    FAILED: {e}")
            fail_count += 1

        # Be polite to YouTube's servers
        time.sleep(1.5)

# ---------- SAVE ----------
# Combine with any previous results and write the final file
df_new = pd.DataFrame(new_records)
df_final = pd.concat([df_existing, df_new], ignore_index=True)
df_final.to_csv(OUT, index=False, encoding="utf-8")

# ---------- FINAL REPORT ----------
print("\n" + "=" * 65)
print("ENRICHMENT COMPLETE")
print("=" * 65)
print(f"Successfully processed this run : {success_count}")
print(f"Failed this run                 : {fail_count}")
print(f"Total videos in final file     : {len(df_final)}")
print(f"Metadata saved to              : {OUT}")
print("=" * 65)