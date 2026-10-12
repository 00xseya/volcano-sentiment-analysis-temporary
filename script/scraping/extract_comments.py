import pandas as pd
import time
from pathlib import Path
from ytscrape import YouTube, CommentSort

META = Path("data/raw/videos_metadata.csv")
OUT = Path("data/raw/comments_raw.csv")

meta = pd.read_csv(META)
all_comments = []

with YouTube(language="id", region="ID") as yt:
    for i, row in meta.iterrows():
        vid = row["video_id"]
        print(f"[{i+1}/{len(meta)}] {vid} | {row['video_title'][:60]}")
        try:
            count = 0
            for c in yt.comments(
                row["url"],
                sort=CommentSort.NEWEST,
                max_results=20,
            ):
                all_comments.append({
                    "comment_id":       c.id if hasattr(c, "id") else None,
                    "video_id":         vid,
                    "video_title":      row["video_title"],
                    "channel_name":     row["channel_name"],
                    "channel_type":     row["channel_type"],
                    "upload_date":      row["upload_date"],
                    "volcano_name":     row["volcano_name"],
                    "eruption_year":    row["eruption_year"],
                    "comment_text":     c.text,
                    "like_count":       getattr(c, "like_count", None),
                    "comment_timestamp": getattr(c, "published_time", None),
                    "is_reply":         getattr(c, "is_reply", False),
                })
                count += 1
                if count >= 20:
                    break
            print(f"    collected {count} comments")
        except Exception as e:
            print(f"    FAILED: {e}")

        # Rate-limit politeness
        time.sleep(2)

pd.DataFrame(all_comments).to_csv(OUT, index=False, encoding="utf-8")
print(f"\nSaved {len(all_comments)} comments -> {OUT}")