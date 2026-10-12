"""
clean_urls.py
-------------
Reads a messy raw URL list (with duplicates, terminal prompt lines, 
duplicated blocks, etc.) and produces a clean, deduplicated list.

Usage:
    python script/clean_urls.py
"""
import re
from pathlib import Path

# ---------- CONFIG ----------
INPUT  = Path("data/urls/video_urls.txt")
OUTPUT = Path("data/raw/video_urls_clean.txt")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# YouTube ID is always 11 chars from [A-Za-z0-9_-]
URL_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)

# Lines that must be discarded (shell prompts, log noise, etc.)
JUNK_MARKERS = [
    "volcano-sentiment-analysis-v1",
    "python script/",
    ">>> ",
    "PS C:\\",
    "C:\\Users\\",
]

# ---------- READ + CLEAN ----------
seen_ids = set()          # for deduplication
clean_lines = []          # output records
stats = {
    "total_lines": 0,
    "blank_lines": 0,
    "junk_lines": 0,
    "no_url_lines": 0,
    "duplicate_lines": 0,
    "kept_lines": 0,
}

with INPUT.open(encoding="utf-8") as f:
    for raw in f:
        stats["total_lines"] += 1
        line = raw.strip()

        # 1. Skip blank lines
        if not line:
            stats["blank_lines"] += 1
            continue

        # 2. Skip shell-prompt / log junk lines
        if any(j in line for j in JUNK_MARKERS):
            stats["junk_lines"] += 1
            continue

        # 3. Skip lines without a YouTube URL
        m = URL_PATTERN.search(line)
        if not m:
            stats["no_url_lines"] += 1
            continue

        vid = m.group(1)

        # 4. Skip duplicate video IDs (keep first occurrence)
        if vid in seen_ids:
            stats["duplicate_lines"] += 1
            continue
        seen_ids.add(vid)

        # 5. Keep a normalized line: "title - https://www.youtube.com/watch?v=ID"
        title = line[:m.start()].rstrip(" -").strip()
        canonical_url = f"https://www.youtube.com/watch?v={vid}"
        clean_lines.append(f"{title} - {canonical_url}")
        stats["kept_lines"] += 1

# ---------- WRITE ----------
OUTPUT.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")

# ---------- REPORT ----------
print("=" * 60)
print("CLEANUP REPORT")
print("=" * 60)
for k, v in stats.items():
    print(f"{k:<20} : {v}")
print("-" * 60)
print(f"Unique videos written to: {OUTPUT}")
print("=" * 60)