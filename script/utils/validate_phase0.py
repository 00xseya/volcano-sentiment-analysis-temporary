import pandas as pd
from pathlib import Path

META = Path("data/raw/videos_metadata.csv")
COMM = Path("data/raw/comments_raw.csv")

print("=" * 65)
print("PHASE 0 VALIDATION")
print("=" * 65)

# ---------- Load ----------
meta = pd.read_csv(META)
comm = pd.read_csv(COMM)

# ---------- 1. Basic sizes ----------
print("\n[1] FILE SIZES")
print(f"  Videos in metadata       : {len(meta)}")
print(f"  Unique video_id in meta  : {meta['video_id'].nunique()}")
print(f"  Comments total           : {len(comm)}")
print(f"  Unique videos in comments: {comm['video_id'].nunique()}")

# ---------- 2. Duplicates ----------
print("\n[2] DUPLICATES")
meta_dupes = meta['video_id'].duplicated().sum()
comm_dupes = comm['comment_id'].duplicated().sum() if 'comment_id' in comm else 0
print(f"  Duplicate video_id in meta      : {meta_dupes}")
print(f"  Duplicate comment_id in comments: {comm_dupes}")

# ---------- 3. Coverage ----------
print("\n[3] COVERAGE")
meta_ids = set(meta['video_id'].astype(str))
comm_ids = set(comm['video_id'].astype(str))
missing_comments = meta_ids - comm_ids
missing_meta = comm_ids - meta_ids
print(f"  Videos in meta but NOT in comments: {len(missing_comments)}")
print(f"  Videos in comments but NOT in meta: {len(missing_meta)}")
if missing_comments:
    print(f"    -> {sorted(missing_comments)[:10]}")

# ---------- 4. Comments per video ----------
print("\n[4] COMMENTS PER VIDEO")
per_video = comm.groupby('video_id').size()
print(f"  min={per_video.min()}, max={per_video.max()}, mean={per_video.mean():.1f}")
under_20 = per_video[per_video < 20]
print(f"  Videos with <20 comments: {len(under_20)}")
if len(under_20):
    print(f"    -> {under_20.to_dict()}")

# ---------- 5. Null / empty comment_text ----------
print("\n[5] COMMENT TEXT QUALITY")
print(f"  Null comment_text   : {comm['comment_text'].isna().sum()}")
empty = (comm['comment_text'].astype(str).str.strip() == "").sum()
print(f"  Empty comment_text  : {empty}")

# ---------- 6. Temporal scope ----------
print("\n[6] TEMPORAL SCOPE (2020-2025)")
meta['upload_year'] = pd.to_datetime(
    meta['upload_date'], errors='coerce'
).dt.year
out_of_scope = meta[
    (meta['upload_year'] < 2020) | (meta['upload_year'] > 2025)
]
print(f"  Videos outside 2020-2025: {len(out_of_scope)}")
if len(out_of_scope):
    print(out_of_scope[['video_id', 'upload_date', 'video_title']].to_string())

# ---------- 7. Volcano distribution ----------
print("\n[7] VOLCANO DISTRIBUTION (videos / comments)")
vids_per_vol = meta['volcano_name'].value_counts()
comm_per_vol = comm['volcano_name'].value_counts()
for vol in vids_per_vol.index:
    print(f"  {str(vol):<20} videos={vids_per_vol[vol]:>3}  comments={comm_per_vol.get(vol, 0):>4}")

# ---------- 8. Channel distribution ----------
print("\n[8] CHANNEL TYPE DISTRIBUTION")
print(meta['channel_type'].value_counts(dropna=False).to_string())

# ---------- 9. Null volcano_name ----------
print("\n[9] UNTAGGED VIDEOS")
untagged = meta[meta['volcano_name'].isna()]
print(f"  Videos with no volcano_name: {len(untagged)}")
if len(untagged):
    print(untagged[['video_id', 'video_title']].to_string())

print("\n" + "=" * 65)
print("VALIDATION COMPLETE")
print("=" * 65)