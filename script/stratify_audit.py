import pandas as pd

df = pd.read_csv("data/raw/videos_filtered.csv")

print("=== By Volcano ===")
print(df["volcano_name"].value_counts(dropna=False).to_string())

print("\n=== By Year ===")
print(df["upload_year"].value_counts().sort_index().to_string())

print("\n=== By Channel (top 20) ===")
print(df["channel_name"].value_counts().head(20).to_string())

print("\nTotal videos:", len(df))