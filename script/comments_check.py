import pandas as pd
df = pd.read_csv("data/processed/comments_clean.csv")
print(df[["comment_text", "clean_light", "clean_heavy"]].head(10).to_string())
print("\nColumns:", list(df.columns))