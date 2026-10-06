import pandas as pd
print(pd.read_csv("data/features/top_bigrams.csv").head(15).to_string(index=False))
print()
print(pd.read_csv("data/features/top_trigrams.csv").head(15).to_string(index=False))