import pandas as pd
from scipy.sparse import load_npz

X = load_npz("data/features/tfidf_heavy_unigram.npz")
print("TF-IDF shape:", X.shape)
print("Density:", X.nnz / (X.shape[0] * X.shape[1]))
print("Row sums (should ≈ 1 due to L2 norm):", X.sum(axis=1).A1[:5])

hand = pd.read_csv("data/features/handcrafted_features.csv")
print("\nHandcrafted sample:")
print(hand.head(3).to_string())