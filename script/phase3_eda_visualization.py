"""
Fase 5 — Exploratory Data Analysis (EDA) & Visualisasi
Skrip ini akan memuat dataset yang sudah dilabeli (train & test) dan menghasilkan:
1. Distribusi sentimen secara keseluruhan (Bar Chart).
2. Distribusi sentimen berdasarkan gunung berapi (Stacked Bar Chart).
3. WordCloud untuk masing-masing sentimen (Positif, Negatif, Netral).

Cara pakai (dari ROOT proyek):
    python script/phase3_eda_visualization.py
"""
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

# Konfigurasi Path
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "data_train_labeled.csv"
TEST_PATH = DATA_DIR / "data_test_labeled.csv"
OUTPUT_DIR = Path("reports/figures")

# Stopwords bahasa Indonesia untuk WordCloud
ID_STOPWORDS = {
    "yang", "di", "ke", "dari", "ini", "itu", "dan", "atau", "juga", "dengan",
    "untuk", "pada", "adalah", "ada", "tidak", "ya", "yg", "aja", "nya", "buat",
    "sama", "sudah", "udah", "lagi", "terus", "kalau", "kalo", "bisa", "lebih",
    "sih", "kok", "dong", "kan", "kita", "kamu", "aku", "saya", "dia", "mereka",
    "kami", "kalian", "gak", "nggak", "ga", "enggak", "belum", "begitu", "seperti",
    "video", "orang", "jadi"
}


def load_data() -> pd.DataFrame:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print(f"[ERROR] Data belum ada. Harap selesaikan pelabelan dan split (Tahap 1 & 2).")
        sys.exit(1)

    df_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    df_test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")

    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Data dimuat: {len(df)} baris berlabel.")
    return df


def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="gold_label", order=["Positif", "Netral", "Negatif"], palette="pastel")

    plt.title("Distribusi Gold Label Sentimen (Data Anotasi Manual)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Sentimen", fontsize=11, fontweight="bold")
    plt.ylabel("Jumlah Komentar", fontsize=11, fontweight="bold")

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="baseline", fontsize=11, color="black", xytext=(0, 5),
                    textcoords="offset points")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "eda_sentiment_distribution.png"
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Plot disimpan ke {out_path}")
    plt.close()


def plot_sentiment_per_volcano(df: pd.DataFrame) -> None:
    if "volcano_name" not in df.columns:
        print("[SKIP] Kolom volcano_name tidak ditemukan.")
        return

    # Hitung jumlah tiap label per gunung
    volcano_counts = df.groupby(["volcano_name", "gold_label"]).size().unstack(fill_value=0)

    # Urutkan berdasarkan total komentar terbanyak
    volcano_counts["Total"] = volcano_counts.sum(axis=1)
    volcano_counts = volcano_counts.sort_values("Total", ascending=False).drop(columns=["Total"])

    # Pastikan urutan kolom label
    cols = [c for c in ["Positif", "Netral", "Negatif"] if c in volcano_counts.columns]
    volcano_counts = volcano_counts[cols]

    color_map = {"Positif": "#55a868", "Netral": "#ddb310", "Negatif": "#c44e52"}
    colors = [color_map.get(col, "#cccccc") for col in volcano_counts.columns]

    ax = volcano_counts.plot(kind="bar", stacked=True, color=colors, figsize=(10, 6))

    plt.title("Distribusi Sentimen per Gunung Berapi (Sampel Anotasi Manual)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Gunung Berapi", fontsize=11, fontweight="bold")
    plt.ylabel("Jumlah Komentar", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Sentimen")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "eda_sentiment_per_volcano.png"
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Plot disimpan ke {out_path}")
    plt.close()


def generate_wordclouds(df: pd.DataFrame) -> None:
    labels = ["Positif", "Negatif", "Netral"]

    for label in labels:
        text_data = df[df["gold_label"] == label]["comment_text"]
        if len(text_data) == 0:
            continue

        text_combined = " ".join(text_data.dropna().astype(str).tolist())
        cmap = "viridis" if label == "Positif" else ("magma" if label == "Negatif" else "cividis")

        wordcloud = WordCloud(
            width=800, height=400,
            background_color="white",
            stopwords=ID_STOPWORDS,
            colormap=cmap,
            max_words=100,
            random_state=42,
        ).generate(text_combined)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud — Sentimen {label}", fontsize=14, fontweight="bold", pad=10)

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"wordcloud_{label.lower()}.png"
        plt.savefig(out_path, dpi=200)
        print(f"[OK] WordCloud disimpan ke {out_path}")
        plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 50)
    print("FASE 5 — Exploratory Data Analysis & Visualisasi")
    print("=" * 50)

    df = load_data()
    plot_sentiment_distribution(df)
    plot_sentiment_per_volcano(df)
    generate_wordclouds(df)

    print("\nVisualisasi selesai! Silakan periksa folder", OUTPUT_DIR)


if __name__ == "__main__":
    main()
