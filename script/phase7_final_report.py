"""
Phase 7 — Final Report & Data Merging

Menggabungkan hasil Topic Modeling (Fase 2) dan Sentiment Analysis (Fase 6)
menjadi satu dataset master utuh, lalu membuat visualisasi akhir dan tabel agregasi
untuk menjawab pertanyaan penelitian PBA (relasi sentimen dengan topik dan gunung).

Cara pakai (dari ROOT proyek):
    python script/phase7_final_report.py
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SENTIMENT_PATH = Path("data/processed/comments_with_sentiment.csv")
TOPIC_PATH = Path("reports/topics/comments_with_topics.csv")
OUTPUT_DATA = Path("data/processed/final_analysis_dataset.csv")
FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")

TOPIC_NAME_FALLBACK = {
    "Topic_1": "Kondisi Wilayah & Akses Jalan",
    "Topic_2": "Respon Publik & Berita Bencana",
    "Topic_3": "Refleksi Religius & Bencana Alam",
    "Topic_4": "Ketakutan Gempa & Erupsi Berulang",
    "Topic_5": "Doa Keselamatan & Solidaritas Korban",
}

VALID_SENTIMENTS = ["Negatif", "Netral", "Positif"]


def main() -> None:
    print("=" * 60)
    print("FASE 7 — PENGGABUNGAN DATA & LAPORAN AKHIR")
    print("=" * 60)

    if not SENTIMENT_PATH.exists():
        print(f"[ERROR] {SENTIMENT_PATH} tidak ditemukan. Jalankan phase6_inference.py.")
        return
    if not TOPIC_PATH.exists():
        print(f"[ERROR] {TOPIC_PATH} tidak ditemukan. Jalankan phase2_topic_modeling.py.")
        return

    # 1. Load Data
    df_sent = pd.read_csv(SENTIMENT_PATH, encoding="utf-8-sig")
    df_topic = pd.read_csv(TOPIC_PATH, encoding="utf-8-sig")

    if "source_row" not in df_sent.columns:
        df_sent["source_row"] = df_sent.index
    if "source_row" not in df_topic.columns:
        df_topic["source_row"] = df_topic.index

    # 2. Merge Data secara aman menggunakan primary key source_row
    topic_cols = [c for c in ["source_row", "dominant_topic", "dominant_topic_label", "topic_score"] if c in df_topic.columns]
    df_final = pd.merge(
        df_sent.drop(columns=[c for c in topic_cols if c != "source_row"], errors="ignore"),
        df_topic[topic_cols],
        on="source_row",
        how="left",
        validate="many_to_one"
    )

    # Tambahkan nama topik deskriptif jika belum ada
    if "dominant_topic_label" not in df_final.columns and "dominant_topic" in df_final.columns:
        df_final["dominant_topic_label"] = df_final["dominant_topic"].map(TOPIC_NAME_FALLBACK)

    # Simpan Dataset Final
    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_final.to_csv(OUTPUT_DATA, index=False, encoding="utf-8-sig")
    print(f"[OK] Master dataset final ({len(df_final)} baris) disimpan -> {OUTPUT_DATA}")

    # ================= VISUALISASI & TABEL RINGKASAN =================
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # Filter data dengan label sentimen valid untuk penghitungan persentase yang akurat
    df_valid_sent = df_final[df_final["sentiment"].isin(VALID_SENTIMENTS)].copy()

    # 1. Sentimen per Gunung (Persentase)
    if "volcano_name" in df_valid_sent.columns and len(df_valid_sent) > 0:
        ct_volcano_pct = (pd.crosstab(df_valid_sent["volcano_name"], df_valid_sent["sentiment"], normalize="index") * 100).round(2)
        cols_sent = [c for c in VALID_SENTIMENTS if c in ct_volcano_pct.columns]
        ct_volcano_pct = ct_volcano_pct[cols_sent]

        # Simpan tabel CSV agregasi untuk makalah/laporan
        ct_volcano_csv = REPORTS_DIR / "sentiment_by_volcano_crosstab.csv"
        ct_volcano_pct.to_csv(ct_volcano_csv)
        print(f"[OK] Tabel sentimen per gunung disimpan -> {ct_volcano_csv}")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"Negatif": "#c44e52", "Netral": "#ddb310", "Positif": "#55a868"}
        ct_volcano_pct.plot(kind="bar", stacked=True, color=[colors[c] for c in cols_sent], ax=ax)
        plt.title("Distribusi Sentimen per Gunung Berapi (% Seluruh Dataset)", pad=14, fontweight="bold", fontsize=13)
        plt.xlabel("Nama Gunung", fontweight="bold", fontsize=11)
        plt.ylabel("Persentase Komentar (%)", fontweight="bold", fontsize=11)
        plt.legend(title="Sentimen", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out1 = FIGURES_DIR / "final_sentiment_per_volcano.png"
        plt.savefig(out1, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Plot disimpan -> {out1}")

    # 2. Sentimen per Topik Dominan
    topic_col = "dominant_topic_label" if "dominant_topic_label" in df_valid_sent.columns else "dominant_topic"
    if topic_col in df_valid_sent.columns and len(df_valid_sent) > 0:
        ct_topic_pct = (pd.crosstab(df_valid_sent[topic_col], df_valid_sent["sentiment"], normalize="index") * 100).round(2)
        cols_sent = [c for c in VALID_SENTIMENTS if c in ct_topic_pct.columns]
        ct_topic_pct = ct_topic_pct[cols_sent]

        # Simpan tabel CSV agregasi
        ct_topic_csv = REPORTS_DIR / "sentiment_by_topic_crosstab.csv"
        ct_topic_pct.to_csv(ct_topic_csv)
        print(f"[OK] Tabel sentimen per topik disimpan -> {ct_topic_csv}")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"Negatif": "#c44e52", "Netral": "#ddb310", "Positif": "#55a868"}
        ct_topic_pct.plot(kind="bar", stacked=True, color=[colors[c] for c in cols_sent], ax=ax)
        plt.title("Distribusi Sentimen Berdasarkan Tema/Topik LDA (%)", pad=14, fontweight="bold", fontsize=13)
        plt.xlabel("Topik / Framing Bencana", fontweight="bold", fontsize=11)
        plt.ylabel("Persentase Sentimen (%)", fontweight="bold", fontsize=11)
        plt.legend(title="Sentimen", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        out2 = FIGURES_DIR / "final_sentiment_per_topic.png"
        plt.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Plot disimpan -> {out2}")

    # 3. Distribusi Komentar per Topik
    if topic_col in df_final.columns:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        counts = df_final[topic_col].value_counts().sort_values(ascending=False)
        sns.barplot(x=counts.values, y=counts.index, color="#4c72b0", ax=ax)
        for i, v in enumerate(counts.values):
            ax.text(v + 1.5, i, str(v), color="black", va="center", fontweight="bold")
        plt.title("Frekuensi Pembahasan Komentar per Topik (LDA)", pad=14, fontweight="bold", fontsize=13)
        plt.xlabel("Jumlah Komentar", fontweight="bold", fontsize=11)
        plt.ylabel("Tema / Topik", fontweight="bold", fontsize=11)
        plt.tight_layout()
        out3 = FIGURES_DIR / "final_topic_distribution.png"
        plt.savefig(out3, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Plot disimpan -> {out3}")

    print("\nProses Fase 8 selesai! Seluruh hasil telah digabungkan dan visualisasi siap.")


if __name__ == "__main__":
    main()
