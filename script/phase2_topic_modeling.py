"""
Fase 4 — Topic Modeling (LDA)
Menganalisis tema atau framing dari seluruh komentar (Unsupervised Learning).
Skrip ini memuat data komentar bersih hasil Phase 1 (menggunakan kolom clean_heavy),
melakukan evaluasi Topic Coherence (U_mass) & Perplexity untuk berbagai jumlah topik (k),
lalu melatih model LDA akhir dan menyimpan distribusi topik per komentar.

Cara pakai:
    python script/phase2_topic_modeling.py
    python script/phase2_topic_modeling.py --num-topics 5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Konfigurasi Path
INPUT_PATH = Path("data/processed/comments_clean.csv")
OUTPUT_TOPICS_DIR = Path("reports/topics")
OUTPUT_FIGURES_DIR = Path("reports/figures")

# Stopwords domain khusus bencana gunung berapi (termasuk kata slang/stemmed/baku)
DOMAIN_STOPWORDS = [
    "yang", "di", "ke", "dari", "ini", "itu", "dan", "atau", "juga", "dengan",
    "untuk", "pada", "adalah", "ada", "tidak", "ya", "yg", "aja", "nya", "buat",
    "sama", "sudah", "udah", "lagi", "terus", "kalau", "kalo", "bisa", "lebih",
    "sih", "kok", "dong", "kan", "kita", "kamu", "aku", "saya", "dia", "mereka",
    "kami", "kalian", "gak", "nggak", "ga", "enggak", "belum", "begitu", "seperti",
    "gunung", "meletus", "letus", "erupsi", "semeru", "meru", "merapi", "rapi",
    "rap", "marapi", "video", "orang", "jadi", "tau", "tahu", "tuh", "nih"
]

# Label interpretatif manusia untuk konfigurasi default k=5 (random_state=42)
DEFAULT_TOPIC_LABELS_K5 = {
    "Topic_1": "Kondisi Wilayah & Akses Jalan",
    "Topic_2": "Respon Publik & Berita Bencana",
    "Topic_3": "Refleksi Religius & Bencana Alam",
    "Topic_4": "Ketakutan Gempa & Erupsi Berulang",
    "Topic_5": "Doa Keselamatan & Solidaritas Korban",
}


def calculate_umass_coherence(lda: LatentDirichletAllocation, X_binary, doc_counts, top_n: int = 10) -> float:
    """Menghitung U_mass coherence score intrinsik dari matriks ko-okurensi kata."""
    total_coherence = 0.0
    num_topics = lda.n_components
    for comp in lda.components_:
        top_indices = comp.argsort()[:-top_n - 1:-1]
        topic_score = 0.0
        for m_idx, m in enumerate(top_indices[1:], start=1):
            for l in top_indices[:m_idx]:
                co_doc = (X_binary[:, m].multiply(X_binary[:, l])).sum()
                topic_score += np.log((co_doc + 1.0) / (doc_counts[l] + 1.0))
        total_coherence += topic_score
    return float(total_coherence / num_topics)


def evaluate_topic_numbers(X, k_range=range(3, 8)) -> pd.DataFrame:
    """Mengevaluasi variasi jumlah topik k dengan Perplexity dan U_mass Coherence."""
    print("Mengevaluasi variasi jumlah topik (k)...")
    X_binary = (X > 0).astype(int)
    doc_counts = np.asarray(X_binary.sum(axis=0)).flatten()

    eval_results = []
    for k in sorted(set(k_range)):
        model = LatentDirichletAllocation(
            n_components=k,
            random_state=42,
            max_iter=25,
            learning_method="batch",
            n_jobs=-1,
        )
        model.fit(X)
        perp = model.perplexity(X)
        ll = model.score(X)
        umass = calculate_umass_coherence(model, X_binary, doc_counts, top_n=10)
        eval_results.append({
            "k": k,
            "perplexity": round(perp, 2),
            "log_likelihood": round(ll, 2),
            "umass_coherence": round(umass, 3),
        })

    eval_df = pd.DataFrame(eval_results)

    # Plot evaluasi
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color = "tab:blue"
    ax1.set_xlabel("Jumlah Topik (k)", fontweight="bold")
    ax1.set_ylabel("U_mass Coherence (Makin tinggi makin baik)", color=color, fontweight="bold")
    ax1.plot(eval_df["k"], eval_df["umass_coherence"], marker="o", color=color, linewidth=2, label="U_mass Coherence")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Perplexity (Makin rendah makin baik)", color=color, fontweight="bold")
    ax2.plot(eval_df["k"], eval_df["perplexity"], marker="s", linestyle="--", color=color, linewidth=2, label="Perplexity")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Evaluasi Jumlah Topik LDA (Topic Coherence vs Perplexity)", pad=14, fontweight="bold")
    fig.tight_layout()
    plot_path = OUTPUT_FIGURES_DIR / "lda_topic_evaluation.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Grafik evaluasi topik disimpan -> {plot_path}")

    return eval_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — LDA Topic Modeling")
    parser.add_argument("--num-topics", type=int, default=5, help="Jumlah topik LDA (default: 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("FASE 4 — TOPIC MODELING DENGAN LDA")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print(f"[ERROR] Data {INPUT_PATH} tidak ditemukan.")
        print("Pastikan sudah menjalankan script/phase1_preprocess.py sebelumnya.")
        return

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    if "source_row" not in df.columns:
        df["source_row"] = df.index

    # Prioritaskan clean_heavy -> clean_light -> comment_text
    clean_heavy_col = df["clean_heavy"] if "clean_heavy" in df.columns else pd.Series([""] * len(df))
    clean_light_col = df["clean_light"] if "clean_light" in df.columns else df["comment_text"]
    texts = [
        str(h).strip() if pd.notna(h) and str(h).strip()
        else (str(l).strip() if pd.notna(l) else "")
        for h, l in zip(clean_heavy_col, clean_light_col)
    ]

    print(f"Memproses {len(texts)} komentar untuk Topic Modeling...")

    tf_vectorizer = CountVectorizer(
        max_df=0.85,
        min_df=2,
        stop_words=DOMAIN_STOPWORDS,
    )
    tf_matrix = tf_vectorizer.fit_transform(texts)
    print(f"Dimensi matriks kata: {tf_matrix.shape}")

    # 1. Evaluasi kuantitatif variasi k (sertakan num_topics pilihan pengguna)
    k_eval_range = set(range(3, 8))
    k_eval_range.add(args.num_topics)
    eval_df = evaluate_topic_numbers(tf_matrix, k_range=sorted(k_eval_range))
    OUTPUT_TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    eval_csv_path = OUTPUT_TOPICS_DIR / "lda_coherence_evaluation.csv"
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"[OK] Hasil evaluasi disimpan -> {eval_csv_path}")

    # 2. Fit Final LDA Model (menggunakan konfigurasi yang konsisten)
    num_topics = args.num_topics
    print(f"\nMelatih model LDA akhir dengan k = {num_topics} topik...")
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=25,
        learning_method="batch",
        random_state=42,
        n_jobs=-1,
    )
    lda_model.fit(tf_matrix)

    # 3. Kata Kunci Teratas per Topik
    feature_names = tf_vectorizer.get_feature_names_out()
    topics_dict = {}
    print("\nKata Kunci Teratas:")
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topic_key = f"Topic_{topic_idx + 1}"
        topics_dict[topic_key] = top_words
        label_desc = DEFAULT_TOPIC_LABELS_K5.get(topic_key, topic_key) if num_topics == 5 else topic_key
        print(f"[{topic_key}] {label_desc}: {', '.join(top_words)}")

    # Simpan tabel kata kunci
    keywords_df = pd.DataFrame(topics_dict)
    keywords_path = OUTPUT_TOPICS_DIR / "lda_topics_keywords.csv"
    keywords_df.to_csv(keywords_path, index=False)
    print(f"[OK] Kata kunci disimpan -> {keywords_path}")

    # 4. Distribusi Topik Dominan per Dokumen
    doc_topic_dists = lda_model.transform(tf_matrix)
    dominant_topic_ids = np.argmax(doc_topic_dists, axis=1) + 1

    df_output = df.copy()
    df_output["dominant_topic"] = [f"Topic_{t}" for t in dominant_topic_ids]
    if num_topics == 5:
        df_output["dominant_topic_label"] = [DEFAULT_TOPIC_LABELS_K5.get(f"Topic_{t}", f"Topic_{t}") for t in dominant_topic_ids]
    else:
        df_output["dominant_topic_label"] = df_output["dominant_topic"]
    df_output["topic_score"] = np.max(doc_topic_dists, axis=1).round(3)

    out_file = OUTPUT_TOPICS_DIR / "comments_with_topics.csv"
    df_output.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] Dataset dengan topik disimpan -> {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
