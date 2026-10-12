"""
Phase 5 — Evaluasi model sentimen terhadap gold label (data uji hasil anotasi manual).

Cara pakai (dari ROOT proyek):
    python script/phase5_evaluate_model.py --model dummy              # baseline acak
    python script/phase5_evaluate_model.py --model lexicon            # baseline lexicon-based
    python script/phase5_evaluate_model.py --model indobert-zeroshot  # zero-shot Indonesian RoBERTa
    python script/phase5_evaluate_model.py --model tfidf              # supervised TF-IDF + LogReg (Phase 4)
    python script/phase5_evaluate_model.py --model indobert           # supervised IndoBERT Fine-tuned

Output:
    - classification_report di terminal
    - reports/figures/confusion_matrix_<model>.png
    - reports/error_analysis_<model>.csv (analisis kesalahan prediksi)
    - reports/model_comparison.csv (tabel komparasi seluruh model)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
TEST_PATH = Path("data/data_test_labeled.csv")
FIG_DIR = Path("reports/figures")
REPORT_DIR = Path("reports")
COMPARISON_CSV = REPORT_DIR / "model_comparison.csv"
LEXICON_CSV = Path("data/sentiment/comments_sentiment_lexicon.csv")
ZEROSHOT_CSV = Path("data/sentiment/comments_sentiment_transformer.csv")

TEXT_COL = "comment_text"
LABEL_COL = "gold_label"
LABELS = ["Positif", "Netral", "Negatif"]          # urutan tampil di laporan & matriks

INT_TO_LABEL = {0: "Netral", 1: "Positif", 2: "Negatif"}


# ---------------------------------------------------------------------------
# Prediksi
# ---------------------------------------------------------------------------
def predict_dummy(text_list: list[str], seed: int = 42) -> list[str]:
    """Baseline bawah: menebak label acak seragam."""
    rng = np.random.default_rng(seed)
    return rng.choice(LABELS, size=len(text_list)).tolist()


def predict_from_precomputed(df_test: pd.DataFrame, source_csv: Path) -> list[str]:
    """Mengambil prediksi dari hasil inference pre-computed (Lexicon / Zero-shot RoBERTa)."""
    if not source_csv.exists():
        sys.exit(f"[ERROR] File precomputed tidak ditemukan: {source_csv}")
    source_df = pd.read_csv(source_csv)
    # Validasi dan konversi aman ke integer
    numeric_labels = pd.to_numeric(source_df["sentiment_label"], errors="coerce")
    source_df["pred_label"] = numeric_labels.map(INT_TO_LABEL)
    
    # Validasi baris berlabel valid
    valid_source = source_df.dropna(subset=["pred_label", "comment_text"])
    mapping = valid_source.set_index("comment_text")["pred_label"].to_dict()
    
    missing = [t for t in df_test[TEXT_COL] if t not in mapping]
    if missing:
        sys.exit(f"[ERROR] {len(missing)} teks uji tidak ditemukan di {source_csv}")
        
    return [mapping[t] for t in df_test[TEXT_COL]]


# ---------------------------------------------------------------------------
# Visualisasi & Pelaporan
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, labels, title: str, out_path: Path, show: bool = False) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    # Anotasi sel: jumlah absolut + persentase per baris (= recall per kelas).
    annot = np.array([[f"{cm[i, j]}\n({cm_norm[i, j]:.1%})" for j in range(len(labels))]
                      for i in range(len(labels))])

    sns.set_theme(style="white", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "Proporsi per kelas aktual (recall)", "shrink": 0.8},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Prediksi Model", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_ylabel("Gold Label (Anotasi Manual)", fontsize=12, labelpad=10, fontweight="bold")
    ax.set_title(title, fontsize=13, pad=14, fontweight="bold")
    ax.tick_params(axis="both", length=0)
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[OK] Confusion matrix disimpan -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def update_model_comparison(model_name: str, y_true: list[str], y_pred: list[str], test_hash: str) -> None:
    """Menyimpan atau memperbarui ringkasan komparasi model ke reports/model_comparison.csv."""
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

    cls_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    f1_dict = dict(zip(LABELS, cls_f1))

    new_row = {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Macro_Precision": round(macro_p, 4),
        "Macro_Recall": round(macro_r, 4),
        "Macro_F1": round(macro_f1, 4),
        "F1_Positif": round(f1_dict.get("Positif", 0.0), 4),
        "F1_Netral": round(f1_dict.get("Netral", 0.0), 4),
        "F1_Negatif": round(f1_dict.get("Negatif", 0.0), 4),
        "Test_Fingerprint": test_hash,
    }

    if COMPARISON_CSV.exists():
        comp_df = pd.read_csv(COMPARISON_CSV)
        if "Test_Fingerprint" not in comp_df.columns:
            comp_df = pd.DataFrame()
        else:
            comp_df = comp_df[comp_df["Test_Fingerprint"].astype(str) == test_hash]

        if not comp_df.empty:
            comp_df = comp_df[comp_df["Model"] != model_name]
            comp_df = pd.concat([comp_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            comp_df = pd.DataFrame([new_row])
    else:
        comp_df = pd.DataFrame([new_row])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(COMPARISON_CSV, index=False)
    print(f"[OK] Tabel komparasi model diperbarui -> {COMPARISON_CSV}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dummy", "lexicon", "indobert-zeroshot", "tfidf", "indobert"], default="tfidf")
    ap.add_argument("--show", action="store_true", help="Tampilkan visualisasi GUI matplotlib")
    args = ap.parse_args()
    model = args.model

    model_names = {
        "dummy": "Dummy Random Baseline",
        "lexicon": "Lexicon InSet/VADER Baseline",
        "indobert-zeroshot": "Indo-RoBERTa (Zero-Shot)",
        "tfidf": "TF-IDF + LogReg (Supervised)",
        "indobert": "IndoBERT Fine-tuned",
    }
    model_name = model_names[model]
    fig_path = FIG_DIR / f"confusion_matrix_{model}.png"

    if not TEST_PATH.exists():
        sys.exit(f"[ERROR] {TEST_PATH} tidak ditemukan. "
                 f"Jalankan: python script/phase0_sampling_and_split.py --stage 2")

    df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Kolom hilang di {TEST_PATH}: {missing}")

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[LABEL_COL].isin(LABELS)].copy()
    if df.empty:
        sys.exit("[ERROR] Tidak ada baris dengan gold label valid di data uji.")

    y_true = df[LABEL_COL].tolist()
    texts = df[TEXT_COL].tolist()

    if model == "dummy":
        y_pred = predict_dummy(texts)
    elif model == "lexicon":
        y_pred = predict_from_precomputed(df, LEXICON_CSV)
    elif model == "indobert-zeroshot":
        y_pred = predict_from_precomputed(df, ZEROSHOT_CSV)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sentiment_model import predict
        y_pred = predict(texts, model=model)

    df["predicted_label"] = y_pred

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

    print("=" * 60)
    print(f"EVALUASI MODEL: {model_name}")
    print(f"Data uji      : {TEST_PATH} ({len(df)} komentar)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0))
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}   <- metrik utama (adil untuk kelas tidak seimbang)")

    # 1. Confusion Matrix
    plot_confusion_matrix(
        y_true, y_pred, LABELS,
        title=f"Confusion Matrix — {model_name}\n"
              f"Accuracy {acc:.1%}  |  Macro-F1 {macro_f1:.3f}  |  n = {len(df)}",
        out_path=fig_path,
        show=args.show,
    )

    # 2. Error Analysis Export
    errors_df = df[df[LABEL_COL] != df["predicted_label"]].copy()
    err_cols = [c for c in ["sample_id", "source_row", TEXT_COL, LABEL_COL, "predicted_label", "volcano_name"] if c in df.columns]
    errors_df = errors_df[err_cols]
    err_path = REPORT_DIR / f"error_analysis_{model}.csv"
    errors_df.to_csv(err_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Error analysis ({len(errors_df)} kesalahan) disimpan -> {err_path}")

    # 3. Model Comparison Table
    test_hash = hashlib.sha256(TEST_PATH.read_bytes()).hexdigest()[:8]
    update_model_comparison(model_name, y_true, y_pred, test_hash)


if __name__ == "__main__":
    main()
