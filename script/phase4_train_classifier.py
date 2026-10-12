"""
Phase 6 — Training classifier sentimen (supervised) dari data anotasi manual.

Cara pakai (dari ROOT proyek):
    python script/phase4_train_classifier.py --model tfidf      # cepat, CPU, baseline kuat
    python script/phase4_train_classifier.py --model indobert   # butuh: pip install torch transformers accelerate

Aturan anti-leakage:
    - HANYA data/data_train_labeled.csv yang dipakai untuk training & tuning.
    - data/data_test_labeled.csv TIDAK disentuh di sini; dipakai sekali di
      script/X_evaluate_model.py untuk angka akhir.
    - Tuning hyperparameter memakai Stratified 5-Fold CV di dalam data train.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_model import (ID2LABEL, INDOBERT_DIR, LABEL2ID, LABELS,  # noqa: E402
                             MODEL_DIR, TFIDF_PATH, normalize)

TRAIN_PATH = Path("data/data_train_labeled.csv")
SEED = 42


def load_train() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        sys.exit(f"[ERROR] {TRAIN_PATH} tidak ditemukan.")
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    df["gold_label"] = df["gold_label"].astype(str).str.strip()
    df = df[df["gold_label"].isin(LABELS)].copy()
    df["text"] = df["comment_text"].apply(normalize)
    df = df[df["text"].str.len() > 0]
    print(f"Data train: {len(df)} baris | distribusi: {df['gold_label'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
def train_tfidf(df: pd.DataFrame) -> None:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import FeatureUnion, Pipeline

    # Gabungan n-gram kata + n-gram karakter: karakter tahan terhadap typo &
    # variasi ejaan gaul ("kasian/kasihan/ksian"), penting untuk data kecil.
    pipe = Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ])),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
    ])
    grid = {
        "clf__C": [0.5, 1, 2, 5, 10],
        "feats__word__ngram_range": [(1, 1), (1, 2)],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    gs.fit(df["text"], df["gold_label"])

    print(f"Best params : {gs.best_params_}")
    print(f"CV macro-F1 : {gs.best_score_:.4f} (± {gs.cv_results_['std_test_score'][gs.best_index_]:.4f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, TFIDF_PATH)  # sudah di-refit pada seluruh data train
    print(f"[OK] Model disimpan -> {TFIDF_PATH}")


# ---------------------------------------------------------------------------
def train_indobert(df: pd.DataFrame, base: str, epochs: int) -> None:
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  DataCollatorWithPadding, EarlyStoppingCallback,
                                  Trainer, TrainingArguments, set_seed)
    except ImportError as e:
        sys.exit(f"[ERROR] {e}. Install: pip install torch transformers datasets accelerate")

    set_seed(SEED)
    # Validation split DARI data train (bukan data test!) untuk early stopping.
    try:
        tr, va = train_test_split(df, test_size=0.15, stratify=df["gold_label"], random_state=SEED)
    except ValueError:
        # Fallback to random split if data is too small to stratify validation set
        tr, va = train_test_split(df, test_size=0.15, random_state=SEED)

    tok = AutoTokenizer.from_pretrained(base)

    def to_ds(d):
        ds = Dataset.from_dict({"text": d["text"].tolist(),
                                "label": d["gold_label"].map(LABEL2ID).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {"f1_macro": f1_score(p.label_ids, preds, average="macro")}

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "_indobert_ckpt"),
        num_train_epochs=epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=to_ds(tr), eval_dataset=to_ds(va),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print(f"Device: {'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'}")
    trainer.train()
    print(f"Val macro-F1 terbaik: {trainer.evaluate()['eval_f1_macro']:.4f}")

    INDOBERT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(INDOBERT_DIR))
    tok.save_pretrained(str(INDOBERT_DIR))
    print(f"[OK] Model disimpan -> {INDOBERT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tfidf", "indobert"], default="tfidf")
    ap.add_argument("--base", default="indobenchmark/indobert-base-p1")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()

    df = load_train()
    if a.model == "tfidf":
        train_tfidf(df)
    else:
        train_indobert(df, a.base, a.epochs)
    print(f"\nLangkah berikutnya: python script/X_evaluate_model.py --model {a.model}")


if __name__ == "__main__":
    main()
