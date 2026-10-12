"""
Tahap 0 — Sampling untuk anotasi manual & split train/test berlabel.

Tahap 1 : Ambil 360 komentar acak dari data mentah, tambahkan kolom kosong
          `gold_label`, simpan ke data/sample_for_annotation.csv.
Tahap 2 : Muat sampel yang SUDAH dianotasi, buang baris tanpa label,
          lakukan stratified split 80:20 -> data_train_labeled.csv / data_test_labeled.csv.

Cara pakai (jalankan dari ROOT proyek, sama seperti skrip phase lainnya):
    python script/phase0_sampling_and_split.py --stage 1
    # ... anotasi manual mengikuti annotation_guidelines.md ...
    python script/phase0_sampling_and_split.py --stage 2

Pedoman pelabelan: lihat annotation_guidelines.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
SAMPLE_SIZE = 360
TEST_SIZE = 0.20

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "data_raw.csv"
# Dipakai bila RAW_PATH belum ada (struktur repo saat ini menyimpan data mentah di sini).
FALLBACK_RAW_PATHS = [DATA_DIR / "raw" / "comments_raw.csv"]

SAMPLE_PATH = DATA_DIR / "sample_for_annotation.csv"
TRAIN_PATH = DATA_DIR / "data_train_labeled.csv"
TEST_PATH = DATA_DIR / "data_test_labeled.csv"

TEXT_COL = "comment_text"
LABEL_COL = "gold_label"
NOTE_COL = "annotator_note"
VALID_LABELS = ("Positif", "Negatif", "Netral")

# Toleransi variasi penulisan dari anotator (case-insensitive).
LABEL_ALIASES = {
    "positif": "Positif", "positive": "Positif", "pos": "Positif",
    "negatif": "Negatif", "negative": "Negatif", "neg": "Negatif",
    "netral": "Netral", "neutral": "Netral", "neu": "Netral",
}

# utf-8-sig: aman untuk emoji & bisa dibuka benar oleh Excel / Google Sheets.
CSV_ENCODING = "utf-8-sig"

# Cegah CSV/formula injection: sel yang diawali karakter ini akan dieksekusi
# sebagai formula oleh spreadsheet. Di file anotasi diberi awalan apostrof.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _needs_escape(value) -> bool:
    # Termasuk teks sumber yang SUDAH diawali apostrof + karakter formula,
    # supaya unescape hanya membuang apostrof yang ditambahkan Tahap 1.
    return isinstance(value, str) and (
        value.startswith(FORMULA_PREFIXES)
        or (value.startswith("'") and value[1:].lstrip("'").startswith(FORMULA_PREFIXES))
    )


def _escape_cell(value):
    return "'" + value if _needs_escape(value) else value


def _unescape_cell(value):
    if isinstance(value, str) and value.startswith("'") and _needs_escape(value[1:]):
        return value[1:]
    return value


def escape_formulas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(_escape_cell)
    return out


def unescape_formulas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(_unescape_cell)
    return out


# ---------------------------------------------------------------------------
# Tahap 1 — Sampling
# ---------------------------------------------------------------------------
def resolve_raw_path() -> Path:
    for path in [RAW_PATH, *FALLBACK_RAW_PATHS]:
        if path.exists():
            if path != RAW_PATH:
                print(f"[INFO] {RAW_PATH} tidak ditemukan, memakai fallback: {path}")
            return path
    sys.exit(f"[ERROR] Data mentah tidak ditemukan. Dicari: "
             f"{[str(p) for p in [RAW_PATH, *FALLBACK_RAW_PATHS]]}")


def stage1_sample(force: bool = False) -> None:
    print("=" * 60)
    print("TAHAP 1 — Sampling untuk anotasi manual")
    print("=" * 60)

    # Lindungi hasil kerja anotator dari tertimpa tanpa sengaja.
    if SAMPLE_PATH.exists() and not force:
        sys.exit(f"[STOP] {SAMPLE_PATH} sudah ada dan mungkin sudah berisi label manual.\n"
                 f"       Gunakan --force bila memang ingin menimpanya.")

    src = resolve_raw_path()
    df = pd.read_csv(src)
    if TEXT_COL not in df.columns:
        sys.exit(f"[ERROR] Kolom '{TEXT_COL}' tidak ada di {src}. Kolom: {list(df.columns)}")

    n_loaded = len(df)
    # Komentar kosong tidak bisa dilabeli; duplikat (spam) akan membiaskan evaluasi.
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df = df[df[TEXT_COL] != ""]
    n_nonempty = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL])
    print(f"Dimuat: {n_loaded} baris | non-kosong: {n_nonempty} | unik: {len(df)}")

    if len(df) < SAMPLE_SIZE:
        sys.exit(f"[ERROR] Hanya {len(df)} komentar valid, kurang dari SAMPLE_SIZE={SAMPLE_SIZE}.")

    sample = (
        df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
          .reset_index()
          .rename(columns={"index": "source_row"})   # indeks baris di file sumber
    )
    sample.insert(0, "sample_id", range(1, len(sample) + 1))
    sample[LABEL_COL] = pd.NA
    sample[NOTE_COL] = pd.NA

    # Kolom yang dibutuhkan anotator diletakkan paling depan.
    front = ["sample_id", TEXT_COL, LABEL_COL, NOTE_COL]
    sample = sample[front + [c for c in sample.columns if c not in front]]

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    escape_formulas(sample).to_csv(SAMPLE_PATH, index=False, encoding=CSV_ENCODING)

    print(f"[OK] {len(sample)} baris disimpan -> {SAMPLE_PATH}")
    if "volcano_name" in sample.columns:
        print("\nSebaran sampel per gunung:")
        print(sample["volcano_name"].value_counts().to_string())
    print(f"\nLangkah berikutnya: isi kolom '{LABEL_COL}' dengan {VALID_LABELS}, "
          f"lalu jalankan --stage 2.")


# ---------------------------------------------------------------------------
# Tahap 2 — Validasi label & stratified split
# ---------------------------------------------------------------------------
def normalize_label(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    return LABEL_ALIASES.get(text.lower(), text)


def stage2_split() -> None:
    print("=" * 60)
    print("TAHAP 2 — Validasi label & stratified split 80:20")
    print("=" * 60)

    if not SAMPLE_PATH.exists():
        sys.exit(f"[ERROR] {SAMPLE_PATH} tidak ditemukan. Jalankan --stage 1 terlebih dahulu.")

    # Kembalikan teks asli (hapus apostrof anti-formula dari Tahap 1).
    df = unescape_formulas(pd.read_csv(SAMPLE_PATH, encoding=CSV_ENCODING))
    if LABEL_COL not in df.columns:
        sys.exit(f"[ERROR] Kolom '{LABEL_COL}' tidak ditemukan di {SAMPLE_PATH}.")

    n_total = len(df)
    df[LABEL_COL] = df[LABEL_COL].map(normalize_label)
    df = df.dropna(subset=[LABEL_COL]).copy()
    print(f"Total baris: {n_total} | berlabel: {len(df)} | dibuang (kosong/NaN): {n_total - len(df)}")

    # Label di luar skema = kesalahan input -> hentikan, jangan diam-diam dibuang.
    invalid = ~df[LABEL_COL].isin(VALID_LABELS)
    if invalid.any():
        cols = [c for c in ["sample_id", LABEL_COL, TEXT_COL] if c in df.columns]
        print(f"\n[ERROR] {invalid.sum()} baris memiliki label tidak valid "
              f"(harus salah satu dari {VALID_LABELS}):")
        print(df.loc[invalid, cols].to_string(index=False, max_colwidth=60))
        sys.exit(1)

    counts = df[LABEL_COL].value_counts()
    print("\nDistribusi gold label:")
    print(pd.DataFrame({"n": counts, "pct": (counts / counts.sum() * 100).round(1)}).to_string())

    missing_labels = [lbl for lbl in VALID_LABELS if lbl not in counts.index]
    if missing_labels:
        sys.exit(f"[ERROR] Kelas berikut belum punya sampel berlabel: {missing_labels}.")
    if counts.min() < 2:
        sys.exit("[ERROR] Setiap kelas butuh minimal 2 sampel untuk stratified split.")

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            stratify=df[LABEL_COL],
            random_state=RANDOM_STATE,
            shuffle=True,
        )
    except ValueError as exc:
        sys.exit(f"[ERROR] Stratified split gagal (data berlabel terlalu sedikit?): {exc}")

    for name, part in (("train", train_df), ("test", test_df)):
        absent = set(VALID_LABELS) - set(part[LABEL_COL])
        if absent:
            sys.exit(f"[ERROR] Split '{name}' tidak memuat kelas {sorted(absent)}. "
                     f"Tambah anotasi untuk kelas tersebut. File output TIDAK ditulis.")

    escape_formulas(train_df).to_csv(TRAIN_PATH, index=False, encoding=CSV_ENCODING)
    escape_formulas(test_df).to_csv(TEST_PATH, index=False, encoding=CSV_ENCODING)

    summary = pd.DataFrame({
        "train": train_df[LABEL_COL].value_counts(),
        "test": test_df[LABEL_COL].value_counts(),
    }).fillna(0).astype(int)
    summary.loc["TOTAL"] = summary.sum()
    print("\nRingkasan split:")
    print(summary.to_string())
    print(f"\n[OK] Train ({len(train_df)}) -> {TRAIN_PATH}")
    print(f"[OK] Test  ({len(test_df)}) -> {TEST_PATH}")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                        help="1 = sampling untuk anotasi, 2 = split train/test")
    parser.add_argument("--force", action="store_true",
                        help="Tahap 1: timpa sample_for_annotation.csv yang sudah ada")
    args = parser.parse_args()

    if args.stage == 1:
        stage1_sample(force=args.force)
    else:
        stage2_split()


if __name__ == "__main__":
    main()
