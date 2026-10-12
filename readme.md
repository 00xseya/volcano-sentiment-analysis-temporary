# 🌋 Analisis Sentimen & Topik Komentar YouTube (Bencana Gunung Berapi Indonesia)

![Status](https://img.shields.io/badge/Status-Selesai-success)
![Python](https://img.shields.io/badge/Python-3.9%20to%203.12-blue)
![NLP](https://img.shields.io/badge/NLP-Supervised%20%26%20LDA-orange)

Proyek ini bertujuan untuk menganalisis opini dan respon publik terhadap bencana erupsi gunung berapi di Indonesia (seperti Gunung Marapi, Semeru, Merapi, Ruang, Lewotobi, dll) antara tahun 2020–2025 melalui komentar video berita YouTube nasional dan regional.

Sistem memadukan dua metodologi utama *Natural Language Processing* (NLP):
1. **Supervised Machine Learning (Klasifikasi Sentimen)**: Mengklasifikasikan sentimen komentar ke dalam kategori **Positif**, **Netral**, atau **Negatif** menggunakan model yang dilatih dari data berlabel emas (*gold label*) manual.
2. **Unsupervised Learning (Topic Modeling LDA)**: Mengekstraksi tema pembicaraan netizen (*framing issues*) tanpa supervisi manusia menggunakan algoritma *Latent Dirichlet Allocation* (LDA) yang dioptimasi dengan metrik *Topic Coherence* ($U_{mass}$) dan *Perplexity*.

---

## 📌 Penjelasan Urutan Fase Pipeline

Pipeline telah dirapikan secara sekuensial (Fase 0 hingga Fase 7) untuk mencerminkan alur kerja penelitian NLP yang bersih, bebas kebocoran data (*data leakage*), dan mudah direproduksi:

```
[Data Mentah YouTube]
        │
        ▼
[Fase 0: Sampling & Gold Annotation] ────► data_train_labeled.csv (80%) & data_test_labeled.csv (20%)
        │
        ▼
[Fase 1: Preprocessing Teks] ────────────► data/processed/comments_clean.csv (clean_light & clean_heavy)
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
[Fase 2: Topic Modeling LDA]             [Fase 3: Visualisasi EDA]
  - Grid search k (Coherence Umass)        - Distribusi Sentimen
  - Label semantik topik                   - WordCloud per kelas
        │                                         │
        ▼                                         ▼
[Fase 4: Training Classifier] ───────────► models/tfidf_logreg.joblib (TF-IDF + Logistic Regression)
        │
        ▼
[Fase 5: Evaluasi & Benchmark] ──────────► Benchmarking vs Dummy, Lexicon, Zero-Shot & Error Analysis
        │
        ▼
[Fase 6: Batch Inference] ───────────────► Pelabelan sisa korpus komentar non-anotasi
        │
        ▼
[Fase 7: Final Report & Merge] ──────────► Master dataset final & grafik kontingensi (Topik x Sentimen)
```

> **Catatan Historis Mengenai Fase 2 & 3 Lama:**  
> Pada versi awal proyek, terdapat skrip `phase2_features.py` (ekstraksi fitur manual ke format `.npz`) dan `phase3_sentiment_lexicon.py` (analisis sentimen berbasis aturan kamus). Karena proyek ditingkatkan menggunakan pendekatan **Supervised Learning dengan Gold Label** dan pipeline `scikit-learn` terintegrasi, skrip lama tersebut telah diarsipkan ke folder `script/archive/` agar alur utama menjadi linear dan tidak redundant.

---

## 📁 Struktur Direktori

```text
volcano-sentiment-analysis/
├── data/
│   ├── raw/                 # Data mentah awal (comments_raw.csv, slang_dict.txt)
│   ├── processed/           # Data terproses (comments_clean.csv, final_analysis_dataset.csv)
│   ├── sample_for_annotation.csv # 360 sampel acak untuk proses anotasi
│   ├── data_train_labeled.csv    # 80% data berlabel untuk melatih model
│   └── data_test_labeled.csv     # 20% data berlabel untuk evaluasi independen
├── models/                  # File artefak model biner (tfidf_logreg.joblib)
├── reports/                 # Output visualisasi dan laporan kuantitatif
│   ├── figures/             # Gambar grafik resolusi tinggi (.png)
│   ├── topics/              # Kata kunci topik dan tabel evaluasi koherensi LDA
│   ├── model_comparison.csv # Tabel komparasi performa seluruh model
│   ├── error_analysis_*.csv # Analisis kesalahan prediksi untuk kajian kualitatif
│   ├── sentiment_by_volcano_crosstab.csv # Tabel kontingensi sentimen per gunung
│   └── sentiment_by_topic_crosstab.csv   # Tabel kontingensi sentimen per topik
├── script/                  # Seluruh skrip pipeline yang terurut rapi
│   ├── phase0_sampling_and_split.py # Fase 0: Ekstraksi sampel & stratified split
│   ├── phase1_preprocess.py         # Fase 1: Pembersihan teks media sosial
│   ├── phase2_topic_modeling.py     # Fase 2: Topic modeling LDA & evaluasi k
│   ├── phase3_eda_visualization.py  # Fase 3: Analisis data eksploratif & WordCloud
│   ├── phase4_train_classifier.py   # Fase 4: Pelatihan model klasifikasi supervised
│   ├── phase5_evaluate_model.py     # Fase 5: Evaluasi independen & error analysis
│   ├── phase6_inference.py          # Fase 6: Prediksi batch pada korpus besar
│   ├── phase7_final_report.py       # Fase 7: Penggabungan master & laporan akhir
│   ├── sentiment_model.py           # Shared module untuk inferensi konsisten
│   ├── archive/                     # Skrip eksperimen legacy (lexicon, manual features)
│   ├── scraping/                    # Utilitas scraping data YouTube
│   └── utils/                       # Skrip validasi integritas dataset
├── requirements.txt         # Dependensi standar proyek
├── requirements-deeplearning.txt # Dependensi opsional (Torch/Transformers)
├── annotation_guidelines.md # Pedoman baku pelabelan sentimen bencana
└── readme.md                # Dokumentasi lengkap proyek
```

---

## 🛠 Instalasi & Lingkungan Kerja

Gunakan Virtual Environment dengan Python 3.9 s.d. 3.12:

```bash
# 1. Buat dan aktifkan virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Di Windows: .venv\Scripts\activate

# 2. Pasang dependensi standar
pip install -r requirements.txt
```

*(Opsional)* Jika ingin menjalankan model Transformer Deep Learning (IndoBERT fine-tuning/zero-shot):
```bash
pip install -r requirements-deeplearning.txt
```

---

## 🚀 Panduan Eksekusi Lengkap (Fase demi Fase)

Jalankan seluruh skrip dari direktori akar (*root*) proyek:

### 1. Fase 0 — Sampling & Split Anotasi Manual
Skrip ini mengisolasi data untuk diberi label manusia sebagai tolak ukur (*gold standard*):
```bash
# Tahap 1: Ekstraksi 360 baris acak terlindungi dari CSV injection
python script/phase0_sampling_and_split.py --stage 1

# [Proses Pelabelan]: Tim mengisi data/sample_for_annotation.csv mengikuti annotation_guidelines.md

# Tahap 2: Validasi integritas label dan stratified split 80:20
python script/phase0_sampling_and_split.py --stage 2
```
* **Input**: `data/raw/comments_raw.csv`
* **Output**: `data/data_train_labeled.csv` (288 baris) & `data/data_test_labeled.csv` (72 baris).

---

### 2. Fase 1 — Preprocessing Teks
Tahap pembersihan teks bahasa Indonesia non-formal:
- Case folding & pembersihan URL, mention, karakter berulang (*word elongation*).
- Normalisasi kamus slang domain umum & domain bencana (`data/raw/slang_dict.txt`).
- Demojize (konversi emoji ke deskripsi tekstual).
- Stemming morfologis PySastrawi.
- Stopword removal adaptif konteks bencana (menjaga kata negasi dan sentimen seperti *tidak, jangan, waspada, semoga*).
- Menghasilkan dua representasi: `clean_light` (untuk klasifikasi) dan `clean_heavy` (untuk topic modeling).
- Mempertahankan kolom `source_row` sebagai primary key baris asli.
```bash
python script/phase1_preprocess.py
```
* **Output**: `data/processed/comments_clean.csv`.

---

### 3. Fase 2 — Topic Modeling LDA & Coherence Evaluation
Mengekstraksi topik laten dari teks komentar `clean_heavy`:
- Melakukan grid-search otomatis untuk jumlah topik $k \in [3, 4, 5, 6, 7]$.
- Mengukur kualitas topik dengan **$U_{mass}$ Topic Coherence** dan **Perplexity**.
- Memilih model terbaik ($k=5$) dan memetakan kata kunci ke label semantik tematik.
```bash
python script/phase2_topic_modeling.py
# Atau spesifikasikan jumlah topik manual:
# python script/phase2_topic_modeling.py --num-topics 5
```
* **Output**:
  - `reports/figures/lda_topic_evaluation.png`: Grafik optimasi koherensi topik.
  - `reports/topics/lda_coherence_evaluation.csv`: Tabel metrik kuantitatif.
  - `reports/topics/lda_topics_keywords.csv`: Bobot kata kunci teratas tiap topik.
  - `reports/topics/comments_with_topics.csv`: Dataset dengan kolom `dominant_topic` & `dominant_topic_label`.

---

### 4. Fase 3 — Analisis Eksploratif (EDA) & Visualisasi
Menganalisis sebaran data dan pola linguistik:
- Menghitung frekuensi label Positif, Netral, dan Negatif.
- Visualisasi sebaran sentimen per gunung berapi.
- Pembuatan WordCloud terpisah untuk masing-masing kelas sentimen.
```bash
python script/phase3_eda_visualization.py
```
* **Output**: Gambar tersimpan di `reports/figures/` (`eda_sentiment_distribution.png`, `eda_sentiment_per_volcano.png`, `wordcloud_positif.png`, `wordcloud_negatif.png`, `wordcloud_netral.png`).

---

### 5. Fase 4 — Pelatihan Model Klasifikasi Supervised
Melatih model machine learning menggunakan data latih hasil anotasi:
- Mencegah *data leakage* menggunakan `sklearn.pipeline.Pipeline` (TF-IDF Vectorizer + Logistic Regression).
- Optimasi hiperparameter (C, ngram_range, sublinear_tf) via Stratified 5-Fold Cross Validation.
```bash
python script/phase4_train_classifier.py --model tfidf
```
* **Output**: `models/tfidf_logreg.joblib`.

---

### 6. Fase 5 — Evaluasi Model & Error Analysis (Benchmarking)
Menguji performa model secara objektif pada data uji yang belum pernah dilihat (`data_test_labeled.csv`):
```bash
# 1. Baseline Acak (Uniform Random)
python script/phase5_evaluate_model.py --model dummy

# 2. Baseline Berbasis Aturan Kamus (InSet Lexicon)
python script/phase5_evaluate_model.py --model lexicon

# 3. Model Transformer Pre-trained Zero-Shot (Indo-RoBERTa)
python script/phase5_evaluate_model.py --model indobert-zeroshot

# 4. Model Supervised Domain-Adapted (TF-IDF + Logistic Regression)
python script/phase5_evaluate_model.py --model tfidf
```
* **Output**:
  - Matriks kebingungan (*Confusion Matrix*) di `reports/figures/confusion_matrix_<model>.png`.
  - Ekspor komentar yang salah diprediksi untuk analisis kualitatif di `reports/error_analysis_<model>.csv`.
  - Tabel perbandingan seluruh metrik di `reports/model_comparison.csv`.

---

### 7. Fase 6 — Batch Inference pada Seluruh Dataset
Mengaplikasikan model supervised yang telah teruji ke seluruh data komentar non-anotasi:
- Menjaga data gold label asli agar tidak tertimpa.
- Memetakan hasil klasifikasi kembali ke indeks baris asli (`source_row`).
```bash
python script/phase6_inference.py --model tfidf
```
* **Output**: `data/processed/comments_with_sentiment.csv`.

---

### 8. Fase 7 — Penggabungan Master & Laporan Akhir
Mengintegrasikan hasil analisis sentimen (Fase 6) dan pemodelan topik (Fase 2) menjadi satu tabel terpadu:
- Penggabungan aman dengan validasi kardinalitas `many_to_one` berbasis `source_row`.
- Menghasilkan tabel kontingensi (crosstab) antara topik dan sentimen, serta gunung dan sentimen.
- Memproduksi visualisasi komprehensif untuk laporan/publikasi ilmiah.
```bash
python script/phase7_final_report.py
```
* **Output**:
  - `data/processed/final_analysis_dataset.csv` (Master Dataset Final)
  - `reports/sentiment_by_volcano_crosstab.csv`
  - `reports/sentiment_by_topic_crosstab.csv`
  - `reports/figures/final_sentiment_per_volcano.png`
  - `reports/figures/final_sentiment_per_topic.png`
  - `reports/figures/final_topic_distribution.png`

---

## 📊 Hasil Evaluasi & Temuan Kunci

### Tabel Komparasi Model (Benchmark pada Data Uji $n=72$)

| Model | Pendekatan | Akurasi | Macro Precision | Macro Recall | **Macro F1** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dummy Random** | Baseline Heuristik | 45.83% | 0.4591 | 0.4516 | **0.4528** |
| **InSet Lexicon** | Unsupervised Rule-Based | 41.67% | 0.4792 | 0.4434 | **0.4134** |
| **Indo-RoBERTa** | Zero-Shot Transformer | 48.61% | 0.4741 | 0.4659 | **0.4656** |
| **TF-IDF + LogReg** | **Supervised Domain-Adapted** | **61.11%** | **0.6202** | **0.6015** | **0.6072** |

### Analisis Temuan:
1. **Pentingnya Adaptasi Domain Bencana**:  
   Model lexicon dan model zero-shot pra-latih gagal menangani nuansa teks bencana alam Indonesia, di mana kata-kata seperti *"azab"*, *"innalillahi"*, *"merinding"*, atau doa keselamatan kerap disalahartikan. Model **TF-IDF + Logistic Regression** yang dilatih secara supervised melampaui lexicon sebesar **+19.4% Macro F1**.
2. **Kesesuaian Tema Topik LDA ($k=5$)**:  
   Optimasi nilai $U_{mass}$ Coherence membuktikan $k=5$ menghasilkan skor koherensi tertinggi (-84.09), membedakan 5 dimensi diskursus publik:
   - **Topik 1 (Kondisi Wilayah & Akses Jalan)**: Kerusakan infrastruktur dan penutupan jalur evakuasi.
   - **Topik 2 (Respon Publik & Berita Bencana)**: Update informasi terkini dan tayangan video erupsi.
   - **Topik 3 (Refleksi Religius & Bencana Alam)**: Persepsi bencana sebagai peringatan spiritual / azab.
   - **Topik 4 (Ketakutan Gempa & Erupsi Berulang)**: Kecemasan warga terhadap getaran seismik dan abu vulkanik.
   - **Topik 5 (Doa Keselamatan & Solidaritas Korban)**: Ungkapan empati, doa perlindungan, dan dukungan moril.
