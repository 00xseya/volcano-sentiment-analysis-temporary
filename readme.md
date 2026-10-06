
# Indonesian Volcano News Sentiment Analysis: A Phase-by-Phase Walkthrough

## Introduction

This project sits at the intersection of computational linguistics, disaster communication studies, and Indonesian NLP. Your objective—collecting 90 YouTube video news items with 20 comments each (1,800 comments total) covering Indonesian volcanic eruptions from 2020–2025—creates a rich corpus for sentiment and framing analysis. Below is a detailed, actionable walkthrough for each phase.

---

## Phase 0: Data Collection (Pre-Pipeline)

Before the four-stage NLP pipeline can begin, you need a structured corpus. This phase is foundational because the quality of your downstream analysis depends entirely on how you collect and organize the data.

### 0.1 Video Selection Strategy

Your target is 90 videos spanning 2020–2025. To ensure balanced coverage:

- **Volcano diversity**: Aim for at least 5–8 major volcanoes (e.g., Merapi, Semeru, Sinabung, Anak Krakatau, Lewotobi Laki-laki, Raung, Dukono). The 2020–2022 Semeru eruptions and the 2025 Lewotobi Laki-laki activity are particularly well-documented in news media.
- **News source diversity**: Include videos from major Indonesian news channels (Kompas TV, Metro TV, tvOne, CNN Indonesia, Detik, Tribun News) as well as regional channels.
- **Temporal spread**: Distribute videos across the five-year window to capture evolving framing—early coverage (2020–2021) vs. recent coverage (2024–2025) may reveal shifts in tone.

### 0.2 YouTube Comment Extraction

For extracting 20 comments per video, you have several Python options. **ytscrape** is particularly suitable because it requires no API key and no quota, using YouTube's internal InnerTube API with transparent pagination and typed dataclasses. A minimal extraction pattern:

```python
from ytscrape import YouTube, CommentSort

with YouTube(language="id", region="ID") as yt:
    for comment in yt.comments(
        "https://youtu.be/VIDEO_ID",
        sort=CommentSort.NEWEST,
        max_results=20
    ):
        print(comment.author, comment.text)
```

**tapline** is an alternative if you prefer an API-key-based approach with higher reliability and explicit credit management. The official YouTube Data API v3 is another option but comes with quota constraints.

**Important consideration**: Collect top-level comments sorted by "Newest" to avoid algorithmic bias toward highly-liked comments, which may overrepresent polarizing or viral reactions. Store metadata (video title, channel, upload date, volcano name, comment count) alongside each comment.

### 0.3 Data Structure

Organize your data as a flat table:

| comment_id | video_id | video_title | channel | upload_date | volcano | comment_text | like_count |
| ---------- | -------- | ----------- | ------- | ----------- | ------- | ------------ | ---------- |

This structure supports both sentiment analysis (per-comment) and framing analysis (per-video aggregation).

---

## Phase 1: Text Preprocessing

Preprocessing transforms raw Indonesian comment text—often informal, code-mixed, and noisy—into normalized form suitable for feature extraction. Indonesian social media text presents unique challenges: slang (e.g., "gak", "banget", "udah"), abbreviations, and code-switching with English.

### 1.1 Recommended Libraries

- **Sastrawi**: The de facto standard for Indonesian stemming and stopword removal. It implements the Nazief-Adriani algorithm and provides a default Indonesian stopword list including words like "yang", "di", "dari".
- **nahiarhdNLP**: A modern, configurable pipeline library offering HTML/URL/mention removal, emoji handling, spell correction, slang normalization, stemming, and stopword removal in a single configurable pipeline.
- **indo-text-preprocessing**: A lightweight alternative (numpy-only dependency) with 100% Sastrawi-compatible stemming at ~5× the speed, plus built-in slang dictionaries (4,534 entries) and 736 stopwords.

### 1.2 Step-by-Step Preprocessing

**Step 1 — Lowercasing**: Convert all text to lowercase. This is straightforward but essential for consistent token matching.

**Step 2 — Contraction and Slang Expansion**: Indonesian YouTube comments are heavy on informal contractions. Build or use a slang mapping:

- "gak" → "tidak"
- "udah" → "sudah"
- "banget" → "sangat"
- "emangnya" → "memangnya"

The `indo-text-preprocessing` library ships with a comprehensive slang dictionary derived from Kamus-Alay and colloquial Indonesian sources.

**Step 3 — Punctuation and Special Character Removal**: Remove punctuation, URLs, mentions (@username), hashtags, and emojis. However, consider preserving certain punctuation for sentiment cues—excessive exclamation marks ("!!") can signal strong emotion. A hybrid approach: strip punctuation for feature extraction but retain a "punctuation_count" feature for sentiment modeling.

**Step 4 — Tokenization**: Split text into word tokens. For Indonesian, simple whitespace tokenization works reasonably well, but you may need to handle hyphenated reduplication (e.g., "buku-buku" → "buku buku").

**Step 5 — Stopword Removal**: Remove high-frequency function words. Sastrawi provides a default Indonesian stopword list, and `indo-text-preprocessing` offers 736 stopwords with customizable keep/remove lists.

**Step 6 — Stemming vs. Lemmatization**: For Indonesian, stemming (reducing "memperbaiki" → "baik", "makananmu" → "makan") is more commonly used than lemmatization, as reliable Indonesian lemmatizers are scarce. Sastrawi's stemmer is the standard, though `indo-text-preprocessing` offers a faster, compatible alternative.

**Step 7 — Spelling Correction**: Indonesian social media text contains frequent typos. A spell corrector using a dictionary of standard Indonesian words plus a slang corpus can normalize these. The `nahiarhdNLP` library provides both word-level and sentence-level spell correction.

**Step 8 — Rare Word and Common Word Removal**: After building your corpus vocabulary, remove words appearing in fewer than 2 documents (rare words) and words appearing in more than 90% of documents (common words that carry little discriminative information). This step is typically applied after initial tokenization and before vectorization.

### 1.3 Pipeline Configuration Example

```python
from indo_text_preprocessing import IndoPreprocessor

p = IndoPreprocessor(
    remove_url=True,
    remove_mention=True,
    replace_slang=True,
    remove_stopwords=True,
    stem=True
)
cleaned = p.preprocess_batch(comments_list)
```

**Caveat**: Aggressive preprocessing can strip sentiment-bearing words. For sentiment analysis, consider a "light" pipeline (lowercase + slang expansion + stopword removal only) and a "heavy" pipeline (full stemming + rare word removal) for topic modeling and clustering, where lexical diversity matters less.

---

## Phase 2: Feature Engineering

Feature engineering converts preprocessed text into numerical representations that machine learning models can consume.

### 2.1 Term Frequency (TF) and TF-IDF

TF measures how often a term appears in a document. IDF downweights terms that appear across many documents. TF-IDF combines them: \( \text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log(N / \text{DF}(t)) \). For Indonesian text classification, TF-IDF with `ngram_range=(1,2)` and `sublinear_tf=True` has proven effective.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)
X_tfidf = vectorizer.fit_transform(cleaned_texts)
```

### 2.2 N-grams

Bigrams and trigrams capture multi-word expressions that unigrams miss. In Indonesian, n-grams are particularly valuable for capturing fixed phrases like "gunung meletus" (volcano erupts), "status awas" (alert status), and "abu vulkanik" (volcanic ash). Research on Indonesian text shows that n-gram range up to (1,3) works well, as Indonesian uses 3-letter acronyms (e.g., "kartu tanda penduduk" → KTP).

### 2.3 Bag of Words (BoW)

A simple count-based representation. While TF-IDF generally outperforms BoW for classification, BoW is useful for topic modeling (LDA expects count data) and for interpretability.

### 2.4 Auxiliary Features

Beyond lexical features, extract:

- **Word count**: Total tokens per comment
- **Character count**: Total characters (excluding spaces)
- **Average word length**: Mean character count per word
- **Punctuation density**: Ratio of punctuation to total characters (sentiment proxy)
- **Uppercase ratio**: Proportion of uppercase characters (emotional intensity proxy)

These features can be concatenated with TF-IDF vectors for hybrid models.

### 2.5 Document Similarity

For document similarity, TF-IDF vectors can be compared using cosine similarity. For semantic similarity, consider sentence embeddings from IndoBERT or multilingual models.

---

## Phase 3: Modeling

Modeling applies statistical and machine learning algorithms to the engineered features to uncover patterns. For your project, different tasks require different modeling approaches.

### 3.1 Traditional Machine Learning (for Classification and Sentiment)

**Naive Bayes**: A strong baseline for Indonesian sentiment analysis, especially with TF-IDF features. It is fast, interpretable, and performs well on short text like YouTube comments.

**Support Vector Machine (SVM)**: Often outperforms Naive Bayes on text classification with TF-IDF. Linear SVM with C=1 and TF-IDF (min_df=0.05, max_df=0.75, n-gram (1,2)) has been shown effective for Indonesian text.

**Logistic Regression**: Provides probabilistic outputs, useful for confidence-based sentiment scoring.

### 3.2 Transformer-Based Models (for Higher Accuracy)

**IndoBERT**: A BERT model pre-trained on Indonesian text. Fine-tuning IndoBERT for sentiment classification typically yields F1-scores in the 80–90% range for Indonesian social media text. A study on earthquake-related YouTube comments achieved competitive results comparing BiLSTM with IndoBERT, with positive sentiment at 41.2%, negative at 35.5%, and neutral at 23.2%.

```python
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="indobenchmark/indobert-base-p1"
)
```

**BiLSTM with FastText embeddings**: A lighter alternative to BERT, achieving strong performance on Indonesian NER tasks (F1 up to 79.64%) when combined with CRF.

### 3.3 Clustering

For unsupervised grouping of comments by theme or sentiment:

- **K-Means** on TF-IDF or SVD-reduced features
- **DBSCAN** for density-based clustering (finds arbitrarily shaped clusters, useful for identifying outlier comments)
- **Hierarchical clustering** for dendrogram visualization

### 3.4 Topic Modeling

Two approaches for Indonesian news and comments:

**LDA (Latent Dirichlet Allocation)**: Probabilistic generative model that treats each document as a mixture of topics. Works well with BoW features. Fast but struggles with short texts like YouTube comments.

**BERTopic**: Uses BERT embeddings and advanced clustering (HDBSCAN) to generate coherent topics. Research on Indonesian news corpora shows BERTopic outperforms LDA in topic coherence and interpretability, especially for short, unstructured text. A comparative study on Indonesian YouTube protest discourse found that LDA is faster in time metrics while BERTopic produces more coherent topics.

### 3.5 Named Entity Recognition (NER)

For extracting entities from news coverage, fine-tune IndoBERT for NER with entity types:

- **PER** (Person): Officials, volcanologists, local residents
- **ORG** (Organization): PVMBG, BNPB, local government
- **LOC** (Location): Volcano names, villages, evacuation zones
- **DIS** (Disaster-related terms): Eruption, ashfall, pyroclastic flow

IndoBERT achieves best-in-class performance for Indonesian NER, particularly for disease entities (F1 79.64% in one study). For your project, you would define volcano-specific entity types.

---

## Phase 4: Task Execution

This phase applies the trained models to produce actionable insights.

### 4.1 Sentiment Analysis

Classify each comment as positive, negative, or neutral. For Indonesian YouTube comments on disaster news, negative and critical sentiment often dominates—a study on LPG subsidy comments found 197 negative/critical comments vs. 55 positive, with hostile sentiment at 87 comments. Expect similar patterns for volcano coverage, though framing differences across channels may produce variations.

**Deliverable**: Per-comment sentiment labels, aggregated per video, per volcano, and per time period.

### 4.2 Framing Analysis

Framing analysis goes beyond sentiment polarity to examine how news videos frame volcanic events. Common frames:

- **Disaster frame**: Focus on destruction, casualties, evacuation
- **Scientific frame**: Focus on volcanic activity, alert levels, geological explanation
- **Human interest frame**: Focus on residents' experiences, community response
- **Political frame**: Focus on government response, policy, blame

Combine topic modeling (BERTopic) with manual frame annotation to identify dominant frames in your corpus.

### 4.3 Classification

Train a classifier to predict, for example:

- Volcano type from news text
- Alert level from coverage
- Whether a video is from a national vs. regional channel

### 4.4 Clustering

Cluster videos or comments by:

- **Sentiment profile**: Videos with predominantly negative comments vs. mixed
- **Topic profile**: Videos clustering around evacuation, ashfall, government response
- **Temporal profile**: Videos from different eruption phases

### 4.5 Entity Extraction

Extract and link entities across videos:

- **Volcano entities**: Merapi, Semeru, Sinabung, etc.
- **Organization entities**: PVMBG, BNPB, BMKG
- **Location entities**: Yogyakarta, Lumajang, Karo Regency

Build a co-occurrence network of entities to reveal which volcanoes are associated with which organizations and locations.

### 4.6 Document Similarity

Compute pairwise cosine similarity between video comment sets. Videos covering the same volcano eruption from different channels should cluster together. This can reveal:

- Which channels cover similar stories in similar ways
- Which videos are outliers (unique framing)
- How coverage of the same event evolves over time

### 4.7 Topic Modeling Output

For each topic, extract:

- Top 10 terms (using c-TF-IDF for BERTopic)
- Representative documents
- Topic size and temporal distribution

Interpret topics in the context of Indonesian volcano discourse: "evacuation and displacement", "ashfall and health", "government response and aid", "volcanic tourism and danger", etc.

---

## Tools and Environment Summary

| Phase                  | Primary Libraries                                    |
| ---------------------- | ---------------------------------------------------- |
| Data Collection        | ytscrape, tapline, google-api-python-client          |
| Preprocessing          | Sastrawi, nahiarhdNLP, indo-text-preprocessing, NLTK |
| Feature Engineering    | scikit-learn (TfidfVectorizer, CountVectorizer)      |
| Modeling (Traditional) | scikit-learn (NaiveBayes, SVM, LogisticRegression)   |
| Modeling (Transformer) | transformers (IndoBERT, BERTopic)                    |
| Topic Modeling         | gensim (LDA), BERTopic                               |
| NER                    | transformers (IndoBERT fine-tuned), pycrfsuite       |
| Visualization          | matplotlib, seaborn, wordcloud, pyLDAvis             |

All libraries are compatible with Python 3.11 and can be run in VSCode `.ipynb` notebooks. For Google Colab, GPU acceleration is available for transformer-based models.

---

## Suggested Workflow Sequence

1. **Week 1–2**: Data collection (90 videos, 20 comments each) + metadata structuring
2. **Week 3**: Preprocessing pipeline development and validation
3. **Week 4**: Feature engineering + exploratory data analysis
4. **Week 5–6**: Modeling (sentiment classifier, topic model, NER)
5. **Week 7**: Task execution (clustering, similarity, framing analysis)
6. **Week 8**: Interpretation, visualization, and reporting

This phased approach ensures that each stage builds on a validated foundation, and that you can iterate on preprocessing and feature engineering without re-running expensive modeling steps.
