"""
Modul bersama untuk klasifikasi sentimen (dipakai phase4, phase5, phase6).

Kenapa modul terpisah?  Preprocessing saat TRAINING, EVALUASI, dan INFERENSI
harus 100% identik. Kalau ditulis ulang di tiap script, mudah terjadi
"training-serving skew" yang diam-diam menurunkan akurasi.

Normalisasi sengaja RINGAN (tanpa stemming/stopword): kata seperti "tidak",
"jangan", "semoga" justru pembawa sentimen utama.
"""
from __future__ import annotations

import re
from pathlib import Path

LABELS = ["Positif", "Netral", "Negatif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

MODEL_DIR = Path("models")
TFIDF_PATH = MODEL_DIR / "tfidf_logreg.joblib"
INDOBERT_DIR = MODEL_DIR / "indobert-sentiment"
SLANG_FILE = Path("data/raw/slang_dict.txt")

# Slang inti domain bencana (melengkapi slang_dict.txt bila ada).
BASE_SLANG = {
    "yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "ngga": "tidak", "tdk": "tidak", "g": "tidak", "dgn": "dengan", "utk": "untuk",
    "bgt": "banget", "aja": "saja", "sdh": "sudah", "udh": "sudah", "udah": "sudah",
    "blm": "belum", "krn": "karena", "karna": "karena", "jd": "jadi", "tp": "tapi",
    "smg": "semoga", "smoga": "semoga", "aamiin": "amin", "aamin": "amin",
    "amiin": "amin", "aminn": "amin", "allah": "allah", "pemda": "pemerintah daerah",
    "pemkab": "pemerintah kabupaten", "bnpb": "bnpb", "sy": "saya", "gw": "saya",
    "gue": "saya", "lu": "kamu", "lo": "kamu", "org": "orang", "dr": "dari",
    "sm": "sama", "trs": "terus", "klo": "kalau", "kalo": "kalau", "emg": "memang",
}

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_REPEAT = re.compile(r"(\w)\1{2,}")          # "kasiaaaan" -> "kasiaan"
_NONWORD = re.compile(r"[^a-z_\s]")
_SPACE = re.compile(r"\s+")


def _load_slang() -> dict[str, str]:
    slang = dict(BASE_SLANG)
    if SLANG_FILE.exists():
        for line in SLANG_FILE.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                slang.setdefault(parts[0].lower(), parts[1].lower())
    return slang


_SLANG = _load_slang()


def normalize(text) -> str:
    """Teks mentah -> teks bersih ringan. Emoji DIKONVERSI jadi token kata
    (mis. 😭 -> 'emo_loudly_crying_face') karena emoji membawa sinyal emosi."""
    t = "" if text is None else str(text)
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    try:
        import emoji
        t = emoji.demojize(t, delimiters=(" emo_", " "))
    except ImportError:  # pip install emoji  (disarankan)
        pass
    t = t.lower()
    t = _REPEAT.sub(r"\1\1", t)
    t = _NONWORD.sub(" ", t)
    tokens = [_SLANG.get(tok, tok) for tok in _SPACE.split(t.strip()) if tok]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Prediksi
# ---------------------------------------------------------------------------
def predict(texts: list[str], model: str = "tfidf", batch_size: int = 32) -> list[str]:
    if model == "tfidf":
        import joblib

        if not TFIDF_PATH.exists():
            raise FileNotFoundError(f"{TFIDF_PATH} belum ada. Jalankan: "
                                    "python script/phase4_train_classifier.py --model tfidf")
        pipe = joblib.load(TFIDF_PATH)
        return pipe.predict([normalize(t) for t in texts]).tolist()

    if model == "indobert":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not INDOBERT_DIR.exists():
            raise FileNotFoundError(f"{INDOBERT_DIR} belum ada. Jalankan: "
                                    "python script/phase4_train_classifier.py --model indobert")
        tok = AutoTokenizer.from_pretrained(INDOBERT_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(INDOBERT_DIR).eval()
        out: list[str] = []
        clean = [normalize(t) for t in texts]
        with torch.no_grad():
            for i in range(0, len(clean), batch_size):
                enc = tok(clean[i:i + batch_size], truncation=True, max_length=128,
                          padding=True, return_tensors="pt")
                ids = mdl(**enc).logits.argmax(-1).tolist()
                out.extend(ID2LABEL[j] for j in ids)
        return out

    raise ValueError(f"Model tidak dikenal: {model}")
