# Pedoman Anotasi Sentimen — Komentar YouTube Berita Erupsi Gunung Berapi Indonesia

**Versi:** 1.0
**Berlaku untuk:** `data/sample_for_annotation.csv` (kolom `gold_label`)
**Label yang diizinkan:** `Positif`, `Negatif`, `Netral` (tulis persis seperti ini, huruf depan kapital)

---

## 1. Tujuan

Dokumen ini adalah aturan baku untuk melabeli sentimen komentar YouTube pada video berita erupsi gunung berapi di Indonesia. Hasil anotasi dipakai sebagai **gold label** untuk:

1. Mengukur akurasi model otomatis (Lexicon dan Transformer) yang sudah ada di proyek ini.
2. Menjadi data latih/uji untuk fine-tuning model di tahap berikutnya.

Karena gold label menjadi "kunci jawaban", **konsistensi antar-anotator lebih penting daripada kecepatan.** Kalau ragu, ikuti urutan keputusan di Bagian 5 dan isi kolom `annotator_note`.

---

## 2. Prinsip Utama

> **Yang dilabeli adalah sikap atau emosi PENULIS komentar, bukan topik yang dibicarakan.**

Topik bencana memang sudah negatif (korban, kerusakan, pengungsian). Kalau labelnya ikut topik, hampir semua komentar akan jadi Negatif dan labelnya tidak bisa membedakan apa-apa. Jadi pertanyaannya selalu:

> *"Apa yang dirasakan atau ingin disampaikan penulis: harapan/dukungan, emosi buruk/kritik, atau sekadar informasi?"*

Contoh: *"Korban meninggal sudah 13 orang"* → **Netral** (melaporkan fakta), padahal topiknya sangat negatif.

---

## 3. Definisi Label dalam Konteks Bencana Alam

### 3.1 Positif

Penulis menunjukkan sikap **konstruktif, suportif, atau penuh harapan**. Termasuk:

- **Doa dan harapan keselamatan**: "semoga", "ya Allah lindungi", "aamiin", "moga cepat reda".
- **Dukungan dan penguatan untuk korban**: "yang kuat ya", "sabar dan tabah", "kalian ga sendiri".
- **Apresiasi** untuk relawan, tim SAR, polisi, BPBD, PVMBG, atau jurnalis.
- **Rasa syukur**: "alhamdulillah keluarga selamat", "syukurlah sudah turun status".
- **Ajakan solidaritas atau donasi**: "yuk bantu saudara kita di Lumajang".
- **Kekaguman pada fenomena alam** tanpa nada mengejek: "masyaAllah dahsyat ciptaan-Mu".
- **Ajakan refleksi diri yang bernada harapan**: "ayo perbanyak istighfar, semoga kita dijauhkan dari musibah".

### 3.2 Negatif

Penulis menunjukkan **emosi negatif, penolakan, atau serangan**. Termasuk:

- **Ketakutan dan kengerian**: "ngeri bgt", "serem woy", "merinding liatnya".
- **Kesedihan dan duka yang tidak disertai doa atau harapan**: "sedih bgt liatnya 😭", "ya ampun kasian bgt".
- **Kemarahan dan kritik** kepada pemerintah, aparat, media, warga, atau pihak lain: "pemerintah kemana aja??", "tvOne beritanya diulang2 mulu".
- **Menyalahkan atau narasi azab**: "ini azab gara2 maksiat", "Tuhan murka sama pejabat korup".
- **Sarkasme dan ejekan**: "mantap, warganya malah nonton wedus gembel 👏👏".
- **Hinaan, umpatan, atau kata kasar**: "goblok", "tolol", "anjir" (yang dipakai untuk mencela).
- **Fatalisme yang meremehkan nyawa**: "biarin aja meletus semua, kiamat sekalian".

### 3.3 Netral

Penulis **tidak menunjukkan sikap emosional yang jelas**. Termasuk:

- **Informasi atau laporan fakta**: "status Semeru naik ke level IV", "abu sampai Surabaya".
- **Pertanyaan faktual**: "ini kejadian tgl brp ya?", "jaraknya brp km dr Malang?".
- **Penjelasan ilmiah atau edukasi** tanpa emosi: "itu awan panas guguran, bukan letusan eksplosif".
- **Komentar di luar topik tanpa polaritas**: "PDI.jokowi", "KA Argo Semeru namanya dari gunung ini".
- **Murni spam, promosi, atau tautan** (tanpa isi lain): "cek promo di bio", daftar link media sosial. Jika ada isi substantif di samping link, misalnya ajakan donasi beserta link rekening, nilai sentimen isinya (→ biasanya **Positif**).

> **Bukan Netral — dikosongkan:** komentar yang **tidak dipahami** anotator (aksara asing, bahasa daerah yang tidak dimengerti). Biarkan `gold_label` **kosong** dan tulis `"tidak paham"` di `annotator_note`. Baris kosong otomatis dibuang di Tahap 2, sehingga kelas Netral tidak tercemar tebakan.

---

## 4. Contoh Pelabelan

### 4.1 Contoh Positif

| # | Komentar | Label | Alasan |
|---|----------|-------|--------|
| P1 | `Ya Allah kuatkan & selamatkan sodara2ku di Lumajang 🤲` | **Positif** | Doa keselamatan dan penguatan untuk korban. |
| P2 | `Salut sm pak polisi yg turun lgsg ngarahin warga, sehat sll pak 🙏` | **Positif** | Apresiasi untuk aparat ditambah doa. |
| P3 | `Moga cpt reda ya, yg ngungsi tetep semangat!! kalian gak sendiri` | **Positif** | Harapan dan dukungan moral. |
| P4 | `Alhamdulillah kluarga gw di Pronojiwo aman smua, mksh relawan` | **Positif** | Rasa syukur dan terima kasih. |
| P5 | `Turut berduka, semoga almarhum husnul khotimah & kluarga dikasih ketabahan` | **Positif** | Ada duka, tetapi tujuan utamanya doa dan penguatan (lihat EC-2). |

### 4.2 Contoh Negatif

| # | Komentar | Label | Alasan |
|---|----------|-------|--------|
| N1 | `Ngeri bgt anjir, merinding gw liatnya 😱` | **Negatif** | Ketakutan atau kengerian. "anjir" di sini hanya penanda kaget, bukan pemicu label. |
| N2 | `Pemerintah kmn aja?? Tiap taun meletus tp jalur evakuasi ga pernah dibenerin` | **Negatif** | Kritik dan kemarahan kepada pemerintah. |
| N3 | `Udh genting msh aja diwawancara.. gk ada akhlak` | **Negatif** | Kritik keras kepada media. |
| N4 | `Ini azab, kebanyakan maksiat sih` | **Negatif** | Narasi menyalahkan atau azab. |
| N5 | `Warganya malah nonton wedus gembel, terakhir waras kapan sih??` | **Negatif** | Sarkasme dan celaan kepada warga. |
| N6 | `Sedih bgt liatnya 😭😭` | **Negatif** | Kesedihan murni tanpa doa atau harapan. |

### 4.3 Contoh Netral

| # | Komentar | Label | Alasan |
|---|----------|-------|--------|
| T1 | `Ini kejadian hari apa ya? thn brp?` | **Netral** | Pertanyaan faktual tanpa emosi. |
| T2 | `Abu vulkaniknya sampe Surabaya, mobil sm genteng ketutup debu` | **Netral** | Laporan kondisi atau fakta. |
| T3 | `Itu APG (awan panas guguran), beda sm letusan eksplosif` | **Netral** | Penjelasan ilmiah. |
| T4 | `KA Argo Semeru namanya diambil dr gunung ini` | **Netral** | Di luar topik, tanpa polaritas. |
| T5 | `Respect. Try -> 'promo sm'!` | **Netral** | Spam atau promosi. Kata "respect" tidak dihitung karena hanya umpan spam. |

---

## 5. Urutan Keputusan (Wajib Diikuti)

Jalankan langkah berikut **berurutan**. Berhenti di langkah pertama yang cocok.

```
0. Apakah komentar kosong atau TIDAK DAPAT DIPAHAMI anotator?
      → gold_label DIKOSONGKAN  (annotator_note: "tidak paham" / "bahasa asing")

1. Apakah komentar HANYA berisi spam/promosi/link tanpa isi lain?
      → Netral  (annotator_note: "spam")
      (Jika ada isi substantif di samping link, misalnya ajakan donasi → lanjut ke langkah 2.)

2. Apakah ada sarkasme, hinaan, atau narasi azab/menyalahkan?
      → Negatif  (meskipun ada kata religius atau kata positif)

3. Apakah ada lebih dari satu sentimen (campuran)?
      → Tentukan TUJUAN UTAMA penulis (lihat Bagian 6, EC-1 & EC-2).
        Jika benar-benar seimbang → Negatif, dan isi annotator_note: "campuran".

4. Apakah ada doa, harapan, dukungan, apresiasi, atau syukur?
      → Positif

5. Apakah ada ketakutan, kesedihan, kemarahan, atau kritik?
      → Negatif

6. Selain itu (fakta, pertanyaan, info, di luar topik tanpa emosi)
      → Netral
```

---

## 6. Kasus Khusus / Edge Cases

### EC-1 — Doa digabung kritik

Tentukan **bagian mana yang menjadi isi utama** komentar.

| Komentar | Label | Alasan |
|----------|-------|--------|
| `Semoga korban selamat. Tp pemerintah lelet bgt, udh 3 hari bantuan blm nyampe, mana janjinya??` | **Negatif** | Kritik lebih panjang, spesifik, dan punya sasaran. Doanya hanya pembuka. |
| `Semoga semua selamat & cepet pulih ya, sabar sodaraku. Pemerintah tolong lbh sigap ya 🙏` | **Positif** | Doa dan dukungan mendominasi. Permintaannya sopan, bukan serangan. |
| `Pejabat sibuk naik gaji rakyat taruhan nyawa. Mengharap ridho Tuhan YME.` | **Negatif** | Penutup religius tidak mengubah isi utamanya, yaitu kemarahan. |

> **Aturan tie-break:** jika porsi doa dan kritik setara dan anotator tetap ragu → **Negatif** + catat `"campuran"`. Alasannya, kritik yang eksplisit adalah sinyal yang lebih informatif untuk analisis framing.

### EC-2 — Ungkapan duka ("Innalillahi", "turut berduka")

- **Hanya ungkapan duka**: `Innalillahi wa inna ilaihi rojiun 😢` → **Negatif** (duka atau kesedihan).
- **Duka + doa atau penguatan**: `Innalillahi, semoga korban diterima di sisi-Nya, aamiin` → **Positif** (tujuan utamanya mendoakan).
- **Duka + narasi kiamat/azab**: `Innalillahi, kiamat sdh dekat, ini peringatan Allah` → **Negatif** (menakut-nakuti).

### EC-3 — Konten religius dan narasi akhir zaman

- `Ayo perbanyak istighfar, semoga kita semua dilindungi` → **Positif** (ajakan + harapan).
- `Kalau 7 gunung meletus barengan, dukhon gelap 40 hari 40 malam` → **Netral** jika disampaikan sebagai "informasi" tanpa emosi. Jika ada nada menakut-nakuti seperti `siap2 kalian!!` → **Negatif**.

### EC-4 — Kekaguman yang berpotensi tidak peka

- `MasyaAllah dahsyat bgt ciptaan Allah` → **Positif** (kekaguman).
- `Keren bgt awan panasnya buat konten 😂` → **Negatif** jika menertawakan atau meremehkan korban. **Positif** jika murni kagum tanpa ejekan. Kalau ragu → isi catatan.

### EC-5 — Kata kasar sebagai penanda kaget

Kata seperti "anjir", "gila", "buset" sering hanya menunjukkan **keterkejutan**, bukan hinaan.
- `Gila gede bgt letusannya` → **Negatif** (kengerian/kaget), **bukan** karena kata "gila".
- `Gila nih orang2, udh dibilang ngungsi masih ngeyel` → **Negatif** (celaan).
- Intinya: label ditentukan oleh **emosi keseluruhan**, bukan oleh satu kata.

### EC-6 — Pertanyaan retoris vs pertanyaan faktual

- `Jaraknya brp km dr kawah?` → **Netral** (pertanyaan faktual).
- `Pemerintah kemana aja selama ini???` → **Negatif** (retoris, isinya kritik).

### EC-7 — Komentar hanya emoji

- Emoji dengan emosi jelas: `😭😭😭` → **Negatif**, `🙏🙏` / `🤲` → **Positif**.
- Emoji acak, campuran, atau "seni emoji" → **Netral** + catat `"emoji ambigu"`.
- Emoji tertawa (😂🤣) pada konteks bencana tanpa teks → **Netral** + catat. Jangan menebak sarkasme tanpa bukti.

### EC-8 — Bahasa daerah dan campuran bahasa (code-mixing)

- Terjemahkan dulu maknanya. `Ojo cedak-cedak, bahaya rek!` (Jawa: "jangan dekat-dekat, bahaya") → **Positif**, karena isinya imbauan yang peduli keselamatan (lihat EC-10).
- Jika anotator **tidak memahami** sebagian besar isinya → **kosongkan `gold_label`** + catat `"tidak paham"`. Jangan menebak.
- Bahasa Inggris dilabeli dengan aturan yang sama: `Pray for Semeru 🙏` → **Positif**.

### EC-9 — Kritik ke media (bukan ke bencana)

Kritik kepada stasiun TV, reporter, atau judul clickbait tetap **Negatif**. Contoh: `Judulnya lebay, clickbait`.

### EC-10 — Peringatan / imbauan keselamatan

- `Warga tolong jauhi jalur lahar ya, jaga keselamatan` → **Positif** (kepedulian).
- `Jangan nonton di jembatan, bahaya!` → **Positif** jika nadanya peduli. **Negatif** jika disertai celaan (`dasar bebal`).

---

## 7. Prosedur Kerja Anotator

1. **Buka** `data/sample_for_annotation.csv` di Google Sheets atau LibreOffice. Hindari Excel jika memungkinkan, karena Excel sering merusak emoji atau encoding saat menyimpan.
2. **Baca kolom `comment_text` (teks asli).** Jangan memakai kolom hasil preprocessing, karena emoji dan tanda baca yang membawa emosi sudah dihapus di sana.
3. **Jangan melihat prediksi model** (file di `data/sentiment/`) sebelum selesai, supaya tidak terbias.
4. Isi kolom **`gold_label`** dengan salah satu: `Positif`, `Negatif`, `Netral`.
5. Isi kolom **`annotator_note`** bila kasusnya ragu, campuran, tidak dipahami, atau termasuk edge case. Contoh: `"EC-1 campuran"`, `"tidak paham"`.
6. **Jangan mengubah, menghapus, atau mengurutkan ulang** kolom lain, terutama `sample_id`.
7. Simpan kembali sebagai **CSV (UTF-8)** dengan nama file yang sama.
8. Baris yang tidak dipahami atau benar-benar tidak bisa dilabeli **dikosongkan** (`gold_label` kosong + catatan). Baris itu otomatis dibuang di Tahap 2 skrip split.

---

## 8. Kontrol Kualitas (Disarankan)

- **Anotasi ganda:** minimal 2 anotator melabeli subset yang sama (misalnya 60 baris pertama) secara independen.
- **Ukur kesepakatan** dengan *Cohen's Kappa*:
  ```python
  from sklearn.metrics import cohen_kappa_score
  cohen_kappa_score(labels_anotator_A, labels_anotator_B)
  ```
  Target **κ ≥ 0.61** (*substantial agreement*). Jika κ < 0.61, diskusikan baris yang tidak sepakat, perbarui pedoman ini (naikkan versinya), lalu ulangi.
- **Adjudikasi:** ketidaksepakatan diputuskan oleh anotator ketiga atau lewat diskusi tim. Label akhir dicatat di `gold_label`.

---

## 9. Pemetaan ke Skema Numerik Proyek

Skrip `phase3_sentiment.py` dan `phase3_sentiment_lexicon.py` memakai label angka. Pemetaannya:

| Gold label | Kode numerik (`sentiment_label`) |
|------------|----------------------------------|
| Netral     | 0 |
| Positif    | 1 |
| Negatif    | 2 |
