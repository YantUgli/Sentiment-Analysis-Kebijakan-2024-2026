# Catatan Progres — Submission Analisis Sentimen (Dicoding)

> Terakhir diperbarui: 24 Agustus 2026, ±17:30 WIB
> Status: **Scraping selesai. Eksekusi final notebook belum dilakukan.**

---

## Ringkasan Keputusan Proyek

| Aspek | Keputusan |
|---|---|
| Sumber data | Twitter/X via **twscrape** (login cookie akun pribadi, tanpa API key) |
| Topik | Isu ekonomi & kebijakan harga Indonesia (kenaikan harga, sembako, upah minimum, pajak, subsidi, bansos, inflasi, dll.) |
| Target data | 10.000+ sampel → **TERCAPAI: 10.226 tweet unik** |
| Pelabelan | Lexicon InSet + ambang batas skor **±3** (3 kelas: positif/netral/negatif) |
| Framework DL | TensorFlow/Keras (TF 2.21) |
| 3 Skema | SVM+TF-IDF (80/20), RF+Word2Vec (80/20), BiLSTM+Embedding (80/20) |

## Struktur File Saat Ini

```
Sentiment_Analysis/
├── scraping_twitter.py              # Scraper final (versi modifikasi user: kuota per tahun,
│                                    #   jendela bulanan since/until) → SUDAH SELESAI DIJALANKAN
├── sentiment_analysis_training.ipynb # Notebook 27 sel (SUDAH divalidasi 2x smoke test,
│                                    #   tapi BELUM dieksekusi dengan dataset penuh)
├── build_notebook.py                # Generator notebook (nbformat) — sumber notebook
├── requirements.txt                 # ✅ selesai
├── dataset_tweets.csv → data/dataset_tweets.csv  # 10.226 tweet unik ✅
├── data/
│   ├── dataset_tweets.csv           # Dataset final
│   └── arsip_tweet_2013_2014.csv    # Arsip tweet lama (tidak dipakai — hasil indeks terbatas X)
├── lexicon/
│   ├── inset_positive.tsv           # Lexicon positif (fajri91/InSet)
│   ├── inset_negative.tsv           # Lexicon negatif (fajri91/InSet)
│   └── kamus_alay.csv               # Kamus slang (nasalsabila/kamus-alay, ±15rb entri)
├── .env                             # KREDENSIAL — JANGAN ikut di-zip / di-commit!
├── .env.example                     # Template kredensial
├── accounts.db                      # Sesi twscrape — JANGAN ikut di-zip
├── scraping.log                     # Log proses scraping (riwayat lengkap)
├── scraper.pid                      # PID file proses terakhir
└── .venv/                           # Virtual environment (Python 3.12)
```

## Yang Sudah Selesai

1. **Environment**: venv `.venv` + semua dependensi terpasang & teruji
   (twscrape 0.20 + curl_cffi, pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9,
   gensim 4.4, tensorflow 2.21, nltk, Sastrawi, nbformat/nbclient).
2. **Lexicon & kamus** diunduh ke `lexicon/`.
3. **Scraping mandiri** ✅ — 10.226 tweet unik bahasa Indonesia
   (distribusi tahun: 2026=5.510 | 2025=2.272 | 2024=2.251).
4. **Notebook dibuat & tervalidasi** — seluruh 27 sel sukses dieksekusi pada smoke test
   (data parsial 790 & 902 tweet). Tidak ada error kode.
5. **Perbaikan yang sudah masuk desain notebook**:
   - Filter tweet trading kripto/forex (±10% polusi dari bot) di tahap load data.
   - Pelabelan pakai ambang ±3 (dari uji distribusi: proporsi kelas lebih seimbang
     daripada aturan >0/<0/=0 yang membuat netral hanya ~5%).
   - Pembersihan token "RT", normalisasi slang, negasi lexicon (window 3 token),
     stemming Sastrawi dengan cache, class_weight untuk BiLSTM.

## Temuan Penting Selama Eksekusi (pelajaran teknis)

- **snscrape/twikit mati di 2026**; twifork 2.3.5 & upstream twikit git masih error
  (`Couldn't get KEY_BYTE indices`) karena X mengganti bundle JS web client.
- Solusi: **twscrape + backend curl_cffi** (`TWS_HTTP_BACKEND=curl`) untuk menembus Cloudflare.
- Login password gagal (error 399 / verifikasi keamanan) → solusi: **cookie auth_token + ct0** dari browser.
- Operator `lang:id` menyebabkan X mengembalikan indeks lama 2013–2014 → query memakai
  **frasa kata kunci Indonesia saja** (tanpa lang:id), `since:`/`until:` bekerja normal tanpa `lang:id`.
- Kuota pencarian X ±5 request/15 menit/akun → strategi user: kuota per tahun
  (2026:5500, 2025:2250, 2024:2250) + jendela bulanan since/until → selesai 17:15 WIB.
- `wc -l` pada CSV tampak lebih besar dari jumlah baris karena ada tweet multi-baris
  dalam field ter-quote — bukan duplikasi.

---

## LANJUTAN YANG HARUS DILAKUKAN

### Langkah 1 — Eksekusi final notebook (BELUM DILAKUKAN, prioritas tertinggi)
```bash
cd /home/kbuser/bryant_folder/machine_learning/Sentiment_Analysis
.venv/bin/python /tmp/opencode/run_nb.py    # atau jalankan ulang script eksekusi nbclient
```
- Estimasi 15–25 menit (stemming 10rb tweet + training 3 skema).
- Script `/tmp/opencode/run_nb.py` menulis output langsung ke notebook
  (syarat submission: notebook terkirim dalam keadaan SUDAH DIJALANKAN).
- Jika `/tmp/opencode/run_nb.py` sudah hilang, isinya:
  baca notebook via `nbformat`, eksekusi via `nbclient.NotebookClient`
  (`timeout=2400`, `resources={"metadata": {"path": "."}}`), tulis balik ke file.

### Langkah 2 — Verifikasi target akurasi
Cek tabel perbandingan di Bab 5 notebook:
- [ ] Semua skema: **akurasi testing ≥ 85%**
- [ ] Minimal satu skema (target: BiLSTM): **train & test > 92%**

Jika belum tercapai, opsi tuning berurutan:
1. Naikkan kapasitas BiLSTM (Embedding 128, LSTM 128) atau epochs maksimal.
2. Sesuaikan ambang pelabelan (mis. ±4) jika distribusi kelas timpang.
3. Tambah `max_features` TF-IDF (20rb) atau ngram (1,3).
4. Word2Vec: `vector_size=200`, `epochs=20`.
5. Periksa overfitting: turunkan Dropout / tambah data dedupe ketat.

Catatan smoke test terakhir (902 tweet): test acc baru 45–52% — **wajar untuk data kecil**;
dengan 10rb tweet biasanya melonjak jauh di atas 85%. Jangan panik sebelum uji data penuh.

### Langkah 3 — Packaging zip submission
Isi zip (hanya ini):
```
[Analisis Sentimen]_Submission_Bryant.zip
├── sentiment_analysis_training.ipynb   (sudah berisi output)
├── scraping_twitter.py
├── requirements.txt
├── data/dataset_tweets.csv
└── lexicon/                            (inset_positive.tsv, inset_negative.tsv, kamus_alay.csv)
                                         → agar reviewer bisa re-run notebook
```
**JANGAN ikut meng-zip**: `.env`, `accounts.db`, `cookies.json` (jika ada), `.venv/`,
`scraping.log`, `scraper.pid`, `build_notebook.py`, `data/arsip_tweet_2013_2014.csv`.

```bash
zip -r "[Analisis Sentimen]_Submission_Bryant.zip" \
  sentiment_analysis_training.ipynb scraping_twitter.py requirements.txt \
  data/dataset_tweets.csv lexicon/
```

### Langkah 4 — Checklist akhir sebelum dikumpulkan
- [ ] Notebook di-restart & run-all sukses, semua cell ada outputnya (tanpa `*`)
- [ ] Cell inference menghasilkan kelas kategorikal (positif/netral/negatif) — Bab 6
- [ ] Tabel Bab 5 menunjukkan pemenuhan Kriteria 4 + saran akurasi >92%
- [ ] Distribusi kelas 3 kelas tampil di Bab 3 (pie chart)
- [ ] Tidak ada kredensial/cookie ter-cetak di output notebook atau file yang dizip
- [ ] Nama zip jelas (sesuai format kelas masing-masing)

## Perintah Cepat Saat Melanjutkan

```bash
cd /home/kbuser/bryant_folder/machine_learning/Sentiment_Analysis
source .venv/bin/activate

# cek dataset
python -c "import pandas as pd; print(pd.read_csv('data/dataset_tweets.csv', dtype={'tweet_id':'string'}).tweet_id.nunique())"

# eksekusi notebook final (buat ulang run_nb.py sesuai catatan Langkah 1)
python /tmp/opencode/run_nb.py

# setelah metrik OK → zip (Langkah 3)
```
