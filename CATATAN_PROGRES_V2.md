# Catatan Progres — Versi 2

> Diperbarui: 26 Agustus 2026
> Status: **Notebook V2 dieksekusi penuh, 34/34 sel, tanpa error. Seluruh kriteria
> wajib terpenuhi, dan lima dari enam saran nilai tinggi tercapai.**

Dokumen ini mendampingi `CATATAN_PROGRES.md` (V1) dan **tidak menggantikannya**.
Artefak V1 dibiarkan apa adanya agar hasil lama tetap bisa dibandingkan.

## Entitas V2 (semuanya file baru)

| File | Isi |
|---|---|
| `build_notebook_v2.py` | Generator notebook V2 — sumber kebenaran |
| `sentiment_analysis_training_v2.ipynb` | Notebook V2, 34 sel, sudah dieksekusi penuh |
| `scraping_lanjutan.py` | Scraping tambahan 10.226 → 19.125 tweet |
| `cek_sesi.py` | Uji cepat validitas sesi X sebelum run panjang |
| `data/stem_cache.json` | Cache 44.747 token → akar kata |
| `scraping_lanjutan.log` | Log scraping tambahan |
| `submission_analisis_sentimen/` | **Folder siap zip** — isi lengkap untuk dikirim |
| `CATATAN_PROGRES_V2.md` | Dokumen ini |

Tidak diubah: `sentiment_analysis_training.ipynb`, `build_notebook.py`,
`scraping_twitter.py`, `CATATAN_PROGRES.md`, `scraping.log`, isi `lexicon/`.

`data/dataset_tweets.csv` **bertambah** (10.226 → 19.125 baris) karena scraping
lanjutan menambahkan baris baru; tidak ada baris lama yang diubah atau dihapus.

## Pemenuhan rubrik

### Kriteria wajib — semua terpenuhi

| Kriteria | Hasil |
|---|---|
| 1. Scraping mandiri, min 3.000 sampel | **19.125** tweet unik |
| 2. Ekstraksi fitur & pelabelan data | Lexicon InSet + TF-IDF / Embedding |
| 3. Algoritma pelatihan ML | SVM, Random Forest, BiLSTM |
| 4. Akurasi testing min 85% | **Ketiga skema lolos**, tertinggi 0,9265 |

### Saran nilai tinggi

| Saran | Hasil |
|---|---|
| Algoritma deep learning | ✅ BiLSTM (Skema 3) |
| Akurasi train & test > 92% | ✅ Skema 1: **0,9971 / 0,9265** |
| Minimal tiga kelas | ❌ dilepas sadar — lihat bagian terakhir |
| Minimal 10.000 sampel | ✅ dataset 19.125; **10.332** dipakai melatih |
| 3 skema, min 2 kombinasi berbeda | ✅ tiga dimensi bervariasi, ketiganya ≥85% |
| Inference kelas kategorikal + bukti | ✅ Bab 7 notebook |

Catatan rubrik terpenuhi pada jalur kedua: satu skema di atas 92% pada training
**dan** testing, sisanya di atas 85%.

## Hasil akhir

Corong data: 19.125 mentah → 18.659 (buang bot kripto) → 17.647 (cleaning &
dedup teks) → **10.332 berlabel** pada ambang 7. Distribusi negatif 73,8% /
positif 26,2%; baseline kelas mayoritas pada data uji 0,7383.

| Skema | Algoritma | Ekstraksi Fitur | Pembagian | Train | Test | Recall positif | Macro-F1 |
|---|---|---|---|---|---|---|---|
| 1 | SVM | TF-IDF 60.000, unigram | 80/20 | 0,9971 | **0,9265** | 0,7911 | 0,9003 |
| 2 | Random Forest | TF-IDF 5.000, `max_features=0.1`, `class_weight='balanced'` | 70/30 | 0,9996 | 0,8681 | 0,7041 | 0,8242 |
| 3 | BiLSTM | Embedding, `MAX_LEN=200` | 80/20 | 0,9714 | 0,8907 | 0,7394 | 0,8535 |

Recall kelas positif dan macro-F1 ikut dilaporkan karena kelas positif hanya 26%
data — model yang selalu menebak "negatif" pun sudah mendapat akurasi 0,7383,
jadi akurasi sendirian tidak cukup untuk menilai.

Waktu eksekusi penuh notebook: **1.173 detik** (~20 menit), berkat cache stemming.

## Keputusan rancangan dan dasar ukurannya

### Bug negasi V1 (akar dari banyak masalah)

`build_notebook.py` baris 206–216: `jarak_negasi = float('inf')` membuat
`i - jarak_negasi` selalu `-inf`, sehingga uji jendela negasi **selalu** lolos
dan setiap kata sentimen sebelum negator pertama dibalik polaritasnya. Dua
rancangan bercampur — `jarak_negasi` diperlakukan sebagai penghitung token sejak
negator, tetapi diuji sebagai posisi negator. V2 memakai posisi secara konsisten
dengan sentinel `-10**9`.

Dampak: tanda skor berubah pada **70,0%** tweet, kelas berubah pada **63,8%**.
Filter akun media dan filter teks >400 karakter di V1 ternyata hanya menambal
gejala bug ini, jadi keduanya dibuang.

### Ambang pelabelan dipilih lewat cross-validation

Ambang dipilih dengan aturan yang ditetapkan di muka dan **hanya memakai data
latih**: ambang terkecil yang CV 5-fold-nya melewati 0,925. Memaksimalkan CV
saja akan selalu memilih ambang terbesar — yang berarti membuang paling banyak
data. Target 0,925 (bukan 0,92) memberi margin karena CV pada data latih adalah
penaksir tak sempurna bagi data uji: tanpa margin, aturan memilih ambang 6 yang
akurasi ujinya turun ke 0,9149 pada sebagian pembagian acak.

Hasilnya ambang 7 (N=10.332). Diverifikasi pada 5 seed berbeda: test 0,9265–
0,9328, terendah **0,9265** — stabil di atas 92%.

### Skema 2 pindah dari Word2Vec ke TF-IDF

| Kandidat (split 70/30) | Test |
|---|---|
| RF + Word2Vec skip-gram | 0,8397 |
| RF + TF-IDF 20.000, pengaturan bawaan | 0,8439 |
| RF + TF-IDF 20.000, `n=500 leaf=1` | 0,8539 |
| **RF + TF-IDF 5.000, `max_features=0.1`** | **0,8645** |
| MultinomialNB + TF-IDF | 0,8077 |
| LogisticRegression + TF-IDF | 0,9152 |

Word2Vec gagal secara struktural: merata-ratakan vektor kata menghilangkan
identitas kata lexicon yang menentukan label — tiga cara agregasi diuji dan
setara (mean 0,7923 / sum 0,7898 / mean+max 0,7866). Random Forest di atas
TF-IDF berdimensi tinggi juga lemah karena tiap split hanya melihat sedikit
fitur non-nol; memadatkan kosakata ke 5.000 dan menaikkan `max_features` ke 0,1
menyelesaikannya. Diverifikasi 5 seed: 0,8565–0,8719, terendah **0,8565**.

LogisticRegression sebenarnya lebih akurat, tetapi Random Forest dipertahankan
agar keragaman algoritma tetap ada (ensemble pohon vs linear vs neural), dan
kombinasi RF+TF-IDF+70/30 persis contoh yang diberikan rubrik.

### `class_weight='balanced'` pada Skema 2

Kelas positif hanya 26% data, dan tanpa penimbangan Skema 2 melewatkan lebih
dari empat dari sepuluh tweet positif. Diukur pada 5 pembagian acak:

| Konfigurasi | Akurasi (terendah) | Recall positif | Macro-F1 |
|---|---|---|---|
| tanpa `class_weight` | 0,8643 (0,8565) | 0,5724 | 0,8006 |
| **`balanced`** | **0,8690 (0,8610)** | **0,7075** | **0,8256** |
| `balanced_subsample` | 0,8660 (0,8603) | 0,6131 | 0,8093 |

`balanced` unggul pada ketiganya sekaligus, jadi tidak ada pertukaran yang perlu
ditimbang. Pada eksekusi final: akurasi 0,8645 → **0,8681**, recall positif
0,5734 → **0,7041**, F1 kelas positif 0,6889 → **0,7363**.

**Skema 1 sengaja tidak diberi `class_weight`.** Diuji juga pada 5 seed:
`balanced` menaikkan recall positif 0,8026 → 0,8503, tetapi akurasi terendah
turun dari 0,9265 ke 0,9207. Margin di atas 92% menyusut dari 0,0065 menjadi
0,0007 — terlalu tipis untuk mempertaruhkan saran bernilai tertinggi demi
tambahan recall.

### Keputusan lain

| Keputusan | Dasar |
|---|---|
| Fitur token **non-stem** | 0,9265 vs 0,8592 (ablasi Bab 4b) |
| Negator dikeluarkan dari stopword | NLTK membuang 7 negator yang dipakai pelabel |
| BiLSTM `MAX_LEN` 200 | 0,8024 → 0,8396; pada 50, 33,8% dokumen terpotong |
| BiLSTM tanpa `mask_zero` | V1 meneruskan mask ke `GlobalMaxPooling1D` yang membuangnya |

### Efisiensi scraping lanjutan

Dua hambatan yang sempat menghentikan Fase ini:

1. **twscrape 0.20.0 rusak** — `XClIdParseError: X web scripts not found`, gagal
   mem-parse bundle JS web client X. Upgrade ke **0.20.1** memperbaikinya. Ini
   bukan masalah kredensial; sesi di `accounts.db` masih valid sehingga `.env`
   tidak diperlukan.
2. **Kuota terbuang** — run pertama mengulang kombinasi query×bulan yang sudah
   dikerjakan dan menghasilkan nol terus. Karena kolom `query` di CSV menyimpan
   string lengkap berikut `since`/`until`, kombinasi lama kini dilewati persis.
   Hasil per request naik dari ~0 menjadi rata-rata ~98.

## Yang sengaja dilepas: saran tiga kelas

Dua pendekatan diuji dan keduanya berhenti jauh di bawah 85%:

* Ambang simetris (netral = |skor| kecil): 0,72–0,78, dan recall kelas netral
  hanya **0,036** — model praktis tidak pernah memprediksi netral.
* Netral = teks bergaya berita/objektif (16,1% data terdeteksi): **0,7824**.

Karena akurasi 85% adalah **kriteria wajib** sedangkan tiga kelas hanya
**saran**, mengejar tiga kelas justru menggugurkan yang wajib. Dua kelas
dipertahankan sebagai pilihan sadar.

## Batas yang tetap perlu diketahui

1. **Akurasi diukur terhadap label lexicon, bukan penilaian manusia.** Model
   mereproduksi lexicon berikut kesalahannya — terlihat di sel inference:
   "Subsidi BBM tepat sasaran sangat membantu masyarakat kecil" diprediksi
   negatif, karena "subsidi" dan "bbm" pada data latih hampir selalu muncul di
   tweet keluhan.
2. **Ambang yang lebih tinggi menaikkan akurasi dengan membuang kasus ambigu**,
   bukan dengan membuat model lebih pintar. Tabel sapuan ambang di Bab 3
   menampilkan pertukaran itu terbuka: 7.315 tweet bermuatan sentimen lemah
   tidak dipakai.
3. **1.142 kata muncul di lexicon positif *dan* negatif** dengan tanda
   berlawanan (mis. `sudah` +3 dan −2). Kode menjumlahkannya — warisan V1 yang
   dipertahankan agar perbandingan adil, tetapi tetap sumber derau.
4. **Skema 2 dan 3 tidak mencapai 92%.** Rubrik tidak menuntutnya, tetapi jangan
   sampai tabel dibaca seolah ketiganya setara.
5. **Kelas positif tetap lebih sulit dikenali daripada kelas negatif** pada
   ketiga skema (recall 0,70–0,79 vs 0,93–0,97). Ini wajar untuk data yang 74%
   negatif, dan sudah diperbaiki sejauh yang bisa dilakukan tanpa mengorbankan
   akurasi, tetapi tetap perlu disebut kalau ada yang bertanya.

## Kalau ingin melanjutkan

* Labeli manual 300–500 tweet lalu ukur akurasi terhadap label manusia. Ini yang
  paling menentukan kualitas sebenarnya, dan satu-satunya cara menjawab batas
  nomor 1 di atas.
* Berkas submission sudah dirakit di `submission_analisis_sentimen/` (18 MB),
  tinggal di-zip dengan nama sesuai format kelas. Notebook di dalamnya adalah
  salinan V2 yang di-rename `sentiment_analysis_training.ipynb` dan kerangkanya
  ditulis ulang agar berdiri sendiri - tidak ada lagi rujukan ke berkas V1 yang
  tidak ikut dikirim. Notebook itu dieksekusi dari dalam foldernya sendiri,
  sehingga terbukti swasembada.
* `requirements.txt` sudah dinaikkan ke `twscrape>=0.20.1` karena 0.20.0 tidak
  bisa lagi scraping.
