# -*- coding: utf-8 -*-
"""Lanjutan scraping untuk menambah dataset dari ~10.200 ke ~19.000 tweet.

Entitas terpisah; `scraping_twitter.py` tidak disentuh agar bukti scraping asli
untuk Kriteria 1 tetap utuh. Seluruh logika query, jendela bulanan, penulisan
CSV, dan dedup dipakai ulang dari skrip asli lewat import.

Dua perbedaan saja terhadap skrip asli:

1. Tidak membaca `.env`. Mesin ini tidak punya berkas itu (scraping pertama
   dijalankan di mesin lain), tetapi `accounts.db` ikut terbawa dan memuat dua
   akun aktif berikut cookie sesinya. Jadi pool dipakai apa adanya, tanpa
   `del_account`/`add_account` yang menuntut kredensial.
2. Kuota per periode dinaikkan, dengan menjaga proporsi era yang sudah
   dirancang di skrip asli.

Skrip ini resume-safe: hasil ditambahkan bertahap ke data/dataset_tweets.csv dan
duplikat dibuang berdasarkan tweet_id, sama seperti skrip asli.

    python scraping_lanjutan.py
"""
import asyncio
import os
import random
import time
from datetime import datetime

os.environ.setdefault("TWS_HTTP_BACKEND", "curl")

from scraping_twitter import (  # noqa: E402
    DATA_PATH,
    DB_PATH,
    QUERIES,
    QUERIES_TAMBAHAN,
    _month_windows,
    append_rows,
    load_seen_ids,
    polite_sleep,
)

# Kuota lama: 2026=5500, 2025=2250, 2024=2250 (total 10.000, tercapai 10.226).
# Dinaikkan agar total ~19.000 tweet mentah, yang setelah pembersihan dan
# pelabelan menyisakan >=10.000 sampel terlatih.
PERIODS = [
    {"label": "2026", "year": 2026, "quota": 10500},
    {"label": "2025", "year": 2025, "quota": 4250},
    {"label": "2024", "year": 2024, "quota": 4250},
]

MAX_RUNTIME_MINUTES = int(os.getenv("MAX_RUNTIME_MINUTES", 420))


def query_sudah_terpakai():
    """Kombinasi query x jendela yang sudah menghasilkan tweet di run pertama.

    Kolom `query` pada CSV menyimpan string lengkap berikut since/until, jadi
    kombinasi yang sudah dikerjakan bisa dilewati persis. Ini penting karena
    batas kuota X hanya ~5 request/15 menit/akun: mengulang kombinasi lama
    berarti membuang jam untuk hasil nol.

    Kombinasi yang pernah dicoba tetapi tidak menghasilkan baris baru tidak
    tercatat di sini dan akan dicoba lagi - jumlahnya minoritas.
    """
    import csv as _csv

    dipakai = set()
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                q = row.get("query")
                if q:
                    dipakai.add(q)
    return dipakai


def bangun_rencana():
    """Query x jendela bulanan per periode, melewati yang sudah dikerjakan."""
    semua = QUERIES + QUERIES_TAMBAHAN
    dipakai = query_sudah_terpakai()
    print(f"[plan] {len(dipakai):,} kombinasi query sudah dikerjakan run "
          f"sebelumnya - akan dilewati")

    rencana = {}
    for p in PERIODS:
        entries = []
        for (sejak, sampai) in _month_windows(p["year"]):
            for q in semua:
                entri = f"{q} since:{sejak} until:{sampai}"
                if entri not in dipakai:
                    entries.append(entri)
        rng = random.Random(2026)
        rng.shuffle(entries)
        rencana[p["label"]] = entries
    return rencana


async def build_api_dari_db():
    """Pakai sesi yang sudah tersimpan di accounts.db, tanpa kredensial."""
    from twscrape import API, AccountsPool

    pool = AccountsPool(db_file=DB_PATH, wait_timeout=1800, wait_interval=15)
    akun = await pool.get_all()
    aktif = [a for a in akun if a.active]
    if not aktif:
        raise SystemExit(
            "Tidak ada akun aktif di accounts.db. Jalankan cek_sesi.py untuk "
            "memastikan, atau isi .env lalu pakai scraping_twitter.py."
        )
    print(f"[auth] {len(aktif)} akun aktif dari accounts.db: "
          f"{', '.join(a.username for a in aktif)}")
    return API(pool)


async def main():
    api = await build_api_dari_db()
    seen, per_year = load_seen_ids()
    start_ts = time.time()
    rencana = bangun_rencana()

    ptr = {p["label"]: 0 for p in PERIODS}
    quota = {p["label"]: p["quota"] for p in PERIODS}
    progress = {p["label"]: per_year.get(p["year"], 0) for p in PERIODS}

    print("[plan] Kuota & progress awal per periode:")
    for p in PERIODS:
        lb = p["label"]
        print(f"   {lb}: {progress[lb]:,}/{quota[lb]:,} "
              f"| {len(rencana[lb]):,} entri query tersedia")

    def pilih_periode():
        kandidat = [
            p["label"] for p in PERIODS
            if progress[p["label"]] < quota[p["label"]]
            and ptr[p["label"]] < len(rencana[p["label"]])
        ]
        if not kandidat:
            return None
        return min(kandidat, key=lambda lb: progress[lb] / quota[lb])

    while True:
        elapsed_min = (time.time() - start_ts) / 60
        if elapsed_min > MAX_RUNTIME_MINUTES:
            print(f"[stop] Batas waktu {MAX_RUNTIME_MINUTES} menit tercapai")
            break

        label = pilih_periode()
        if label is None:
            print("[stop] Semua kuota periode terpenuhi atau rencana habis")
            break

        query = rencana[label][ptr[label]]
        ptr[label] += 1
        batch_target = min(200, quota[label] - progress[label])

        kept = []
        try:
            async for tweet in api.search(query, limit=batch_target):
                text = (tweet.rawContent or "").strip()
                tid = str(tweet.id)
                if not text or tid in seen:
                    continue
                seen.add(tid)
                thn = tweet.date.year
                per_year[thn] = per_year.get(thn, 0) + 1
                if str(thn) in progress:
                    progress[str(thn)] += 1
                kept.append({
                    "created_at": str(tweet.date),
                    "username": tweet.user.username if tweet.user else "",
                    "full_text": text,
                    "tweet_id": tid,
                    "query": query,
                })
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc.__class__.__name__}: {str(exc)[:150]}")
            print("[info] Menunggu sebelum mencoba lagi...")
            await asyncio.sleep(90)
            continue

        if kept:
            append_rows(kept)
        ts = datetime.now().strftime("%H:%M:%S")
        prog_str = " | ".join(
            f"{p['label']}:{progress[p['label']]}/{quota[p['label']]}"
            for p in PERIODS
        )
        print(f"[{ts}] [{label}] '{query}' -> {len(kept)} baru "
              f"|| total:{len(seen):,} || {prog_str}", flush=True)

        polite_sleep()

    print(f"\n[selesai] Total tweet unik di {DATA_PATH}: {len(seen):,}")
    for p in PERIODS:
        print(f"   {p['label']}: {progress[p['label']]:,}/{quota[p['label']]:,}")


if __name__ == "__main__":
    asyncio.run(main())
