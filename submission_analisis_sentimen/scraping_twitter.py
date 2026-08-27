"""
Scraping tweet bahasa Indonesia dari X (Twitter) menggunakan twscrape.
Topik: isu ekonomi & kebijakan harga di Indonesia.

Cara pakai:
1. Isi kredensial pada file .env:
       X_USERNAME=username_anda
       X_PASSWORD=password_anda
       X_EMAIL=email_anda

   Jika login via password gagal karena verifikasi keamanan X,
   gunakan metode cookie (lebih andal):
   - Login ke https://x.com di browser, lalu buka DevTools (F12)
     > Application > Cookies > https://x.com
   - Salin nilai cookie 'auth_token' dan 'ct0', lalu tambahkan ke .env:
       X_AUTH_TOKEN=nilai_auth_token
       X_CT0=nilai_ct0

2. Jalankan:
       python scraping_twitter.py

Catatan:
- Tanpa API key berbayar; memakai sesi akun pribadi.
- Skrip resume-safe: hasil ditambahkan bertahap ke data/dataset_tweets.csv,
  duplikat dibuang berdasarkan tweet_id.
"""

import asyncio
import csv
import os
import random
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_TWEETS = int(os.getenv("TARGET_TWEETS", 10000))
MAX_RUNTIME_MINUTES = int(os.getenv("MAX_RUNTIME_MINUTES", 720))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_tweets.csv")
DB_PATH = os.path.join(BASE_DIR, "accounts.db")

QUERIES = [
    # bernuansa keluhan/negatif
    "kenaikan harga",
    "harga sembako naik",
    "upah minimum gaji",
    "pajak naik rakyat",
    "gaji tidak cukup",
    "harga BBM naik",
    "listrik mahal",
    # bernuansa positif
    "bantuan sosial pemerintah",
    "harga turun murah",
    "gaji naik terima kasih",
    "thr karyawan",
    "subsidi membantu masyarakat",
    # bernuansa netral/informatif
    "inflasi Indonesia",
    "subsidi pemerintah",
    "belanja hemat tips",
    "kebijakan ekonomi pemerintah",
]

QUERIES_TAMBAHAN = [
    "harga cabai", "harga beras", "harga minyak goreng", "tarif tol naik",
    "bbm subsidi", "harga tiket naik", "makan murah kenyang", "hemat pengeluaran",
    "kenaikan tarif", "tagihan listrik", "bayar sekolah", "gajian bonus",
    "naik jabatan gaji", "promo diskon harga", "harga telur", "uang jajan anak",
]


PERIODS = [
    {"label": "2026", "year": 2026, "quota": 5500},  # era Prabowo
    {"label": "2025", "year": 2025, "quota": 2250},  # transisi Jokowi->Prabowo
    {"label": "2024", "year": 2024, "quota": 2250},  # akhir era Jokowi
]


def _month_windows(year):
    from datetime import date, timedelta

    hari_ini = date.today()
    besok = hari_ini + timedelta(days=1)
    windows = []
    for m in range(1, 13):
        awal = date(year, m, 1)
        if awal > hari_ini:
            break
        akhir = date(year + 1, 1, 1) if m == 12 else date(year, m + 1, 1)
        if akhir > besok:
            akhir = besok
        windows.append((awal.isoformat(), akhir.isoformat()))
    return windows


def bangun_rencana_per_periode():
    semua = QUERIES + QUERIES_TAMBAHAN
    rencana = {}
    for p in PERIODS:
        entries = []
        for (sejak, sampai) in _month_windows(p["year"]):
            for q in semua:
                entries.append(f"{q} since:{sejak} until:{sampai}")
        rng = random.Random(42)
        rng.shuffle(entries)
        rencana[p["label"]] = entries
    return rencana

FIELDS = ["created_at", "username", "full_text", "tweet_id", "query"]


def load_env():
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    accounts = []
    idx = 1
    while True:
        suffix = "" if idx == 1 else f"_{idx}"
        username = os.getenv(f"X_USERNAME{suffix}")
        if not username:
            break
        accounts.append({
            "username": username,
            "password": os.getenv(f"X_PASSWORD{suffix}"),
            "email": os.getenv(f"X_EMAIL{suffix}"),
            "auth_token": os.getenv(f"X_AUTH_TOKEN{suffix}"),
            "ct0": os.getenv(f"X_CT0{suffix}"),
        })
        idx += 1

    if not accounts:
        sys.exit("Isi minimal X_USERNAME, X_PASSWORD, dan X_EMAIL pada file .env")
    for a in accounts:
        if not all([a["username"], a["password"], a["email"]]):
            sys.exit(
                f"Akun '{a['username']}' kurang lengkap: butuh username, "
                "password, dan email."
            )
    return accounts


async def build_api():
    import json

    from twscrape import AccountsPool, API

    accounts = load_env()
    pool = AccountsPool(db_file=DB_PATH, wait_timeout=1800, wait_interval=15)

    need_login = []
    for creds in accounts:
        # Hapus akun lama agar bisa ditambah ulang dengan data terbaru
        try:
            await pool.del_account(creds["username"])
        except Exception:  # noqa: BLE001
            pass

        if creds["auth_token"] and creds["ct0"]:
            cookies_json = json.dumps({
                "auth_token": creds["auth_token"],
                "ct0": creds["ct0"],
            })
            await pool.add_account(
                creds["username"],
                creds["password"],
                creds["email"],
                "dummy",
                cookies=cookies_json,
            )
            print(f"[auth] {creds['username']}: memakai cookies dari .env")
        else:
            await pool.add_account(
                creds["username"], creds["password"], creds["email"], "dummy"
            )
            need_login.append(creds["username"])

    if need_login:
        print(f"[auth] Login via username/password untuk: {', '.join(need_login)}")
        await pool.login_all()

    active = 0
    for creds in accounts:
        acc = await pool.get(creds["username"])
        if acc and acc.active:
            active += 1
        else:
            print(f"[warn] Akun '{creds['username']}' TIDAK aktif — dilewati.")
    if active == 0:
        sys.exit(
            "Tidak ada akun aktif. Coba metode cookie: isi X_AUTH_TOKEN dan "
            "X_CT0 (mis. _2 untuk akun kedua) di .env."
        )
    print(f"[auth] {active} akun aktif siap dipakai (rotasi otomatis).")
    return API(pool)


def load_seen_ids():
    seen = set()
    per_year = {}
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["tweet_id"])
                y = row["created_at"][:4]
                if y.isdigit():
                    per_year[int(y)] = per_year.get(int(y), 0) + 1
    print(f"[data] {len(seen):,} tweet sudah ada di dataset")
    return seen, per_year


def append_rows(rows):
    new_file = not os.path.exists(DATA_PATH)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def polite_sleep(low=0.5, high=1.5):
    time.sleep(random.uniform(low, high))


async def main():
    api = await build_api()
    seen, per_year = load_seen_ids()
    start_ts = time.time()
    rencana = bangun_rencana_per_periode()

    ptr = {p["label"]: 0 for p in PERIODS}
    quota = {p["label"]: p["quota"] for p in PERIODS}
    progress = {p["label"]: per_year.get(p["year"], 0) for p in PERIODS}

    print("[plan] Kuota & progress awal per periode:")
    for p in PERIODS:
        lb = p["label"]
        print(f"   {lb}: {progress[lb]:,}/{quota[lb]:,}")

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
                if thn in per_year:
                    per_year[thn] += 1
                else:
                    per_year[thn] = 1
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
        print(f"[{ts}] [{label}] '{query}' -> {len(kept)} baru || {prog_str}")

        polite_sleep()

    print(f"\n[selesai] Total tweet unik di {DATA_PATH}: {len(seen):,}")
    for p in PERIODS:
        print(f"   {p['label']}: {progress[p['label']]:,}/{quota[p['label']]:,}")


if __name__ == "__main__":
    os.environ.setdefault("TWS_HTTP_BACKEND", "curl")
    asyncio.run(main())
