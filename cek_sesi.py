# -*- coding: utf-8 -*-
"""Uji cepat: apakah sesi tersimpan di accounts.db masih valid?

Menjalankan SATU request pencarian kecil. Dipakai sebelum run scraping panjang
supaya kegagalan autentikasi ketahuan dalam hitungan detik, bukan jam.

    python cek_sesi.py
"""
import asyncio
import os

os.environ.setdefault("TWS_HTTP_BACKEND", "curl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "accounts.db")


async def main():
    from twscrape import API, AccountsPool

    pool = AccountsPool(db_file=DB_PATH, wait_timeout=120, wait_interval=15)
    akun = await pool.get_all()
    if not akun:
        print("[gagal] accounts.db kosong.")
        return 1

    print(f"[info] {len(akun)} akun di pool:")
    for a in akun:
        print(f"   - {a.username} | active={a.active} | last_used={a.last_used}")

    aktif = [a for a in akun if a.active]
    if not aktif:
        print("[gagal] Tidak ada akun aktif.")
        return 1

    api = API(pool)
    print("\n[uji] Menjalankan satu request pencarian kecil...")
    n = 0
    try:
        async for tweet in api.search("kenaikan harga", limit=5):
            n += 1
            if n == 1:
                # Konsol Windows memakai cp1252; teks tweet bisa memuat emoji,
                # jadi dipaksa aman lebih dulu agar print tidak melempar error.
                cuplikan = (tweet.rawContent or "")[:70]
                cuplikan = cuplikan.encode("ascii", "replace").decode("ascii")
                penulis = tweet.user.username if tweet.user else "?"
                print(f"   contoh: [{tweet.date}] @{penulis} - {cuplikan}")
    except Exception as exc:  # noqa: BLE001
        print(f"[gagal] {exc.__class__.__name__}: {str(exc)[:200]}")
        return 1

    if n == 0:
        print("[gagal] Request berhasil tetapi tidak mengembalikan tweet — "
              "kemungkinan sesi ditolak diam-diam.")
        return 1

    print(f"\n[ok] Sesi masih valid — {n} tweet berhasil diambil.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
