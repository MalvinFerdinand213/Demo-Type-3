"""Penghubung model skoring aspek ke aplikasi.

Model: aspect_model.py + aspect_model.json (dilatih oleh train_model.py dari qualified_reviews.csv).
Model ini hanya membaca TEKS. Foto diterima tetapi belum dianalisis.

Kontrak yang dipakai app:
  - text   : str, isi review
  - images : list[bytes], foto review (belum dipakai)
  - star   : int, bintang hotel (belum dipakai)
  - return : {aspek: skor 1.0-5.0, atau None kalau aspek tidak disebut di teks}
"""
import csv
import hashlib
import os
import random
import re

import aspect_model as am
from hotel_core import ASPECTS

assert list(am.ASPECTS) == list(ASPECTS), "Daftar aspek di aspect_model.py dan hotel_core.py harus sama."

_MODEL = None


def _get_model() -> dict:
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = am.load_model(am.DEFAULT_MODEL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                "aspect_model.json tidak ditemukan. Jalankan: python train_model.py "
                "(atau taruh file itu di folder yang sama dengan app_hotel.py)."
            ) from None
    return _MODEL


def analyze_review(text: str, images: list, star: int) -> dict:
    """{aspek: {"score": float | None, "evidence": [potongan teks yang dipakai]}}"""
    detail = am.analyze(text or "", _get_model())
    for d in detail.values():
        if d["score"] is not None:
            d["score"] = round(d["score"], 2)
    return detail


def score_review(text: str, images: list, star: int) -> dict:
    return {a: d["score"] for a, d in analyze_review(text, images, star).items()}


def demo_scores(text: str, images: list, star: int) -> dict:
    """Skor DUMMY (acak, bukan dari model), hanya untuk mencoba tampilan."""
    key = f"{text}|{len(images)}|{star}".encode()
    rng = random.Random(int(hashlib.sha256(key).hexdigest(), 16) % (2**32))
    return {a: round(min(5.0, max(1.0, rng.gauss(2.2 + 0.4 * star, 0.9))), 1) for a in ASPECTS}
