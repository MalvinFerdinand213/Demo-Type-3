"""Logika inti (tanpa Streamlit): benchmark tier, penggabungan skor, perbandingan.

Skala semua skor: 1-5.
"""
import os

import numpy as np
import pandas as pd

ASPECTS = [
    "Lokasi",
    "Kebersihan",
    "Pelayanan",
    "Kamar dan fasilitas",
    "Harga",
    "Makanan dan minuman",
]
TIERS = [1, 2, 3, 4, 5]
BENCHMARK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.csv")
DISAGREE_THRESHOLD = 1.5  # selisih model vs rating pengguna yang ditandai "berbeda jauh"


# ---------- benchmark (hasil analisis skripsi per bintang) ----------
def demo_benchmark() -> pd.DataFrame:
    """DUMMY, bukan hasil analisismu. Hanya supaya app bisa dicoba sebelum benchmark.csv ada."""
    rows = []
    for t in TIERS:
        for i, a in enumerate(ASPECTS):
            rows.append(
                {
                    "bintang": t,
                    "aspek": a,
                    "rata_rata": round(2.6 + 0.35 * (t - 1) + 0.05 * ((i % 3) - 1), 2),
                    "std": 0.8,
                }
            )
    return pd.DataFrame(rows)


def load_benchmark(path: str = BENCHMARK_PATH):
    """Return (DataFrame, is_demo). Kolom CSV: bintang, aspek, rata_rata, std (std opsional)."""
    if not os.path.exists(path):
        return demo_benchmark(), True

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"bintang", "aspek", "rata_rata"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: kolom wajib hilang: {sorted(missing)}")
    if "std" not in df.columns:
        df["std"] = np.nan

    df["bintang"] = pd.to_numeric(df["bintang"], errors="coerce")
    df["rata_rata"] = pd.to_numeric(df["rata_rata"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")

    canon = {a.lower(): a for a in ASPECTS}
    df["aspek"] = df["aspek"].astype(str).str.strip().str.lower().map(canon)

    if df["aspek"].isna().any():
        raise ValueError(f"{path}: ada nama aspek yang tidak dikenal. Gunakan: {ASPECTS}")
    if df["bintang"].isna().any() or df["rata_rata"].isna().any():
        raise ValueError(f"{path}: kolom bintang / rata_rata masih ada yang kosong atau bukan angka.")
    if not df["bintang"].isin(TIERS).all():
        raise ValueError(f"{path}: bintang harus bernilai 1-5.")
    if df.duplicated(["bintang", "aspek"]).any():
        raise ValueError(f"{path}: ada kombinasi bintang + aspek yang ganda.")
    return df[["bintang", "aspek", "rata_rata", "std"]], False


# ---------- penggabungan model + rating pengguna ----------
def _clip(x: float) -> float:
    return float(min(5.0, max(1.0, x)))


def fuse(model_scores: dict, user_ratings: dict, w_user: float) -> dict:
    """Gabungkan skor model dan rating pengguna per aspek.

    gabungan = (1 - w_user) * model + w_user * rating_pengguna
    Kalau salah satu tidak ada, dipakai yang ada. Kalau keduanya tidak ada -> None.
    """
    out = {}
    for a in ASPECTS:
        m, u = model_scores.get(a), user_ratings.get(a)
        if m is not None and u is not None:
            f, src = (1 - w_user) * m + w_user * u, "gabungan"
        elif m is not None:
            f, src = m, "model saja"
        elif u is not None:
            f, src = u, "rating saja"
        else:
            f, src = None, "-"
        out[a] = {"model": m, "user": u, "fused": None if f is None else _clip(f), "sumber": src}
    return out


# ---------- perbandingan dengan benchmark ----------
def compare(fused: dict, bench: pd.DataFrame, tier: int, band: float = 0.3, z_band: float = 0.5):
    """Tabel per aspek: skor gabungan vs rata-rata hotel bintang `tier`."""
    b = bench[bench["bintang"] == tier].set_index("aspek")
    rows = []
    for a in ASPECTS:
        r = fused[a]
        f = r["fused"]
        row = {
            "aspek": a,
            "model": np.nan if r["model"] is None else r["model"],
            "user": np.nan if r["user"] is None else r["user"],
            "fused": np.nan if f is None else f,
            "tier_mean": np.nan,
            "gap": np.nan,
            "status": "-",
        }
        if f is not None and a in b.index:
            mean, std = float(b.loc[a, "rata_rata"]), b.loc[a, "std"]
            gap = f - mean
            if pd.notna(std) and std > 0:
                neutral = abs(gap / std) <= z_band
            else:
                neutral = abs(gap) <= band
            row["tier_mean"], row["gap"] = mean, gap
            row["status"] = (
                "Sekitar rata-rata" if neutral else ("Di atas rata-rata" if gap > 0 else "Di bawah rata-rata")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def overall(table: pd.DataFrame):
    """Rata-rata keseluruhan (hanya aspek yang punya skor dan benchmark)."""
    t = table.dropna(subset=["fused", "tier_mean"])
    if t.empty:
        return None, None
    return float(t["fused"].mean()), float(t["tier_mean"].mean())


def nearest_tier(fused: dict, bench: pd.DataFrame):
    """Bintang mana yang profilnya paling mirip (rata-rata selisih absolut terkecil)."""
    dists = {}
    for t in TIERS:
        b = bench[bench["bintang"] == t].set_index("aspek")
        diffs = [
            abs(fused[a]["fused"] - float(b.loc[a, "rata_rata"]))
            for a in ASPECTS
            if fused[a]["fused"] is not None and a in b.index
        ]
        if diffs:
            dists[t] = float(np.mean(diffs))
    if not dists:
        return None, {}
    return min(dists, key=dists.get), dists


def disagreements(fused: dict, threshold: float = DISAGREE_THRESHOLD) -> list:
    """Aspek yang skor modelnya berbeda jauh dari rating pengguna."""
    return [
        a
        for a in ASPECTS
        if fused[a]["model"] is not None
        and fused[a]["user"] is not None
        and abs(fused[a]["model"] - fused[a]["user"]) >= threshold
    ]


def available_tiers(bench: pd.DataFrame) -> list:
    """Bintang yang punya pembanding di benchmark."""
    return sorted(int(t) for t in bench["bintang"].unique())


# ---------- ringkasan otomatis (tanpa API) ----------
def rule_summary(table: pd.DataFrame, ov, near, tier: int, diff: list) -> str:
    """Ringkasan singkat berbasis aturan; selalu konsisten dengan angka di tabel."""
    t = table.dropna(subset=["fused", "tier_mean"])
    if t.empty or ov[0] is None:
        return ""
    gap = ov[0] - ov[1]
    if abs(gap) <= 0.15:
        pos = "sekitar rata-rata"
    elif gap > 0:
        pos = "di atas rata-rata"
    else:
        pos = "di bawah rata-rata"
    parts = [f"Skor keseluruhan **{ov[0]:.2f}/5**, {pos} hotel {tier}★ ({gap:+.2f})."]
    if len(t) >= 2:
        best, worst = t.loc[t["gap"].idxmax()], t.loc[t["gap"].idxmin()]
        bits = []
        if best["gap"] > 0:
            bits.append(f"Aspek terkuat: **{best['aspek']}** ({best['gap']:+.2f})")
        else:
            bits.append(f"Tidak ada aspek di atas rata-rata, yang paling dekat: **{best['aspek']}** ({best['gap']:+.2f})")
        if worst["gap"] < 0:
            bits.append(f"terlemah: **{worst['aspek']}** ({worst['gap']:+.2f})")
        parts.append(", ".join(bits) + ".")
    if near is not None:
        parts.append(f"Profilnya paling mirip hotel {near}★.")
    if diff:
        parts.append("Skor model dan rating kamu berbeda jauh pada " + ", ".join(diff) + ".")
    return " ".join(parts)
