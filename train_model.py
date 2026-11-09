"""Melatih model sentimen untuk aspect_model.py dan membuat benchmark.csv dari dataset.

Pakai:
    pip install scikit-learn scipy pandas numpy
    python train_model.py --data qualified_reviews.csv           # latih + tulis aspect_model.json & benchmark.csv
    python train_model.py --data qualified_reviews.csv --tune    # bandingkan beberapa konfigurasi (CV)

Catatan penting:
  * Dataset hanya punya rating keseluruhan per review, TIDAK ada label per aspek. Model dilatih memprediksi
    rating dari teks (jendela kata di sekitar kata kunci aspek diberi label rating review). Hasilnya dicampur
    dengan kamus sentimen. Ini pendekatan weak supervision, bukan model aspek yang dilatih dengan label aspek.
  * Evaluasi memakai GroupKFold per hotel: hotel di data uji tidak pernah ada di data latih.
  * benchmark.csv dihitung dari skor out-of-fold, jadi tidak terlalu optimistis.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import aspect_model as am

HERE = os.path.dirname(os.path.abspath(__file__))
MODE, ALPHA, WIN_WEIGHT, MIN_DF = "both", 1.0, 0.5, 2
MIN_N_BENCH = 5  # aspek dengan kurang dari 5 review tidak dimasukkan ke benchmark


def load_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["review_text", "review_rating", "hotel_stars", "place_id"])
    df["review_text"] = df["review_text"].astype(str)
    return df[df["review_text"].str.strip().str.len() > 0].reset_index(drop=True)


def build_samples(df, mode, win_weight=WIN_WEIGHT):
    texts, y, w = [], [], []
    for text, rating in zip(df["review_text"], df["review_rating"]):
        if mode in ("whole", "both"):
            texts.append(" ".join(am.TOKEN_RE.findall(am.normalize(text))))
            y.append(rating)
            w.append(1.0)
        if mode in ("windows", "both"):
            for windows in am.extract_windows(text).values():
                for win in windows:
                    texts.append(" ".join(win))
                    y.append(rating)
                    w.append(win_weight)
    return texts, np.asarray(y, float), np.asarray(w, float)


def fit(texts, y, w, alpha):
    vec = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=MIN_DF, sublinear_tf=True)
    X = vec.fit_transform(texts)
    ridge = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
    vocab = {t: [round(float(vec.idf_[i]), 6), round(float(ridge.coef_[i]), 6)] for t, i in vec.vocabulary_.items()}
    return {"vocab": vocab, "intercept": float(ridge.intercept_)}, vec, ridge


def score_oof(df, mode, alpha, n_splits=5):
    """Skor setiap review memakai model yang TIDAK dilatih pada hotel review itu."""
    rows = []
    for tr, te in GroupKFold(n_splits).split(df, groups=df["place_id"]):
        texts, y, w = build_samples(df.iloc[tr], mode)
        model, _, _ = fit(texts, y, w, alpha)
        train_mean = float(df["review_rating"].iloc[tr].mean())
        for i in te:
            text = df["review_text"].iloc[i]
            whole = " ".join(am.TOKEN_RE.findall(am.normalize(text)))
            an = am.analyze(text, model)
            got = {a: r["score"] for a, r in an.items() if r["score"] is not None}
            rows.append(
                {
                    "i": i,
                    "rating": float(df["review_rating"].iloc[i]),
                    "stars": int(df["hotel_stars"].iloc[i]),
                    "ml_whole": am.ml_predict(whole, model),
                    "base": train_mean,
                    "overall": float(np.mean(list(got.values()))) if got else np.nan,
                    "aspects": got,
                }
            )
    return pd.DataFrame(rows).sort_values("i").reset_index(drop=True)


def report(oof, label=""):
    ok = oof.dropna(subset=["overall"])
    y_low = (ok["rating"] <= 3).astype(int)
    out = {
        "cakupan_review": len(ok) / len(oof),
        "spearman_ml_whole": spearmanr(oof["ml_whole"], oof["rating"])[0],
        "mae_ml_whole": float((oof["ml_whole"] - oof["rating"]).abs().mean()),
        "mae_baseline_rata2": float((oof["base"] - oof["rating"]).abs().mean()),
        "spearman_pipeline": spearmanr(ok["overall"], ok["rating"])[0],
        "auc_rating<=3_vs_>=4": roc_auc_score(y_low, -ok["overall"]) if y_low.nunique() == 2 else np.nan,
    }
    print(f"{label:<28}" + " | ".join(f"{k}={v:.3f}" for k, v in out.items()))
    return out


# ---------------- uji kewajaran (bukan akurasi) ----------------
SANITY = [
    ("Kamar kotor dan bau apek, tapi staf ramah banget", {"Kebersihan": "-", "Pelayanan": "+"}),
    ("Lokasi strategis dekat stasiun, harga murah", {"Lokasi": "+", "Harga": "+"}),
    ("Harga mahal tidak sesuai dengan fasilitas", {"Harga": "-"}),
    ("Sarapannya enak dan banyak pilihan", {"Makanan dan minuman": "+"}),
    ("Makanan tidak enak dan pelayanannya lambat", {"Makanan dan minuman": "-", "Pelayanan": "-"}),
    ("AC mati, kamar mandi bocor, wifi susah", {"Kamar dan fasilitas": "-"}),
    ("Resepsionis jutek dan cuek", {"Pelayanan": "-"}),
    ("Kamar bersih, nyaman, wangi", {"Kebersihan": "+", "Kamar dan fasilitas": "+"}),
    ("Tempatnya jauh dari mana-mana, susah akses", {"Lokasi": "-"}),
    ("Staff friendly and helpful, room is clean", {"Pelayanan": "+", "Kebersihan": "+", "Kamar dan fasilitas": "+"}),
    ("The room was dirty and smelly, very noisy at night", {"Kebersihan": "-", "Kamar dan fasilitas": "-"}),
    ("Not clean at all", {"Kebersihan": "-"}),
    ("Tidak bersih, banyak nyamuk", {"Kebersihan": "-"}),
    ("Kamarnya tidak bau dan tidak berisik", {"Kebersihan": "+"}),
    ("Kurang bersih sedikit tapi masih oke", {"Kebersihan": "-"}),
    ("Harganya terjangkau banget untuk lokasi segini", {"Harga": "+"}),
    ("Kasur empuk, tidur nyenyak", {"Kamar dan fasilitas": "+"}),
    ("Kamar sempit dan panas", {"Kamar dan fasilitas": "-"}),
    ("Parkir luas, lokasi mudah dicari", {"Kamar dan fasilitas": "+", "Lokasi": "+"}),
    ("Mas resepsionis sangat membantu dan sopan", {"Pelayanan": "+"}),
    ("Murah tapi kamar kotor", {"Harga": "+", "Kebersihan": "-", "Kamar dan fasilitas": "-"}),
    ("Check in lama, staf cuek", {"Pelayanan": "-"}),
    ("Sprei kotor ada noda", {"Kebersihan": "-"}),
    ("Overpriced untuk kamar seperti ini", {"Harga": "-"}),
    ("Breakfast was delicious", {"Makanan dan minuman": "+"}),
    ("Lokasi jauh banget dari pusat kota, macet", {"Lokasi": "-"}),
    ("Toilet bersih dan air panas lancar", {"Kebersihan": "+", "Kamar dan fasilitas": "+"}),
    ("Pelayanan buruk, kecewa", {"Pelayanan": "-"}),
    ("Sangat puas dengan pelayanannya", {"Pelayanan": "+"}),
    ("Kamar luas dan nyaman", {"Kamar dan fasilitas": "+"}),
    ("Banyak kecoa di kamar mandi", {"Kebersihan": "-"}),
    ("Wifi lancar, ac dingin", {"Kamar dan fasilitas": "+"}),
]


def sanity(model, verbose=False):
    total = detected = correct = 0
    for text, expected in SANITY:
        res = am.score_text(text, model)
        for aspect, sign in expected.items():
            total += 1
            s = res[aspect]
            if s is None:
                if verbose:
                    print(f"  TIDAK TERDETEKSI  {aspect:<20} <- {text}")
                continue
            detected += 1
            good = (sign == "+" and s > 3.2) or (sign == "-" and s < 2.8)
            correct += good
            if verbose and not good:
                print(f"  SALAH ({s:.2f}, harap {sign})  {aspect:<20} <- {text}")
    print(f"Uji kewajaran ({len(SANITY)} kalimat buatan sendiri): terdeteksi {detected}/{total}, "
          f"tanda benar {correct}/{detected}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "qualified_reviews.csv"))
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    df = load_data(args.data)
    print(f"Data: {len(df)} review, {df['place_id'].nunique()} hotel, bintang: "
          f"{df['hotel_stars'].value_counts().sort_index().to_dict()}")

    if args.tune:
        for mode in ("whole", "windows", "both"):
            for alpha in (1.0, 3.0, 10.0):
                report(score_oof(df, mode, alpha), f"{mode} alpha={alpha:g}")
        return

    print("\nEvaluasi (GroupKFold per hotel, 5 lipatan):")
    oof = score_oof(df, MODE, ALPHA)
    metrics = report(oof, f"{MODE} alpha={ALPHA:g}")

    # benchmark dari skor out-of-fold
    recs = [
        {"bintang": r.stars, "aspek": a, "skor": s}
        for r in oof.itertuples()
        for a, s in r.aspects.items()
    ]
    g = pd.DataFrame(recs).groupby(["bintang", "aspek"])["skor"].agg(["mean", "std", "count"]).reset_index()
    g = g[g["count"] >= MIN_N_BENCH]
    order = {a: i for i, a in enumerate(am.ASPECTS)}
    g = g.sort_values(["bintang", "aspek"], key=lambda c: c.map(order) if c.name == "aspek" else c)
    bench = pd.DataFrame(
        {"bintang": g["bintang"], "aspek": g["aspek"], "rata_rata": g["mean"].round(3),
         "std": g["std"].round(3), "n": g["count"]}
    )
    bench.to_csv(os.path.join(HERE, "benchmark.csv"), index=False)
    print("\nbenchmark.csv:\n" + bench.to_string(index=False))

    # model final (semua data) + cek konsistensi dengan sklearn
    texts, y, w = build_samples(df, MODE)
    model, vec, ridge = fit(texts, y, w, ALPHA)
    sample = texts[:80]
    ref = np.clip(ridge.predict(vec.transform(sample)), 1, 5)
    mine = np.array([am.ml_predict(t, {**model, "intercept": float(ridge.intercept_)}) for t in sample])
    print(f"\nSelisih maks inferensi murni-Python vs sklearn: {np.abs(ref - mine).max():.2e} "
          "(kecil karena pembulatan bobot ke 6 desimal)")

    model["meta"] = {
        "mode": MODE, "alpha": ALPHA, "win_weight": WIN_WEIGHT, "n_reviews": int(len(df)),
        "n_hotels": int(df["place_id"].nunique()), "n_features": len(model["vocab"]),
        "cv": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in metrics.items()},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(HERE, "aspect_model.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, separators=(",", ":"))
    print(f"aspect_model.json: {len(model['vocab'])} fitur, "
          f"{os.path.getsize(os.path.join(HERE, 'aspect_model.json')) / 1024:.0f} KB\n")
    sanity(model, verbose=True)


if __name__ == "__main__":
    sys.exit(main())
