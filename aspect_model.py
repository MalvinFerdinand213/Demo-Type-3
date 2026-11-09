"""Model skoring aspek ringan (tanpa sklearn saat dijalankan).

Alur:
  1. teks dinormalisasi lalu dipecah jadi klausa
  2. tiap klausa dicari kata kunci aspek -> diambil jendela kata di sekitarnya
  3. tiap jendela diberi skor 1-5 = campuran
       - model TF-IDF + Ridge (bobot dibaca dari aspect_model.json, dilatih oleh train_model.py)
       - kamus kata sentimen (positif/negatif, negasi, penguat)
  4. skor aspek = rata-rata skor jendela; None kalau aspek tidak disebut

Hanya butuh pustaka standar Python.
"""
import json
import math
import os
import re

ASPECTS = [
    "Lokasi",
    "Kebersihan",
    "Pelayanan",
    "Kamar dan fasilitas",
    "Harga",
    "Makanan dan minuman",
]

ASPECT_KEYWORDS = {
    "Lokasi": {
        "lokasi", "strategis", "akses", "dijangkau", "dekat", "jauh", "gang", "stasiun", "bandara",
        "halte", "mall", "mal", "lingkungan", "sekitar", "macet", "location", "near", "nearby",
        "close", "walking", "area", "daerah", "alamat", "terminal", "transjakarta", "tol", "pusat",
    },
    "Kebersihan": {
        "bersih", "kotor", "jorok", "bau", "apek", "lembab", "sampah", "debu", "berdebu", "noda",
        "kecoa", "kecoak", "nyamuk", "kutu", "semut", "tikus", "wangi", "harum", "higienis", "kumuh",
        "clean", "dirty", "smell", "smelly", "odor", "dust", "dusty", "hygiene", "cleanliness",
        "bedbug", "jamur", "moldy", "mold", "sprei", "seprai", "sprai", "handuk", "kebersihan",
        "pengap",
    },
    "Pelayanan": {
        "staf", "pelayanan", "layanan", "ramah", "resepsionis", "petugas", "karyawan", "pegawai",
        "pemilik", "owner", "service", "pelayan", "satpam", "penjaga", "checkin", "checkout",
        "respon", "responsif", "sopan", "jutek", "judes", "cuek", "mas", "mbak", "admin",
        "friendly", "helpful", "rude", "manager", "pengelola", "melayani", "keramahan",
    },
    "Kamar dan fasilitas": {
        "kamar", "room", "kasur", "bed", "ranjang", "bantal", "ac", "kipas", "kamarmandi", "mandi",
        "toilet", "wc", "shower", "airpanas", "tv", "televisi", "wifi", "internet", "sinyal",
        "fasilitas", "facility", "facilities", "kolam", "pool", "parkir", "lift", "tangga",
        "balkon", "lemari", "meja", "colokan", "listrik", "lampu", "pintu", "jendela", "ruangan",
        "bangunan", "gedung", "lantai",
    },
    "Harga": {
        "harga", "murah", "mahal", "tarif", "biaya", "bayar", "rp", "ribu", "rb", "worth", "value",
        "terjangkau", "ekonomis", "budget", "promo", "diskon", "price", "cheap", "expensive",
        "affordable", "overprice", "overpriced", "semalam", "pricey",
    },
    "Makanan dan minuman": {
        "makan", "makanan", "sarapan", "breakfast", "minum", "minuman", "kopi", "teh", "menu",
        "restoran", "restaurant", "warung", "kantin", "resto", "food", "drink", "masakan", "nasi",
        "lauk", "mie", "cafe", "kafe", "snack", "roti", "telur", "kuliner", "dinner", "lunch",
    },
}

POS = {
    "bagus", "baik", "bersih", "rapi", "nyaman", "enak", "ramah", "murah", "terjangkau", "strategis",
    "dekat", "mudah", "lengkap", "luas", "besar", "wangi", "harum", "sejuk", "tenang", "aman",
    "cepat", "sigap", "membantu", "puas", "mantap", "oke", "ok", "okay", "recommended", "worth",
    "sesuai", "memuaskan", "sopan", "hangat", "lancar", "nyenyak", "empuk", "terawat", "baru",
    "modern", "cantik", "indah", "lezat", "nikmat", "fresh", "segar", "terbaik", "keren", "asik",
    "asyik", "suka", "senang", "cocok", "juara", "top", "sempurna", "istimewa", "helpful",
    "friendly", "clean", "comfortable", "cozy", "cheap", "affordable", "spacious", "good", "great",
    "nice", "excellent", "amazing", "tidy", "convenient", "delicious", "tasty", "perfect", "best",
    "lovely", "quiet", "safe", "fast", "responsif", "higienis", "ekonomis", "terang", "praktis",
    "profesional", "betah", "menyenangkan", "kinclong", "sehat", "bersahabat", "adem", "dingin",
}
NEG = {
    "kotor", "jorok", "bau", "apek", "lembab", "bocor", "rusak", "mati", "jelek", "buruk", "mahal",
    "jauh", "sempit", "berisik", "bising", "ribet", "susah", "sulit", "lambat", "jutek", "cuek",
    "judes", "ketus", "kasar", "kecewa", "parah", "zonk", "kecoa", "kecoak", "nyamuk", "semut",
    "tikus", "kutu", "kumuh", "lusuh", "usang", "dirty", "smelly", "noisy", "rude", "expensive",
    "small", "broken", "bad", "terrible", "awful", "worst", "horrible", "poor", "mold", "moldy",
    "disappointing", "disappointed", "uncomfortable", "slow", "unfriendly", "overprice",
    "overpriced", "menyesal", "sampah", "debu", "berdebu", "noda", "pengap", "gelap", "kecil",
    "bobrok", "seram", "angker", "panas", "pricey", "macet", "mengecewakan",
}
NEGATORS = {"tidak", "bukan", "belum", "kurang", "tanpa", "no", "not", "never"}
INTENSIFIERS = {"banget", "sekali", "sangat", "amat", "very", "really", "super", "extremely", "terlalu", "paling"}
DAMPENERS = {"cukup", "lumayan", "agak", "sedikit", "quite", "fairly"}

SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak",
    "engga": "tidak", "gk": "tidak", "tdk": "tidak", "tak": "tidak", "bgt": "banget",
    "bngt": "banget", "yg": "yang", "dgn": "dengan", "krn": "karena", "utk": "untuk",
    "sdh": "sudah", "blm": "belum", "jg": "juga", "bgus": "bagus", "kmr": "kamar", "hrg": "harga",
    "mantab": "mantap", "mantul": "mantap", "rekomen": "recommended", "recomended": "recommended",
    "recommend": "recommended", "rekomendasi": "recommended", "direkomendasikan": "recommended",
    "staff": "staf", "reseptionis": "resepsionis", "receptionist": "resepsionis",
    "reception": "resepsionis", "okee": "oke", "kamr": "kamar", "bersihh": "bersih",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
FEATURE_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # sama dengan default TfidfVectorizer
CLAUSE_SPLIT = re.compile(
    r"[.!?;\n,:()/]+|\b(?:tapi|namun|tetapi|sayangnya|but|however|although|cuma|hanya saja)\b"
)
PRE, POST = 3, 4  # jendela: 3 kata sebelum dan 4 kata sesudah kata kunci aspek
W_LEX, W_ML, W_ML_NO_LEX, NEUTRAL = 0.65, 0.35, 0.6, 3.0


# ------------------------- teks -------------------------
def normalize(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"<br\s*/?>", ". ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)  # mantaaap -> mantap
    t = re.sub(r"check[\s-]?in", "checkin", t)
    t = re.sub(r"check[\s-]?out", "checkout", t)
    t = re.sub(r"air\s+panas", "airpanas", t)
    t = re.sub(r"kamar\s+mandi", "kamarmandi", t)
    t = re.sub(r"wi-?\s?fi", "wifi", t)
    return re.sub(r"[a-z0-9]+", lambda m: SLANG.get(m.group(0), m.group(0)), t)


def _match(token: str, vocab) -> bool:
    return token in vocab or (token.endswith("nya") and len(token) > 5 and token[:-3] in vocab)


def extract_windows(text: str) -> dict:
    """{aspek: [jendela (list token), ...]} untuk seluruh klausa pada teks."""
    out = {a: [] for a in ASPECTS}
    for clause in CLAUSE_SPLIT.split(normalize(text)):
        toks = TOKEN_RE.findall(clause)
        if not toks:
            continue
        for aspect, kw in ASPECT_KEYWORDS.items():
            hits = [i for i, t in enumerate(toks) if _match(t, kw)]
            if not hits:
                continue
            merged = []
            for i in hits:
                a, b = max(0, i - PRE), min(len(toks), i + POST + 1)
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            out[aspect].extend(toks[a:b] for a, b in merged)
    return out


# ------------------------- skor kamus -------------------------
def lex_score(tokens: list):
    """Skor 1-5 dari kamus sentimen, atau None kalau tidak ada kata sentimen."""
    s, hits = 0.0, 0
    for j, t in enumerate(tokens):
        base = t[:-3] if t.endswith("nya") and len(t) > 5 else t
        pol = 1.0 if base in POS else -1.0 if base in NEG else 0.0
        if pol == 0.0:
            continue
        hits += 1
        prev = tokens[max(0, j - 2):j]
        near = tokens[max(0, j - 1):j] + tokens[j + 1:j + 2]
        if any(p in NEGATORS for p in prev):
            pol *= -0.8
        if any(n in INTENSIFIERS for n in near):
            pol *= 1.5
        elif any(n in DAMPENERS for n in prev):
            pol *= 0.6
        s += pol
    if hits == 0:
        return None
    return 3.0 + 2.0 * math.tanh(s / 1.5)


# ------------------------- skor ML -------------------------
def load_model(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ml_predict(window_text: str, model: dict) -> float:
    """TF-IDF (unigram+bigram, sublinear tf, l2) lalu regresi linear. Skala 1-5."""
    toks = FEATURE_TOKEN_RE.findall(window_text)
    grams = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    counts = {}
    for g in grams:
        if g in model["vocab"]:
            counts[g] = counts.get(g, 0) + 1
    if not counts:
        return min(5.0, max(1.0, model["intercept"]))
    num, sq = 0.0, 0.0
    for g, c in counts.items():
        idf, coef = model["vocab"][g]
        w = (1.0 + math.log(c)) * idf
        num += w * coef
        sq += w * w
    return min(5.0, max(1.0, model["intercept"] + num / math.sqrt(sq)))


def score_window(tokens: list, model: dict) -> float:
    ml = ml_predict(" ".join(tokens), model)
    lex = lex_score(tokens)
    if lex is None:
        s = W_ML_NO_LEX * ml + (1 - W_ML_NO_LEX) * NEUTRAL
    else:
        s = W_LEX * lex + W_ML * ml
    return min(5.0, max(1.0, s))


# ------------------------- API -------------------------
def analyze(text: str, model: dict) -> dict:
    """{aspek: {"score": float | None, "evidence": [str, ...]}}"""
    result = {}
    for aspect, windows in extract_windows(text).items():
        if not windows:
            result[aspect] = {"score": None, "evidence": []}
            continue
        scores = [score_window(w, model) for w in windows]
        result[aspect] = {
            "score": sum(scores) / len(scores),
            "evidence": [" ".join(w) for w in windows][:3],
        }
    return result


def score_text(text: str, model: dict) -> dict:
    return {a: r["score"] for a, r in analyze(text, model).items()}


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aspect_model.json")
