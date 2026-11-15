"""Roman Urdu spelling normalisation and lightweight language identification."""
import re

# Variants -> canonical spelling. Only words that are NOT common English words.
VARIANTS = {
    "kitny": "kitni", "kitnee": "kitni", "kitna": "kitni",
    "kia": "kya", "kiya": "kya", "kea": "kya",
    "hy": "hai", "hae": "hai", "hain": "hai",
    "nhi": "nahi", "nahin": "nahi", "nai": "nahi", "nae": "nahi",
    "bohot": "bohat", "buhat": "bohat", "bht": "bohat", "bhot": "bohat",
    "ziyada": "zyada", "zyadah": "zyada", "zyda": "zyada",
    "mei": "mein", "mn": "mein",
    "kesi": "kaisi", "kaisee": "kaisi", "kesa": "kaisa",
    "btayen": "batayen", "bataen": "batayen",
    "krna": "karna", "karnaa": "karna",
    "acha": "achi", "achha": "achi",
    "pls": "please", "plz": "please", "plss": "please",
}

# Unambiguous Roman Urdu marker words (used for language identification)
MARKERS = {
    "hai", "kya", "kitni", "kitne", "mein", "nahi", "bohat", "zyada", "kab", "dein",
    "yeh", "mujhe", "abhi", "tak", "achi", "kaisa", "kaisi", "batayen", "milega", "hoga",
    "hogi", "kapra", "mashallah", "zabardast", "khoobsurat", "aya", "gaya", "rahi", "raha",
    "kuch", "wala", "wali", "nikalta", "jayega", "karna", "lag",
}
# Appear in Roman Urdu but also in English text or names, so they count for less
AMBIGUOUS = {"ka", "ki", "ko", "se", "kar", "ye", "kam", "din", "aur"}

_TOKEN = re.compile(r"[a-z']+")


def normalize_roman_urdu(text: str) -> str:
    return " ".join(VARIANTS.get(tok, tok) for tok in text.split())


def roman_urdu_score(text: str) -> float:
    toks = _TOKEN.findall(text.lower())
    if not toks:
        return 0.0
    hits = sum(1.0 for t in toks if t in MARKERS) + sum(0.5 for t in toks if t in AMBIGUOUS)
    return hits / len(toks)
