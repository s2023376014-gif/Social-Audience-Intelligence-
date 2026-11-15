"""Reusable text cleaning. Keeps prices, numbers, cities and Urdu script; removes noise only."""
import re

from src.preprocessing.roman_urdu import normalize_roman_urdu, roman_urdu_score

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\uFE0F\u200d]")
_URDU = re.compile(r"[\u0600-\u06FF]")
_URDU_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_NUM_COMMA = re.compile(r"(?<=\d),(?=\d{3})")
_REPEAT = re.compile(r"(.)\1{2,}")
_KEEP = re.compile(r"[^a-z0-9\u0600-\u06FF?.'\s]")


def count_emojis(text) -> int:
    return len(_EMOJI.findall(text)) if isinstance(text, str) else 0


def clean_text(text) -> str:
    """Lowercase, drop URLs/mentions/emojis, keep numbers (5,990 -> 5990), collapse 'pleaseeee'."""
    if not isinstance(text, str):
        return ""
    s = _URL.sub(" ", text)
    s = _MENTION.sub(" ", s)
    s = _EMOJI.sub(" ", s)
    s = s.replace("#", " ")
    s = s.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")  # Arabic yeh/kaf -> Urdu forms
    s = _URDU_DIACRITICS.sub("", s)
    s = s.lower()
    s = _NUM_COMMA.sub("", s)
    s = _REPEAT.sub(r"\1\1", s)
    s = _KEEP.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_roman_urdu(s)


def detect_language(clean: str) -> str:
    """Script check for Urdu, marker-word check for Roman Urdu, otherwise English. 'Unknown' if no letters."""
    letters = [c for c in clean if c.isalpha()] if clean else []
    if not letters:
        return "Unknown"
    if sum(1 for c in letters if _URDU.match(c)) / len(letters) > 0.3:
        return "Urdu"
    if roman_urdu_score(clean) >= 0.2:
        return "Roman Urdu"
    return "English"
