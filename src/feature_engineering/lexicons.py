"""Keyword patterns for customer needs (English, Roman Urdu, Urdu).
These are transparent weak signals. Phase 3 adds embeddings to go beyond keywords."""
import re

_LATIN = {
    "delivery": r"deliver\w*|parcel|shipping|courier|delay\w*|late|abhi tak|still waiting|dispatch\w*",
    "discount": r"discount\w*|sale|offer\w*|deal|off|kam price|sasta|bulk|voucher|promo\w*",
    "purchase": r"want to buy|where can i (?:order|buy)|order karna|buy now|inbox|dm me|send details|details send|i will order",
    "price": r"price\w*|rate|cost\w*|how much|kitni hai|kitne ka|qeemat|pkr|rs\.? ?\d+",
    "quality": r"quality|fabric|stitch\w*|material|fade\w*|colou?r nikalta|kapra|soft|cotton",
    "availability": r"available|availability|in stock|stock|restock\w*|size|sold out|store",
}
_URDU = {
    "delivery": "ڈیلیوری|ابھی تک|پارسل",
    "discount": "ڈسکاؤنٹ|سیل|آفر",
    "purchase": "آرڈر کرنا|خریدنا",
    "price": "قیمت|ریٹ",
    "quality": "کپڑا|کوالٹی|سلائی",
    "availability": "دستیاب|سائز|اسٹاک",
}
NEEDS = list(_LATIN)
# When a comment matches several needs, the first one in this list becomes its primary_need
PRIORITY = ["delivery", "discount", "purchase", "price", "quality", "availability"]
PATTERNS = {k: re.compile(rf"\b(?:{_LATIN[k]})\b|{_URDU[k]}") for k in NEEDS}
