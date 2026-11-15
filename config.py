"""Central configuration. Every other module imports paths and constants from here."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = DATA_DIR / "app.db"

RANDOM_SEED = 42
CURRENCY = "PKR"

# Required columns per dataset (used later by the validation module)
POST_REQUIRED = [
    "post_id", "brand", "platform", "campaign", "campaign_type", "content_type",
    "product_category", "likes", "comments_count", "shares", "views",
    "days_since_post", "post_date", "source_url", "data_source_type",
]
COMMENT_REQUIRED = [
    "comment_id", "post_id", "comment_text", "language_type", "topic",
    "purchase_intent", "sentiment", "comment_date", "data_source_type",
]

# Provenance label stamped on every generated row
SYNTHETIC_LABEL = "SYNTHETIC_DEMONSTRATION_DATA"
