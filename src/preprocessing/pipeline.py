"""Comment preprocessing pipeline."""
import pandas as pd

from src.preprocessing.text_cleaning import clean_text, count_emojis, detect_language


def preprocess_comments(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns (processed df, stats). Only duplicate comment_ids are dropped: identical wording on a
    post is a genuine repeated customer need, not noise."""
    stats = {"rows_in": len(df)}
    df = df.copy()
    if "comment_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["comment_id"])
        stats["duplicate_ids_removed"] = before - len(df)
    df["emoji_count"] = df["comment_text"].map(count_emojis)
    df["comment_clean"] = df["comment_text"].map(clean_text)
    empty = df["comment_clean"].eq("")
    stats["empty_after_cleaning_removed"] = int(empty.sum())
    df = df[~empty].copy()
    df["language_detected"] = df["comment_clean"].map(detect_language)
    if "language_type" in df.columns and df["language_type"].notna().any():
        known = df[df["language_type"].notna()]
        stats["language_agreement_with_label"] = round(float((known["language_type"] == known["language_detected"]).mean()), 4)
    stats["rows_out"] = len(df)
    return df.reset_index(drop=True), stats
