"""Comment-level need flags and post-level aggregates."""
import pandas as pd

from src.feature_engineering.lexicons import NEEDS, PATTERNS, PRIORITY


def add_need_flags(comments: pd.DataFrame, text_col: str = "comment_clean") -> pd.DataFrame:
    """Multi-label flags (a comment can ask about price AND delivery) plus one primary_need."""
    df = comments.copy()
    for need in NEEDS:
        df[f"need_{need}"] = df[text_col].fillna("").map(lambda t, p=PATTERNS[need]: int(bool(p.search(t))))

    def primary(row):
        for need in PRIORITY:
            if row[f"need_{need}"]:
                return need
        return "general"

    df["primary_need"] = df.apply(primary, axis=1) if len(df) else pd.Series(dtype=str)
    return df


def aggregate_to_posts(comments: pd.DataFrame) -> pd.DataFrame:
    """Share of stored comments per need for each post. Rates (not counts) so posts are comparable."""
    g = comments.groupby("post_id")
    out = pd.DataFrame({"n_comments_analysed": g.size()})
    for need in NEEDS:
        out[f"rate_{need}"] = g[f"need_{need}"].mean()
    if "emoji_count" in comments:
        out["avg_emoji_count"] = g["emoji_count"].mean()
    return out.reset_index()
