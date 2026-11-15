"""Post performance features. Every formula is NaN-safe: zero or missing denominators give NaN, never inf."""
import numpy as np
import pandas as pd

MIN_PEERS = 10  # minimum posts in a peer group before we trust its median


def safe_divide(num, den) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def add_post_features(posts: pd.DataFrame) -> pd.DataFrame:
    df = posts.copy()
    for c in ["likes", "comments_count", "shares", "views", "days_since_post"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # total_engagement needs all three parts; if any is missing it stays NaN (we do not invent values)
    df["total_engagement"] = df[["likes", "comments_count", "shares"]].sum(axis=1, min_count=3)
    # engagement_rate = (likes + comments + shares) / views
    df["engagement_rate"] = safe_divide(df["total_engagement"], df["views"])
    # like_to_comment_ratio = likes / comments  (NaN when comments == 0)
    df["like_to_comment_ratio"] = safe_divide(df["likes"], df["comments_count"])
    # share_rate = shares / views
    df["share_rate"] = safe_divide(df["shares"], df["views"])
    # comment_rate = comments / views
    df["comment_rate"] = safe_divide(df["comments_count"], df["views"])
    # views_per_engagement = views / total_engagement (higher = less engaging)
    df["views_per_engagement"] = safe_divide(df["views"], df["total_engagement"])
    # comments_per_day = comments / max(days_since_post, 1)
    df["comments_per_day"] = safe_divide(df["comments_count"], df["days_since_post"].clip(lower=1))
    return _relative_engagement(df)


def _relative_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """engagement_rate divided by the median of comparable posts (same content_type + platform).
    Falls back to content_type, then to all posts, when a peer group has fewer than MIN_PEERS posts."""
    df["relative_engagement"] = np.nan
    df["peer_group_level"] = ""
    levels = [("content_type+platform", ["content_type", "platform"]), ("content_type", ["content_type"]), ("all_posts", [])]
    for name, keys in levels:
        if keys and not all(k in df.columns and df[k].notna().any() for k in keys):
            continue
        if keys:
            grp = df.groupby(keys)["engagement_rate"]
            med, cnt = grp.transform("median"), grp.transform("count")
        else:
            med = pd.Series(df["engagement_rate"].median(), index=df.index)
            cnt = pd.Series(df["engagement_rate"].count(), index=df.index)
        todo = df["peer_group_level"].eq("") & df["engagement_rate"].notna() & (cnt >= MIN_PEERS) & (med > 0)
        df.loc[todo, "relative_engagement"] = df.loc[todo, "engagement_rate"] / med[todo]
        df.loc[todo, "peer_group_level"] = name
    return df
