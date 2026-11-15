"""Campaign / content / brand comparisons. Observational data: differences are associations, not causes."""
import numpy as np
import pandas as pd
from scipy import stats

from src.feature_engineering.lexicons import NEEDS

METRICS = ["engagement_rate", "comment_rate", "share_rate", "views"]


def group_performance(posts: pd.DataFrame, by: str, min_posts: int = 5) -> pd.DataFrame:
    if by not in posts.columns:
        return pd.DataFrame()
    g = posts.groupby(by)
    out = pd.DataFrame({"n_posts": g.size()})
    for m in METRICS:
        if m in posts.columns:
            out[f"median_{m}"] = g[m].median()
    if "comments_count" in posts.columns:
        out["total_comments"] = g["comments_count"].sum()
    out = out[out["n_posts"] >= min_posts]
    return out.sort_values("median_engagement_rate", ascending=False).reset_index() if "median_engagement_rate" in out else out.reset_index()


def difference_test(posts: pd.DataFrame, by: str, metric: str = "engagement_rate") -> dict:
    """Kruskal-Wallis across groups (non-parametric: engagement is skewed). A small p-value says the groups
    differ in this sample; it does NOT say the format caused the difference."""
    groups = [g[metric].dropna().to_numpy() for _, g in posts.groupby(by) if g[metric].notna().sum() >= 5]
    if len(groups) < 2:
        return {"p_value": np.nan, "n_groups": len(groups)}
    stat, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return {"p_value": float(p), "n_groups": len(groups), "epsilon_squared": float((stat - len(groups) + 1) / (n - len(groups)))}


def need_rates(comments: pd.DataFrame, posts: pd.DataFrame, by: str) -> pd.DataFrame:
    """Share of comments mentioning each need, per campaign / brand / content type."""
    if by not in posts.columns:
        return pd.DataFrame()
    m = comments.merge(posts[["post_id", by]], on="post_id", how="left")
    flags = [f"need_{n}" for n in NEEDS if f"need_{n}" in m.columns]
    out = m.groupby(by)[flags].mean().round(3)
    out.insert(0, "n_comments", m.groupby(by).size())
    return out.reset_index()


def segment_by_group(comments: pd.DataFrame, posts: pd.DataFrame, by: str) -> pd.DataFrame:
    if by not in posts.columns or "segment" not in comments.columns:
        return pd.DataFrame()
    m = comments.merge(posts[["post_id", by]], on="post_id", how="left")
    return pd.crosstab(m[by], m["segment"], normalize="index").round(3).reset_index()


def monthly_trend(posts: pd.DataFrame) -> pd.DataFrame:
    d = posts.copy()
    d["month"] = pd.to_datetime(d["post_date"], errors="coerce").dt.to_period("M").astype(str)
    return d[d["month"] != "NaT"].groupby("month").agg(posts=("post_id", "size"), median_engagement_rate=("engagement_rate", "median")).reset_index()
