"""Turn raw cluster ids into business segments using actual cluster content (never arbitrary names)."""
import numpy as np
import pandas as pd

from src.feature_engineering.lexicons import NEEDS

SEGMENT_NAMES = {
    "price": "Price-Sensitive Customers", "discount": "Discount-Focused Customers",
    "delivery": "Delivery-Conscious Customers", "availability": "Product Availability Seekers",
    "quality": "Quality-Conscious Customers", "purchase": "High Purchase-Intent Customers",
    "general": "General Engagement Audience",
}
PURITY = 0.5  # a need must appear in >= 50% of a cluster's comments to name it


def profile_clusters(comments: pd.DataFrame, labels: np.ndarray, X: np.ndarray, n_reps: int = 5) -> pd.DataFrame:
    df = comments.copy()
    df["cluster"] = labels
    overall = {n: df[f"need_{n}"].mean() for n in NEEDS}
    rows = []
    for c, g in df.groupby("cluster"):
        rates = {n: g[f"need_{n}"].mean() for n in NEEDS}
        rates["general"] = float((g[[f"need_{n}" for n in NEEDS]].sum(axis=1) == 0).mean())
        top = max(rates, key=rates.get)
        if c == -1:
            name = "Unclassified (DBSCAN noise)"
        elif rates[top] >= PURITY:
            name = SEGMENT_NAMES[top]
        else:
            name = "Mixed Behaviour"
        lift = {n: (rates[n] / overall[n] if overall[n] > 0 else np.nan) for n in NEEDS}
        centroid = X[(labels == c)].mean(axis=0)
        d = np.linalg.norm(X[(labels == c)] - centroid, axis=1)
        reps = g.iloc[np.argsort(d)]["comment_text"].drop_duplicates().head(n_reps).tolist()
        rows.append({"cluster": c, "segment": name, "size": len(g), "share_pct": round(100 * len(g) / len(df), 2),
                     "dominant_need": top, "dominant_need_share": round(rates[top], 3),
                     **{f"share_{n}": round(rates[n], 3) for n in NEEDS},
                     "top_lift": max(lift, key=lambda k: -1 if np.isnan(lift[k]) else lift[k]),
                     "languages": g["language_detected"].value_counts(normalize=True).round(2).to_dict() if "language_detected" in g else {},
                     "representative_comments": " | ".join(reps)})
    out = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    # make duplicate segment names unique so the dashboard can tell clusters apart
    dup = out["segment"].duplicated(keep=False)
    out.loc[dup, "segment"] = out.loc[dup].apply(lambda r: f"{r['segment']} (cluster {r['cluster']})", axis=1)
    return out


def explain_comment(row: pd.Series, profile: pd.DataFrame, X_row: np.ndarray, centroid: np.ndarray) -> str:
    """Plain-language reason a comment sits in its cluster."""
    p = profile[profile["cluster"] == row["cluster"]].iloc[0]
    flags = [n for n in NEEDS if row.get(f"need_{n}", 0) == 1] or ["no specific need keyword"]
    dist = float(np.linalg.norm(X_row - centroid))
    return (f"Assigned to '{p['segment']}' because the comment mentions {', '.join(flags)}; "
            f"{p['dominant_need_share']:.0%} of this cluster shares '{p['dominant_need']}' behaviour. "
            f"Distance to cluster centre: {dist:.2f}.")


def explain_comment_simple(row: pd.Series, profile: pd.DataFrame) -> str:
    """Dashboard-friendly reason a comment belongs to its segment (no feature matrix needed)."""
    p = profile[profile["cluster"] == row["cluster"]].iloc[0]
    flags = [n for n in NEEDS if row.get(f"need_{n}", 0) == 1]
    seen = f"mentions {', '.join(flags)}" if flags else "contains no specific need keyword"
    return (f"This comment {seen}. Its cluster is named '{p['segment']}' because {p['dominant_need_share']:.0%} of the "
            f"{int(p['size']):,} comments in it share '{p['dominant_need']}' behaviour.")
