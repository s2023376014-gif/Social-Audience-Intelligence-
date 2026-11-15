"""One reusable entry point for the whole clustering stage (used by scripts and the dashboard)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import config
from src.clustering import clusterers as C
from src.clustering.profiling import profile_clusters


def cluster_comments(comments: pd.DataFrame, X: np.ndarray, k_max: int = 10) -> dict:
    n = len(X)
    if n < 30:
        raise ValueError("Need at least 30 comments to cluster meaningfully.")
    sweep = C.kmeans_sweep(X, range(2, min(k_max, n // 10) + 1))
    k = C.pick_k(sweep)
    db = C.dbscan_sweep(X)
    db_ok = db.dropna(subset=["silhouette"])
    best_db = db_ok.sort_values("silhouette", ascending=False).iloc[0] if len(db_ok) else db.iloc[0]
    km = KMeans(k, n_init=10, random_state=config.RANDOM_SEED).fit(X)
    runs = {"K-Means": (km.labels_, C.kmeans_stability(X, k)),
            "DBSCAN": (C.run_dbscan(X, best_db["eps"], int(best_db["min_samples"])), np.nan),
            "Agglomerative (ward)": (C.run_agglomerative(X, k), np.nan)}
    summary = pd.DataFrame([{"algorithm": a, **C.internal_metrics(X, lab), "stability_ari": st} for a, (lab, st) in runs.items()])
    final = C.select_final(summary)
    labels = runs[final][0]
    profile = profile_clusters(comments, labels, X)
    return {"sweep": sweep, "dbscan": db, "summary": summary, "final": final, "k": k,
            "labels": labels, "profile": profile, "kmeans": km}
