"""K-Means, DBSCAN and Agglomerative clustering with internal-metric comparison."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.neighbors import NearestCentroid, NearestNeighbors

import config

SIL_SAMPLE = 5000       # silhouette is O(n^2): evaluate on a sample
AGG_SAMPLE = 3000       # Agglomerative needs an n x n matrix: 17.9k^2 floats ~ 2.5 GB, too much for 8 GB RAM


def internal_metrics(X, labels, rng_seed=config.RANDOM_SEED) -> dict:
    """Metrics ignore DBSCAN noise (-1). Returns NaNs when fewer than 2 clusters exist."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    out = {"n_clusters": n_clusters, "noise_pct": round(float((~mask).mean() * 100), 2),
           "largest_cluster_pct": round(float(pd.Series(labels[mask]).value_counts(normalize=True).max() * 100), 2) if mask.any() else np.nan}
    if n_clusters < 2 or mask.sum() < 10:
        return {**out, "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    Xm, lm = X[mask], labels[mask]
    return {**out,
            "silhouette": round(float(silhouette_score(Xm, lm, sample_size=min(SIL_SAMPLE, len(lm)), random_state=rng_seed)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(Xm, lm)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(Xm, lm)), 1)}


def kmeans_sweep(X, k_range=range(2, 11)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(k, n_init=5, random_state=config.RANDOM_SEED).fit(X)
        rows.append({"k": k, "inertia": round(float(km.inertia_), 2), **internal_metrics(X, km.labels_)})
    return pd.DataFrame(rows)


def kmeans_stability(X, k, seeds=(0, 1, 2, 3, 4)) -> float:
    """Mean pairwise Adjusted Rand Index across random restarts (1.0 = identical every time)."""
    labs = [KMeans(k, n_init=1, random_state=s).fit_predict(X) for s in seeds]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return round(float(np.mean(aris)), 4)


def pick_k(sweep: pd.DataFrame) -> int:
    """Best silhouette; ties within 0.01 go to the smaller k (simpler = more interpretable)."""
    s = sweep.dropna(subset=["silhouette"])
    best = s["silhouette"].max()
    return int(s[s["silhouette"] >= best - 0.01]["k"].min())


def dbscan_sweep(X, min_samples=10, percentiles=(80, 90, 95)) -> pd.DataFrame:
    """eps candidates come from the k-distance curve (distance to the min_samples-th neighbour)."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.choice(len(X), min(len(X), 8000), replace=False)
    kd = NearestNeighbors(n_neighbors=min_samples).fit(X[idx]).kneighbors(X[idx])[0][:, -1]
    rows = []
    for p in percentiles:
        eps = float(np.percentile(kd, p))
        lab = DBSCAN(eps=max(eps, 1e-3), min_samples=min_samples).fit_predict(X)
        rows.append({"eps": round(eps, 4), "min_samples": min_samples, "percentile": p, **internal_metrics(X, lab)})
    return pd.DataFrame(rows)


def run_dbscan(X, eps, min_samples=10):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def run_agglomerative(X, k, linkage="ward"):
    """Fit on a sample, then assign everyone else to the nearest cluster centroid (documented approximation)."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.choice(len(X), min(len(X), AGG_SAMPLE), replace=False)
    lab_s = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X[idx])
    nc = NearestCentroid().fit(X[idx], lab_s)
    return nc.predict(X)


def select_final(summary: pd.DataFrame) -> str:
    """Rule-based choice. An algorithm is eligible if it has 3-12 clusters, <20% noise and no cluster >70%.
    Among eligible, rank by silhouette (higher), Davies-Bouldin (lower) and stability (higher)."""
    ok = summary[(summary["n_clusters"].between(3, 12)) & (summary["noise_pct"] < 20) & (summary["largest_cluster_pct"] < 70)].copy()
    if ok.empty:
        return "K-Means"  # most predictable fallback
    ok["rank"] = (ok["silhouette"].rank(ascending=False) + ok["davies_bouldin"].rank() +
                  ok["stability_ari"].fillna(0).rank(ascending=False))
    winner = ok.sort_values("rank").iloc[0]
    km = ok[ok["algorithm"] == "K-Means"]
    # Business/efficiency tie-break: K-Means scales, is repeatable and can label NEW comments directly
    # (Agglomerative cannot predict, DBSCAN has no cluster centres). Only override it for a clear quality gain.
    if winner["algorithm"] != "K-Means" and not km.empty and winner["silhouette"] - km.iloc[0]["silhouette"] < 0.05:
        return "K-Means"
    return winner["algorithm"]
