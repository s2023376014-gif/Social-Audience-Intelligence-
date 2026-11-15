"""End-to-end pipeline used by the dashboard and scripts/run_pipeline.py."""
import numpy as np
import pandas as pd

import config
from src.anomaly_detection.detector import detect_anomalies
from src.campaign_analytics.analytics import group_performance, need_rates
from src.clustering.runner import cluster_comments
from src.data_ingestion.validation import validate
from src.database.db import save_tables
from src.feature_engineering.behaviour_features import add_need_flags, aggregate_to_posts
from src.feature_engineering.embeddings import embed_texts
from src.feature_engineering.feature_space import build_comment_features
from src.feature_engineering.post_features import add_post_features
from src.feature_engineering.semantic_needs import semantic_need_assignment
from src.preprocessing.pipeline import preprocess_comments
from src.recommendation_engine.rules import generate_recommendations


def run_pipeline(posts_raw: pd.DataFrame, comments_raw: pd.DataFrame, backend: str = "auto", save: bool = True, progress=lambda m: None) -> dict:
    posts, rp = validate(posts_raw, "posts")
    comments, rc = validate(comments_raw, "comments")
    errors = rp.errors + rc.errors
    if errors:
        raise ValueError("; ".join(errors))
    progress("Cleaning comments")
    comments, pstats = preprocess_comments(comments)
    comments = add_need_flags(comments)
    progress("Computing post features")
    posts = add_post_features(posts)
    posts = posts.merge(aggregate_to_posts(comments), on="post_id", how="left")
    progress("Creating embeddings (cached after first run)")
    E, used = embed_texts(comments["comment_clean"].tolist(), backend=backend)
    comments = semantic_need_assignment(comments, E)
    X, weights = build_comment_features(comments, posts, E)
    progress("Clustering")
    cl = cluster_comments(comments, X)
    comments["cluster"] = cl["labels"]
    comments = comments.merge(cl["profile"][["cluster", "segment"]], on="cluster", how="left")
    progress("Detecting anomalies")
    anomalies, _ = detect_anomalies(posts)
    progress("Generating recommendations")
    recs = generate_recommendations(posts, comments, anomalies)
    res = {"posts": anomalies, "comments": comments, "profile": cl["profile"], "comparison": cl["summary"], "kmeans_sweep": cl["sweep"],
           "recommendations": recs, "validation": {"posts": rp, "comments": rc}, "preprocess_stats": pstats,
           "meta": {"embedding_backend": used, "final_algorithm": cl["final"], "k": cl["k"], "weights": weights,
                    "n_posts": len(posts), "n_comments": len(comments),
                    "data_source_types": sorted(set(posts["data_source_type"].dropna().astype(str)))}}
    res["embeddings_2d"] = _pca2(X, comments)
    if save:
        meta = res["meta"]
        meta_df = pd.DataFrame([{"embedding_backend": meta["embedding_backend"], "final_algorithm": meta["final_algorithm"], "k": meta["k"],
                                 "n_posts": meta["n_posts"], "n_comments": meta["n_comments"],
                                 "data_source_types": ",".join(meta["data_source_types"])}])
        save_tables({"posts": anomalies, "comments": comments, "cluster_profiles": cl["profile"], "cluster_comparison": cl["summary"],
                     "recommendations": recs, "pca_2d": res["embeddings_2d"], "meta": meta_df})
    return res


def _pca2(X, comments, max_points=4000):
    from sklearn.decomposition import PCA
    idx = np.random.default_rng(config.RANDOM_SEED).choice(len(X), min(len(X), max_points), replace=False)
    P = PCA(2, random_state=config.RANDOM_SEED).fit_transform(X[idx])
    out = comments.iloc[idx][["comment_id", "comment_text", "segment", "post_id"]].copy()
    out["pc1"], out["pc2"] = P[:, 0], P[:, 1]
    return out.reset_index(drop=True)
