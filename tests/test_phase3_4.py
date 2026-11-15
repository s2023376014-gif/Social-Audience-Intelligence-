import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.detector import detect_anomalies, evaluate_against_truth, robust_z
from src.clustering import clusterers as C
from src.clustering.profiling import profile_clusters
from src.feature_engineering.embeddings import embed_texts
from src.feature_engineering.feature_space import build_comment_features
from src.feature_engineering.lexicons import NEEDS
from src.feature_engineering.semantic_needs import semantic_need_assignment


def _comments(n=300):
    rng = np.random.default_rng(0)
    needs = rng.choice(NEEDS, n)
    df = pd.DataFrame({"post_id": [f"P{i % 20}" for i in range(n)], "emoji_count": 0,
                       "comment_text": [f"{k} question number {i % 7}" for i, k in enumerate(needs)],
                       "language_detected": "English"})
    df["comment_clean"] = df["comment_text"]
    for k in NEEDS:
        df[f"need_{k}"] = (needs == k).astype(int)
    return df


def test_embeddings_tfidf_shape_cache_and_repeat(tmp_path):
    texts = ["price please", "price please", "delivery lahore", "is it in stock", "kitni price hai", "delivery karachi"] * 5
    E, used = embed_texts(texts, backend="tfidf", cache_dir=tmp_path)
    assert E.shape[0] == len(texts) and used == "tfidf-char-svd"
    assert np.allclose(E[0], E[1])                      # identical text -> identical vector
    E2, _ = embed_texts(texts, backend="tfidf", cache_dir=tmp_path)  # served from cache
    assert np.allclose(E, E2)


def test_embeddings_handle_empty_and_null_text(tmp_path):
    E, _ = embed_texts(["hello world", None, "", "price", "hello"] * 4, backend="tfidf", cache_dir=tmp_path)
    assert len(E) == 20 and np.isfinite(E).all()


def test_feature_space_blocks_have_balanced_weight():
    c = _comments()
    posts = pd.DataFrame({"post_id": [f"P{i}" for i in range(20)], "engagement_rate": np.linspace(.01, .1, 20)})
    E = np.random.default_rng(0).normal(size=(len(c), 64))
    X, w = build_comment_features(c, posts, E, weights={"text": 1.0, "behaviour": 1.0, "post": 1.0})
    var = X.var(axis=0)
    text_var, beh_var = var[:20].sum(), var[20:20 + len(NEEDS) + 1].sum()
    assert 0.5 < text_var / beh_var < 2.0               # text does not dominate despite 20 vs 7 columns


def test_semantic_need_returns_labels():
    c = _comments()
    E = np.eye(len(NEEDS))[[NEEDS.index(n) for n in c[[f"need_{k}" for k in NEEDS]].idxmax(axis=1).str[5:]]]
    out = semantic_need_assignment(c, E)
    assert (out["semantic_need"] == c[[f"need_{k}" for k in NEEDS]].idxmax(axis=1).str[5:]).all()


def test_kmeans_recovers_planted_clusters_and_stability():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(m, .3, (100, 5)) for m in (0, 4, 8)])
    sweep = C.kmeans_sweep(X, range(2, 6))
    assert C.pick_k(sweep) == 3
    assert C.kmeans_stability(X, 3) > 0.95


def test_dbscan_marks_outliers_as_noise():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0, .2, (200, 3)), rng.normal(5, .2, (200, 3)), [[20, 20, 20]]])
    lab = C.run_dbscan(X, eps=0.8, min_samples=5)
    assert lab[-1] == -1


def test_metrics_with_one_cluster_do_not_crash():
    m = C.internal_metrics(np.random.rand(50, 3), np.zeros(50, dtype=int))
    assert np.isnan(m["silhouette"])


def test_small_dataset_clustering_does_not_crash():
    X = np.random.rand(15, 4)
    sweep = C.kmeans_sweep(X, range(2, 4))
    assert len(sweep) == 2


def test_cluster_names_come_from_content():
    c = _comments()
    labels = c[[f"need_{k}" for k in NEEDS]].to_numpy().argmax(axis=1)
    X = c[[f"need_{k}" for k in NEEDS]].to_numpy(float)
    prof = profile_clusters(c, labels, X)
    price_cluster = prof[prof["dominant_need"] == "price"].iloc[0]
    assert price_cluster["segment"] == "Price-Sensitive Customers" and "price question" in price_cluster["representative_comments"]


def test_mixed_cluster_gets_mixed_label():
    c = _comments(120)
    prof = profile_clusters(c, np.zeros(len(c), dtype=int), np.zeros((len(c), 2)))
    assert prof.loc[0, "segment"] in ("Mixed Behaviour",) or prof.loc[0, "dominant_need_share"] >= 0.5


# ---------- anomaly detection ----------
def _posts(n=300, seed=3):
    rng = np.random.default_rng(seed)
    views = rng.lognormal(9, .4, n)
    return pd.DataFrame({
        "post_id": [f"P{i}" for i in range(n)], "content_type": rng.choice(["Reel", "Photo"], n),
        "campaign_type": "Sale", "views": views, "engagement_rate": rng.normal(.05, .005, n),
        "comment_rate": rng.normal(.004, .0005, n), "share_rate": rng.normal(.003, .0004, n),
        "comments_count": views * .004, "rate_delivery": rng.choice([0, .1, .2], n),
        "rate_price": rng.choice([0, .2, .3], n), "rate_discount": rng.choice([0, .1, .2], n)})


def test_injected_anomalies_are_found_and_categorised():
    p = _posts()
    p.loc[0, ["views", "engagement_rate", "comments_count"]] = [p.views.max() * 20, .5, 50000]
    p.loc[1, ["views", "engagement_rate"]] = [50, .0005]
    p.loc[2, "rate_delivery"] = 5.0
    p["_anomaly_truth"] = ""
    p.loc[[0, 1, 2], "_anomaly_truth"] = "x"
    res, _ = detect_anomalies(p)
    assert res.loc[[0, 1, 2], "is_anomaly"].all()
    assert res.loc[0, "anomaly_category"] == "Engagement spike"
    assert res.loc[1, "anomaly_category"] == "Low performance"
    assert res.loc[2, "anomaly_category"] == "Delivery complaint surge"
    assert evaluate_against_truth(res)["recall"] == 1.0
    assert "fraud" not in " ".join(res["anomaly_explanation"]).lower()


def test_anomaly_small_dataset_and_missing_metrics_do_not_crash():
    res, iso = detect_anomalies(_posts(10))
    assert iso is None and not res["is_anomaly"].any()
    p = _posts().drop(columns=["share_rate", "rate_price", "rate_discount", "rate_delivery", "comment_rate"])
    res, iso = detect_anomalies(p)           # only 3 metrics left: still runs
    assert "anomaly_score" in res


def test_robust_z_constant_series():
    assert (robust_z(pd.Series([5.0] * 10)) == 0).all()
