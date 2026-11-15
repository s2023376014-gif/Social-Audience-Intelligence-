"""Combined feature space for clustering comments: PCA-reduced text + weighted behaviour + metadata.

Why: 384 raw embedding dims would drown ~10 behaviour columns. We (1) reduce text with PCA, (2) standardise
each block, (3) divide each block by sqrt(block width) so a block's TOTAL variance is 1, then (4) multiply
by an explicit weight. Weights are therefore interpretable: 'text counts twice as much as engagement'."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from src.feature_engineering.lexicons import NEEDS

DEFAULT_WEIGHTS = {"text": 1.0, "behaviour": 1.0, "post": 0.5}


def build_comment_features(comments: pd.DataFrame, posts: pd.DataFrame, E: np.ndarray,
                           weights: dict = None, n_pca: int = 20):
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    n_pca = max(2, min(n_pca, E.shape[1], len(comments) - 1))
    T = PCA(n_pca, random_state=config.RANDOM_SEED).fit_transform(E)
    B = comments[[f"need_{n}" for n in NEEDS]].to_numpy(float)
    B = np.hstack([B, comments[["emoji_count"]].to_numpy(float)]) if "emoji_count" in comments else B
    post_cols = [c for c in ["engagement_rate", "relative_engagement"] if c in posts.columns]
    P = comments[["post_id"]].merge(posts[["post_id"] + post_cols], on="post_id", how="left")[post_cols]
    P = np.log1p(P.fillna(P.median()).fillna(0).clip(lower=0).to_numpy(float)) if post_cols else np.zeros((len(comments), 1))
    blocks = []
    for X, key in ((T, "text"), (B, "behaviour"), (P, "post")):
        Z = StandardScaler().fit_transform(X)
        Z = np.nan_to_num(Z) / np.sqrt(Z.shape[1])
        blocks.append(Z * w[key])
    return np.hstack(blocks), w
