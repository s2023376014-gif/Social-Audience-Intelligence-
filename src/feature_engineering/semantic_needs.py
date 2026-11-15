"""Semantic need grouping: assign each comment to the nearest need 'prototype' (centroid of keyword-
confident comments). Catches comments that keywords miss. Prototypes are learned from the data itself."""
import numpy as np
import pandas as pd

from src.feature_engineering.lexicons import NEEDS


def semantic_need_assignment(comments: pd.DataFrame, E: np.ndarray, min_sim: float = 0.35):
    """Returns df with semantic_need (or 'general'/'unclear') and semantic_confidence. Uses keyword-flagged
    comments as seeds only; never uses the generator's 'topic' column."""
    protos, names = [], []
    for need in NEEDS:
        mask = (comments[f"need_{need}"] == 1).to_numpy()
        if mask.sum() >= 5:
            c = E[mask].mean(axis=0)
            protos.append(c / (np.linalg.norm(c) + 1e-9))
            names.append(need)
    out = comments.copy()
    if not protos:
        out["semantic_need"], out["semantic_confidence"] = "unclear", 0.0
        return out
    S = E @ np.vstack(protos).T
    best = S.argmax(axis=1)
    conf = S.max(axis=1)
    out["semantic_need"] = np.where(conf >= min_sim, np.array(names)[best], "unclear")
    out["semantic_confidence"] = conf.round(4)
    return out
