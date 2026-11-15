"""Post-level anomaly detection: robust Z-score vs comparable posts + Isolation Forest.
Anomalies are patterns to investigate, NOT fraud, bots or failures."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config

FEATURES = ["views", "engagement_rate", "comment_rate", "share_rate", "comments_count", "rate_delivery", "rate_price", "rate_discount"]
Z_THRESHOLD = 3.5      # standard robust-Z cut-off (Iglewicz & Hoaglin)
MIN_GROUP = 15


def robust_z(s: pd.Series) -> pd.Series:
    """(x - median) / (1.4826 * MAD); falls back to std when MAD is 0; 0 when the metric is constant."""
    med = s.median()
    scale = 1.4826 * (s - med).abs().median()
    if not scale or np.isnan(scale):
        scale = s.std()
    return (s - med) / scale if scale and not np.isnan(scale) else pd.Series(0.0, index=s.index)


def _grouped_z(df: pd.DataFrame, col: str, keys: list) -> pd.Series:
    """Robust Z inside each peer group (same content type and campaign type) when the group has enough posts."""
    z = pd.Series(np.nan, index=df.index)
    if keys:
        for _, idx in df.groupby(keys).groups.items():
            if len(idx) >= MIN_GROUP:
                z.loc[idx] = robust_z(df.loc[idx, col])
    return z.fillna(robust_z(df[col]))


def detect_anomalies(posts: pd.DataFrame, contamination: float = 0.04, peer_keys=("content_type", "campaign_type")):
    df = posts.copy()
    feats = [f for f in FEATURES if f in df.columns and df[f].notna().any()]
    if len(df) < 30 or len(feats) < 3:
        df["anomaly_score"], df["is_anomaly"], df["anomaly_category"] = 0.0, False, ""
        df["anomaly_explanation"] = "Not enough data for anomaly detection (need >=30 posts and >=3 metrics)."
        return df, None
    keys = [k for k in peer_keys if k in df.columns and df[k].notna().any()]
    X = df[feats].copy()
    for c in ("views", "comments_count"):
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=config.RANDOM_SEED).fit(Xs)
    raw = -iso.score_samples(Xs)                       # higher = more unusual
    df["iso_score"] = np.round((raw - raw.min()) / (raw.max() - raw.min() + 1e-12), 4)
    # Robust Z on log scale for count metrics: raw counts are right-skewed, so a plain MAD can never flag LOW values
    zdf = df.copy()
    for c in ("views", "comments_count"):
        if c in zdf:
            zdf[c] = np.log1p(zdf[c].clip(lower=0))
    z = {f: _grouped_z(zdf, f, keys) for f in feats}
    for f in feats:
        df[f"z_{f}"] = z[f].round(2)
    zmax = pd.concat([z[f].abs() for f in feats], axis=1).max(axis=1)
    df["max_abs_z"] = zmax.round(2)
    # combined score in [0,1]: half Isolation Forest, half capped robust-Z
    df["anomaly_score"] = (0.5 * df["iso_score"] + 0.5 * (zmax.clip(upper=10) / 10)).round(4)
    iso_flag = iso.predict(Xs) == -1
    df["is_anomaly"] = iso_flag & (zmax >= Z_THRESHOLD)    # both methods must agree: fewer false alarms
    df["anomaly_category"], df["anomaly_explanation"] = "", ""
    for i in df.index[df["is_anomaly"]]:
        cat, why = _categorise(df.loc[i], z, i)
        df.at[i, "anomaly_category"], df.at[i, "anomaly_explanation"] = cat, why
    return df, iso


def _categorise(row, z, i) -> tuple:
    g = lambda f: z[f].get(i, 0) if f in z else 0
    if g("rate_delivery") >= Z_THRESHOLD:
        return "Delivery complaint surge", (f"Delivery-related share of comments is {g('rate_delivery'):.1f} robust-SDs above comparable posts. "
                                           "Check fulfilment/dispatch status for this campaign before drawing conclusions.")
    if g("rate_price") >= Z_THRESHOLD:
        return "Price inquiry surge", "Price questions are unusually frequent; the price may be missing or unclear in the post."
    if g("engagement_rate") >= Z_THRESHOLD or g("views") >= Z_THRESHOLD:
        return "Engagement spike", (f"Engagement/views far above comparable posts (z={max(g('engagement_rate'), g('views')):.1f}). "
                                    "Could be organic virality, a paid boost, a repost or an influencer share: investigate the source.")
    if g("comments_count") >= Z_THRESHOLD:
        return "Unexpected comment volume", "Comment count is far above comparable posts; check for a giveaway, controversy or tagging campaign."
    if g("engagement_rate") <= -Z_THRESHOLD or g("views") <= -Z_THRESHOLD:
        return "Low performance", "Views/engagement far below comparable posts; check posting time, targeting or whether the post was restricted."
    return "Unusual metric combination", "The combination of metrics is rare even though no single metric is extreme."


def evaluate_against_truth(df: pd.DataFrame, truth_col: str = "_anomaly_truth") -> dict:
    """Only valid when injected ground truth exists (synthetic data)."""
    if truth_col not in df.columns:
        return {}
    t = df[truth_col].notna() & df[truth_col].astype(str).ne("")
    p = df["is_anomaly"]
    tp = int((t & p).sum())
    prec = tp / max(int(p.sum()), 1)
    rec = tp / max(int(t.sum()), 1)
    return {"injected": int(t.sum()), "flagged": int(p.sum()), "true_positives": tp, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "recall_by_type": df[t].groupby(truth_col)["is_anomaly"].mean().round(3).to_dict()}
