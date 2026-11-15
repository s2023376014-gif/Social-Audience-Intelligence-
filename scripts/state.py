"""Session state helpers: results live in st.session_state and are reloaded from SQLite on first visit."""
import pandas as pd
import streamlit as st

from src.database import db

NEEDED = {"posts", "comments", "cluster_profiles", "cluster_comparison", "recommendations", "pca_2d", "meta"}


def load_from_db():
    try:
        if not NEEDED <= set(db.list_tables()):
            return None
        t = {n: db.load_table(n) for n in NEEDED}
    except Exception:
        return None
    posts = t["posts"]
    posts["is_anomaly"] = posts["is_anomaly"].astype(bool)
    m = t["meta"].iloc[0]
    meta = {"embedding_backend": m["embedding_backend"], "final_algorithm": m["final_algorithm"], "k": int(m["k"]),
            "n_posts": int(m["n_posts"]), "n_comments": int(m["n_comments"]), "data_source_types": str(m["data_source_types"]).split(",")}
    return {"posts": posts, "comments": t["comments"], "profile": t["cluster_profiles"], "comparison": t["cluster_comparison"],
            "recommendations": t["recommendations"], "embeddings_2d": t["pca_2d"], "meta": meta}


def get_results():
    if "results" not in st.session_state:
        st.session_state["results"] = load_from_db()
    return st.session_state["results"]
