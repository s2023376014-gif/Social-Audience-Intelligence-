import pandas as pd
import streamlit as st


def apply_filters(res: dict, key: str):
    """Brand / platform / campaign / category / date filters. Returns (posts, comments) restricted accordingly."""
    posts, comments = res["posts"], res["comments"]
    with st.expander("Filters", expanded=False):
        cols = st.columns(4)
        mask = pd.Series(True, index=posts.index)
        for col, label, c in zip(["brand", "platform", "campaign", "product_category"], ["Brand", "Platform", "Campaign", "Product category"], cols):
            if col in posts.columns and posts[col].notna().any():
                opts = sorted(posts[col].dropna().astype(str).unique())
                chosen = c.multiselect(label, opts, key=f"{key}_{col}")
                if chosen:
                    mask &= posts[col].astype(str).isin(chosen)
        dates = pd.to_datetime(posts.get("post_date"), errors="coerce")
        if dates.notna().any():
            lo, hi = dates.min().date(), dates.max().date()
            rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi, key=f"{key}_dates")
            if isinstance(rng, tuple) and len(rng) == 2:
                mask &= dates.between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1])) | dates.isna()
    p = posts[mask]
    c = comments[comments["post_id"].isin(p["post_id"])]
    if p.empty:
        st.warning("No posts match these filters.")
    return p, c
