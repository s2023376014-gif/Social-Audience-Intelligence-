import pandas as pd
import streamlit as st

import config
from src.data_ingestion.loader import DataLoadError, load_synthetic, load_table
from src.data_ingestion.validation import validate
from src.pipeline import run_pipeline


def _report(df, kind):
    fixed, rep = validate(df, kind)
    (st.success if rep.ok else st.error)(f"{kind.title()}: {rep.n_rows:,} rows. " + ("Valid." if rep.ok else " ".join(rep.errors)))
    for w in rep.warnings:
        st.warning(w)
    if rep.ok:
        miss = pd.DataFrame({"column": list(rep.missing_pct), "missing_%": list(rep.missing_pct.values())})
        st.dataframe(miss[miss["missing_%"] > 0].sort_values("missing_%", ascending=False), hide_index=True, use_container_width=True)
    return rep


def render(res):
    st.title("Data Management")
    st.caption("Upload CSV/Excel files (posts and comments), or load the synthetic demo data. Required columns are listed in the docs; missing optional columns are filled with nulls.")
    c1, c2 = st.columns(2)
    up_p = c1.file_uploader("Posts file", type=["csv", "xlsx"], key="up_posts")
    up_c = c2.file_uploader("Comments file", type=["csv", "xlsx"], key="up_comments")
    try:
        if up_p is not None:
            st.session_state["raw_posts"] = load_table(up_p, up_p.name)
            st.session_state["source"] = "uploaded"
        if up_c is not None:
            st.session_state["raw_comments"] = load_table(up_c, up_c.name)
            st.session_state["source"] = "uploaded"
    except DataLoadError as e:
        st.error(str(e))
    if st.button("Load synthetic demo data"):
        try:
            st.session_state["raw_posts"], st.session_state["raw_comments"] = load_synthetic(config.SYNTHETIC_DIR)
            st.session_state["source"] = "synthetic"
        except DataLoadError as e:
            st.error(str(e))

    rp, rc = st.session_state.get("raw_posts"), st.session_state.get("raw_comments")
    if rp is None or rc is None:
        st.info("Load both a posts and a comments dataset to continue.")
        return
    st.subheader("Preview")
    t1, t2 = st.tabs(["Posts", "Comments"])
    t1.dataframe(rp.head(100), use_container_width=True)
    t2.dataframe(rc.head(100), use_container_width=True)
    st.subheader("Validation and data quality")
    a, b = st.columns(2)
    with a:
        r1 = _report(rp, "posts")
    with b:
        r2 = _report(rc, "comments")
    if st.button("Remove duplicate IDs"):
        st.session_state["raw_posts"] = rp.drop_duplicates(subset=[c for c in rp.columns if c.lower() == "post_id"][:1] or None)
        st.session_state["raw_comments"] = rc.drop_duplicates(subset=[c for c in rc.columns if c.lower() == "comment_id"][:1] or None)
        st.rerun()
    st.divider()
    backend = st.selectbox("Embedding backend", ["auto (MiniLM if installed, else TF-IDF fallback)", "tfidf (fast, not semantic)", "transformer (MiniLM only)"])
    if st.button("Run analysis", type="primary", disabled=not (r1.ok and r2.ok)):
        box = st.empty()
        try:
            with st.spinner("Running pipeline... the first run with MiniLM downloads the model once."):
                out = run_pipeline(rp, rc, backend=backend.split()[0], progress=lambda m: box.info(m))
            st.session_state["results"] = out
            box.success("Analysis complete. Open the other pages from the sidebar.")
        except Exception as e:
            box.error(f"Analysis failed: {e}")
