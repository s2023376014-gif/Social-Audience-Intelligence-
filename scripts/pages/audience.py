import streamlit as st

from dashboard.components.filters import apply_filters
from dashboard.visualizations import charts
from src.clustering.profiling import explain_comment_simple


def render(res):
    st.title("Audience Intelligence")
    posts, comments = apply_filters(res, "aud")
    prof = res["profile"]
    st.caption(f"Algorithm: {res['meta']['final_algorithm']} | embeddings: {res['meta']['embedding_backend']}. Segment names are derived from cluster content (the dominant customer need), never assigned arbitrarily.")
    if comments.empty:
        return
    dist = comments["segment"].value_counts().rename_axis("segment").reset_index(name="comments")
    k1, k2 = st.columns(2)
    k1.metric("Discovered clusters", prof["cluster"].nunique())
    k2.metric("Comments in view", f"{len(comments):,}")
    st.plotly_chart(charts.bar(dist, "segment", "comments", "Cluster distribution", horizontal=True), use_container_width=True)
    st.subheader("Cluster characteristics")
    cols = ["segment", "size", "share_pct", "dominant_need", "dominant_need_share"] + [c for c in prof.columns if c.startswith("share_")] + ["representative_comments"]
    st.dataframe(prof[[c for c in dict.fromkeys(cols)]], hide_index=True, use_container_width=True)
    pca = res["embeddings_2d"]
    pca = pca[pca["post_id"].isin(posts["post_id"])]
    if len(pca):
        st.plotly_chart(charts.scatter_pca(pca), use_container_width=True)
    st.subheader("Behaviour profile comparison")
    need_cols = [c for c in prof.columns if c.startswith("share_")]
    st.plotly_chart(charts.heatmap(prof.set_index("segment")[need_cols].reset_index(), "segment", "Share of each need within each cluster"), use_container_width=True)
    st.subheader("Why is a comment in its cluster?")
    sample = comments.sample(min(25, len(comments)), random_state=1)
    pick = st.selectbox("Pick a comment", sample["comment_text"].tolist())
    row = sample[sample["comment_text"] == pick].iloc[0]
    st.info(explain_comment_simple(row, prof))
    with st.expander("Algorithm comparison (internal metrics)"):
        st.dataframe(res["comparison"], hide_index=True, use_container_width=True)
        st.caption("Silhouette, Davies-Bouldin and Calinski-Harabasz measure geometric separation, not business usefulness. DBSCAN here splits into many tiny clusters (one per repeated phrase), so it is not selected.")
