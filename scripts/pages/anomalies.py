import pandas as pd
import plotly.express as px
import streamlit as st

import config
from dashboard.visualizations import charts
from src.anomaly_detection.detector import evaluate_against_truth


def render(res):
    st.title("Anomaly Detection")
    posts = res["posts"]
    st.caption("Isolation Forest plus robust Z-scores against comparable posts (same content type and campaign type). A post is flagged only when both agree. Anomalies are patterns to investigate, not fraud or failure.")
    a = posts[posts["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    k1, k2 = st.columns(2)
    k1.metric("Anomalies flagged", len(a))
    k2.metric("Share of posts", f"{len(a) / max(len(posts), 1):.1%}")
    if a.empty:
        st.info("No post was flagged by both methods.")
    else:
        cat = a["anomaly_category"].value_counts().rename_axis("category").reset_index(name="posts")
        l, r = st.columns(2)
        l.plotly_chart(charts.bar(cat, "category", "posts", "Anomaly categories", horizontal=True), use_container_width=True)
        r.plotly_chart(charts.hist(posts, "anomaly_score", "Anomaly score distribution"), use_container_width=True)
        show = [c for c in ["post_id", "brand", "campaign", "content_type", "anomaly_score", "anomaly_category", "views", "engagement_rate", "comments_count", "rate_delivery", "rate_price", "anomaly_explanation"] if c in a]
        st.dataframe(a[show], hide_index=True, use_container_width=True)
        fig = px.scatter(posts, x="views", y="engagement_rate", color="is_anomaly", log_x=True, hover_data=["post_id", "anomaly_category"], title="Views vs engagement rate", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Investigate a post")
        pid = st.selectbox("Post", a["post_id"].tolist())
        row = a[a["post_id"] == pid].iloc[0]
        st.write(f"**{row['anomaly_category']}**: {row['anomaly_explanation']}")
        z = {c[2:]: row[c] for c in row.index if c.startswith("z_") and pd.notna(row[c])}
        st.dataframe(pd.DataFrame({"metric": list(z), "robust_z_vs_peers": list(z.values())}), hide_index=True)
    if "_anomaly_truth" in posts.columns:
        with st.expander("Evaluation against injected synthetic anomalies"):
            st.json(evaluate_against_truth(posts))
            st.caption("Only possible because the demo data contains deliberately injected anomalies. Real data has no such labels, so precision/recall cannot be computed.")
