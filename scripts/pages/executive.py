import streamlit as st

from dashboard.visualizations import charts
from src.campaign_analytics.analytics import group_performance


def render(res):
    posts, comments, meta = res["posts"], res["comments"], res["meta"]
    st.title("Executive Dashboard")
    if any("SYNTHETIC" in t for t in meta["data_source_types"]):
        st.warning("SYNTHETIC DEMONSTRATION DATA: generated for testing. Brands are fictional; nothing here reflects real customers.")
    c = st.columns(7)
    vals = [("Posts", f"{len(posts):,}"), ("Comments", f"{len(comments):,}"), ("Brands", posts["brand"].nunique()), ("Campaigns", posts["campaign"].nunique()),
            ("Avg engagement rate", f"{posts['engagement_rate'].mean():.2%}"), ("Audience clusters", res["profile"]["cluster"].nunique()), ("Anomalies", int(posts["is_anomaly"].sum()))]
    for col, (l, v) in zip(c, vals):
        col.metric(l, v)
    left, right = st.columns(2)
    camp = group_performance(posts, "campaign")
    if not camp.empty:
        left.plotly_chart(charts.bar(camp, "campaign", "median_engagement_rate", "Median engagement rate by campaign", pct=True), use_container_width=True)
    ct = group_performance(posts, "content_type")
    if not ct.empty:
        right.plotly_chart(charts.bar(ct, "content_type", "median_engagement_rate", "Median engagement rate by content type", pct=True), use_container_width=True)
    st.subheader("Key insights")
    recs = res["recommendations"]
    if recs.empty:
        st.info("No recommendation met its evidence threshold.")
    for _, r in recs.head(4).iterrows():
        st.markdown(f"**[{r['priority']}] {r['scope']}**: {r['pattern']}  \n_{r['metric']}_ (evidence: {r['evidence_level']})")
