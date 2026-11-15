import pandas as pd
import streamlit as st

from dashboard.components.filters import apply_filters
from dashboard.visualizations import charts
from src.campaign_analytics.analytics import difference_test, group_performance, segment_by_group


def render(res):
    st.title("Campaign Intelligence")
    posts, comments = apply_filters(res, "camp")
    if posts.empty:
        return
    st.caption("Comparisons are observational. A difference between groups is an association; it does not prove the content type or campaign caused it.")
    ct = group_performance(posts, "content_type")
    if not ct.empty:
        a, b = st.columns(2)
        a.plotly_chart(charts.bar(ct, "content_type", "median_engagement_rate", "Engagement rate by content type (median)", pct=True), use_container_width=True)
        b.plotly_chart(charts.box(posts.dropna(subset=["engagement_rate"]), "content_type", "engagement_rate", "Engagement rate distribution"), use_container_width=True)
        t = difference_test(posts, "content_type")
        if pd.notna(t["p_value"]):
            st.info(f"Kruskal-Wallis test across content types: p = {t['p_value']:.3g}. " + ("Differences are unlikely to be chance alone in this sample." if t["p_value"] < 0.05 else "Differences could be due to chance."))
        st.dataframe(ct, hide_index=True, use_container_width=True)
    st.subheader("Campaigns and brands")
    for by in ("campaign", "brand", "platform"):
        g = group_performance(posts, by)
        if not g.empty:
            st.plotly_chart(charts.bar(g, by, "median_engagement_rate", f"Median engagement rate by {by}", pct=True), use_container_width=True)
    seg = segment_by_group(comments, posts, "campaign")
    if not seg.empty:
        st.subheader("Audience segment mix per campaign")
        st.plotly_chart(charts.heatmap(seg, "campaign", "Share of each segment within a campaign"), use_container_width=True)
