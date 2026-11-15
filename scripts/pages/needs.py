import pandas as pd
import streamlit as st

from dashboard.components.filters import apply_filters
from dashboard.visualizations import charts
from src.campaign_analytics.analytics import need_rates
from src.feature_engineering.lexicons import NEEDS


def render(res):
    st.title("Customer Needs Analysis")
    posts, comments = apply_filters(res, "needs")
    if comments.empty:
        return
    share = pd.DataFrame({"need": NEEDS, "share": [comments[f"need_{n}"].mean() for n in NEEDS]})
    share["comments"] = [int(comments[f"need_{n}"].sum()) for n in NEEDS]
    a, b = st.columns(2)
    a.plotly_chart(charts.bar(share, "need", "share", "Share of comments mentioning each need (keyword flags; a comment can count for several)", pct=True), use_container_width=True)
    if "semantic_need" in comments:
        sem = comments["semantic_need"].value_counts().rename_axis("need").reset_index(name="comments")
        b.plotly_chart(charts.bar(sem, "need", "comments", "Semantic grouping (nearest need prototype)"), use_container_width=True)
        b.caption("'unclear' = no prototype was similar enough. With the TF-IDF fallback this grouping is character-based, not truly semantic.")
    st.subheader("Needs by campaign")
    nr = need_rates(comments, posts, "campaign")
    if not nr.empty:
        st.plotly_chart(charts.heatmap(nr.drop(columns="n_comments"), "campaign", "Share of comments by need and campaign"), use_container_width=True)
    if "language_detected" in comments:
        lang = comments["language_detected"].value_counts().rename_axis("language").reset_index(name="comments")
        st.plotly_chart(charts.bar(lang, "language", "comments", "Comment language mix"), use_container_width=True)
    st.subheader("Example comments")
    need = st.selectbox("Need", NEEDS)
    st.dataframe(comments[comments[f"need_{need}"] == 1][["comment_text", "language_detected", "segment"]].drop_duplicates("comment_text").head(20), hide_index=True, use_container_width=True)
