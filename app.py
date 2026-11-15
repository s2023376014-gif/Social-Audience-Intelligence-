"""Pakistan Social Audience & Campaign Intelligence Engine. Run with: streamlit run app.py"""
import streamlit as st

from dashboard.pages import anomalies, audience, campaign, data_management, executive, needs, recommendations, reports
from dashboard.state import get_results

st.set_page_config(page_title="Audience & Campaign Intelligence", page_icon="📊", layout="wide")

PAGES = {"1. Executive Dashboard": executive, "2. Data Management": data_management, "3. Audience Intelligence": audience,
         "4. Customer Needs": needs, "5. Campaign Intelligence": campaign, "6. Anomaly Detection": anomalies,
         "7. Recommendations": recommendations, "8. Reports": reports}

st.sidebar.title("Audience Intelligence")
page = st.sidebar.radio("Navigate", list(PAGES), key="nav")
res = get_results()
st.sidebar.caption("Unsupervised ML: clustering, anomaly detection, rule-based recommendations.")
if res is not None:
    st.sidebar.success(f"{res['meta']['n_posts']:,} posts | {res['meta']['n_comments']:,} comments")

try:
    if page.endswith("Data Management"):
        PAGES[page].render(res)
    elif res is None:
        st.title("No analysis yet")
        st.info("Go to **2. Data Management**, load the demo data or upload files, then press **Run analysis**.")
    else:
        PAGES[page].render(res)
except Exception as exc:
    st.error(f"Something went wrong on this page: {exc}")
    st.caption("Check that your files contain the required columns (see docs/DATASET.md).")
