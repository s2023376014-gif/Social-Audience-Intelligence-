import streamlit as st


def render(res):
    st.title("Marketing Recommendations")
    recs = res["recommendations"]
    st.caption("Every recommendation is a rule triggered by a measured pattern. Evidence level reflects sample size. Content-format findings are correlational.")
    if recs.empty:
        st.info("No pattern met its evidence threshold, so no recommendation was generated.")
        return
    pri = st.multiselect("Priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    for _, r in recs[recs["priority"].isin(pri)].iterrows():
        with st.expander(f"[{r['priority']}] {r['scope']}: {r['pattern']}"):
            st.markdown(f"**Supporting metric:** {r['metric']}  \n**Relevant segment:** {r['segment']}  \n**Suggested action:** {r['suggested_action']}  \n"
                        f"**Business relevance:** {r['business_relevance']}  \n**Evidence level:** {r['evidence_level']}")
