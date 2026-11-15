import streamlit as st

from src.reporting.exporters import summary_frame, to_csv_bytes, to_excel_bytes, to_pdf_bytes


def render(res):
    st.title("Reports")
    st.caption("Reports are generated from the current analysis results.")
    st.dataframe(summary_frame(res), hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("CSV")
        st.download_button("Recommendations", to_csv_bytes(res["recommendations"]), "recommendations.csv", "text/csv")
        st.download_button("Audience segments", to_csv_bytes(res["profile"]), "audience_segments.csv", "text/csv")
        st.download_button("Anomalies", to_csv_bytes(res["posts"][res["posts"]["is_anomaly"]]), "anomalies.csv", "text/csv")
    with c2:
        st.subheader("Excel")
        if st.button("Build Excel report"):
            st.session_state["xlsx"] = to_excel_bytes(res)
        if "xlsx" in st.session_state:
            st.download_button("Download .xlsx", st.session_state["xlsx"], "audience_intelligence_report.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.subheader("PDF")
        if st.button("Build PDF summary"):
            try:
                st.session_state["pdf"] = to_pdf_bytes(res)
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
        if "pdf" in st.session_state:
            st.download_button("Download .pdf", st.session_state["pdf"], "audience_intelligence_summary.pdf", "application/pdf")
