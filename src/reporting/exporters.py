"""CSV / Excel / PDF report builders (all return bytes so Streamlit can offer downloads)."""
import io
from xml.sax.saxutils import escape

import pandas as pd

from src.campaign_analytics.analytics import group_performance

LIMITATION = ("Findings are associations in the supplied data, not proof of cause. Clusters come from unsupervised learning, so internal "
              "metrics measure geometric separation, not business value. Anomalies are patterns to investigate, not confirmed problems.")


def summary_frame(res: dict) -> pd.DataFrame:
    p, m = res["posts"], res["meta"]
    return pd.DataFrame({"metric": ["Posts", "Comments analysed", "Brands", "Campaigns", "Average engagement rate", "Audience clusters",
                                    "Anomalies flagged", "Clustering algorithm", "Embedding backend", "Data source type(s)"],
                         "value": [len(p), m["n_comments"], p["brand"].nunique(), p["campaign"].nunique(), f"{p['engagement_rate'].mean():.2%}",
                                   res["profile"]["cluster"].nunique(), int(p["is_anomaly"].sum()), m["final_algorithm"], m["embedding_backend"],
                                   ", ".join(m["data_source_types"])]})


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")   # BOM keeps Urdu readable in Excel


def to_excel_bytes(res: dict) -> bytes:
    buf = io.BytesIO()
    prof = res["profile"].copy()
    prof["languages"] = prof["languages"].astype(str)
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        summary_frame(res).to_excel(xl, sheet_name="Summary", index=False)
        prof.to_excel(xl, sheet_name="Audience segments", index=False)
        res["recommendations"].to_excel(xl, sheet_name="Recommendations", index=False)
        res["posts"][res["posts"]["is_anomaly"]].to_excel(xl, sheet_name="Anomalies", index=False)
        group_performance(res["posts"], "content_type").to_excel(xl, sheet_name="Content performance", index=False)
        group_performance(res["posts"], "campaign").to_excel(xl, sheet_name="Campaign performance", index=False)
        res["comparison"].to_excel(xl, sheet_name="Algorithm comparison", index=False)
        res["comments"].drop(columns=["comment_clean"], errors="ignore").head(50000).to_excel(xl, sheet_name="Comments", index=False)
    return buf.getvalue()


def _ascii(s) -> str:
    """The built-in PDF fonts cannot draw Urdu script, so PDF text is limited to Latin characters."""
    return escape(str(s).encode("ascii", "ignore").decode())


def to_pdf_bytes(res: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    small = ss["BodyText"].clone("small", fontSize=8, leading=10)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    el = [Paragraph("Pakistan Social Audience &amp; Campaign Intelligence Engine", ss["Title"]), Paragraph("Summary report", ss["Heading2"])]
    if any("SYNTHETIC" in t for t in res["meta"]["data_source_types"]):
        el.append(Paragraph("<b>SYNTHETIC DEMONSTRATION DATA:</b> these results come from generated data, not real customers or brands.", ss["BodyText"]))
    el.append(Spacer(1, 8))

    def table(data, widths=None):
        t = Table([[Paragraph(_ascii(c), small) for c in row] for row in data], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    s = summary_frame(res)
    el += [Paragraph("Key figures", ss["Heading3"]), table([["Metric", "Value"]] + s.values.tolist(), [6 * cm, 11 * cm]), Spacer(1, 8)]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        prof = res["profile"].sort_values("size")
        fig, ax = plt.subplots(figsize=(6.5, 3))
        ax.barh([_ascii(x)[:34] for x in prof["segment"]], prof["share_pct"], color="#2F6BB3")
        ax.set_xlabel("% of comments"); ax.set_title("Audience segments")
        fig.tight_layout()
        img = io.BytesIO(); fig.savefig(img, format="png", dpi=150); plt.close(fig); img.seek(0)
        el += [Image(img, width=15 * cm, height=6.9 * cm)]
    except Exception:
        pass

    ct = group_performance(res["posts"], "content_type")
    if not ct.empty:
        el += [Paragraph("Content type performance (median)", ss["Heading3"]),
               table([["Content type", "Posts", "Engagement rate", "Comment rate"]] +
                     [[r["content_type"], int(r["n_posts"]), f"{r['median_engagement_rate']:.2%}", f"{r.get('median_comment_rate', float('nan')):.3%}"] for _, r in ct.iterrows()])]
    el += [Paragraph("Audience segments", ss["Heading3"]),
           table([["Segment", "Comments", "Share"]] + [[r["segment"], int(r["size"]), f"{r['share_pct']}%"] for _, r in res["profile"].iterrows()], [9 * cm, 3 * cm, 3 * cm])]
    recs = res["recommendations"].head(12)
    el += [Paragraph("Top recommendations", ss["Heading3"]),
           table([["Priority", "Pattern", "Evidence", "Suggested action"]] +
                 [[r["priority"], f"{r['scope']}: {r['pattern']} ({r['metric']})", r["evidence_level"], r["suggested_action"]] for _, r in recs.iterrows()], [1.6 * cm, 6.4 * cm, 1.6 * cm, 8 * cm]),
           Spacer(1, 8), Paragraph("Limitations", ss["Heading3"]), Paragraph(_ascii(LIMITATION), small)]
    doc.build(el)
    return buf.getvalue()
