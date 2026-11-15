"""Rule-based recommendations. Every rule needs a measured pattern, a threshold and enough data; otherwise
nothing is generated. Evidence level reflects sample size; content findings are correlational."""
import pandas as pd

from src.campaign_analytics.analytics import difference_test, group_performance, need_rates

NEED_RULES = {
    "price": ("Price inquiries are frequent",
              "Consider publishing clear product prices in campaign posts and captions.",
              "Fewer repeated price questions and quicker path to purchase.", 0.15),
    "discount": ("Discount-related questions are frequent",
                 "Test a limited-time promotional offer and compare its engagement with previous campaigns.",
                 "Shows whether price promotions lift engagement or just shift timing.", 0.10),
    "delivery": ("Delivery concerns are frequent",
                 "Improve delivery information visibility (timelines, cities, charges) in product descriptions and campaign posts.",
                 "Lower support load and fewer pre-purchase doubts.", 0.10),
    "availability": ("Availability/stock questions are frequent",
                     "Provide clearer stock, size and city-wise availability information.",
                     "Avoids lost interest when customers cannot tell if an item is obtainable.", 0.10),
    "quality": ("Quality/fabric questions are frequent",
                "Add fabric, stitching and colour-fastness details (and wash-care info) to posts.",
                "Reduces hesitation for customers who cannot judge quality online.", 0.10),
    "purchase": ("Many comments show buying intent",
                 "Make the ordering path obvious (pinned link, clear 'how to order' line) so interested customers do not have to ask.",
                 "Captures demand already expressed in comments.", 0.10),
}
SEGMENT_FOR = {"price": "Price-Sensitive Customers", "discount": "Discount-Focused Customers", "delivery": "Delivery-Conscious Customers",
               "availability": "Product Availability Seekers", "quality": "Quality-Conscious Customers", "purchase": "High Purchase-Intent Customers"}
MIN_COMMENTS = 100


def _evidence(n: int) -> str:
    return "High" if n >= 500 else "Medium" if n >= MIN_COMMENTS else "Low"


def _priority(value: float, threshold: float) -> str:
    return "High" if value >= 1.5 * threshold else "Medium" if value >= threshold else "Low"


def generate_recommendations(posts: pd.DataFrame, comments: pd.DataFrame, anomalies: pd.DataFrame = None) -> pd.DataFrame:
    recs = []
    n = len(comments)
    if n >= MIN_COMMENTS:
        for need, (pattern, action, relevance, thr) in NEED_RULES.items():
            col = f"need_{need}"
            if col not in comments:
                continue
            share = comments[col].mean()
            if share >= thr:
                recs.append(dict(scope="All campaigns", pattern=pattern, metric=f"{share:.1%} of {n:,} comments", segment=SEGMENT_FOR[need],
                                 suggested_action=action, business_relevance=relevance, evidence_level=_evidence(n),
                                 priority=_priority(share, thr), score=share / thr))
        # campaign-specific spikes: campaign share well above the overall share
        if "campaign" in posts.columns:
            nr = need_rates(comments, posts, "campaign")
            for _, row in nr.iterrows():
                if row["n_comments"] < MIN_COMMENTS:
                    continue
                for need, (pattern, action, relevance, thr) in NEED_RULES.items():
                    if f"need_{need}" not in comments:
                        continue
                    overall = comments[f"need_{need}"].mean()
                    v = row.get(f"need_{need}", 0)
                    if overall > 0 and v >= thr and v >= 1.25 * overall:
                        recs.append(dict(scope=f"Campaign: {row['campaign']}", pattern=f"{pattern} in this campaign ({v / overall:.1f}x the overall rate)",
                                         metric=f"{v:.1%} of {int(row['n_comments']):,} comments vs {overall:.1%} overall", segment=SEGMENT_FOR[need],
                                         suggested_action=action, business_relevance=relevance, evidence_level=_evidence(int(row["n_comments"])),
                                         priority=_priority(v, thr), score=v / thr))
    # content format finding (correlational)
    perf = group_performance(posts, "content_type")
    if len(perf) >= 2:
        test = difference_test(posts, "content_type")
        best, overall_med = perf.iloc[0], posts["engagement_rate"].median()
        if test["p_value"] < 0.05 and best["median_engagement_rate"] >= 1.1 * overall_med:
            recs.append(dict(scope="Content strategy", pattern=f"'{best['content_type']}' posts have the highest median engagement rate in this dataset",
                             metric=f"{best['median_engagement_rate']:.2%} vs {overall_med:.2%} overall (Kruskal-Wallis p={test['p_value']:.3g}, n={int(best['n_posts'])} posts)",
                             segment="All audiences", suggested_action=f"Consider testing more '{best['content_type']}' content and compare against current mix. "
                             "This is an association in observational data, not proof the format causes higher engagement.",
                             business_relevance="Helps allocate content effort towards formats that currently resonate.",
                             evidence_level="High" if best["n_posts"] >= 100 else "Medium", priority="Medium", score=1.0))
    if anomalies is not None and "is_anomaly" in anomalies:
        a = anomalies[anomalies["is_anomaly"]]
        for cat, g in a.groupby("anomaly_category"):
            recs.append(dict(scope="Anomaly review", pattern=f"{len(g)} post(s) flagged as '{cat}'",
                             metric=f"Highest anomaly score {g['anomaly_score'].max():.2f}; posts: {', '.join(g.sort_values('anomaly_score', ascending=False)['post_id'].head(3))}",
                             segment="n/a", suggested_action="Investigate these posts (source of traffic, fulfilment status, posting details) before acting. Anomalies are patterns to check, not confirmed problems.",
                             business_relevance="Early warning of delivery issues or learning from unusually strong posts.",
                             evidence_level="Medium" if len(g) >= 3 else "Low", priority="High" if "Delivery" in cat else "Medium", score=0.9))
    out = pd.DataFrame(recs)
    if out.empty:
        return pd.DataFrame(columns=["scope", "pattern", "metric", "segment", "suggested_action", "business_relevance", "evidence_level", "priority", "score"])
    order = {"High": 0, "Medium": 1, "Low": 2}
    return out.sort_values(by=["priority", "score"], key=lambda s: s.map(order) if s.name == "priority" else -s).reset_index(drop=True)
