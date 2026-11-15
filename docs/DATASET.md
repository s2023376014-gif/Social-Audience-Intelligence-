# Dataset documentation
**Type: SYNTHETIC_DEMONSTRATION_DATA**, created by `scripts/generate_synthetic_data.py` (seed 42, regenerable). Brands are fictional.
Real data you upload must use `data_source_type` values such as `ACTUAL_COLLECTED` or `PUBLIC_DATASET` and, where available, `source_url` and collection date.

**Posts** required: post_id, brand, platform, campaign, campaign_type, content_type, product_category, likes, comments_count, shares, views,
days_since_post, post_date, source_url, data_source_type. Optional: product_price, campaign_budget, engagement_rate.
**Comments** required: comment_id, post_id, comment_text, language_type, topic, purchase_intent, sentiment, comment_date, data_source_type. Optional: delivery_region.

Only `post_id` (and `comment_text` for comments) are critical. Other missing columns become nulls with a warning; metrics that need them become NaN, never invented.
`topic`, `purchase_intent` and `_anomaly_truth` are generator labels used only for sanity checks/evaluation, never as model features.
Designed patterns: sale campaigns -> more discount questions; launches -> availability and price; Eid -> delivery; about 1 post in 60 is an injected anomaly (viral spike, dead post, delivery storm).
Only up to 6 comments per post are stored although `comments_count` can be larger.
