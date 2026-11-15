import numpy as np
import pandas as pd
import pytest

from src.data_ingestion.loader import DataLoadError, load_table
from src.data_ingestion.validation import validate
from src.feature_engineering.behaviour_features import add_need_flags, aggregate_to_posts
from src.feature_engineering.post_features import add_post_features, safe_divide
from src.preprocessing.pipeline import preprocess_comments
from src.preprocessing.text_cleaning import clean_text, detect_language


# ---------- loading / validation ----------
def test_unsupported_file_type():
    with pytest.raises(DataLoadError):
        load_table("data.txt")


def test_empty_dataset_gives_error_not_crash():
    _, rep = validate(pd.DataFrame(), "posts")
    assert not rep.ok


def test_missing_critical_column():
    _, rep = validate(pd.DataFrame({"brand": ["a"] * 40}), "posts")
    assert not rep.ok and any("post_id" in e for e in rep.errors)


def test_missing_optional_columns_filled_with_nan():
    df, rep = validate(pd.DataFrame({"post_id": [f"P{i}" for i in range(40)], "views": range(40)}), "posts")
    assert rep.ok and "likes" in df.columns and df["likes"].isna().all()


def test_negative_values_become_nan():
    df, _ = validate(pd.DataFrame({"post_id": [f"P{i}" for i in range(40)], "likes": [-5] + [1] * 39}), "posts")
    assert np.isnan(df.loc[0, "likes"])


def test_small_dataset_warns():
    _, rep = validate(pd.DataFrame({"post_id": ["P1"]}), "posts")
    assert any("small" in w.lower() for w in rep.warnings)


# ---------- text cleaning ----------
def test_url_emoji_and_mentions_removed():
    assert clean_text("Price? 😍😍 @sapphire http://x.com/a") == "price?"


def test_price_numbers_preserved():
    assert "5990" in clean_text("Rs. 5,990 only??")


def test_repeated_chars_collapsed():
    assert clean_text("pleaseeeee") == "pleasee"


def test_roman_urdu_variants_normalised():
    assert clean_text("Price kitny hy") == "price kitni hai"


def test_urdu_script_kept():
    assert "قیمت" in clean_text("قیمت کیا ہے؟")


@pytest.mark.parametrize("text,lang", [
    ("How much does it cost?", "English"),
    ("Is it in stock?", "English"),
    ("Price kitni hai?", "Roman Urdu"),
    ("Delivery Karachi mein available hai?", "Roman Urdu"),
    ("قیمت کیا ہے؟", "Urdu"),
    ("😍😍", "Unknown"),
])
def test_language_detection(text, lang):
    assert detect_language(clean_text(text)) == lang


def test_pipeline_handles_nulls_and_duplicates():
    df = pd.DataFrame({"comment_id": ["C1", "C1", "C2", "C3"], "post_id": ["P"] * 4,
                       "comment_text": ["Price?", "Price?", None, "😍"]})
    out, stats = preprocess_comments(df)
    assert len(out) == 1 and stats["duplicate_ids_removed"] == 1


# ---------- behaviour features ----------
@pytest.mark.parametrize("text,need", [
    ("price kitni hai?", "price"),
    ("is there any discount?", "discount"),
    ("delivery karachi mein available hai?", "delivery"),
    ("ordered last week still waiting", "delivery"),
    ("large size available hai?", "availability"),
    ("fabric kaisa hai?", "quality"),
    ("inbox mein details send kar dein", "purchase"),
    ("mashallah lovely", "general"),
])
def test_primary_need(text, need):
    out = add_need_flags(pd.DataFrame({"comment_clean": [clean_text(text)]}))
    assert out.loc[0, "primary_need"] == need


def test_multi_label_flags():
    out = add_need_flags(pd.DataFrame({"comment_clean": ["price and delivery to lahore?"]}))
    assert out.loc[0, "need_price"] == 1 and out.loc[0, "need_delivery"] == 1


def test_aggregate_rates_between_0_and_1():
    c = add_need_flags(pd.DataFrame({"post_id": ["P1", "P1"], "comment_clean": ["price?", "lovely"]}))
    agg = aggregate_to_posts(c)
    assert agg.loc[0, "rate_price"] == 0.5


# ---------- post features ----------
def test_safe_divide_zero_and_missing():
    r = safe_divide(pd.Series([1, 1, 1]), pd.Series([0, np.nan, 2]))
    assert np.isnan(r[0]) and np.isnan(r[1]) and r[2] == 0.5


def _posts(n=30, **over):
    d = {"post_id": [f"P{i}" for i in range(n)], "content_type": ["Reel"] * n, "platform": ["Instagram"] * n,
         "likes": [100] * n, "comments_count": [10] * n, "shares": [5] * n, "views": [1000] * n, "days_since_post": [5] * n}
    d.update(over)
    return pd.DataFrame(d)


def test_engagement_rate_formula():
    out = add_post_features(_posts())
    assert out.loc[0, "engagement_rate"] == pytest.approx(0.115)
    assert out.loc[0, "comments_per_day"] == pytest.approx(2.0)


def test_zero_views_and_missing_metrics_do_not_create_inf():
    out = add_post_features(_posts(views=[0] + [1000] * 29, shares=[np.nan] + [5] * 29))
    assert np.isnan(out.loc[0, "engagement_rate"])
    assert not np.isinf(out.select_dtypes("number")).any().any()


def test_relative_engagement_falls_back_for_small_groups():
    p = _posts(12)
    p.loc[:2, "platform"] = "TikTok"  # group of 3 < MIN_PEERS -> falls back to content_type
    out = add_post_features(p)
    assert out.loc[0, "peer_group_level"] == "content_type"
    assert out.loc[0, "relative_engagement"] == pytest.approx(1.0)


def test_missing_views_column_all_nan_no_crash():
    out = add_post_features(_posts(views=[np.nan] * 30))
    assert out["engagement_rate"].isna().all()
