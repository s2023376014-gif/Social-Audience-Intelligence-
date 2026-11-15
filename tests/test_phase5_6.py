import io

import numpy as np
import pandas as pd
import pytest

import config
from src.campaign_analytics.analytics import difference_test, group_performance, need_rates
from src.database import db
from src.recommendation_engine.rules import generate_recommendations
from src.reporting.exporters import to_csv_bytes, to_excel_bytes, to_pdf_bytes


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Full pipeline on a small freshly generated synthetic dataset (isolated DB and caches)."""
    import importlib.util, sys
    from src.pipeline import run_pipeline
    tmp = tmp_path_factory.mktemp("run")
    spec = importlib.util.spec_from_file_location("gen", config.BASE_DIR / "scripts" / "generate_synthetic_data.py")
    gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
    posts, comments, _ = gen.generate(400, 7)
    config.MODELS_DIR = tmp
    config.DB_PATH = tmp / "t.db"
    import src.database.db as dbm
    return run_pipeline(posts, comments, backend="tfidf", save=True)


def test_pipeline_outputs(results):
    assert {"posts", "comments", "profile", "recommendations"} <= results.keys()
    assert results["meta"]["n_posts"] == 400
    assert results["comments"]["segment"].notna().all()
    assert "SYNTHETIC_DEMONSTRATION_DATA" in results["meta"]["data_source_types"]


def test_database_roundtrip(results):
    assert "posts" in db.list_tables()
    assert len(db.load_table("posts")) == 400
    with pytest.raises(KeyError):
        db.load_table("nope")


def test_analytics(results):
    p = results["posts"]
    g = group_performance(p, "content_type", min_posts=1)
    assert g["n_posts"].sum() == 400
    assert 0 <= difference_test(p, "content_type")["p_value"] <= 1
    assert group_performance(p, "missing_col").empty
    nr = need_rates(results["comments"], p, "campaign")
    assert (nr.filter(like="need_").to_numpy() <= 1).all()


def test_recommendations_have_required_fields_and_evidence(results):
    r = results["recommendations"]
    for col in ["pattern", "metric", "segment", "suggested_action", "business_relevance", "evidence_level", "priority"]:
        assert col in r and r[col].notna().all()
    assert set(r["evidence_level"]) <= {"High", "Medium", "Low"}


def test_recommendations_empty_for_tiny_data():
    posts = pd.DataFrame({"post_id": ["P1"], "content_type": ["Reel"], "engagement_rate": [.05], "campaign": ["A"]})
    comments = pd.DataFrame({"post_id": ["P1"], "need_price": [1]})
    assert generate_recommendations(posts, comments).empty


def test_discount_campaign_triggers_discount_rule():
    rng = np.random.default_rng(0)
    n = 600
    posts = pd.DataFrame({"post_id": [f"P{i}" for i in range(n)], "campaign": ["Sale"] * 300 + ["Other"] * 300,
                          "content_type": "Reel", "engagement_rate": .05})
    c = pd.DataFrame({"post_id": posts["post_id"].repeat(2).to_numpy()})
    c["need_discount"] = np.where(c["post_id"].str[1:].astype(int) < 300, rng.random(len(c)) < .4, rng.random(len(c)) < .05).astype(int)
    r = generate_recommendations(posts, c)
    assert r["pattern"].str.contains("Discount").any()


def test_exports_open_correctly(results):
    xl = pd.ExcelFile(io.BytesIO(to_excel_bytes(results)))
    assert {"Summary", "Recommendations", "Audience segments"} <= set(xl.sheet_names)
    assert to_pdf_bytes(results)[:4] == b"%PDF"
    assert to_csv_bytes(results["profile"]).startswith(b"\xef\xbb\xbf")


def test_dashboard_pages_render_without_exceptions(results):
    from streamlit.testing.v1 import AppTest
    for page in ["1. Executive Dashboard", "2. Data Management", "3. Audience Intelligence", "4. Customer Needs",
                 "5. Campaign Intelligence", "6. Anomaly Detection", "7. Recommendations", "8. Reports"]:
        at = AppTest.from_file(str(config.BASE_DIR / "app.py"), default_timeout=60)
        at.session_state["results"] = results
        at.run()
        at.radio(key="nav").set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"
        assert not at.error, f"{page}: {[e.value for e in at.error]}"


def test_app_without_results_shows_guidance():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.BASE_DIR / "app.py"), default_timeout=30)
    at.session_state["results"] = None
    at.run()
    assert not at.exception and any("No analysis" in t.value for t in at.title)
