# Pakistan Social Audience & Campaign Intelligence Engine

Unsupervised-ML marketing analytics for Pakistani online fashion brands: it finds hidden audience segments in social-media
comments, flags unusual posts, compares content and campaigns, and turns the findings into evidence-backed recommendations.

> **Data warning.** The bundled dataset is **SYNTHETIC DEMONSTRATION DATA** (fictional brands). Nothing in it comes from
> Sapphire, Khaadi, Gul Ahmed or any real customer. Findings on it show that the pipeline works, not how real customers behave.

## Business problem
Brands see thousands of comments but cannot read them all. Which customer needs dominate (price, discount, delivery, stock,
quality, buying intent)? Which posts behave unusually? Which content formats attract more engagement? What should the team test next?

## Key features
- CSV/Excel upload, validation, data-quality report, SQLite storage (PostgreSQL-ready via SQLAlchemy)
- Cleaning for English, Roman Urdu and Urdu; Roman Urdu spelling normalisation; language detection
- Post features (engagement rate, comment rate, relative engagement vs comparable posts) and customer-need features
- Sentence embeddings (all-MiniLM-L6-v2) with disk cache, plus an offline TF-IDF fallback
- Clustering: K-Means vs DBSCAN vs Agglomerative, compared on internal metrics and stability, segments named from cluster content
- Anomaly detection: Isolation Forest + robust Z-score against comparable posts
- Campaign/content analytics with non-parametric significance tests (associations, not causation)
- Rule-based recommendations with pattern, metric, segment, action, relevance, evidence level, priority
- 8-page Streamlit dashboard; CSV, Excel and PDF reports

## Quick start (Windows 11, PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\generate_synthetic_data.py --posts 3000
python scripts\run_pipeline.py          # analysis + SQLite (first run downloads MiniLM)
streamlit run app.py
python -m pytest tests -q               # 55 tests
```
More detail: `docs/INSTALL_WINDOWS.md`. Step-by-step phase scripts: `run_phase2.py`, `run_phase3.py`, `run_phase4.py`.

## Architecture
`src/data_ingestion` -> `src/preprocessing` -> `src/feature_engineering` -> `src/clustering` / `src/anomaly_detection` ->
`src/campaign_analytics` -> `src/recommendation_engine` -> `src/database` / `src/reporting` -> `dashboard/` + `app.py`.
`src/pipeline.py` wires everything together. See `docs/TECHNICAL.md`.

## Evaluation methodology
- Clustering: silhouette, Davies-Bouldin, Calinski-Harabasz, size distribution, K-Means restart stability (ARI), manual reading of representative comments.
- Anomalies: score distribution, robust-Z breakdown per flagged post; precision/recall **only** on the injected synthetic anomalies.
- NLP: similarity checks on hand-written pairs, language-detection agreement, manual review.
- Internal metrics measure geometry, not business value. No "accuracy" is claimed for unsupervised models.

## Limitations
- Synthetic data uses a small phrase pool, so clusters are cleaner than real comments would give; keyword/segment agreement near 100% is a property of the generator.
- Keyword lexicons are English/Roman Urdu/Urdu heuristics and will miss slang; embeddings help but MiniLM is weak on Roman Urdu (compare the multilingual model, see `docs/MODELS.md`).
- Only up to 6 comments per post are stored in the demo data; real exports differ.
- Platforms restrict comment/analytics access; this tool analyses data you are permitted to use and does no scraping.
- PDF text is Latin-only (built-in fonts cannot draw Urdu script); Excel/CSV keep Urdu.

## Future improvements
Multilingual embeddings fine-tuned on Roman Urdu, UMAP view, topic modelling per cluster, time-series anomaly detection per campaign,
official-API connectors (Meta/TikTok business APIs), PostgreSQL deployment, new product categories.

## Screenshots
Add screenshots of each page here after running the app (`docs/screenshots/`).
