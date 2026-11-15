# Windows 11 installation and execution guide
1. Install Python 3.10-3.12 from python.org (tick "Add python.exe to PATH") and VS Code.
2. Open the project folder in VS Code, then Terminal > New Terminal (PowerShell).
3. `python -m venv .venv` then `.venv\Scripts\Activate.ps1`
   (if blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`).
4. `pip install -r requirements.txt` (sentence-transformers pulls PyTorch, about 200 MB; if it fails, delete that last line: the app falls back to TF-IDF).
5. `python scripts\generate_synthetic_data.py --posts 3000`
6. `python scripts\run_pipeline.py` (first run downloads the MiniLM model, about 90 MB, needs internet once).
7. `streamlit run app.py` and open http://localhost:8501
8. Tests: `python -m pytest tests -q`
Memory: embeddings are computed once per unique comment and cached in `models/`; 8 GB RAM is enough. Agglomerative clustering is run on a 3,000-comment sample on purpose (a full distance matrix would need about 2.5 GB).
Troubleshooting: `ModuleNotFoundError: src` -> run commands from the project root. Urdu looks garbled in Excel -> open the CSV via Data > From Text/CSV with UTF-8.
