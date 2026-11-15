"""Load CSV / Excel files (from a path or a Streamlit upload) into DataFrames."""
from pathlib import Path

import pandas as pd


class DataLoadError(Exception):
    """Raised with a readable message when a file cannot be loaded."""


def load_table(source, filename: str | None = None) -> pd.DataFrame:
    """`source` is a path or a file-like object. `filename` is needed for uploads."""
    name = (filename or str(source)).lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(source, encoding="utf-8-sig")
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(source)
    except Exception as exc:  # corrupt file, wrong encoding, etc.
        raise DataLoadError(f"Could not read '{name}': {exc}") from exc
    raise DataLoadError(f"Unsupported file type: '{Path(name).suffix}'. Use .csv or .xlsx")


def load_synthetic(synthetic_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    posts_p, comm_p = Path(synthetic_dir) / "posts.csv", Path(synthetic_dir) / "comments.csv"
    if not posts_p.exists() or not comm_p.exists():
        raise DataLoadError("Synthetic data not found. Run: python scripts/generate_synthetic_data.py")
    return load_table(posts_p), load_table(comm_p)
