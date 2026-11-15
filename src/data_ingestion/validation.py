"""Validate uploaded data without crashing on optional columns."""
from dataclasses import dataclass, field

import pandas as pd

import config

CRITICAL = {"posts": ["post_id"], "comments": ["post_id", "comment_text"]}
REQUIRED = {"posts": config.POST_REQUIRED, "comments": config.COMMENT_REQUIRED}


@dataclass
class ValidationReport:
    kind: str
    n_rows: int
    errors: list = field(default_factory=list)       # block processing
    warnings: list = field(default_factory=list)     # processing continues
    missing_columns: list = field(default_factory=list)
    missing_pct: dict = field(default_factory=dict)
    duplicate_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, ValidationReport]:
    """Return (df with all required columns present, report). Missing optional columns become NaN."""
    if kind not in REQUIRED:
        raise ValueError("kind must be 'posts' or 'comments'")
    if df is None or df.empty:
        report = ValidationReport(kind=kind, n_rows=0)
        report.errors.append("The dataset is empty.")
        return pd.DataFrame(columns=REQUIRED[kind]), report
    report = ValidationReport(kind=kind, n_rows=len(df))

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    id_col = "post_id" if kind == "posts" else "comment_id"

    for col in CRITICAL[kind]:
        if col not in df.columns:
            report.errors.append(f"Critical column missing: '{col}'.")
    if report.errors:
        return df, report

    report.missing_columns = [c for c in REQUIRED[kind] if c not in df.columns]
    for col in report.missing_columns:
        df[col] = float("nan")
        report.warnings.append(f"Column '{col}' missing: filled with nulls; features that need it will be NaN.")

    if df[id_col].isna().all():
        df[id_col] = [f"{id_col[0].upper()}{i:06d}" for i in range(len(df))]
        report.warnings.append(f"'{id_col}' was empty: generated sequential IDs.")

    report.missing_pct = (df[REQUIRED[kind]].isna().mean() * 100).round(2).to_dict()
    report.duplicate_rows = int(df.duplicated(subset=[id_col]).sum())
    if report.duplicate_rows:
        report.warnings.append(f"{report.duplicate_rows} duplicate '{id_col}' values found.")
    if len(df) < 30:
        report.warnings.append("Very small dataset (<30 rows): clustering results will be unreliable.")

    if kind == "posts":
        for col in ["likes", "comments_count", "shares", "views", "days_since_post"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if (df[col] < 0).any():
                report.warnings.append(f"Negative values in '{col}' set to null.")
                df.loc[df[col] < 0, col] = float("nan")
    return df, report
