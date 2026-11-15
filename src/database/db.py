"""SQLite storage through SQLAlchemy (switching to PostgreSQL later = changing the connection URL)."""
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect

import config


def get_engine(url: str = None):
    return create_engine(url or f"sqlite:///{Path(config.DB_PATH)}")


def save_tables(tables: dict, engine=None) -> None:
    engine = engine or get_engine()
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out = df.copy()
        for c in out.columns:                       # dict columns (e.g. language shares) -> text for SQL
            if out[c].map(lambda v: isinstance(v, (dict, list))).any():
                out[c] = out[c].astype(str)
        out.to_sql(name, engine, if_exists="replace", index=False)


def load_table(name: str, engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    if name not in inspect(engine).get_table_names():
        raise KeyError(f"Table '{name}' not found. Run the analysis first.")
    return pd.read_sql_table(name, engine)


def list_tables(engine=None) -> list:
    return inspect(engine or get_engine()).get_table_names()
