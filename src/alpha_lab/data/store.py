"""Parquet-хранилище на D:\\alpha-lab\\data.

Раскладка: <root>/bars/<symbol>/<freq>/<symbol>-<freq>-<YYYY-MM>.parquet
           <root>/funding/<symbol>/<symbol>-funding.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.data.schema import FUNDING_COLUMNS, normalize_bars

DEFAULT_ROOT = Path(r"D:\alpha-lab\data")


def bars_dir(root: Path, symbol: str, freq: str) -> Path:
    return Path(root) / "bars" / symbol / freq


def write_bars(df: pd.DataFrame, root: Path, symbol: str, freq: str) -> list[Path]:
    df = normalize_bars(df)
    out_dir = bars_dir(root, symbol, freq)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for period, chunk in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        path = out_dir / f"{symbol}-{freq}-{period}.parquet"
        chunk.reset_index(drop=True).to_parquet(path, index=False, compression="zstd")
        written.append(path)
    return sorted(written)


def read_bars(root: Path, symbol: str, freq: str,
              start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out_dir = bars_dir(root, symbol, freq)
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Нет данных: {out_dir}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = normalize_bars(df)

    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["ts"] <= pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)


def write_funding(df: pd.DataFrame, root: Path, symbol: str) -> Path:
    out_dir = Path(root) / "funding" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}-funding.parquet"

    out = df.loc[:, FUNDING_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = (out.drop_duplicates(subset="ts", keep="last")
              .sort_values("ts")
              .reset_index(drop=True))
    out.to_parquet(path, index=False, compression="zstd")
    return path


def read_funding(root: Path, symbol: str) -> pd.DataFrame:
    path = Path(root) / "funding" / symbol / f"{symbol}-funding.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет данных о funding: {path}")
    out = pd.read_parquet(path)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.sort_values("ts").reset_index(drop=True)
