#!/usr/bin/env python3
"""Load the Skylit watchlist backtest data (Jul 1 - Sep 30, 2026, 50 tickers).

Data: Skylit (https://skylit.ai). See data/raw/watchlist/README.txt.
Requires: pip install pandas pyarrow
"""
import os
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
# SKYLIT_DATA=oos switches every study to data/raw/oos and results/tables/oos.
_SET = os.environ.get("SKYLIT_DATA", "")
DATA = _ROOT / "data" / "raw" / (_SET or "watchlist")
TABLES = _ROOT / "results" / "tables" / _SET


def gamma():
    """Heatseeker gamma snapshots: time_et, symbol, spot, strike, net_gamma, node_type."""
    return pd.read_parquet(DATA / "gamma_snapshots.parquet")


def bars():
    """One-minute regular-session bars: time_et, symbol, open, high, low, close."""
    return pd.read_parquet(DATA / "price_bars_1min.parquet")


def kings():
    """One row per snapshot with the king strike and its net gamma."""
    g = gamma()
    return (g[g.node_type == "king"]
            .rename(columns={"strike": "king", "net_gamma": "king_gamma"})
            .drop(columns="node_type")
            .reset_index(drop=True))


if __name__ == "__main__":
    g, b = gamma(), bars()
    print(f"gamma: {len(g):,} rows, {g.symbol.nunique()} symbols, "
          f"{g.time_et.dt.date.nunique()} days")
    print(f"bars:  {len(b):,} rows, {b.time_et.min()} -> {b.time_et.max()}")
