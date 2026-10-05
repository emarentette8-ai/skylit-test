#!/usr/bin/env python3
"""Score the hypotheses in results/oos_plan.md on both datasets.

Run after node_claims.py, rug.py, confluence.py and trade_features.py have been
run for both datasets (default and SKYLIT_DATA=oos).

Usage: python3 scripts/oos_verdict.py
Writes results/tables/oos/verdict.csv.
"""
import numpy as np
import pandas as pd

from confluence import NEAR, build_sr, near_sr
from load import _ROOT
from node_claims import scale_at, typical_range
from stats import day_ci

SETS = {"Jul-Sep 2026": "", "Oct 2025-Jun 2026": "oos"}


def bars_for(tag):
    return pd.read_parquet(_ROOT / "data" / "raw" / (tag or "watchlist") / "price_bars_1min.parquet")


def score(tag):
    t = _ROOT / "results" / "tables" / tag
    f = pd.read_csv(t / "trade_features.csv", parse_dates=["map_time"])
    king = pd.read_csv(t / "node_claims_king.csv")
    p = bars_for(tag)
    p["symbol"] = p.symbol.astype(str)
    p["date"] = p.time_et.dt.normalize()
    scale, sr = typical_range(p), build_sr(p)
    f["sr"] = [near_sr(sr, r.symbol, r.map_time.normalize(), r.map_time, r.level,
                       NEAR * scale_at(scale, r.symbol, r.map_time) * r.level) for r in f.itertuples()]
    setups = f[~f.setup.str.endswith("control")]
    rows = []

    def add(h, desc, df, passed, extra=""):
        lo, hi = day_ci(df)
        rows.append(dict(hypothesis=h, test=desc, n=len(df), mean_r_net=df.r_net.mean(),
                         ci95_low=lo, ci95_high=hi, extra=extra, passed=passed(df.r_net.mean(), lo)))

    neg, pos = f[f.king_sign == "-"], f[f.king_sign == "+"]
    add("H1", "negative King, all Rug-family trades", neg,
        lambda m, lo: bool(m > 0 and lo > 0 and m > pos.r_net.mean()),
        f"positive King mean {pos.r_net.mean():.3f} (n={len(pos)})")
    rug_sr = f[(f.setup == "rug") & f.sr]
    add("H2", "Rug at price S/R", rug_sr, lambda m, lo: bool(m > 0 and lo > 0))
    between = setups[setups.king_zone == "between"]
    behind = setups[(setups.king_zone == "behind") & (setups.king_is_level == 0)]
    add("H3", "Rug+Reverse Rug, King between entry and target", between,
        lambda m, lo: bool(m > behind.r_net.mean()),
        f"King behind (not the level) mean {behind.r_net.mean():.3f} (n={len(behind)})")
    h15 = king[(king.by == "hour") & (king.group.astype(str) == "15")].iloc[0]
    rows.append(dict(hypothesis="H4", test="15:00 map: close ends nearer the King", n=int(h15.n),
                     mean_r_net=np.nan, ci95_low=np.nan, ci95_high=np.nan,
                     extra=f"toward King {h15.toward_king:.3f} vs mirror {h15.toward_mirror:.3f}",
                     passed=bool(h15.toward_king > h15.toward_mirror)))
    for s in ("rug", "reverse_rug"):
        add("H5", f"{s} on SPY/QQQ/SPXW", setups[setups.setup == s],
            lambda m, lo: bool(m > 0 and lo > 0))
    return pd.DataFrame(rows)


def main():
    out = pd.concat([score(tag).assign(period=name) for name, tag in SETS.items()])
    out = out[["hypothesis", "test", "period", "n", "mean_r_net", "ci95_low", "ci95_high", "extra", "passed"]]
    path = _ROOT / "results" / "tables" / "oos" / "verdict.csv"
    out.round(4).to_csv(path, index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
