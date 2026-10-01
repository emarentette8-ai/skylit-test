#!/usr/bin/env python3
"""Does lining a Heatseeker node up with price support/resistance help?

Price S/R known at a given moment:
  - prior day high, low and close,
  - today's opening range (9:30-10:00 high/low), once it is complete,
  - swing highs/lows on 15-minute bars (2 bars each side), once confirmed.
A level has confluence when an S/R level sits within 0.25 typical 30-minute
ranges of it.

Tests:
  1. Touch hold rate (same rule as node_claims.py) for named nodes and ordinary
     strikes, with and without confluence, plus S/R levels on their own.
  2. Rug / Reverse Rug trades from rug.py, split by confluence at the level.

Usage: python3 scripts/confluence.py  (run rug.py first)
Writes results/tables/confluence_{touch,rug}.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from node_claims import INDEXES, NAMED, prepare, typical_range

OUT = Path(__file__).resolve().parent.parent / "results" / "tables"
NEAR = 0.25            # confluence tolerance, in typical ranges


def build_sr(p):
    """Per (symbol, date): list of (known_from, price) S/R levels."""
    sr = {}
    for sym, sp in p.groupby("symbol"):
        days = sorted(sp.date.unique())
        prev = None
        for d in days:
            b = sp[sp.date == d].sort_values("time_et")
            if b.empty:
                continue
            start = b.time_et.iloc[0]
            lv = []
            if prev is not None:
                lv += [(start, prev.high.max()), (start, prev.low.min()), (start, prev.close.iloc[-1])]
            orng = b[b.time_et < start + pd.Timedelta(minutes=30)]
            ready = start + pd.Timedelta(minutes=30)
            lv += [(ready, orng.high.max()), (ready, orng.low.min())]
            q = b.set_index("time_et").resample("15min").agg({"high": "max", "low": "min"}).dropna()
            h, l, ts = q.high.values, q.low.values, q.index
            for i in range(2, len(q) - 2):
                known = ts[i + 2] + pd.Timedelta(minutes=15)
                if h[i] == h[i - 2:i + 3].max():
                    lv.append((known, h[i]))
                if l[i] == l[i - 2:i + 3].min():
                    lv.append((known, l[i]))
            sr[(sym, d)] = lv
            prev = b
    return sr


def near_sr(sr, sym, d, t, lvl, tol):
    return any(k <= t and abs(px - lvl) <= tol for k, px in sr.get((sym, d), ()))


def first_touch(b, t, lvl, x):
    """Hold rule from node_claims.py, first touch only. Returns 1, 0 or None."""
    hi, lo, cl, tm = b.high.values, b.low.values, b.close.values, b.time_et.values
    i = np.searchsorted(tm, np.datetime64(t))
    if i >= len(cl):
        return None
    below = cl[i] < lvl
    for k in range(i + 1, len(cl)):
        if lo[k] <= lvl <= hi[k]:
            for j in range(k + 1, len(cl)):
                if (hi[j] >= lvl + x) if below else (lo[j] <= lvl - x):
                    return 0
                if (lo[j] <= lvl - x) if below else (hi[j] >= lvl + x):
                    return 1
            return None
    return None


def touch_test(g, day_bars, scale, sr):
    g = g.join(scale.rename("scale"), on="symbol")
    g["dz"] = (g.strike - g.spot).abs() / g.spot / g.scale
    c = (g[(g.dz > 0.5) & (g.dz < 3)].sort_values("time_et")
         .drop_duplicates(["symbol", "date", "strike"]))
    c["node"] = np.where(c.node_type.isin(NAMED), "named", "normal")
    rows = []
    for row in c.itertuples():
        b = day_bars.get((row.symbol, row.date))
        if b is None:
            continue
        u = row.scale * row.strike
        held = first_touch(b, row.time_et, row.strike, 0.5 * u)
        if held is None:
            continue
        conf = near_sr(sr, row.symbol, row.date, row.time_et, row.strike, NEAR * u)
        rows.append((row.symbol in INDEXES, row.node, "+" if row.net_gamma > 0 else "-", conf, held))
    # S/R levels on their own, from the 10:00 map time, same distance filter
    for (sym, d), lv in sr.items():
        b = day_bars.get((sym, d))
        if b is None or sym == "VIX":
            continue
        t = d + pd.Timedelta(hours=10)
        at = b[b.time_et >= t]
        if at.empty:
            continue
        spot = at.close.iloc[0]
        for k, px in lv:
            if k > t or not 0.5 < abs(px - spot) / spot / scale[sym] < 3:
                continue
            held = first_touch(b, t, px, 0.5 * scale[sym] * px)
            if held is not None:
                rows.append((sym in INDEXES, "price_sr_only", "", True, held))
    t = pd.DataFrame(rows, columns=["index", "node", "sign", "confluence", "held"])
    return (t.groupby(["index", "node", "sign", "confluence"]).held
            .agg(n="size", hold_rate="mean").reset_index())


def rug_split(sr, scale):
    tr = pd.read_csv(OUT / "rug_trades.csv", parse_dates=["map_time"])
    tr["confluence"] = [near_sr(sr, r.symbol, r.map_time.normalize(), r.map_time, r.level,
                                NEAR * scale[r.symbol] * r.level) for r in tr.itertuples()]
    return (tr.groupby(["setup", "confluence"])
            .agg(trades=("r", "size"), win_rate=("r", lambda s: (s > 0).mean()),
                 mean_r_net=("r_net", "mean"), total_r_net=("r_net", "sum")).reset_index())


def main():
    g, p, day_bars = prepare()
    scale = typical_range(p)
    sr = build_sr(p)
    results = {"touch": touch_test(g, day_bars, scale, sr), "rug": rug_split(sr, scale)}
    pd.set_option("display.width", 160)
    for name, df in results.items():
        df.round(4).to_csv(OUT / f"confluence_{name}.csv", index=False)
        print(f"\n== {name}\n{df.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
