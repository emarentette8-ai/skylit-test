#!/usr/bin/env python3
"""Backtest the Rug and Reverse Rug on SPY, QQQ and SPX (SPXW maps).

Rug (short): the largest node above price, within W, is a named positive node
(the ceiling), and below price, within 2W, the largest node is negative and
net gamma is negative (no floor). Reverse Rug (long) is the mirror image.
W is 1.5 typical 30-minute ranges.

Rules, applied in the 30 minutes a map is current:
  - price must tap the level (come within 0.1 typical range of it),
  - entry on the first 1-minute close back 0.25 typical range away from it
    (the rejection is confirmed),
  - stop 0.5 typical range beyond the level, target 3x the risk,
    otherwise exit at the 15:59 close. One open trade per symbol at a time.
Control: same rules at a named positive ceiling (or floor) whose far side is
net positive, i.e. the level without the Rug structure.
Results are in R (multiples of risk), gross and after a round-trip cost.

Usage: python3 scripts/rug.py
Writes results/tables/rug_trades.csv and results/tables/rug_summary.csv.
"""

import numpy as np
import pandas as pd

from load import TABLES
from node_claims import prepare, typical_range

OUT = TABLES
SYMBOLS = ("SPY", "QQQ", "SPXW")
COST = 0.0002          # round trip, fraction of price
WINDOW = 1.5           # search width for the ceiling/floor, in typical ranges
TAP, CONFIRM, STOP, REWARD = 0.1, 0.25, 0.5, 3.0


def classify(s, spot, w):
    """Return (setup, level) for one map: 'rug', 'reverse_rug', control variants or None."""
    found = []
    for side in ("above", "below"):
        near = s[(s.strike > spot) & (s.strike <= spot * (1 + w))] if side == "above" \
            else s[(s.strike < spot) & (s.strike >= spot * (1 - w))]
        far = s[(s.strike < spot) & (s.strike >= spot * (1 - 2 * w))] if side == "above" \
            else s[(s.strike > spot) & (s.strike <= spot * (1 + 2 * w))]
        if near.empty or far.empty:
            continue
        lvl = near.loc[near.net_gamma.abs().idxmax()]
        if lvl.net_gamma <= 0 or lvl.node_type == "normal":
            continue
        far_top = far.loc[far.net_gamma.abs().idxmax()]
        rug = far_top.net_gamma < 0 and far.net_gamma.sum() < 0
        ctrl = far.net_gamma.sum() > 0
        name = "rug" if side == "above" else "reverse_rug"
        if rug:
            found.append((name, lvl.strike))
        elif ctrl:
            found.append((name + "_control", lvl.strike))
    return found


def simulate(b, start, end, lvl, short, scale):
    """Walk 1-minute bars; return a trade dict or None if never triggered."""
    hi, lo, cl, tm = b.high.values, b.low.values, b.close.values, b.time_et.values
    i = np.searchsorted(tm, np.datetime64(start))
    stop_i = np.searchsorted(tm, np.datetime64(end))
    u = scale * lvl
    tapped = False
    for k in range(i + 1, min(stop_i, len(cl))):
        if not tapped:
            tapped = hi[k] >= lvl - TAP * u if short else lo[k] <= lvl + TAP * u
            continue
        if (short and cl[k] <= lvl - CONFIRM * u) or (not short and cl[k] >= lvl + CONFIRM * u):
            entry = cl[k]
            stop = lvl + STOP * u if short else lvl - STOP * u
            risk = abs(stop - entry)
            target = entry - REWARD * risk if short else entry + REWARD * risk
            exit_px, how, exit_t = cl[-1], "close", tm[-1]
            for m in range(k + 1, len(cl)):
                if (short and hi[m] >= stop) or (not short and lo[m] <= stop):
                    exit_px, how, exit_t = stop, "stop", tm[m]
                    break
                if (short and lo[m] <= target) or (not short and hi[m] >= target):
                    exit_px, how, exit_t = target, "target", tm[m]
                    break
            pnl = (entry - exit_px) if short else (exit_px - entry)
            return dict(entry_time=tm[k], entry=entry, stop=stop, target=target,
                        exit_time=exit_t, exit=exit_px,
                        exit_how=how, r=pnl / risk, r_net=(pnl - COST * entry) / risk)
    return None


def run():
    g, p, day_bars = prepare()
    scale = typical_range(p)
    g = g[g.symbol.isin(SYMBOLS)]
    trades = []
    busy_until = {}
    for (sym, t), s in g.sort_values("time_et").groupby(["symbol", "time_et"], sort=True):
        b = day_bars.get((sym, t.normalize()))
        if b is None:
            continue
        at = b[b.time_et >= t]
        if at.empty:
            continue
        spot = at.close.iloc[0]
        for setup, lvl in classify(s, spot, WINDOW * scale[sym]):
            key = (sym, setup)
            if busy_until.get(key, pd.Timestamp.min) > t:
                continue
            tr = simulate(b, t, t + pd.Timedelta(minutes=30), lvl,
                          short=setup.startswith("rug"), scale=scale[sym])
            if tr:
                busy_until[key] = pd.Timestamp(tr["exit_time"])
                trades.append(dict(symbol=sym, map_time=t, setup=setup, level=lvl, **tr))
    return pd.DataFrame(trades)


def summarize(tr):
    rng = np.random.default_rng(0)

    def boot(df):
        days = df.groupby(df.map_time.dt.date).r_net.sum()
        n = df.groupby(df.map_time.dt.date).size()
        means = []
        for _ in range(2000):
            idx = rng.integers(0, len(days), len(days))
            means.append(days.values[idx].sum() / n.values[idx].sum())
        return np.percentile(means, [2.5, 97.5])

    rows = []
    for keys, df in [(("all", s), d) for s, d in tr.groupby("setup")] + \
                    [((sym, s), d) for (sym, s), d in tr.groupby(["symbol", "setup"])]:
        lo, hi = boot(df)
        rows.append(dict(symbol=keys[0], setup=keys[1], trades=len(df), days=df.map_time.dt.date.nunique(),
                         win_rate=(df.r > 0).mean(), target_rate=(df.exit_how == "target").mean(),
                         mean_r=df.r.mean(), mean_r_net=df.r_net.mean(), total_r_net=df.r_net.sum(),
                         ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def main():
    tr = run()
    s = summarize(tr)
    OUT.mkdir(parents=True, exist_ok=True)
    tr.to_csv(OUT / "rug_trades.csv", index=False)
    s.round(4).to_csv(OUT / "rug_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
