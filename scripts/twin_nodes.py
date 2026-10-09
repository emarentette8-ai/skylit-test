#!/usr/bin/env python3
"""Twin nodes on single stocks: two similar-size nodes with a gap between them.

On every map of a single stock (index and sector ETFs excluded):
  A = largest |gamma| node; B = largest node at least 2 typical 30-minute
  ranges from A. ratio = |B| / |A| (twins: ratio >= 0.7). Gap: no node strictly
  between them (ignoring strikes within 0.5 typical range of A or B, i.e. their
  own clusters) larger than half of |B|. Price must be between A and B.
Trade (edges.simulate): the nearer node is tapped and a 1-minute close 0.25
typical range back off it shows it held; enter toward the far node, stop 0.5
typical range beyond the near node, target the far node, else exit at the close.
Split by ratio bucket, node signs, and the day's opening gap versus the previous
close (at least one typical range): trading toward the still-unfilled gap,
against it, gap already filled, or no gap.

Usage: python3 scripts/twin_nodes.py  (SKYLIT_DATA=oos for the new data)
Writes twin_nodes_trades.csv and twin_nodes_summary.csv to the tables folder.
"""
import numpy as np
import pandas as pd

from edges import simulate
from load import TABLES
from node_claims import prepare, scale_at, typical_range
from stats import day_ci

OUT = TABLES
ETFS = {"SPY", "QQQ", "SPXW", "IWM", "SMH", "VIX"}
MIN_APART, CLUSTER, GAP_MAX = 2.0, 0.5, 0.5


def prev_closes(p):
    last = p.sort_values("time_et").groupby(["symbol", "date"]).close.last()
    return last.groupby(level="symbol").shift(1).dropna()


def pair(s, spot, u):
    """(near, far, ratio) for the map, or None."""
    a = s.loc[s.net_gamma.abs().idxmax()]
    away = s[(s.strike - a.strike).abs() >= MIN_APART * u]
    if away.empty:
        return None
    b = away.loc[away.net_gamma.abs().idxmax()]
    lo, hi = sorted((a.strike, b.strike))
    if not lo < spot < hi:
        return None
    inner = s[(s.strike > lo + CLUSTER * u) & (s.strike < hi - CLUSTER * u)]
    if len(inner) and inner.net_gamma.abs().max() > GAP_MAX * abs(b.net_gamma):
        return None
    near, far = (a, b) if abs(spot - a.strike) <= abs(spot - b.strike) else (b, a)
    return near, far, abs(b.net_gamma) / abs(a.net_gamma)


def gap_state(b, entry_time, prev, long, u):
    if prev is None or np.isnan(prev):
        return "no prior close"
    open_ = b.open.values[0]
    if abs(open_ - prev) < u:
        return "no gap"
    before = b[b.time_et < entry_time]
    filled = before.high.max() >= prev if open_ < prev else before.low.min() <= prev
    if filled:
        return "gap already filled"
    toward = (open_ < prev and long) or (open_ > prev and not long)
    return "toward unfilled gap" if toward else "against unfilled gap"


def trades(g, day_bars, scale, pc):
    out, busy = [], {}
    stocks = g[~g.symbol.isin(ETFS)]
    for (sym, t), s in stocks.sort_values("time_et").groupby(["symbol", "time_et"], sort=True):
        b = day_bars.get((sym, t.normalize()))
        sc = scale_at(scale, sym, t)
        if b is None or np.isnan(sc) or b[b.time_et >= t].empty or busy.get(sym, pd.Timestamp.min) > t:
            continue
        spot = b[b.time_et >= t].close.iloc[0]
        u = sc * spot
        found = pair(s, spot, u)
        if found is None:
            continue
        near, far, ratio = found
        long = far.strike > near.strike
        tr = simulate(b, t, t + pd.Timedelta(minutes=30), near.strike, long, far.strike, u)
        if tr is None:
            continue
        busy[sym] = pd.Timestamp(tr["exit_time"])
        prev = pc.get((sym, t.normalize()), np.nan)
        out.append(dict(symbol=sym, map_time=t, side="long" if long else "short",
                        near=near.strike, near_val=near.net_gamma, far=far.strike, far_val=far.net_gamma,
                        ratio=ratio, signs=("+" if near.net_gamma > 0 else "-") + "/" + ("+" if far.net_gamma > 0 else "-"),
                        prev_close=prev, prev_close_near_target=bool(abs(prev - far.strike) <= u) if not np.isnan(prev) else False,
                        gap=gap_state(b, pd.Timestamp(tr["entry_time"]), prev, long, u), **tr))
    return pd.DataFrame(out)


def summarize(t):
    t = t.assign(bucket=pd.cut(t.ratio, [0, 0.5, 0.7, 1.0], labels=["< 0.5", "0.5-0.7", "twins >= 0.7"]))
    tw = t[t.bucket == "twins >= 0.7"]
    splits = [("all pairs", t)] + [(f"ratio {b}", t[t.bucket == b]) for b in ("< 0.5", "0.5-0.7", "twins >= 0.7")]
    splits += [(f"twins, signs near/far {sg}", tw[tw.signs == sg]) for sg in ("+/+", "+/-", "-/+", "-/-")]
    splits += [(f"twins, {gs}", tw[tw.gap == gs]) for gs in
               ("toward unfilled gap", "against unfilled gap", "gap already filled", "no gap")]
    splits += [("twins, prev close at the target", tw[tw.prev_close_near_target]),
                ("all pairs, toward unfilled gap", t[t.gap == "toward unfilled gap"])]
    rows = []
    for name, x in splits:
        lo, hi = day_ci(x)
        rows.append(dict(split=name, trades=len(x), win_rate=(x.r > 0).mean() if len(x) else np.nan,
                         target_rate=(x.exit_how == "target").mean() if len(x) else np.nan,
                         median_reward_risk=x.reward_risk.median(), mean_r_net=x.r_net.mean(),
                         ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def main():
    g, p, day_bars = prepare()
    scale = typical_range(p)
    t = trades(g, day_bars, scale, prev_closes(p))
    s = summarize(t)
    t.to_csv(OUT / "twin_nodes_trades.csv", index=False)
    s.round(4).to_csv(OUT / "twin_nodes_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{t.symbol.nunique()} stocks")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
