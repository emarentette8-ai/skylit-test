#!/usr/bin/env python3
"""King rejection with Trinity rolls (SPXW / SPY / QQQ).

Long (short is the mirror image), for each index and each pair of consecutive
maps t0 -> t1 (30 minutes apart, same day):
  1. tap: price comes down to the t0 King (within 0.1 typical range of it)
     between t0 and t1, and is back at least 0.25 typical range above it at t1
     (the rejection);
  2. Kings roll: at t1 the King strike is higher than at t0 on >= 2 of the 3
     indexes ("all 3" also reported);
  3. floors roll: the largest node below price is at a higher strike at t1 than
     at t0 on >= 2 of the 3 indexes;
  4. entry at the t1 price; stop 0.1 typical range below the rejection wick
     (lowest low since the tap); target the t1 King, which must be above entry;
     otherwise exit at the close. A bar that hits stop and target is a loss.
Trades are reported for the rejection alone and with each condition added.

Usage: python3 scripts/king_reject.py  (SKYLIT_DATA=oos for the new data)
Writes king_reject_trades.csv and king_reject_summary.csv to the tables folder.
"""
import numpy as np
import pandas as pd

from load import TABLES
from node_claims import prepare, scale_at, typical_range
from stats import day_ci

OUT = TABLES
TRIO = ("SPXW", "SPY", "QQQ")
TAP, CONFIRM, BUFFER, COST = 0.1, 0.25, 0.1, 0.0002


def boards(g, day_bars):
    """King and floor/ceiling strike for each index map, plus the price at map time."""
    rows = []
    for (sym, t), s in g[g.symbol.isin(TRIO)].groupby(["symbol", "time_et"]):
        b = day_bars.get((sym, t.normalize()))
        if b is None or b[b.time_et >= t].empty:
            continue
        spot = b[b.time_et >= t].close.iloc[0]
        below, above = s[s.strike < spot], s[s.strike > spot]
        if below.empty or above.empty:
            continue
        rows.append(dict(symbol=sym, time_et=t, spot=spot,
                         king=s.loc[s.node_type == "king", "strike"].iloc[0],
                         floor=below.loc[below.net_gamma.abs().idxmax(), "strike"],
                         ceiling=above.loc[above.net_gamma.abs().idxmax(), "strike"]))
    return pd.DataFrame(rows).set_index(["time_et", "symbol"]).sort_index()


def run(b, t1, entry, stop, target, long):
    hi, lo, cl, tm = b.high.values, b.low.values, b.close.values, b.time_et.values
    k = np.searchsorted(tm, np.datetime64(t1))
    exit_px, how = cl[-1], "close"
    for m in range(k + 1, len(cl)):
        if (long and lo[m] <= stop) or (not long and hi[m] >= stop):
            exit_px, how = stop, "stop"
            break
        if (long and hi[m] >= target) or (not long and lo[m] <= target):
            exit_px, how = target, "target"
            break
    risk = abs(entry - stop)
    pnl = (exit_px - entry) if long else (entry - exit_px)
    return dict(exit=exit_px, exit_how=how, reward_risk=abs(target - entry) / risk,
                r=pnl / risk, r_net=(pnl - COST * entry) / risk)


def trades(views, day_bars, scale):
    times = views.index.get_level_values(0).unique().sort_values()
    out = []
    for t0, t1 in zip(times[:-1], times[1:]):
        if t1 - t0 != pd.Timedelta(minutes=30):
            continue
        try:
            v0, v1 = views.loc[t0], views.loc[t1]
        except KeyError:
            continue
        common = v0.index.intersection(v1.index)
        king_up = int((v1.king[common] > v0.king[common]).sum())
        king_dn = int((v1.king[common] < v0.king[common]).sum())
        floor_up = int((v1.floor[common] > v0.floor[common]).sum())
        ceil_dn = int((v1.ceiling[common] < v0.ceiling[common]).sum())
        for sym in common:
            sc = scale_at(scale, sym, t0)
            b = day_bars.get((sym, t0.normalize()))
            if np.isnan(sc) or b is None:
                continue
            k0, k1, spot1 = v0.king[sym], v1.king[sym], v1.spot[sym]
            u = sc * k0
            w = b[(b.time_et > t0) & (b.time_et <= t1)]
            if w.empty:
                continue
            for long in (True, False):
                tap = (w.low <= k0 + TAP * u) if long else (w.high >= k0 - TAP * u)
                if not tap.any():
                    continue
                after = w[tap.values.argmax():]
                rejected = spot1 >= k0 + CONFIRM * u if long else spot1 <= k0 - CONFIRM * u
                if not rejected:
                    continue
                wick = after.low.min() if long else after.high.max()
                stop = wick - BUFFER * u if long else wick + BUFFER * u
                if (long and k1 <= spot1) or (not long and k1 >= spot1):
                    continue                                    # no King ahead to target
                res = run(b, t1, spot1, stop, k1, long)
                out.append(dict(symbol=sym, map_time=t1, side="long" if long else "short",
                                king_before=k0, king_after=k1, entry=spot1, stop=stop, target=k1,
                                kings_rolled=king_up if long else king_dn,
                                floors_rolled=floor_up if long else ceil_dn, **res))
    return pd.DataFrame(out)


def summarize(t):
    splits = [
        ("King rejection only", slice(None)),
        ("+ Kings rolled 2 of 3", t.kings_rolled >= 2),
        ("+ floors rolled 2 of 3", t.floors_rolled >= 2),
        ("+ Kings and floors rolled 2 of 3 (your setup)", (t.kings_rolled >= 2) & (t.floors_rolled >= 2)),
        ("+ Kings rolled all 3 and floors 2 of 3", (t.kings_rolled == 3) & (t.floors_rolled >= 2)),
        ("your setup, target >= 1R away", (t.kings_rolled >= 2) & (t.floors_rolled >= 2) & (t.reward_risk >= 1)),
    ]
    rows = []
    for name, mask in splits:
        x = t[mask]
        lo, hi = day_ci(x)
        rows.append(dict(filter=name, trades=len(x), days=x.map_time.dt.date.nunique(),
                         win_rate=(x.r > 0).mean(), target_rate=(x.exit_how == "target").mean(),
                         median_reward_risk=x.reward_risk.median(), mean_r_net=x.r_net.mean(),
                         ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def main():
    g, p, day_bars = prepare()
    scale = typical_range(p)
    t = trades(boards(g, day_bars), day_bars, scale)
    s = summarize(t)
    t.to_csv(OUT / "king_reject_trades.csv", index=False)
    s.round(4).to_csv(OUT / "king_reject_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print("\nby symbol, your setup:")
    y = t[(t.kings_rolled >= 2) & (t.floors_rolled >= 2)]
    print(y.groupby(["symbol", "side"]).agg(trades=("r", "size"), win_rate=("r", lambda s: (s > 0).mean()),
                                            mean_r_net=("r_net", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
