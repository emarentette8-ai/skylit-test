#!/usr/bin/env python3
"""Rugs and Reverse Rugs whose level is not the King, with different targets.

Setups come from rug.classify() (positive named ceiling with negative gamma and
no floor below = Rug, short; mirror = Reverse Rug, long). Only levels that are
NOT the map's King are kept. Entry and stop are rug.py's: tap the level, a
1-minute close 0.25 typical range back off it, stop 0.5 typical range beyond it.
The same entries are then run to three targets:
  king    the map's King, when it lies beyond the entry in the trade direction
  swing   the latest confirmed 15-minute swing low (shorts) / high (longs) beyond
          the entry, else the session low / high so far if beyond the entry
  3R      rug.py's fixed 3x-risk target, for comparison
Else exit at the close; a bar hitting stop and target counts as the stop.
One open trade per symbol, setup and target at a time.

Usage: python3 scripts/rug_targets.py  (SKYLIT_DATA=oos for the new data)
Writes rug_targets_trades.csv and rug_targets_summary.csv to the tables folder.
"""
import numpy as np
import pandas as pd

from load import TABLES
from node_claims import INDEXES, prepare, scale_at, typical_range
from ote import pivots
from rug import CONFIRM, COST, STOP, TAP, WINDOW, classify
from stats import day_ci

OUT = TABLES
TARGETS = ("king", "swing", "3R")


def entry_for(b, start, end, lvl, short, u):
    """rug.py's trigger: tap the level, then a confirming close. Returns (index, entry, stop)."""
    hi, lo, cl = b.high.values, b.low.values, b.close.values
    tm = b.time_et.values
    i, j = np.searchsorted(tm, np.datetime64(start)), np.searchsorted(tm, np.datetime64(end))
    tapped = False
    for k in range(i + 1, min(j, len(cl))):
        if not tapped:
            tapped = hi[k] >= lvl - TAP * u if short else lo[k] <= lvl + TAP * u
            continue
        if (short and cl[k] <= lvl - CONFIRM * u) or (not short and cl[k] >= lvl + CONFIRM * u):
            return k, cl[k], (lvl + STOP * u if short else lvl - STOP * u)
    return None


def swing_target(piv, b, k, entry, short):
    t = pd.Timestamp(b.time_et.values[k])
    kind = "L" if short else "H"
    beyond = [p[3] for p in piv if p[0] <= t and p[2] == kind
              and ((short and p[3] < entry) or (not short and p[3] > entry))]
    if beyond:
        return beyond[-1], "swing"
    done = b.iloc[:k + 1]
    ext = done.low.min() if short else done.high.max()
    if (short and ext < entry) or (not short and ext > entry):
        return ext, "session"
    return None, None


def run_to(b, k, entry, stop, target, short):
    hi, lo, cl, tm = b.high.values, b.low.values, b.close.values, b.time_et.values
    exit_px, how, exit_t = cl[-1], "close", tm[-1]
    for m in range(k + 1, len(cl)):
        if (short and hi[m] >= stop) or (not short and lo[m] <= stop):
            exit_px, how, exit_t = stop, "stop", tm[m]
            break
        if (short and lo[m] <= target) or (not short and hi[m] >= target):
            exit_px, how, exit_t = target, "target", tm[m]
            break
    risk = abs(stop - entry)
    pnl = (entry - exit_px) if short else (exit_px - entry)
    return dict(exit=exit_px, exit_how=how, exit_time=exit_t, reward_risk=abs(target - entry) / risk,
                r=pnl / risk, r_net=(pnl - COST * entry) / risk)


def trades(g, day_bars, scale):
    out, busy, piv_cache = [], {}, {}
    for (sym, t), s in g.sort_values("time_et").groupby(["symbol", "time_et"], sort=True):
        b = day_bars.get((sym, t.normalize()))
        sc = scale_at(scale, sym, t)
        if b is None or np.isnan(sc) or b[b.time_et >= t].empty:
            continue
        spot = b[b.time_et >= t].close.iloc[0]
        king = s.loc[s.node_type == "king"].iloc[0]
        for setup, lvl in classify(s, spot, WINDOW * sc):
            if setup not in ("rug", "reverse_rug") or lvl == king.strike:
                continue
            short = setup == "rug"
            u = sc * lvl
            e = entry_for(b, t, t + pd.Timedelta(minutes=30), lvl, short, u)
            if e is None:
                continue
            k, entry, stop = e
            risk = abs(stop - entry)
            key = (sym, t.normalize())
            if key not in piv_cache:
                piv_cache[key] = pivots(b)
            swing, swing_src = swing_target(piv_cache[key], b, k, entry, short)
            king_ok = (short and king.strike < entry) or (not short and king.strike > entry)
            goals = {"king": king.strike if king_ok else None, "swing": swing,
                     "3R": entry - 3 * risk if short else entry + 3 * risk}
            for name, target in goals.items():
                if target is None or busy.get((sym, setup, name), pd.Timestamp.min) > t:
                    continue
                res = run_to(b, k, entry, stop, target, short)
                busy[(sym, setup, name)] = pd.Timestamp(res["exit_time"])
                out.append(dict(symbol=sym, index=sym in INDEXES, map_time=t, setup=setup,
                                level=lvl, level_type=s.loc[s.strike == lvl, "node_type"].iloc[0],
                                king=king.strike, king_sign="+" if king.net_gamma > 0 else "-",
                                target_kind=name, target_source=swing_src if name == "swing" else name,
                                entry_time=b.time_et.values[k], entry=entry, stop=stop, target=target,
                                **res))
    return pd.DataFrame(out)


def summarize(t):
    rows = []
    for scope, df in (("indexes", t[t["index"]]), ("all symbols", t)):
        for setup in ("rug", "reverse_rug", "both"):
            x0 = df if setup == "both" else df[df.setup == setup]
            for name in TARGETS:
                x = x0[x0.target_kind == name]
                lo, hi = day_ci(x)
                rows.append(dict(scope=scope, setup=setup, target=name, trades=len(x),
                                 win_rate=(x.r > 0).mean(), target_rate=(x.exit_how == "target").mean(),
                                 median_reward_risk=x.reward_risk.median(), mean_r_net=x.r_net.mean(),
                                 ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def main():
    g, p, day_bars = prepare()
    scale = typical_range(p)
    t = trades(g, day_bars, scale)
    s = summarize(t)
    t.to_csv(OUT / "rug_targets_trades.csv", index=False)
    s.round(4).to_csv(OUT / "rug_targets_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print("\nlevel types traded:", t.drop_duplicates(["symbol", "map_time", "setup"]).level_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
