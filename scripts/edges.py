#!/usr/bin/env python3
"""Rolling floors/ceilings and trading from the edges (Skylit Academy range fade).

On every 30-minute map: floor = largest |gamma| node below price, ceiling =
largest above (Academy "map the board" step 2). Rolling = the floor moved to a
higher strike (bullish) or the ceiling to a lower one (bearish) since the
previous map that day.

Edge trade, in the 30 minutes a map is current: price taps the floor (ceiling)
within 0.1 typical range, then a 1-minute close 0.25 typical range back off it
confirms the bounce: buy the floor / sell the ceiling. Stop 0.5 typical range
beyond the edge, target the floor-ceiling midpoint, else exit at the close.
One open trade per symbol. Trades are split by:
  roll     with the roll / against it / no roll / first map of the day
  sign     edge node positive (yellow) or negative (purple)
  size     edge node >= 20% of the King (the image's "scenery" cut-off)
Also: does a roll predict the rest of the day's direction?

Usage: python3 scripts/edges.py  (SKYLIT_DATA=oos for the new data)
Writes edges_trades.csv, edges_summary.csv and edges_roll_direction.csv.
"""
import numpy as np
import pandas as pd

from load import TABLES
from node_claims import INDEXES, prepare, scale_at, typical_range
from stats import day_ci

OUT = TABLES
TAP, CONFIRM, STOP, COST = 0.1, 0.25, 0.5, 0.0002


def board(g, day_bars):
    """One row per map: spot, floor, ceiling (strike/value), King, roll vs previous map."""
    rows = []
    for (sym, t), s in g.groupby(["symbol", "time_et"]):
        b = day_bars.get((sym, t.normalize()))
        if b is None:
            continue
        at = b[b.time_et >= t]
        if at.empty:
            continue
        spot = at.close.iloc[0]
        below, above = s[s.strike < spot], s[s.strike > spot]
        if below.empty or above.empty:
            continue
        f = below.loc[below.net_gamma.abs().idxmax()]
        c = above.loc[above.net_gamma.abs().idxmax()]
        k = s.loc[s.node_type == "king"].iloc[0]
        rows.append(dict(symbol=sym, time_et=t, date=t.normalize(), spot=spot,
                         floor=f.strike, floor_val=f.net_gamma, ceiling=c.strike, ceiling_val=c.net_gamma,
                         king_abs=abs(k.net_gamma)))
    m = pd.DataFrame(rows).sort_values(["symbol", "time_et"])
    prev = m.groupby(["symbol", "date"])[["floor", "ceiling"]].shift()
    up = m.floor > prev.floor
    down = m.ceiling < prev.ceiling
    m["roll"] = np.select([prev.floor.isna(), up & ~down, down & ~up], ["first map", "bullish", "bearish"],
                          "none/mixed")
    return m


def simulate(b, start, end, lvl, long, target, u):
    hi, lo, cl, tm = b.high.values, b.low.values, b.close.values, b.time_et.values
    i, stop_i = np.searchsorted(tm, np.datetime64(start)), np.searchsorted(tm, np.datetime64(end))
    tapped = False
    for k in range(i + 1, min(stop_i, len(cl))):
        if not tapped:
            tapped = lo[k] <= lvl + TAP * u if long else hi[k] >= lvl - TAP * u
            continue
        if (long and cl[k] >= lvl + CONFIRM * u) or (not long and cl[k] <= lvl - CONFIRM * u):
            entry = cl[k]
            stop = lvl - STOP * u if long else lvl + STOP * u
            if (long and target <= entry) or (not long and target >= entry):
                return None                              # already past the midpoint
            exit_px, how, exit_t = cl[-1], "close", tm[-1]
            for m in range(k + 1, len(cl)):
                if (long and lo[m] <= stop) or (not long and hi[m] >= stop):
                    exit_px, how, exit_t = stop, "stop", tm[m]
                    break
                if (long and hi[m] >= target) or (not long and lo[m] <= target):
                    exit_px, how, exit_t = target, "target", tm[m]
                    break
            risk = abs(entry - stop)
            pnl = (exit_px - entry) if long else (entry - exit_px)
            return dict(entry_time=tm[k], entry=entry, stop=stop, target=target, exit=exit_px,
                        exit_how=how, exit_time=exit_t, reward_risk=abs(target - entry) / risk,
                        r=pnl / risk, r_net=(pnl - COST * entry) / risk)
        if (long and lo[k] < lvl - STOP * u) or (not long and hi[k] > lvl + STOP * u):
            return None                                  # broke through before confirming
    return None


def trades(m, day_bars, scale):
    out, busy = [], {}
    for r in m.itertuples():
        if busy.get(r.symbol, pd.Timestamp.min) > r.time_et:
            continue
        sc = scale_at(scale, r.symbol, r.date)
        if np.isnan(sc):
            continue
        b = day_bars[(r.symbol, r.date)]
        mid = (r.floor + r.ceiling) / 2
        u = sc * r.spot
        best = None
        for long, lvl, val in ((True, r.floor, r.floor_val), (False, r.ceiling, r.ceiling_val)):
            tr = simulate(b, r.time_et, r.time_et + pd.Timedelta(minutes=30), lvl, long, mid, u)
            if tr and (best is None or tr["entry_time"] < best["entry_time"]):
                roll_dir = {"bullish": 1, "bearish": -1}.get(r.roll, 0)
                d = 1 if long else -1
                best = dict(symbol=r.symbol, index=r.symbol in INDEXES, map_time=r.time_et,
                            side="long@floor" if long else "short@ceiling", level=lvl,
                            edge_sign="+" if val > 0 else "-", edge_pct_king=100 * abs(val) / r.king_abs,
                            roll=r.roll,
                            vs_roll=("first map" if r.roll == "first map" else
                                     "no roll" if roll_dir == 0 else
                                     "with roll" if roll_dir == d else "against roll"), **tr)
        if best:
            out.append(best)
            busy[r.symbol] = pd.Timestamp(best["exit_time"])
    return pd.DataFrame(out)


def summarize(t):
    rows = []
    for scope, df in (("all symbols", t), ("indexes", t[t["index"]])):
        splits = [("all edge trades", slice(None))]
        splits += [(f"roll: {v}", df.vs_roll == v) for v in ("no roll", "with roll", "against roll", "first map")]
        splits += [("edge positive (yellow)", df.edge_sign == "+"), ("edge negative (purple)", df.edge_sign == "-"),
                   ("edge >= 20% of King", df.edge_pct_king >= 20), ("edge < 20% of King", df.edge_pct_king < 20),
                   ("positive, >=20% King, not against roll",
                    (df.edge_sign == "+") & (df.edge_pct_king >= 20) & (df.vs_roll != "against roll"))]
        for name, mask in splits:
            x = df[mask]
            lo, hi = day_ci(x)
            rows.append(dict(scope=scope, split=name, trades=len(x), win_rate=(x.r > 0).mean(),
                             target_rate=(x.exit_how == "target").mean(), mean_r_net=x.r_net.mean(),
                             ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def roll_direction(m, day_bars):
    """After a roll, how often does the close end on the roll's side of the map price?"""
    rows = []
    for r in m[m.roll.isin(["bullish", "bearish"])].itertuples():
        close = day_bars[(r.symbol, r.date)].close.iloc[-1]
        d = 1 if r.roll == "bullish" else -1
        rows.append(dict(index=r.symbol in INDEXES, roll=r.roll, hour=r.time_et.hour,
                         followed=np.sign(close - r.spot) == d, move_pct=100 * d * (close - r.spot) / r.spot))
    x = pd.DataFrame(rows)
    return (x.groupby(["index", "roll"]).agg(maps=("followed", "size"), close_followed_roll=("followed", "mean"),
                                             mean_move_pct=("move_pct", "mean")).reset_index())


def main():
    g, p, day_bars = prepare()
    scale = typical_range(p)
    m = board(g, day_bars)
    t = trades(m, day_bars, scale)
    s, rd = summarize(t), roll_direction(m, day_bars)
    t.to_csv(OUT / "edges_trades.csv", index=False)
    s.round(4).to_csv(OUT / "edges_summary.csv", index=False)
    rd.round(4).to_csv(OUT / "edges_roll_direction.csv", index=False)
    pd.set_option("display.width", 200)
    print("maps by roll state:", m.roll.value_counts().to_dict())
    print(s.round(3).to_string(index=False))
    print("\n== Does a roll predict the close?\n" + rd.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
