#!/usr/bin/env python3
"""Test three Skylit Academy claims about Heatseeker nodes on the watchlist data.

A. King magnet: does the close end nearer the King than price at the snapshot?
   Control: a mirror level the same distance away on the other side of price.
B. Regime: is the next hour's range smaller under positive local gamma?
C. Node holds: when price touches a level, how often does it hold, by node
   type and by touch number, versus ordinary strikes?

Usage: python3 scripts/node_claims.py
Writes results/tables/node_claims_{king,regime,touch,taps}.csv.
"""

import numpy as np
import pandas as pd

from load import TABLES, bars as load_bars, gamma as load_gamma

OUT = TABLES
INDEXES = {"SPY", "QQQ", "SPXW"}
NAMED = ("king", "gatekeeper", "pika", "barney")


def prepare():
    g, p = load_gamma(), load_bars()
    g["symbol"] = g.symbol.astype(str)
    p["symbol"] = p.symbol.astype(str)
    g = g[g.symbol != "VIX"].copy()
    g["date"] = g.time_et.dt.normalize()
    p["date"] = p.time_et.dt.normalize()
    day_bars = {k: v.sort_values("time_et").reset_index(drop=True)
                for k, v in p.groupby(["symbol", "date"])}
    return g, p, day_bars


LOOKBACK, MIN_DAYS = 20, 5


def typical_range(p):
    """Typical 30-minute high-low range as a fraction of price, per (symbol, date).

    Trailing: the median of the previous LOOKBACK days' daily medians, so a day
    is scaled only with information from before it. Days with fewer than
    MIN_DAYS prior days have no value and are skipped by the studies.
    """
    def one(v):
        r = (v.set_index("time_et").resample("30min")
             .agg({"high": "max", "low": "min", "close": "last"}).dropna())
        return ((r.high - r.low) / r.close).median()
    daily = p.groupby(["symbol", "date"]).apply(one)
    return (daily.groupby(level="symbol", group_keys=False)
            .apply(lambda s: s.rolling(LOOKBACK, min_periods=MIN_DAYS).median().shift(1))
            .dropna())


def scale_at(scale, sym, date):
    """Typical range for a symbol on a date, NaN if there is not enough history."""
    return scale.get((sym, pd.Timestamp(date).normalize()), np.nan)


def snapshot_table(g, day_bars, scale):
    rows = []
    for (sym, t), s in g.groupby(["symbol", "time_et"]):
        b = day_bars.get((sym, t.normalize()))
        if b is None:
            continue
        fut = b[b.time_et >= t]
        if len(fut) < 30:
            continue
        sc = scale_at(scale, sym, t)
        if np.isnan(sc):
            continue
        spot = fut.close.iloc[0]
        king = s.loc[s.node_type == "king"].iloc[0]
        near = s[(s.strike - spot).abs() / spot < 0.02]
        nxt = fut.iloc[:60]
        rows.append(dict(
            symbol=sym, time_et=t, spot=spot, close=b.close.iloc[-1],
            king=king.strike, king_gamma=king.net_gamma,
            local_net=near.net_gamma.sum(), local_abs=near.net_gamma.abs().sum(),
            range60=(nxt.high.max() - nxt.low.min()) / spot / sc))
    r = pd.DataFrame(rows)
    r["index"] = r.symbol.isin(INDEXES)
    return r


def king_magnet(r):
    r = r.assign(dist=(r.king - r.spot) / r.spot)
    a = r[(r.dist.abs() > 0.002) & (r.dist.abs() < 0.03)].copy()
    mirror = 2 * a.spot - a.king
    a["toward_king"] = (a.close - a.king).abs() < (a.spot - a.king).abs()
    a["toward_mirror"] = (a.close - mirror).abs() < (a.spot - mirror).abs()
    a["hour"] = a.time_et.dt.hour
    a["king_sign"] = np.where(a.king_gamma > 0, "+", "-")
    agg = dict(n=("toward_king", "size"), toward_king=("toward_king", "mean"),
               toward_mirror=("toward_mirror", "mean"))
    out = pd.concat([a.groupby(k).agg(**agg).reset_index().rename(columns={k: "group"})
                     .assign(by=k) for k in ("index", "hour", "king_sign")])
    pin = r[(r.time_et.dt.hour == 15) & (r.dist.abs() < 0.01)]
    pinned = ((pin.close - pin.king).abs() / pin.spot < 0.0025).mean()
    mirror_pinned = ((pin.close - (2 * pin.spot - pin.king)).abs() / pin.spot < 0.0025).mean()
    out = pd.concat([out, pd.DataFrame([dict(by="pin_15:00_within_0.25pct", group="all",
                                             n=len(pin), toward_king=pinned,
                                             toward_mirror=mirror_pinned)])])
    return out[["by", "group", "n", "toward_king", "toward_mirror"]]


def regime(r):
    r = r.assign(bucket=pd.cut(r.local_net / r.local_abs, [-1, -0.3, 0, 0.3, 1]).astype(str))
    return (r.groupby(["index", "bucket"]).range60
            .agg(n="size", median="median", mean="mean").reset_index())


def touches(g, day_bars, scale):
    """Hold = price moves half a typical range back before half a range through.

    Levels are taken from the first snapshot of the day that lists the strike
    0.5-3 typical ranges from price. Up to 3 touches per level; a break ends it.
    """
    g = g.join(scale.rename("scale"), on=["symbol", "date"])
    g["dz"] = (g.strike - g.spot).abs() / g.spot / g.scale
    c = (g[(g.dz > 0.5) & (g.dz < 3)].sort_values("time_et")
         .drop_duplicates(["symbol", "date", "strike"]))
    c["node"] = np.where(c.node_type.isin(NAMED), c.node_type.astype(str), "normal")
    out = []
    for (sym, d), s in c.groupby(["symbol", "date"]):
        b = day_bars.get((sym, d))
        if b is None:
            continue
        hi, lo, cl, tm = b.high.values, b.low.values, b.close.values, b.time_et.values
        for row in s.itertuples():
            x = 0.5 * row.scale * row.strike
            i = np.searchsorted(tm, np.datetime64(row.time_et))
            if i >= len(cl):
                continue
            below = cl[i] < row.strike
            lvl, tap, armed, i = row.strike, 0, True, i + 1
            while i < len(cl) and tap < 3:
                if armed and lo[i] <= lvl <= hi[i]:
                    tap += 1
                    held, j = None, i + 1
                    for j in range(i + 1, len(cl)):
                        through = hi[j] >= lvl + x if below else lo[j] <= lvl - x
                        back = lo[j] <= lvl - x if below else hi[j] >= lvl + x
                        if through:
                            held = 0
                            break
                        if back:
                            held = 1
                            break
                    if held is not None:
                        out.append((sym, sym in INDEXES, row.node,
                                    "+" if row.net_gamma > 0 else "-", tap, held))
                    if held != 1:
                        break
                    armed, i = False, j
                    continue
                if not armed and abs(cl[i] - lvl) >= x:
                    armed = True
                i += 1
    return pd.DataFrame(out, columns=["symbol", "index", "node", "sign", "tap", "held"])


def main():
    g, p, day_bars = prepare()
    scale = typical_range(p)
    r = snapshot_table(g, day_bars, scale)
    t = touches(g, day_bars, scale)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)
    results = {
        "king": king_magnet(r),
        "regime": regime(r),
        "touch": (t[t.tap == 1].groupby(["index", "node", "sign"]).held
                  .agg(n="size", hold_rate="mean").reset_index()),
        "taps": (t.assign(named=t.node != "normal").groupby(["named", "tap"]).held
                 .agg(n="size", hold_rate="mean").reset_index()),
    }
    for name, df in results.items():
        df.round(4).to_csv(OUT / f"node_claims_{name}.csv", index=False)
        print(f"\n== {name}\n{df.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
