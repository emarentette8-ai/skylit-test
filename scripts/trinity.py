#!/usr/bin/env python3
"""Does SPXW / SPY / QQQ agreement ("Trinity") improve index trades?

For every index trade, count how many of the three index maps agree with the
trade's direction at the map current at entry (the traded index included):
  setup  the map shows the same setup (Rug for shorts, Reverse Rug for longs)
         using rug.py's classify()
  king   the King sits on the trade's side of price (below for shorts,
         above for longs)
"Aligned" = at least 2 of 3 agree; "all 3" is also reported.
Trades: Rug-family trades (rug.py) and index OTE trades (ote.py).

Usage: python3 scripts/trinity.py  (SKYLIT_DATA=oos for the new data;
       run rug.py and ote.py first)
Writes trinity_flags.csv and trinity_summary.csv to the tables folder.
"""
import numpy as np
import pandas as pd

from load import TABLES
from node_claims import prepare, scale_at, typical_range
from rug import WINDOW, classify

OUT = TABLES
TRIO = ("SPXW", "SPY", "QQQ")


def map_views(g, day_bars, scale):
    """Per (symbol, map time): bearish/bullish setup flags and King side."""
    rows = []
    for (sym, t), s in g[g.symbol.isin(TRIO)].groupby(["symbol", "time_et"]):
        b = day_bars.get((sym, t.normalize()))
        if b is None:
            continue
        at = b[b.time_et >= t]
        if at.empty:
            continue
        spot, sc = at.close.iloc[0], scale_at(scale, sym, t)
        if np.isnan(sc):
            continue
        setups = {name for name, _ in classify(s, spot, WINDOW * sc)}
        king = s.loc[s.node_type == "king", "strike"].iloc[0]
        rows.append(dict(symbol=sym, time_et=t, rug="rug" in setups,
                         reverse_rug="reverse_rug" in setups, king_side=np.sign(king - spot)))
    return pd.DataFrame(rows).set_index(["time_et", "symbol"]).sort_index()


def agreement(views, t, direction):
    """(setup votes, king votes) out of the three indexes at map time t."""
    if t not in views.index.get_level_values(0):
        return np.nan, np.nan
    v = views.loc[t]
    setup = (v.rug if direction < 0 else v.reverse_rug).sum()
    king = (v.king_side == direction).sum()
    return setup, king


def flag(trades, views, time_col, direction_of):
    times = views.index.get_level_values(0).unique().sort_values()
    out = []
    for r in trades.itertuples():
        t = getattr(r, time_col)
        i = times.searchsorted(t, side="right") - 1
        if i < 0 or times[i].normalize() != t.normalize():
            out.append((np.nan, np.nan))
            continue
        out.append(agreement(views, times[i], direction_of(r)))
    trades = trades.copy()
    trades["setup_votes"], trades["king_votes"] = zip(*out) if out else ([], [])
    return trades.dropna(subset=["king_votes"])


def summarize(df, name):
    rows = []
    for col in ("setup_votes", "king_votes"):
        for label, mask in [("all trades", slice(None)), ("aligned 2 of 3", df[col] >= 2),
                            ("all 3", df[col] == 3), ("not aligned", df[col] < 2)]:
            x = df[mask]
            rows.append(dict(trades_from=name, alignment=col.replace("_votes", ""), filter=label,
                             trades=len(x), win_rate=(x.r > 0).mean() if len(x) else np.nan,
                             mean_r_net=x.r_net.mean() if len(x) else np.nan))
    return rows


def main():
    g, p, day_bars = prepare()
    scale = typical_range(p)
    views = map_views(g, day_bars, scale)

    rug = pd.read_csv(OUT / "rug_trades.csv", parse_dates=["map_time", "entry_time"])
    rug = flag(rug, views, "entry_time", lambda r: -1 if r.setup.startswith("rug") else 1)
    ote = pd.read_csv(OUT / "ote_trades.csv", parse_dates=["entry_time"])
    ote = ote[ote.symbol.isin(TRIO)]
    ote = flag(ote, views, "entry_time", lambda r: -1 if r.side == "short" else 1)

    setups = rug[~rug.setup.str.endswith("control")]
    rows = (summarize(setups, "Rug + Reverse Rug") + summarize(rug, "Rug family incl. controls")
            + summarize(ote, "OTE on indexes"))
    s = pd.DataFrame(rows)
    rug.assign(source="rug").to_csv(OUT / "trinity_flags.csv", index=False)
    s.round(4).to_csv(OUT / "trinity_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
