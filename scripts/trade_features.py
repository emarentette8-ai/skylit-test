#!/usr/bin/env python3
"""Compare the Heatseeker map behind winning and losing Rug-family trades.

For each trade in results/tables/rug_trades.csv, rebuild the map it came from
and measure, in the trade's direction (positive = toward profit):
  king_dist_pct    King distance from entry, % of price
  king_vs_target   King distance / target distance (1 = King at the target)
  king_zone        behind entry / between entry and target / beyond target
  king_is_level    the traded level is itself the King
  level_node_pct   traded level size as % of the King (Skylit's Node %)
  king_share       King |gamma| / total |gamma| on the board (concentration)
  next_node_pct    distance to the first named node in the trade direction, %
  named_nearby     named nodes within 1% of price (map clutter)
  local_net_share  net / gross gamma within 2% (regime)
  hour             map hour
Winners are trades with r_net > 0.

Usage: python3 scripts/trade_features.py  (run rug.py first)
Writes results/tables/trade_features.csv and trade_features_summary.csv.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from load import TABLES, gamma as load_gamma
from node_claims import NAMED

OUT = TABLES
FEATURES = ["king_dist_pct", "king_vs_target", "king_is_level", "level_node_pct",
            "king_share", "next_node_pct", "named_nearby", "local_net_share", "hour"]


def features(tr, g):
    maps = {k: v for k, v in g.groupby(["symbol", "time_et"], observed=True)}
    rows = []
    for t in tr.itertuples():
        s = maps[(t.symbol, t.map_time)]
        d = -1 if t.setup.startswith("rug") else 1          # short = -1
        king = s.loc[s.node_type == "king"].iloc[0]
        lvl = s.loc[(s.strike - t.level).abs().idxmin()]
        kd = d * (king.strike - t.entry)
        td = abs(t.target - t.entry)
        ahead = s[(d * (s.strike - t.entry) > 0) & s.node_type.isin(NAMED)]
        near = s[(s.strike - t.entry).abs() / t.entry < 0.02]
        rows.append(dict(
            king_dist_pct=100 * kd / t.entry,
            king_vs_target=kd / td if td else np.nan,
            king_zone="behind" if kd <= 0 else ("between" if kd <= td else "beyond"),
            king_is_level=float(king.strike == t.level),
            level_node_pct=100 * abs(lvl.net_gamma) / abs(king.net_gamma),
            king_share=abs(king.net_gamma) / s.net_gamma.abs().sum(),
            next_node_pct=100 * (d * (ahead.strike - t.entry)).min() / t.entry if len(ahead) else np.nan,
            named_nearby=int(s[(s.strike - t.entry).abs() / t.entry < 0.01]
                             .node_type.isin(NAMED).sum()),
            local_net_share=near.net_gamma.sum() / near.net_gamma.abs().sum(),
            hour=t.map_time.hour,
            king_sign="+" if king.net_gamma > 0 else "-"))
    return pd.concat([tr.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def compare(f, label):
    w, l = f[f.r_net > 0], f[f.r_net <= 0]
    rows = []
    for c in FEATURES:
        a, b = w[c].dropna(), l[c].dropna()
        p = mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
        rows.append(dict(group=label, feature=c, win_median=a.median(), loss_median=b.median(),
                         win_mean=a.mean(), loss_mean=b.mean(), p_value=p))
    return pd.DataFrame(rows)


def main():
    tr = pd.read_csv(OUT / "rug_trades.csv", parse_dates=["map_time"])
    g = load_gamma()
    g["symbol"] = g.symbol.astype(str)
    g = g[g.symbol.isin(tr.symbol.unique())]
    f = features(tr, g)
    f.to_csv(OUT / "trade_features.csv", index=False)
    setups = f[~f.setup.str.endswith("control")]
    summary = pd.concat([compare(setups, "rug+reverse_rug"), compare(f, "all incl. controls")])
    summary.round(4).to_csv(OUT / "trade_features_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    for name, df in (("rug+reverse_rug", setups), ("all incl. controls", f)):
        print(f"\n== King position vs target ({name})")
        print(df.groupby("king_zone").agg(n=("r_net", "size"), win_rate=("r_net", lambda s: (s > 0).mean()),
                                          mean_r_net=("r_net", "mean")).round(3))
        print(df.groupby("king_sign").agg(n=("r_net", "size"), win_rate=("r_net", lambda s: (s > 0).mean()),
                                          mean_r_net=("r_net", "mean")).round(3))


if __name__ == "__main__":
    main()
