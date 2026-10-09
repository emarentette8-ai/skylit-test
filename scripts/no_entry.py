#!/usr/bin/env python3
"""Re-run every setup with no-entry rules taken from the loser analysis.

Rules (a trade is skipped if any applies):
  late        entry at or after 15:00 ET
  king_heavy  SPY/SPXW/QQQ only: the King's share of the map's total |gamma| (latest
              map at entry) is in the top quarter of index maps; the cut-off is set on
              Jul-Sep index maps and reused unchanged for Oct-Jun
  tight_stop  stop closer than 0.4 typical 30-minute ranges to the entry
  gap_toward  trading toward an unfilled opening gap
Reported per period: all trades, each rule alone, all rules together, and the
same number of trades removed at random (500 draws) as the baseline for "removing
a losing slice always helps a bit".

Usage: python3 scripts/no_entry.py  (after filters.py flags for both periods)
Writes results/tables/oos/no_entry.csv.
"""
import numpy as np
import pandas as pd

from filters import latest
from load import _ROOT
from losers import PER, build
from stats import day_ci

INDEX = {"SPY", "SPXW", "QQQ"}
RULES = ["late", "king_heavy", "tight_stop", "gap_toward"]


def king_share(data):
    g = pd.read_parquet(_ROOT / f"data/raw/{data}/gamma_snapshots.parquet")
    g["symbol"] = g.symbol.astype(str)
    g = g[g.symbol.isin(INDEX)]
    tot = g.assign(a=g.net_gamma.abs()).groupby(["symbol", "time_et"]).a.sum()
    k = g[g.node_type == "king"].set_index(["symbol", "time_et"]).net_gamma.abs()
    return (k / tot).rename("king_share").reset_index()


def tag(f, ks, cut):
    by = {s: x.sort_values("time_et") for s, x in ks.groupby("symbol")}
    share = []
    for r in f.itertuples():
        x = by.get(r.symbol)
        i = latest(x.time_et.values, r.entry_time) if x is not None else None
        share.append(x.king_share.values[i] if i is not None else np.nan)
    f = f.copy()
    f["king_share"] = share
    f["late"] = f.entry_time.dt.hour >= 15
    f["king_heavy"] = f.king_share >= cut
    f["tight_stop"] = f.stop_dist < 0.4
    f["gap_toward"] = f.gap_toward.astype(float).fillna(0).astype(bool)
    return f


def row(name, per, x):
    lo, hi = day_ci(x, "entry_time")
    return dict(period=per, case=name, trades=len(x), win_rate=(x.r > 0).mean(), mean_r_net=x.r_net.mean(),
                ci95_low=lo, ci95_high=hi)


def main():
    F = {per: build(*v) for per, v in PER.items()}
    KS = {per: king_share(v[1]) for per, v in PER.items()}
    cut = KS["jul-sep"].king_share.quantile(0.75)
    rows = []
    rng = np.random.default_rng(0)
    for per, f in F.items():
        f = tag(f, KS[per], cut)
        skip = f[RULES].any(axis=1)
        for scope, x in (("all setups", f), ("indexes", f[f.symbol.isin(INDEX)])):
            sk = skip[x.index]
            rows.append(dict(row("all trades", per, x), scope=scope))
            for r in RULES:
                rows.append(dict(row(f"without {r} ({int(x[r].sum())} skipped)", per, x[~x[r]]), scope=scope))
            kept = x[~sk]
            rows.append(dict(row(f"ALL RULES ({int(sk.sum())} skipped)", per, kept), scope=scope))
            rnd = [x.r_net.values[rng.choice(len(x), len(kept), replace=False)].mean() for _ in range(500)]
            rows.append(dict(period=per, scope=scope, case=f"random removal of {int(sk.sum())}",
                             trades=len(kept), mean_r_net=np.mean(rnd),
                             ci95_low=np.percentile(rnd, 2.5), ci95_high=np.percentile(rnd, 97.5)))
            if scope == "all setups":
                for fam, y in kept.groupby("family"):
                    rows.append(dict(row(f"ALL RULES, {fam}", per, y), scope=scope))
    out = pd.DataFrame(rows)[["period", "scope", "case", "trades", "win_rate", "mean_r_net", "ci95_low", "ci95_high"]]
    out.round(4).to_csv(_ROOT / "results/tables/oos/no_entry.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(f"king_heavy cut-off (Jul-Sep index maps, 75th pct of King share): {cut:.3f}")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
