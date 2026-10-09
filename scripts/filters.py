#!/usr/bin/env python3
"""Score every trade family against every filter tested so far, plus supply/demand.

  python3 scripts/filters.py flags     # per period (SKYLIT_DATA=oos for Oct-Jun)
  python3 scripts/filters.py report    # both periods together

Families (from the study tables): rug (rug.py setups), nonking_rug_{king,swing,3R}
(rug_targets.py), ote, edge_fade, king_reject, twin_nodes. "pooled" = all of them
except the nonking_rug variants (same entries as rug, counted once).

Filters, each from information available at decision time:
  sr            price S/R within 0.25 typical range of the trade level (confluence.py)
  supply_demand fresh demand zone at the level for longs, supply for shorts (supply_demand.py)
  king_ahead    own King beyond the entry in the trade direction (latest map)
  king_negative own King has negative gamma
  level_not_king the traded level is not the King
  trinity_2of3  King on the trade's side of price on >= 2 of SPXW/SPY/QQQ (indexes only)
  ote_zone      entry inside the 61.8-78.6% retracement of the latest swing leg
  roll_with     the map's floor/ceiling rolled in the trade direction at the latest map
  gap_toward    trading toward an unfilled opening gap

Decision rule, fixed before the run: a filter WORKS for a family if, in BOTH periods,
it has >= 20 trades and raises both the win rate and the mean net R versus trades
without it. Working filters are then stacked (pairs, and all together).
Both periods are used to choose, so stacked results have no blind confirmation.
"""
import sys

import numpy as np
import pandas as pd

from confluence import NEAR, build_sr, near_sr
from edges import board
from load import TABLES, _ROOT
from node_claims import INDEXES, prepare, scale_at, typical_range
from ote import in_zone, last_leg, pivots
from stats import day_ci
from supply_demand import build as build_sd, fresh_zone
from trinity import map_views
from twin_nodes import gap_state, prev_closes

FILTERS = ["sr", "supply_demand", "king_ahead", "king_negative", "level_not_king",
           "trinity_2of3", "ote_zone", "roll_with", "gap_toward"]
MIN_N = 20
DIR = {"long": 1, "short": -1, "long@floor": 1, "short@ceiling": -1}


def load_trades():
    T = lambda f, **kw: pd.read_csv(TABLES / f, **kw)
    out = []
    r = T("rug_trades.csv", parse_dates=["map_time", "entry_time"])
    r = r[~r.setup.str.endswith("control")]
    out.append(r.assign(family="rug", direction=np.where(r.setup == "rug", -1, 1), t_dec=r.map_time))
    rt = T("rug_targets_trades.csv", parse_dates=["map_time", "entry_time"])
    for k in ("king", "swing", "3R"):
        x = rt[rt.target_kind == k]
        out.append(x.assign(family=f"nonking_rug_{k}", direction=np.where(x.setup == "rug", -1, 1), t_dec=x.map_time))
    o = T("ote_trades.csv", parse_dates=["entry_time"])
    out.append(o.assign(family="ote", direction=o.side.map(DIR), level=o.entry,
                        map_time=pd.NaT, t_dec=o.entry_time - pd.Timedelta(minutes=1)))
    e = T("edges_trades.csv", parse_dates=["map_time", "entry_time"])
    out.append(e.assign(family="edge_fade", direction=e.side.map(DIR), t_dec=e.map_time))
    k = T("king_reject_trades.csv", parse_dates=["map_time"])
    out.append(k.assign(family="king_reject", direction=k.side.map(DIR), level=k.king_before,
                        entry_time=k.map_time, t_dec=k.map_time - pd.Timedelta(minutes=30)))
    w = T("twin_nodes_trades.csv", parse_dates=["map_time", "entry_time"])
    out.append(w.assign(family="twin_nodes", direction=w.side.map(DIR), level=w.near, t_dec=w.map_time))
    cols = ["family", "symbol", "t_dec", "entry_time", "direction", "entry", "level", "r", "r_net"]
    return pd.concat([x[cols] for x in out], ignore_index=True)


def latest(times, t):
    i = times.searchsorted(np.datetime64(t), side="right") - 1
    return i if i >= 0 and pd.Timestamp(times[i]).normalize() == pd.Timestamp(t).normalize() else None


def flags():
    g, p, day_bars = prepare()
    scale = typical_range(p)
    sr, sd, pc = build_sr(p), build_sd(p, scale), prev_closes(p)
    kings = {s: x.sort_values("time_et") for s, x in g[g.node_type == "king"].groupby("symbol")}
    rolls = {s: x.sort_values("time_et") for s, x in board(g, day_bars).groupby("symbol")}
    views = map_views(g, day_bars, scale)
    vtimes = views.index.get_level_values(0).unique().sort_values().values
    piv = {}
    tr = load_trades()
    rows = []
    for r in tr.itertuples():
        d = r.entry_time.normalize()
        sc = scale_at(scale, r.symbol, d)
        b = day_bars.get((r.symbol, d))
        if np.isnan(sc) or b is None:
            rows.append({f: np.nan for f in FILTERS})
            continue
        u, long = sc * r.entry, r.direction > 0
        f = dict(sr=near_sr(sr, r.symbol, d, r.t_dec, r.level, NEAR * u),
                 supply_demand=fresh_zone(sd, r.symbol, d, r.t_dec, "demand" if long else "supply", r.level, NEAR * u))
        k = kings.get(r.symbol)
        i = latest(k.time_et.values, r.entry_time) if k is not None else None
        if i is not None:
            ks, kv = k.strike.values[i], k.net_gamma.values[i]
            f.update(king_ahead=(ks - r.entry) * r.direction > 0, king_negative=kv < 0,
                     level_not_king=(np.nan if r.family == "ote" else bool(abs(r.level - ks) > 1e-9)))
        else:
            f.update(king_ahead=np.nan, king_negative=np.nan, level_not_king=np.nan)
        j = latest(vtimes, r.entry_time) if r.symbol in INDEXES else None
        f["trinity_2of3"] = (int((views.loc[vtimes[j]].king_side == r.direction).sum()) >= 2) if j is not None else np.nan
        if r.family == "ote":
            f["ote_zone"] = np.nan
        else:
            key = (r.symbol, d)
            if key not in piv:
                piv[key] = pivots(b)
            leg = last_leg(piv[key], r.entry_time, r.direction, u)
            f["ote_zone"] = bool(leg and in_zone(r.entry, leg))
        rr = rolls.get(r.symbol)
        i = latest(rr.time_et.values, r.entry_time) if rr is not None else None
        f["roll_with"] = (rr.roll.values[i] == ("bullish" if long else "bearish")) if i is not None else np.nan
        f["gap_toward"] = gap_state(b, r.entry_time, pc.get((r.symbol, d), np.nan), long, u) == "toward unfilled gap"
        rows.append(f)
    out = pd.concat([tr, pd.DataFrame(rows, index=tr.index)], axis=1)
    out.to_csv(TABLES / "filters_flags.csv", index=False)
    print(out.groupby("family").size().to_string())


def stats(x):
    lo, hi = day_ci(x, "entry_time")
    return dict(n=len(x), win=(x.r > 0).mean() if len(x) else np.nan, r=x.r_net.mean() if len(x) else np.nan,
                lo=lo, hi=hi)


def perm_p(df, mask, n=1000, seed=0):
    """Share of random same-size subsets whose R uplift is at least the filter's."""
    rng = np.random.default_rng(seed)
    v, k = df.r_net.values, int(mask.sum())
    obs = v[mask].mean() - v[~mask].mean()
    hits = 0
    for _ in range(n):
        m = np.zeros(len(v), bool)
        m[rng.choice(len(v), k, replace=False)] = True
        hits += v[m].mean() - v[~m].mean() >= obs
    return hits / n


def null_passes(P, fams, sel, runs=200, seed=1):
    """How many filters pass the WORKS rule for real, and for random same-rate filters."""
    rng = np.random.default_rng(seed)

    def count(mask_of):
        n = 0
        for fam in fams:
            for flt in FILTERS:
                ok = True
                for df in P.values():
                    x = sel(df, fam).dropna(subset=[flt])
                    m = mask_of(x, flt)
                    a, b = x[m], x[~m]
                    if len(a) < MIN_N or len(b) < MIN_N:
                        ok = False
                        break
                    ok &= (a.r > 0).mean() > (b.r > 0).mean() and a.r_net.mean() > b.r_net.mean()
                n += ok
        return n
    real = count(lambda x, f: x[f].astype(bool).values)
    null = [count(lambda x, f: rng.random(len(x)) < x[f].astype(float).mean()) for _ in range(runs)]
    return real, null


def report():
    P = {"jul-sep": pd.read_csv(_ROOT / "results/tables/filters_flags.csv", parse_dates=["entry_time"]),
         "oct-jun": pd.read_csv(_ROOT / "results/tables/oos/filters_flags.csv", parse_dates=["entry_time"])}
    for df in P.values():
        df["pooled"] = ~df.family.str.startswith("nonking_rug")
    fams = sorted(P["jul-sep"].family.unique()) + ["pooled"]
    sel = lambda df, fam: df[df.pooled] if fam == "pooled" else df[df.family == fam]
    rows, works = [], {}
    for fam in fams:
        for flt in FILTERS:
            row, ok = dict(family=fam, filter=flt), True
            for per, df in P.items():
                x = sel(df, fam).dropna(subset=[flt])
                m = x[flt].astype(bool)
                a, b = stats(x[m]), stats(x[~m])
                row.update({f"{per} n_with": a["n"], f"{per} win_with": a["win"], f"{per} win_without": b["win"],
                            f"{per} R_with": a["r"], f"{per} R_without": b["r"],
                            f"{per} perm_p": perm_p(x, m.values) if a["n"] and b["n"] else np.nan})
                ok &= a["n"] >= MIN_N and b["n"] >= MIN_N and a["win"] > b["win"] and a["r"] > b["r"]
            row["works"] = bool(ok)
            rows.append(row)
            if ok:
                works.setdefault(fam, []).append(flt)
    ev = pd.DataFrame(rows)
    combos = []
    for fam, W in works.items():
        sets = [[w] for w in W] + [[a, b] for i, a in enumerate(W) for b in W[i + 1:]] + ([W] if len(W) > 2 else [])
        for s in [[]] + sets:
            row = dict(family=fam, filters=" + ".join(s) or "(no filter)")
            for per, df in P.items():
                x = sel(df, fam).dropna(subset=s) if s else sel(df, fam)
                if s:
                    x = x[x[s].astype(bool).all(axis=1)]
                st = stats(x)
                row.update({f"{per} {k}": v for k, v in st.items()})
            combos.append(row)
    cb = pd.DataFrame(combos)
    real, null = null_passes(P, fams, sel)
    print(f"working filters: {real}; random filters with the same firing rates pass "
          f"{np.mean(null):.1f} on average (95th pct {np.percentile(null, 95):.0f}); "
          f"share of random runs >= real: {np.mean(np.array(null) >= real):.3f}")
    out = _ROOT / "results/tables/oos"
    ev.round(4).to_csv(out / "filters_eval.csv", index=False)
    cb.round(4).to_csv(out / "filters_combos.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    show = ["family", "filter", "jul-sep n_with", "jul-sep win_with", "jul-sep win_without", "jul-sep R_with",
            "jul-sep R_without", "oct-jun n_with", "oct-jun win_with", "oct-jun win_without", "oct-jun R_with",
            "oct-jun R_without", "jul-sep perm_p", "oct-jun perm_p", "works"]
    print(ev[show].round(3).to_string(index=False))
    print("\n== stacked working filters\n" + cb.round(3).to_string(index=False))


if __name__ == "__main__":
    {"flags": flags, "report": report}[sys.argv[1]]()
