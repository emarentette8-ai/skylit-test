#!/usr/bin/env python3
"""Greedy strategy search with a sealed holdout (see results/search_plan.md).

  python3 scripts/search.py            # search on Design A (Jul-Sep 2026) + B (Oct 2025-Jan 2026)
  python3 scripts/search.py holdout    # apply the saved rules once to Feb-Jun 2026

Needs filters.py flags for both periods. Writes results/tables/oos/search_rules.json,
search_log.csv and search_holdout.txt.
"""
import json
import sys

import numpy as np
import pandas as pd

from filters import FILTERS, latest
from load import _ROOT
from losers import PER, build
from no_entry import king_share
from stats import day_ci

INDEX = ["SPXW", "SPY", "QQQ"]
HOLDOUT_START = pd.Timestamp("2026-02-01")
MIN_DESIGN, MIN_PART, MIN_GAIN, MAX_STEPS = 300, 100, 0.01, 6
OUT = _ROOT / "results/tables/oos"


def index_context(data):
    """Per index: King side vs map spot by map time, and price/open by minute."""
    g = pd.read_parquet(_ROOT / f"data/raw/{data}/gamma_snapshots.parquet")
    g["symbol"] = g.symbol.astype(str)
    k = g[g.symbol.isin(INDEX) & (g.node_type == "king")].copy()
    k["side"] = np.sign(k.strike - k.spot)
    kings = {s: x.sort_values("time_et")[["time_et", "side"]] for s, x in k.groupby("symbol")}
    p = pd.read_parquet(_ROOT / f"data/raw/{data}/price_bars_1min.parquet")
    p["symbol"] = p.symbol.astype(str)
    p = p[p.symbol.isin(INDEX)].sort_values("time_et")
    p["day_open"] = p.groupby([p.symbol, p.time_et.dt.normalize()]).open.transform("first")
    p["dir"] = np.sign(p.close - p.day_open)
    prices = {s: x[["time_et", "dir"]] for s, x in p.groupby("symbol")}
    return kings, prices


def votes(table, t, d):
    n = 0
    for s in INDEX:
        x = table.get(s)
        if x is None:
            continue
        i = latest(x.time_et.values, t)
        if i is not None and x.iloc[i, 1] == d:
            n += 1
    return n


def frame(per, tag, data):
    f = build(tag, data)
    kings, prices = index_context(data)
    f["idx_king_votes"] = [votes(kings, r.entry_time, r.direction) for r in f.itertuples()]
    f["idx_price_votes"] = [votes(prices, r.entry_time - pd.Timedelta(minutes=1), r.direction)
                            for r in f.itertuples()]
    ks = king_share(data)
    by = {s: x.sort_values("time_et") for s, x in ks.groupby("symbol")}
    share = []
    for r in f.itertuples():
        x = by.get(r.symbol)
        i = latest(x.time_et.values, r.entry_time) if x is not None else None
        share.append(x.king_share.values[i] if i is not None else np.nan)
    f["king_share"] = share
    f["is_index"] = f.symbol.isin(INDEX)
    f["part"] = "A" if per == "jul-sep" else np.where(f.entry_time < HOLDOUT_START, "B", "holdout")
    # fixed rule: single stocks need >= 2 of 3 index Kings on the trade's side
    return f[f.is_index | (f.idx_king_votes >= 2)].copy()


def buckets(f):
    cut = lambda s, b, l: pd.cut(s, b, labels=l).astype(str)
    B = {
        "setup": f.family, "side": np.where(f.direction > 0, "long", "short"),
        "index vs stock": np.where(f.is_index, "index", "stock"),
        "hour": f.entry_time.dt.hour.clip(10, 15).astype(str),
        "stop distance": cut(f.stop_dist, [0, 0.4, 0.6, 1, 99], ["<0.4", "0.4-0.6", "0.6-1", ">1"]),
        "vs day's move": np.select([f.with_day > 0, f.with_day < 0], ["with", "against"], "flat"),
        "move done": cut(f.move_done, [-1, 1, 3, 99], ["<1", "1-3", ">3"]),
        "day range so far": cut(f.range_so_far, [-1, 3, 6, 99], ["<3", "3-6", ">6"]),
        "king share (indexes)": np.where(~f.is_index, "n/a", np.where(f.king_share >= 0.23, "heavy", "normal")),
        "index price votes": np.where(f.idx_price_votes >= 2, ">=2", "<2"),
        "index king votes": np.where(f.idx_king_votes >= 2, ">=2", "<2"),
    }
    for k in FILTERS:
        B[f"filter {k}"] = f[k].map({True: "yes", False: "no", 1.0: "yes", 0.0: "no"}).fillna("n/a")
    return pd.DataFrame(B, index=f.index)


def candidates(B):
    out = []
    for col in B:
        for v in sorted(set(B[col]) - {"n/a", "nan"}):
            out.append((col, "==", v))
            out.append((col, "!=", v))
    return out


def apply(f, B, rules):
    m = pd.Series(True, index=f.index)
    for col, op, v in rules:
        m &= (B[col] == v) if op == "==" else (B[col] != v)
    return f[m]


def search(D):
    B = buckets(D)
    rules, cur, log, tried = [], D, [], 0
    base = {p: cur[cur.part == p].r_net.mean() for p in ("A", "B")}
    log.append(dict(step=0, rule="(fixed rule only)", n=len(cur), mean_r=cur.r_net.mean(), **{f"R {p}": v for p, v in base.items()}))
    for step in range(1, MAX_STEPS + 1):
        best = None
        for c in candidates(B):
            if c in rules:
                continue
            tried += 1
            x = apply(D, B, rules + [c])
            parts = {p: x[x.part == p] for p in ("A", "B")}
            if len(x) < MIN_DESIGN or min(len(v) for v in parts.values()) < MIN_PART:
                continue
            rp = {p: v.r_net.mean() for p, v in parts.items()}
            if not all(rp[p] > base[p] for p in rp) or x.r_net.mean() < cur.r_net.mean() + MIN_GAIN:
                continue
            if best is None or x.r_net.mean() > best[1].r_net.mean():
                best = (c, x, rp)
        if best is None:
            break
        c, cur, base = best
        rules.append(c)
        log.append(dict(step=step, rule=" ".join(c), n=len(cur), mean_r=cur.r_net.mean(), **{f"R {p}": v for p, v in base.items()}))
    return rules, pd.DataFrame(log), tried


def describe(x, label):
    lo, hi = day_ci(x, "entry_time")
    days = x.groupby(x.entry_time.dt.date).r_net.sum()
    return (f"{label}: {len(x)} trades on {len(days)} days, win {(x.r > 0).mean():.1%}, mean "
            f"{x.r_net.mean():+.3f}R (95% {lo:+.3f}..{hi:+.3f}), positive days {(days > 0).mean():.0%}, "
            f"total {x.r_net.sum():+.1f}R")


def main():
    F = pd.concat([frame(per, *v) for per, v in PER.items()], ignore_index=True)
    if len(sys.argv) > 1 and sys.argv[1] == "holdout":
        rules = [tuple(r) for r in json.loads((OUT / "search_rules.json").read_text())["rules"]]
        H = F[F.part == "holdout"]
        lines = ["rules: " + "; ".join(" ".join(r) for r in rules) if rules else "rules: (none)",
                 describe(H, "holdout, fixed rule only"),
                 describe(apply(H, buckets(H), rules), "holdout, final rules")]
        (OUT / "search_holdout.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    D = F[F.part != "holdout"]
    rules, log, tried = search(D)
    log.round(4).to_csv(OUT / "search_log.csv", index=False)
    (OUT / "search_rules.json").write_text(json.dumps({"rules": rules, "candidates_tried": tried}, indent=1))
    print(log.round(3).to_string(index=False))
    print(f"\ncandidate conditions evaluated: {tried}")
    final = apply(D, buckets(D), rules)
    for p in ("A", "B"):
        print(describe(final[final.part == p], f"design {p}"))


if __name__ == "__main__":
    main()
