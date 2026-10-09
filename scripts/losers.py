#!/usr/bin/env python3
"""What do losing trades have in common? (run after filters.py flags for both periods)

Pools every setup (as filters.py does: non-King rug variants excluded, they repeat
the rug entries) and compares loss rates (net R <= 0) across buckets of:
  hour, weekday, side, index vs stock, setup, stop distance (typical ranges),
  with/against the day's move so far, move already done since the open, the day's
  range so far, and the nine filters.
A factor is CONSISTENT if the same bucket has the highest loss rate in both
periods, at least 3 points above the rest of that factor, with >= 100 trades in
the bucket in both periods. Shuffled outcomes (same rule) show how many factors
pass by chance.

Usage: python3 scripts/losers.py
Writes results/tables/oos/losers.csv and losers_summary.txt.
"""
import numpy as np
import pandas as pd

from filters import FILTERS, load_trades
from load import _ROOT
from node_claims import scale_at, typical_range

PER = {"jul-sep": ("", "watchlist"), "oct-jun": ("oos/", "oos")}
GAP, MIN_N = 0.03, 100


def build(tag, data):
    base = _ROOT / "results/tables" / tag
    f = pd.read_csv(base / "filters_flags.csv", parse_dates=["entry_time"])
    t = load_trades(base, extra=["stop", "exit_how"])
    assert len(t) == len(f) and np.allclose(t.r_net.values, f.r_net.values, equal_nan=True)
    f["stop"], f["exit_how"] = t.stop.values, t.exit_how.values
    f = f[~f.family.str.startswith("nonking_rug")].copy()
    p = pd.read_parquet(_ROOT / f"data/raw/{data}/price_bars_1min.parquet")
    p["symbol"] = p.symbol.astype(str)
    p["date"] = p.time_et.dt.normalize()
    scale = typical_range(p)
    day = {k: v.sort_values("time_et") for k, v in p.groupby(["symbol", "date"])}
    rows = []
    for r in f.itertuples():
        d = r.entry_time.normalize()
        b, sc = day.get((r.symbol, d)), scale_at(scale, r.symbol, d)
        if b is None or np.isnan(sc):
            rows.append((np.nan,) * 4)
            continue
        u = sc * r.entry
        done = b[b.time_et < r.entry_time]
        o = b.open.values[0]
        rows.append((abs(r.entry - r.stop) / u, np.sign(r.entry - o) * r.direction,
                     abs(r.entry - o) / u, (done.high.max() - done.low.min()) / u if len(done) else np.nan))
    f[["stop_dist", "with_day", "move_done", "range_so_far"]] = rows
    f["loss"] = f.r_net <= 0
    return f


def buckets(f):
    return {
        "hour": f.entry_time.dt.hour.clip(10, 15).astype(str),
        "weekday": f.entry_time.dt.day_name().str[:3],
        "side": np.where(f.direction > 0, "long", "short"),
        "index vs stock": np.where(f.symbol.isin(["SPY", "QQQ", "SPXW", "IWM", "SMH"]), "index/ETF", "stock"),
        "setup": f.family,
        "stop distance (typical ranges)": pd.cut(f.stop_dist, [0, 0.4, 0.6, 1, 99], labels=["<0.4", "0.4-0.6", "0.6-1", ">1"]).astype(str),
        "vs the day's move so far": np.select([f.with_day > 0, f.with_day < 0], ["with", "against"], "flat"),
        "move done since open (typical ranges)": pd.cut(f.move_done, [-1, 1, 3, 99], labels=["<1", "1-3", ">3"]).astype(str),
        "day range so far (typical ranges)": pd.cut(f.range_so_far, [-1, 3, 6, 99], labels=["<3", "3-6", ">6"]).astype(str),
        **{f"filter: {k}": f[k].map({True: "yes", False: "no", 1.0: "yes", 0.0: "no"}).fillna("n/a") for k in FILTERS},
    }


def table(P, outcome="loss"):
    rows = []
    B = {per: buckets(f) for per, f in P.items()}
    for factor in B["jul-sep"]:
        for per, f in P.items():
            g = pd.DataFrame({"b": B[per][factor], "loss": f[outcome].values, "r": f.r_net.values})
            g = g[g.b != "n/a"]
            s = g.groupby("b").agg(n=("loss", "size"), loss_rate=("loss", "mean"), mean_r=("r", "mean")).reset_index()
            for x in s.itertuples():
                rest = g[g.b != x.b].loss.mean()
                rows.append(dict(factor=factor, bucket=x.b, period=per, n=x.n, loss_rate=x.loss_rate,
                                 vs_rest=x.loss_rate - rest, mean_r=x.mean_r))
    return pd.DataFrame(rows)


def consistent(t):
    out = []
    for factor, x in t.groupby("factor"):
        w = x.pivot_table(index="bucket", columns="period", values=["loss_rate", "vs_rest", "n"])
        w = w.dropna()
        if w.empty:
            continue
        worst = {per: w["loss_rate"][per].idxmax() for per in PER}
        b = worst["jul-sep"]
        if b == worst["oct-jun"] and all(w["vs_rest"][per][b] >= GAP and w["n"][per][b] >= MIN_N for per in PER):
            out.append((factor, b, *(round(w["loss_rate"][per][b], 3) for per in PER),
                        *(round(w["vs_rest"][per][b], 3) for per in PER)))
    return pd.DataFrame(out, columns=["factor", "worst bucket", "jul-sep loss", "oct-jun loss",
                                      "jul-sep vs rest", "oct-jun vs rest"])


def main():
    P = {per: build(*v) for per, v in PER.items()}
    t = table(P)
    c = consistent(t)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(100):
        Q = {per: f.assign(loss=rng.permutation(f.loss.values)) for per, f in P.items()}
        null.append(len(consistent(table(Q))))
    lines = [f"trades: " + ", ".join(f"{per} {len(f)} ({f.loss.mean():.0%} losers)" for per, f in P.items()),
             "", "CONSISTENT factors (same worst bucket both periods, >= 3 pts above the rest, n >= 100):",
             c.to_string(index=False) if len(c) else "(none)",
             f"\nshuffled outcomes: {np.mean(null):.1f} consistent factors on average, "
             f"95th pct {np.percentile(null, 95):.0f}; real {len(c)}",
             "\nhow losers exit: " + ", ".join(
                 f"{per} " + str(f[f.loss].exit_how.value_counts(normalize=True).round(2).to_dict()) for per, f in P.items())]
    out = _ROOT / "results/tables/oos"
    t.round(4).to_csv(out / "losers.csv", index=False)
    (out / "losers_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    pd.set_option("display.width", 200, "display.max_rows", 300)
    w = t.pivot_table(index=["factor", "bucket"], columns="period", values=["n", "loss_rate", "mean_r"]).round(3)
    print("\n" + w.to_string())


if __name__ == "__main__":
    main()
