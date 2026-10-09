#!/usr/bin/env python3
"""Bias checks on the filter evaluation (run after filters.py flags for both periods).

1. direction    Do working filters just select longs in a rising market? Long share
                with/without each filter, and the R uplift within longs and within shorts.
2. outliers     Best stacks: median R, mean R without the 3 best trades, top symbol share.
3. null         Random-filter pass count with ONE random draw per underlying trade, so
                families sharing entries (the non-King rug targets) share the draw.
4. fills        Rug-family and edge entries re-priced at the next bar's open instead
                of the signal bar's close (exit prices unchanged).
5. universe     Buy-and-hold return of each traded symbol over each period.

Usage: python3 scripts/bias_checks.py
Writes results/tables/oos/bias_checks.txt.
"""
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from filters import FILTERS, MIN_N
from load import _ROOT

PER = {"jul-sep": ("", "watchlist"), "oct-jun": ("oos/", "oos")}
STACKS = {"rug": ["sr", "king_negative"], "king_reject": ["king_negative", "trinity_2of3"]}


def flags(tag):
    d = pd.read_csv(_ROOT / f"results/tables/{tag}filters_flags.csv", parse_dates=["entry_time"])
    d["pooled"] = ~d.family.str.startswith("nonking_rug")
    return d


def sel(d, fam):
    return d[d.pooled] if fam == "pooled" else d[d.family == fam]


def works_table():
    ev = pd.read_csv(_ROOT / "results/tables/oos/filters_eval.csv")
    return ev[ev.works][["family", "filter"]].values.tolist()


def direction(P):
    print("== 1. Direction: long share and uplift within longs / shorts")
    rows = []
    for fam, flt in works_table():
        row = dict(family=fam, filter=flt)
        for per, d in P.items():
            x = sel(d, fam).dropna(subset=[flt])
            m = x[flt].astype(bool)
            row[f"{per} long% with/without"] = f"{(x[m].direction > 0).mean():.0%}/{(x[~m].direction > 0).mean():.0%}"
            for side, s in (("longs", 1), ("shorts", -1)):
                y = x[x.direction == s]
                ym = y[flt].astype(bool)
                ok = ym.sum() >= 10 and (~ym).sum() >= 10
                row[f"{per} uplift {side}"] = round(y[ym].r_net.mean() - y[~ym].r_net.mean(), 3) if ok else np.nan
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    for per, d in P.items():
        x = d[d.pooled]
        print(f"{per}: all pooled trades, longs {x[x.direction > 0].r_net.mean():+.3f}R "
              f"(n={int((x.direction > 0).sum())}), shorts {x[x.direction < 0].r_net.mean():+.3f}R "
              f"(n={int((x.direction < 0).sum())})")


def outliers(P):
    print("\n== 2. Outliers and concentration in the best stacks")
    for fam, fl in STACKS.items():
        for per, d in P.items():
            x = sel(d, fam).dropna(subset=fl)
            x = x[x[fl].astype(bool).all(axis=1)].sort_values("r_net")
            top_sym = x.symbol.value_counts(normalize=True)
            print(f"{fam} + {' + '.join(fl)} [{per}] n={len(x)} mean {x.r_net.mean():+.3f}R  median "
                  f"{x.r_net.median():+.3f}R  mean without best 3 {x.r_net.iloc[:-3].mean():+.3f}R  "
                  f"top symbol {top_sym.index[0]} {top_sym.iloc[0]:.0%}  longs {(x.direction > 0).mean():.0%}")


def null(P, runs=200, seed=1):
    print("\n== 3. Random-filter null with one draw per underlying trade")
    fams = sorted(P["jul-sep"].family.unique()) + ["pooled"]
    rng = np.random.default_rng(seed)

    def count(get_mask):
        n = 0
        for fam in fams:
            for flt in FILTERS:
                ok = True
                for per, d in P.items():
                    x = sel(d, fam).dropna(subset=[flt])
                    m = get_mask(per, x, flt)
                    a, b = x[m], x[~m]
                    if len(a) < MIN_N or len(b) < MIN_N:
                        ok = False
                        break
                    ok &= (a.r > 0).mean() > (b.r > 0).mean() and a.r_net.mean() > b.r_net.mean()
                n += ok
        return n
    real = count(lambda per, x, f: x[f].astype(bool).values)
    keys = {per: d.symbol + "|" + d.entry_time.astype(str) + "|" + d.direction.astype(str) for per, d in P.items()}
    null = []
    for _ in range(runs):
        draw = {per: pd.Series(rng.random(k.nunique()), index=k.unique()) for per, k in keys.items()}
        null.append(count(lambda per, x, f: draw[per][keys[per][x.index]].values < x[f].astype(float).mean()))
    print(f"real passes {real}; shared-draw random filters: mean {np.mean(null):.1f}, 95th pct "
          f"{np.percentile(null, 95):.0f}, share >= real {np.mean(np.array(null) >= real):.3f}")


def fills(P):
    print("\n== 4. Entry at the next bar's open instead of the signal close")
    for per, (tag, data) in PER.items():
        b = pd.read_parquet(_ROOT / f"data/raw/{data}/price_bars_1min.parquet")
        b["symbol"] = b.symbol.astype(str)
        nxt = b.sort_values("time_et").assign(next_open=lambda x: x.groupby("symbol").open.shift(-1))
        nxt = nxt.set_index(["symbol", "time_et"]).next_open
        for f in ("rug_trades", "edges_trades", "rug_targets_trades"):
            t = pd.read_csv(_ROOT / f"results/tables/{tag}{f}.csv", parse_dates=["entry_time"])
            if "target_kind" in t:
                t = t[t.target_kind == "king"]
            if "setup" in t:
                t = t[~t.setup.str.endswith("control")]
                d = np.where(t.setup.str.startswith("rug"), -1, 1)
            else:
                d = np.where(t.side.str.startswith("long"), 1, -1)
            e2 = nxt.reindex(list(zip(t.symbol, t.entry_time))).values
            risk2 = np.abs(e2 - t.stop.values)
            r2 = (d * (t.exit.values - e2) - 0.0002 * e2) / risk2
            ok = ~np.isnan(r2) & ((t.stop.values - e2) * d < 0)
            print(f"{per} {f}: signal-close {t.r_net.mean():+.3f}R  next-open {np.nanmean(r2[ok]):+.3f}R  "
                  f"(n={ok.sum()}, {(~ok).sum()} would already be through the stop)")


def universe():
    print("\n== 5. Buy-and-hold of the traded symbols")
    for per, (tag, data) in PER.items():
        b = pd.read_parquet(_ROOT / f"data/raw/{data}/price_bars_1min.parquet")
        b["symbol"] = b.symbol.astype(str)
        r = b.sort_values("time_et").groupby("symbol").agg(o=("open", "first"), c=("close", "last"))
        ch = 100 * (r.c / r.o - 1)
        print(f"{per}: {len(ch)} symbols, median {ch.median():+.1f}%, rose {int((ch > 0).sum())}, "
              f"fell {int((ch < 0).sum())}; best {ch.idxmax()} {ch.max():+.0f}%, worst {ch.idxmin()} {ch.min():+.0f}%")


def main():
    P = {per: flags(tag) for per, (tag, _) in PER.items()}
    buf = io.StringIO()
    with redirect_stdout(buf):
        pd.set_option("display.width", 250)
        direction(P)
        outliers(P)
        null(P)
        fills(P)
        universe()
    (_ROOT / "results/tables/oos/bias_checks.txt").write_text(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main()
