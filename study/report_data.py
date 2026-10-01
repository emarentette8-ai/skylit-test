"""Collect everything the report needs into results/report_data.json.

Only reads the cached event datasets and result tables; computes a few extra
descriptive breakdowns (time-of-day/0DTE clock, regional cancellation) and selects
example events by a fixed rule (no outcome information used in the selection).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from study import analyze as A
from study import build, spec

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"


def by_bucket(e, sym, col, edges, labels, field):
    rows = []
    r, p = e[e.level == "real"], e[e.level == "placebo"]
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        ra, pa = r[(r[field] >= lo) & (r[field] < hi)], p[(p[field] >= lo) & (p[field] < hi)]
        m = A.boot_mean(ra, col)
        d = A.boot_diff(ra, pa, col) if len(pa) else dict(diff=np.nan, lo=np.nan, hi=np.nan)
        rows.append(dict(symbol=sym, bucket=lab, outcome=col, mean=m["mean"], lo=m["lo"],
                         hi=m["hi"], n=m["n"], days=m["days"], diff=d["diff"],
                         diff_lo=d["lo"], diff_hi=d["hi"], n_placebo=len(pa.dropna(subset=[col]))))
    return rows


def examples():
    """Rule: QQQ holdout trading days in date order; take days #1, #8 and #15; on each,
    the first eligible real-strike entry with decision time >= 11:00 ET."""
    E = A.entries(symbols=["QQQ"])
    E = E[(E.level == "real") & (E.split == "holdout")]
    days = sorted(E.date.unique())
    g, p = build.load()
    p = p[p.symbol == "QQQ"]
    out = []
    for di in (0, 7, 14):
        d = days[di]
        ev = E[(E.date == d) & (E.tau >= 90)].sort_values("tau").iloc[0]
        a = build.day_arrays(p[p.date == d])
        lo, hi = max(0, ev.t_entry_bar - 20), min(build.NMIN - 1, ev.t_entry_bar + 31)
        pts = [[int(m), float(a["close"][m])] for m in range(lo, hi + 1) if not np.isnan(a["close"][m])]
        out.append(dict(date=str(pd.Timestamp(d).date()), K=float(ev.K), s=float(ev.s), b=float(ev.b),
                        t_entry_bar=int(ev.t_entry_bar), tau=int(ev.tau), p0=float(ev.y_p0),
                        r15=float(ev.y_r15), barrier=str(ev.y_barrier),
                        cross_t=None if np.isnan(ev.y_cross_t) else float(ev.y_cross_t),
                        node=next((n for n in ("king", "gatekeeper", "pika", "barney")
                                   if ev[f"f_node_{n}"] == 1), "normal"),
                        loc_share=float(ev.f_loc_share), net_ratio=float(ev.f_net_ratio),
                        snap_age=float(ev.f_snap_age), n_prior=int(ev.n_prior), pts=pts,
                        utc=str(ev.decision_time_utc)))
    return out


REVIEW = ("Independent review. A separate reviewer audited event timing, signs, units, duplicated "
          "observations and the GEX benchmark. It found no look-ahead and no sign or timing errors. "
          "It confirmed that the snapshot spot matches the close of the bar just before the listed time "
          "(median 0.13 bp), so 'latest snapshot at or before the decision' is consistent with the data. "
          "Issues it raised, and what changed: (1) the ridge penalty for QQQ sat at the edge of the grid, "
          "so the primary estimate was near zero by construction. The grid was extended (1e5 to 1e7; the "
          "primary estimate moved from +0.01% to 0.00%), and unpenalised OLS plus a direct "
          "residual-increment test were added. (2) A few robustness cells were individually significant; "
          "they are now reported with a Holm correction. (3) The single-name king/volatility result lacked "
          "symbol fixed effects; with them it shrinks from 3.6% to 1.3% of mean volatility. (4) Placebo "
          "eligibility and strike spacing used the full day's strike list; both now use only strikes "
          "listed at decision time, and the leakage test now covers placebo entries and roundness "
          "features. (5) The trading comparison now carries a confidence interval on M3 minus M0. "
          "Open caveats: in-sample-derived trading thresholds, and SPXW entries priced at the SPX index, "
          "which cannot be traded directly.")


def main():
    E = A.entries()
    extra = []
    for sym in ("QQQ", "SPY", "SPXW"):
        e = E[E.symbol == sym]
        for col in ("y_r15", "y_rv15"):
            extra += by_bucket(e, sym, col, [30, 90, 150, 210, 270, 330, 375],
                               ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"], "tau")
    tod = pd.DataFrame(extra)
    canc = []
    for sym in ("QQQ", "SPY", "SPXW"):
        r = E[(E.symbol == sym) & (E.level == "real")]
        for lo, hi, lab in ((0, .25, "0-25% (heavy offset)"), (.25, .5, "25-50%"), (.5, .75, "50-75%"),
                            (.75, 1.01, "75-100% (one-sided)")):
            sub = r[(r.f_reg_cancel >= lo) & (r.f_reg_cancel < hi)]
            for col in ("y_r15", "y_rv15"):
                m = A.boot_mean(sub, col)
                canc.append(dict(symbol=sym, bucket=lab, outcome=col, mean=m["mean"], lo=m["lo"],
                                 hi=m["hi"], n=m["n"], days=m["days"],
                                 gross_share=sub.f_reg_gross.mean(), net_share=sub.f_reg_net.abs().mean()))
    canc = pd.DataFrame(canc)
    tod.to_csv(TAB / "time_of_day.csv", index=False)
    canc.to_csv(TAB / "cancellation.csv", index=False)
    rd = lambda f: pd.read_csv(TAB / f).replace({np.nan: None}).to_dict("records")
    data = {
        "descriptive": rd("descriptive.csv"), "models": rd("models.csv"),
        "robustness": rd("robustness_models.csv"), "trading": rd("trading.csv"),
        "groups": rd("secondary_groups.csv"), "regime": rd("regime_check.csv"),
        "controlled": rd("controlled_vol.csv"), "curve": rd("distance_curve.csv"),
        "paths": rd("paths.csv"), "counts": rd("event_counts.csv"),
        "tod": tod.replace({np.nan: None}).to_dict("records"),
        "cancel": canc.replace({np.nan: None}).to_dict("records"),
        "crossing": json.load(open(TAB / "crossing_effect.json")),
        "landmarks": sum((rd(f"landmarks_{s}.csv") for s in ("QQQ", "SPY", "SPXW")), []),
        "examples": examples(),
        "resinc": json.load(open(TAB / "residual_increment.json")),
        "review": REVIEW,
        "spec": {"band": spec.BAND, "splits": spec.SPLITS, "min_skill": spec.MIN_MEANINGFUL_SKILL,
                 "cost": spec.ROUND_TRIP_COST_PCT, "barrier": spec.BARRIER_PCT},
    }
    json.dump(data, open(ROOT / "results" / "report_data.json", "w"), default=float)
    print("report data written")


if __name__ == "__main__":
    main()
