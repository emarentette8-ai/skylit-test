"""Supply and demand zones from 5-minute bars (today and the previous day).

Base: one 5-minute bar whose range is at most BASE typical 30-minute ranges.
Demand: within the next two 5-minute bars price closes at least IMPULSE typical
ranges, and at least LEG_OUT times the base's own range, above the base high,
without trading below the base low (supply: the mirror). Thresholds were set from
bar-size frequencies only (median 5-minute range ~0.45 typical range), not outcomes.
Zone = the base's low..high. A zone is known only after the impulse (end of the
second bar after the base) and is fresh until price first trades back into it.
"""
import numpy as np
import pandas as pd

from node_claims import scale_at

BASE, IMPULSE, LEG_OUT = 0.5, 0.5, 2.0


def zones_for(b, u):
    """List of (known, kind, lo, hi, first_touch) from 1-minute bars b (sorted)."""
    q = (b.set_index("time_et").resample("5min")
         .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
    hi, lo, cl, ts = q.high.values, q.low.values, q.close.values, q.index
    out = []
    for j in range(len(q) - 2):
        if ts[j + 2].normalize() != ts[j].normalize() or ts[j + 2] - ts[j] != pd.Timedelta(minutes=10):
            continue
        if hi[j] - lo[j] > BASE * u:
            continue
        need = max(IMPULSE * u, LEG_OUT * (hi[j] - lo[j]))
        if cl[j + 1:j + 3].max() - hi[j] >= need and lo[j + 1:j + 3].min() >= lo[j]:
            kind = "demand"
        elif lo[j] - cl[j + 1:j + 3].min() >= need and hi[j + 1:j + 3].max() <= hi[j]:
            kind = "supply"
        else:
            continue
        known = ts[j + 2] + pd.Timedelta(minutes=5)
        after = b[b.time_et >= known]
        touch = after[after.low <= hi[j]] if kind == "demand" else after[after.high >= lo[j]]
        out.append((known, kind, lo[j], hi[j], touch.time_et.iloc[0] if len(touch) else None))
    return out


def build(p, scale):
    """(symbol, date) -> zones formed on that date or the previous trading day."""
    out = {}
    for sym, sp in p.groupby("symbol"):
        by_day = {d: x.sort_values("time_et") for d, x in sp.groupby("date")}
        days = sorted(by_day)
        for i, d in enumerate(days):
            sc = scale_at(scale, sym, d)
            if np.isnan(sc):
                continue
            cur = by_day[d]
            b = pd.concat([by_day[days[i - 1]], cur]) if i else cur
            out[(sym, d)] = zones_for(b, sc * cur.open.iloc[0])
    return out


def fresh_zone(sd, sym, date, t, kind, price, tol):
    """True if a fresh zone of `kind`, known by t, contains price (+/- tol)."""
    for known, k, lo, hi, first in sd.get((sym, pd.Timestamp(date).normalize()), ()):
        if k == kind and known <= t and (first is None or first >= t) and lo - tol <= price <= hi + tol:
            return True
    return False
