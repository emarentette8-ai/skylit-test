#!/usr/bin/env python3
"""Do OTE and price support/resistance improve Heatseeker trades?

OTE (ICT optimal trade entry): a swing leg on 15-minute bars (pivots with 2
bars each side, confirmed 2 bars later) of at least one typical 30-minute
range; the zone is its 61.8%-78.6% retracement.

Part A - filters on the Rug-family trades from rug.py:
  sr   the traded level has price S/R nearby (confluence.py definition)
  ote  the entry sits in the OTE zone of the latest leg in the trade's direction
Part B - standalone OTE trades on every symbol:
  limit entry at the 70.5% retracement, stop at the leg origin (100%), target
  the leg end (0%), else exit at the close; a fill bar that also hits the stop
  counts as a loss. Only entries from 10:00 (when the first map exists).
  Split by: a named Heatseeker node inside the zone (latest map at entry),
  price S/R inside the zone, both, neither.

Usage: python3 scripts/ote.py  (SKYLIT_DATA=oos for the new data; run rug.py first)
Writes ote_filters.csv, ote_trades.csv and ote_summary.csv to the tables folder.
"""
import numpy as np
import pandas as pd

from confluence import NEAR, build_sr, near_sr
from load import TABLES
from node_claims import INDEXES, NAMED, prepare, typical_range

OUT = TABLES
ZONE = (0.618, 0.786)
ENTRY = 0.705
COST = 0.0002


def pivots(b):
    """Confirmed 15-minute swing points: (known_from, time, 'H'|'L', price)."""
    q = (b.set_index("time_et").resample("15min")
         .agg({"high": "max", "low": "min"}).dropna())
    h, l, ts = q.high.values, q.low.values, q.index
    out = []
    for i in range(2, len(q) - 2):
        known = ts[i + 2] + pd.Timedelta(minutes=15)
        if h[i] == h[i - 2:i + 3].max():
            out.append((known, ts[i], "H", h[i]))
        if l[i] == l[i - 2:i + 3].min():
            out.append((known, ts[i], "L", l[i]))
    return sorted(out)


def last_leg(piv, t, direction, min_size):
    """Latest leg known at t. direction -1 = down leg (for shorts), +1 = up leg (longs).

    Returns (origin, end, end_time) or None. A down leg is a swing high followed by
    a later swing low; an up leg the reverse.
    """
    known = [p for p in piv if p[0] <= t]
    end_kind, start_kind = ("L", "H") if direction < 0 else ("H", "L")
    ends = [p for p in known if p[2] == end_kind]
    if not ends:
        return None
    end = ends[-1]
    starts = [p for p in known if p[2] == start_kind and p[1] < end[1]]
    if not starts:
        return None
    origin = starts[-1]
    if abs(origin[3] - end[3]) < min_size:
        return None
    return origin[3], end[3], end[1]


def in_zone(price, leg):
    origin, end = leg[:2]
    lo, hi = sorted((end + ZONE[0] * (origin - end), end + ZONE[1] * (origin - end)))
    return lo <= price <= hi


def part_a(day_bars, scale, sr):
    tr = pd.read_csv(OUT / "rug_trades.csv", parse_dates=["map_time", "entry_time"])
    cache = {}
    sr_flag, ote_flag = [], []
    for r in tr.itertuples():
        key = (r.symbol, r.map_time.normalize())
        if key not in cache:
            cache[key] = pivots(day_bars[key])
        d = -1 if r.setup.startswith("rug") else 1
        leg = last_leg(cache[key], r.entry_time, d, scale[r.symbol] * r.entry)
        ote_flag.append(bool(leg and in_zone(r.entry, leg)))
        sr_flag.append(near_sr(sr, r.symbol, key[1], r.map_time, r.level, NEAR * scale[r.symbol] * r.level))
    tr["sr"], tr["ote"] = sr_flag, ote_flag
    tr["family"] = np.where(tr.setup.str.endswith("control"), "control", "setup")
    rows = []
    for fam, df in [("setups", tr[tr.family == "setup"]), ("all incl. controls", tr)]:
        for name, mask in [("all trades", slice(None)), ("S/R", df.sr), ("OTE", df.ote),
                           ("S/R + OTE", df.sr & df.ote), ("neither", ~df.sr & ~df.ote)]:
            x = df[mask]
            lo, hi = day_ci(x, "map_time") if len(x) > 1 else (np.nan, np.nan)
            rows.append(dict(group=fam, filter=name, trades=len(x), win_rate=(x.r > 0).mean(),
                             mean_r_net=x.r_net.mean(), ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def latest_map(g, sym, t):
    s = g.get(sym)
    if s is None:
        return None
    times = s.time_et.unique()
    times = times[times <= np.datetime64(t)]
    if not len(times) or pd.Timestamp(times.max()).normalize() != pd.Timestamp(t).normalize():
        return None
    return s[s.time_et == times.max()]


def part_b(g, day_bars, scale, sr):
    gm = {k: v for k, v in g[g.node_type.isin(NAMED)].groupby("symbol")}
    trades = []
    for (sym, d), b in day_bars.items():
        if sym == "VIX" or sym not in scale:
            continue
        piv = pivots(b)
        op, hi, lo, cl, tm = b.open.values, b.high.values, b.low.values, b.close.values, b.time_et.values
        busy = np.datetime64("1970-01-01")
        seen = set()
        for k in range(len(piv)):
            known = piv[k][0]
            for direction in (-1, 1):
                leg = last_leg(piv[:k + 1], known, direction, scale[sym] * piv[k][3])
                if leg is None or (direction, leg) in seen:
                    continue
                seen.add((direction, leg))
                origin, end, end_t = leg
                entry = end + ENTRY * (origin - end)
                i = max(np.searchsorted(tm, np.datetime64(known)), np.searchsorted(tm, busy))
                # skip legs price already retraced to the entry before the swing was confirmed
                k0 = np.searchsorted(tm, np.datetime64(end_t))
                k1 = np.searchsorted(tm, np.datetime64(known))
                if k1 > k0 and ((direction < 0 and hi[k0:k1].max() >= entry) or
                                (direction > 0 and lo[k0:k1].min() <= entry)):
                    continue
                trade = None
                for j in range(i, len(cl)):
                    if pd.Timestamp(tm[j]).time() < pd.Timestamp("10:00").time():
                        if (direction < 0 and lo[j] < end) or (direction > 0 and hi[j] > end):
                            break                       # leg extended before we could trade it
                        continue
                    if (direction < 0 and lo[j] < end) or (direction > 0 and hi[j] > end):
                        break
                    if (direction < 0 and hi[j] >= entry) or (direction > 0 and lo[j] <= entry):
                        trade = j
                        break
                if trade is None:
                    continue
                j = trade
                if (direction < 0 and op[j] > entry) or (direction > 0 and op[j] < entry):
                    entry = op[j]                       # gapped through the limit: filled at the open
                risk = abs(origin - entry)
                if risk <= 0:
                    continue
                if (direction < 0 and hi[j] >= origin) or (direction > 0 and lo[j] <= origin):
                    exit_px, how, m = origin, "stop", j
                else:
                    exit_px, how, m = cl[-1], "close", len(cl) - 1
                    for m in range(j + 1, len(cl)):
                        if (direction < 0 and hi[m] >= origin) or (direction > 0 and lo[m] <= origin):
                            exit_px, how = origin, "stop"
                            break
                        if (direction < 0 and lo[m] <= end) or (direction > 0 and hi[m] >= end):
                            exit_px, how = end, "target"
                            break
                pnl = (entry - exit_px) if direction < 0 else (exit_px - entry)
                t_entry = pd.Timestamp(tm[j])
                zlo, zhi = sorted((end + ZONE[0] * (origin - end), end + ZONE[1] * (origin - end)))
                m_ = latest_map(gm, sym, t_entry)
                node = bool(m_ is not None and ((m_.strike >= zlo) & (m_.strike <= zhi)).any())
                sr_in = any(k_ <= t_entry and zlo <= px <= zhi for k_, px in sr.get((sym, d), ()))
                trades.append(dict(symbol=sym, index=sym in INDEXES, entry_time=t_entry,
                                   side="short" if direction < 0 else "long", entry=entry,
                                   stop=origin, target=end, exit=exit_px, exit_how=how,
                                   r=pnl / risk, r_net=(pnl - COST * entry) / risk,
                                   node_in_zone=node, sr_in_zone=sr_in))
                busy = tm[m]
    return pd.DataFrame(trades)


def day_ci(x, col, n=2000):
    """Day-clustered bootstrap 95% CI of mean r_net."""
    rng = np.random.default_rng(0)
    g = x.groupby(x[col].dt.date).r_net.agg(["sum", "size"])
    s, k = g["sum"].values, g["size"].values
    m = [s[i].sum() / k[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return np.percentile(m, [2.5, 97.5])


def summarize_b(t):
    rows = []
    for scope, df in [("all symbols", t), ("indexes", t[t["index"]])]:
        for name, mask in [("all OTE trades", slice(None)), ("node in zone", df.node_in_zone),
                           ("S/R in zone", df.sr_in_zone),
                           ("node + S/R", df.node_in_zone & df.sr_in_zone),
                           ("neither", ~df.node_in_zone & ~df.sr_in_zone)]:
            x = df[mask]
            lo, hi = day_ci(x, "entry_time") if len(x) > 1 else (np.nan, np.nan)
            rows.append(dict(scope=scope, filter=name, trades=len(x), win_rate=(x.r > 0).mean(),
                             target_rate=(x.exit_how == "target").mean(),
                             mean_r_net=x.r_net.mean(), ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def main():
    g, p, day_bars = prepare()
    scale, sr = typical_range(p), build_sr(p)
    a = part_a(day_bars, scale, sr)
    t = part_b(g, day_bars, scale, sr)
    s = summarize_b(t)
    a.round(4).to_csv(OUT / "ote_filters.csv", index=False)
    t.to_csv(OUT / "ote_trades.csv", index=False)
    s.round(4).to_csv(OUT / "ote_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("== A. Rug-family trades filtered by S/R and OTE\n" + a.round(3).to_string(index=False))
    print("\n== B. Standalone OTE trades\n" + s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
