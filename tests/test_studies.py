"""Checks for the scripts/ trade studies: no look-ahead in the inputs they build,
and correct stop/target handling in the trade simulators (synthetic bars)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from confluence import build_sr                      # noqa: E402
from edges import simulate as edge_sim                # noqa: E402
from node_claims import scale_at, typical_range       # noqa: E402
from ote import pivots                                # noqa: E402
from rug import classify, simulate as rug_sim         # noqa: E402

DAY = pd.Timestamp("2026-03-02")


def bars(closes, start=DAY + pd.Timedelta(hours=9, minutes=30), spread=0.05, symbol="TST"):
    """One-minute bars from a close path; high/low a fixed spread around it."""
    c = np.asarray(closes, dtype=float)
    t = pd.date_range(start, periods=len(c), freq="1min")
    return pd.DataFrame(dict(time_et=t, symbol=symbol, open=c, high=c + spread, low=c - spread, close=c,
                             date=t.normalize()))


def random_days(n_days, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-03-02", periods=n_days)
    out = []
    for d in days:
        path = 100 + np.cumsum(rng.normal(0, 0.05, 390))
        out.append(bars(path, start=d + pd.Timedelta(hours=9, minutes=30), spread=rng.uniform(0.02, 0.2)))
    return pd.concat(out, ignore_index=True)


def test_typical_range_uses_only_earlier_days():
    p = random_days(30)
    base = typical_range(p)
    later = p.date >= p.date.unique()[20]
    p2 = p.copy()
    p2.loc[later, "high"] += 5                         # change only days 20+
    changed = typical_range(p2)
    for d in p.date.unique()[:21]:                     # day 20's value uses days 0-19 only
        assert scale_at(base, "TST", d) == pytest.approx(scale_at(changed, "TST", d), nan_ok=True)
    assert np.isnan(scale_at(base, "TST", p.date.unique()[0]))   # no history yet


def test_sr_levels_known_only_after_they_form():
    p = random_days(3)
    sr = build_sr(p)
    for (sym, d), levels in sr.items():
        day = p[p.date == d]
        for known, px in levels:
            if known == day.time_et.iloc[0]:           # prior-day levels: from the previous day only
                prev = p[p.date < d]
                assert px in set(prev.high) | set(prev.low) | set(prev.close)
            else:                                      # today's level: inside bars already finished
                seen = day[day.time_et < known]
                assert seen.low.min() <= px <= seen.high.max()


def test_swing_pivots_are_confirmed_after_the_swing():
    b = random_days(1)
    for known, t, kind, px in pivots(b):
        assert known >= t + pd.Timedelta(minutes=45)  # pivot bar + two 15-minute bars
        done = b[b.time_et < known]
        assert done.low.min() <= px <= done.high.max()


def test_rug_short_hits_target():
    # tap 101 from below, confirm under it, then fall far: target = 3R
    path = [100.0] * 2 + [100.98] + [100.5] * 2 + list(np.linspace(100.5, 95, 30)) + [95] * 30
    tr = rug_sim(bars(path, spread=0.0), DAY + pd.Timedelta(hours=9, minutes=30),
                 DAY + pd.Timedelta(hours=10), 101.0, short=True, scale=0.01)
    assert tr["exit_how"] == "target" and tr["r"] == pytest.approx(3.0)


def test_rug_same_bar_stop_and_target_counts_as_stop():
    path = [100.0, 100.0, 100.98, 100.5, 100.5, 100.5]
    b = bars(path, spread=0.0)
    b.loc[4, ["high", "low"]] = [102.0, 90.0]           # one bar spans both stop and target
    tr = rug_sim(b, DAY + pd.Timedelta(hours=9, minutes=30), DAY + pd.Timedelta(hours=10),
                 101.0, short=True, scale=0.01)
    assert tr["exit_how"] == "stop" and tr["r"] == pytest.approx(-1.0)


def test_edge_long_from_floor_reaches_midpoint():
    path = [100.5] * 2 + [100.05] + [100.4] * 2 + list(np.linspace(100.4, 102, 20)) + [102] * 10
    tr = edge_sim(bars(path, spread=0.0), DAY + pd.Timedelta(hours=9, minutes=30),
                  DAY + pd.Timedelta(hours=10), 100.0, True, 101.5, 1.0)
    assert tr["exit_how"] == "target" and tr["exit"] == 101.5 and tr["r"] > 0


def test_classify_finds_rug_and_skips_unnamed_ceiling():
    strikes = np.arange(95, 106)
    s = pd.DataFrame(dict(strike=strikes, net_gamma=0.0, node_type="normal"))
    s.loc[s.strike == 101, ["net_gamma", "node_type"]] = [50.0, "gatekeeper"]
    s.loc[s.strike == 99, ["net_gamma", "node_type"]] = [-40.0, "barney"]
    assert ("rug", 101) in classify(s, 100.2, 0.02)
    s.loc[s.strike == 101, "node_type"] = "normal"
    assert not any(name == "rug" for name, _ in classify(s, 100.2, 0.02))
