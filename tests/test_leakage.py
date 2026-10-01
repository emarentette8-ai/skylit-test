"""No-lookahead check: recompute every pre-decision feature from data TRUNCATED at the
decision time (bars after the entry bar and snapshots after tau removed) and require
equality with the stored features. Also checks outcome windows start after tau."""
import numpy as np
import pandas as pd
import pytest

from study import build

SYMS = ["QQQ", "SPXW"]


@pytest.fixture(scope="module")
def data():
    g, p = build.load()
    return g[g.symbol.isin(SYMS)], p[p.symbol.isin(SYMS)]


@pytest.mark.parametrize("sym", SYMS)
def test_features_identical_on_truncated_data(data, sym):
    g, p = data
    days = sorted(p[p.symbol == sym].date.unique())[::8]      # 8 spread-out days
    g, p = g[g.date.isin(days)], p[p.date.isin(days)]
    ev = build.build_symbol(sym, g, p)
    ev = ev[(ev.event == "entry") & (ev.eligible == 1)]
    assert len(ev) > 50 and set(ev.level) == {"real", "placebo"}
    rng = np.random.default_rng(0)
    for _, r in ev.iloc[rng.choice(len(ev), 60, replace=False)].iterrows():
        pday = p[(p.symbol == sym) & (p.date == r.date)]
        trunc = pday[pday.minute <= r.t_entry_bar]                 # nothing after the entry bar
        a = build.day_arrays(trunc)
        f = build.price_features(a, r.t_entry_bar, r.s, r.x_entry)
        gd = g[(g.symbol == sym) & (g.date == r.date) & (g.minute <= r.tau)]
        snaps = {m: {"strike": sd.sort_values("strike").strike.to_numpy(),
                     "gamma": sd.sort_values("strike").net_gamma.to_numpy(float),
                     "node": sd.sort_values("strike").node_type.to_numpy()}
                 for m, sd in gd.groupby("minute")}
        sm = sorted(snaps)
        si = sm[-1]
        assert si == r.snap_min and si <= r.tau
        prev = snaps[sm[-2]] if len(sm) > 1 else None
        f.update(build.gamma_features(snaps[si], prev, r.K, a["close"][r.t_entry_bar], r.s))
        sp = build.spacing(snaps[si]["strike"])
        f["f_round5"] = float(np.isclose((r.K / sp) % 5, 0))
        f["f_round10"] = float(np.isclose((r.K / sp) % 10, 0))
        # eligibility itself only needs the snapshot at tau
        listed = snaps[si]["strike"] if r.level == "real" else build.placebo_levels(snaps[si]["strike"])
        assert np.isclose(listed, r.K).any()
        for k, v in f.items():
            assert np.isclose(v, r[k], equal_nan=True), (k, v, r[k])


def test_outcome_window_starts_after_decision(data):
    g, p = data
    pday = p[(p.symbol == "QQQ")]
    d = pday.date.iloc[0]
    a = build.day_arrays(pday[pday.date == d])
    te, s = 100, 1.0
    y = build.outcomes(a, te, s, a["close"][te], 0.05)
    assert y["y_p0"] == a["open"][te + 1]                   # executable = next bar open
    assert np.isclose(y["y_r1"], 100 * (a["close"][te + 1] / a["open"][te + 1] - 1))
