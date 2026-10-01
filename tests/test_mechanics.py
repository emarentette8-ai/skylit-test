"""Section 12 sanity checks: numerical/logic checks, separate from market results."""
import numpy as np

from study import mechanics as m
from study.events import run_state_machine

BASE = dict(K=100.0, T=5 / 365, sigma=0.25, M=100, r=0.04, q_div=0.01)


def pos(q, right, **kw):
    return {**BASE, **kw, "q": q, "right": right}


def test_long_call_and_put_gamma_positive():
    for S in (90, 99, 100, 101, 110):
        assert m.gamma(S, 100, BASE["T"], 0.25) > 0


def test_position_sign_reverses_hedge_adjustment():
    up_short, _ = m.rehedge_existing(100, 100.5, [pos(-10, "C")])   # dealer short gamma
    up_long, _ = m.rehedge_existing(100, 100.5, [pos(+10, "C")])    # dealer long gamma
    dn_short, _ = m.rehedge_existing(100, 99.5, [pos(-10, "P")])
    assert up_short > 0 and up_long < 0 and dn_short < 0           # table in spec s.3
    assert np.isclose(up_short, -up_long)


def test_synthetic_forward_has_no_gamma_rehedging():
    # dealer long call + short put, same K/T: delta_c - delta_p = exp(-qT) constant
    book = [pos(+10, "C"), pos(-10, "P")]
    for S0, S1 in [(95, 100), (99.9, 100.1), (100, 106)]:   # crosses ATM -> ITM
        net, _ = m.rehedge_existing(S0, S1, book)
        assert abs(net) < 1e-9
    assert abs(m.gex_dollar_1pct(100, book)) < 1e-9


def test_relabeling_moneyness_creates_no_position():
    book = [pos(-10, "C")]
    h_otm = m.hedge(99.0, book)
    h_itm = m.hedge(101.0, book)
    # the same single position; the hedge changes continuously, no jump at K
    mid = m.hedge(100.0, book)
    assert h_otm < mid < h_itm
    eps = 1e-4   # no jump at K: change across the strike matches gamma * dS
    jump = m.hedge(100 + eps, book) - m.hedge(100 - eps, book)
    expect = 10 * 100 * m.gamma(100.0, 100, BASE["T"], 0.25, 0.04, 0.01) * 2 * eps
    assert np.isclose(jump, expect, rtol=1e-3)


def test_zero_inventory_zero_demand_and_units():
    assert m.shock_demand(100, 0.001, [pos(0, "C"), pos(0, "P")]) == (0.0, 0.0)
    one = m.shock_demand(100, 0.001, [pos(-1, "C", M=1)])[0]
    hundred = m.shock_demand(100, 0.001, [pos(-1, "C", M=100)])[0]
    assert np.isclose(hundred, 100 * one)


def test_finite_demand_is_integral_of_gamma():
    book = [pos(-10, "C"), pos(+4, "P", K=98)]
    S0, S1 = 99.0, 100.5
    net, _ = m.rehedge_existing(S0, S1, book)
    grid = np.linspace(S0, S1, 4001)
    g = sum(-p["q"] * p["M"] * m.gamma(grid, p["K"], p["T"], p["sigma"], p["r"], p["q_div"])
            for p in book)
    assert np.isclose(net, np.trapezoid(g, grid), rtol=1e-4)


def test_net_vs_gross_cancellation():
    net, gross = m.rehedge_existing(100, 100.2, [pos(-10, "C"), pos(+10, "P")])
    assert gross > 0 and abs(net) < gross


def test_inventory_identity():
    a, b = m.inventory_split(5, 8, 0.4, 0.6)
    assert np.isclose(a + b, 8 * 0.6 - 5 * 0.4)


def test_one_crossing_is_one_event_and_jumps_separate():
    b = 0.05
    # one strike; price walks up through it once
    x = np.array([[-0.20], [-0.12], [-0.04], [0.01], [0.06], [0.15]])
    ev = run_state_machine(x, np.arange(6), b)
    assert ev == [(2, 0, "entry", 1.0)]
    # same path but skipping the band entirely -> a jump, never an entry
    xj = np.array([[-0.20], [0.08]])
    assert run_state_machine(xj, np.arange(2), b) == [(1, 0, "jump", 1.0)]
    # cooldown: re-arm needs exterior close >= 5 min after entry
    xc = np.array([[-0.2], [0.0], [-0.2], [0.0], [-0.2], [-0.2], [-0.2], [-0.2], [0.0]])
    ev = run_state_machine(xc, np.arange(9), b)
    assert [e[0] for e in ev] == [1, 8]
