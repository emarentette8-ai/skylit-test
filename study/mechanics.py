"""Options hedging mechanics used for the sanity checks (spec section 12).

Black-Scholes-Merton with continuous carry q, European exercise. The empirical study
cannot reprice positions (no chains, IV, OI or dealer inventory in the data), so this
module verifies the *logic* of the hedge-demand definitions, not market results.
"""
import numpy as np
from scipy.stats import norm


def _d1(S, K, T, sigma, r=0.0, q=0.0):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def delta(S, K, T, sigma, right, r=0.0, q=0.0):
    """Delta of ONE LONG option (per unit of underlying)."""
    d1 = _d1(S, K, T, sigma, r, q)
    if right == "C":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, sigma, r=0.0, q=0.0):
    """Gamma of ONE LONG option -- identical for calls and puts, always > 0."""
    d1 = _d1(S, K, T, sigma, r, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def hedge(S, positions):
    """H = -sum q_i M_i delta_i : dealer's delta-neutral hedge in underlying units.

    positions: iterable of dicts {q (signed dealer contracts, + = dealer long), M, K, T,
    sigma, right, r, q_div}. Positive H = hedge is long underlying.
    """
    return -sum(p["q"] * p["M"] * delta(S, p["K"], p["T"], p["sigma"], p["right"],
                                        p.get("r", 0.0), p.get("q_div", 0.0))
                for p in positions)


def rehedge_existing(S0, S1, positions):
    """Delta-H for FROZEN inventory: + = modeled buying, - = modeled selling.
    Returns (net, gross) where gross = sum |q M d_delta| (components can offset)."""
    comps = []
    for p in positions:
        args = (p["K"], p["T"], p["sigma"], p["right"], p.get("r", 0.0), p.get("q_div", 0.0))
        comps.append(-p["q"] * p["M"] * (delta(S1, *args) - delta(S0, *args)))
    return float(sum(comps)), float(sum(abs(c) for c in comps))


def shock_demand(S, r, positions):
    """P_t(r): hedge change under an instantaneous shock S -> S(1+r), other inputs frozen."""
    return rehedge_existing(S, S * (1 + r), positions)


def gex_dollar_1pct(S, positions):
    """GEX_{$,1%} = 0.01 S^2 sum q_i M_i Gamma_i (dealer-signed)."""
    return 0.01 * S**2 * sum(p["q"] * p["M"] * gamma(S, p["K"], p["T"], p["sigma"],
                                                     p.get("r", 0.0), p.get("q_div", 0.0))
                             for p in positions)


def inventory_split(q0, q1, d0, d1):
    """Exact identity q1 d1 - q0 d0 = q0 (d1 - d0) + d1 (q1 - q0) (endpoint convention)."""
    return q0 * (d1 - d0), d1 * (q1 - q0)
