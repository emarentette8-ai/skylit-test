"""Shared statistics for the trade studies."""
import numpy as np


def day_ci(df, time_col="map_time", col="r_net", n=5000, seed=0):
    """Day-clustered bootstrap 95% CI of the per-trade mean of `col`.

    Whole trading days are resampled, because trades on the same day share
    market conditions and are not independent.
    """
    if len(df) < 2:
        return np.nan, np.nan
    g = df.groupby(df[time_col].dt.date)[col].agg(["sum", "size"])
    s, k = g["sum"].values, g["size"].values
    rng = np.random.default_rng(seed)
    means = [s[i].sum() / k[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi
