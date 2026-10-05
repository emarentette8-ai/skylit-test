# Heatseeker trade studies: findings (after the Oct 5 2026 audit)

Data: Skylit Heatseeker gamma maps (nearest 5 expirations, every 30 min 10:00-15:00 ET)
and one-minute bars.

- **Jul-Sep 2026**: 64 days, 49 symbols (`data/raw/watchlist`). Exploratory; most ideas came from here.
- **Oct 2025-Jun 2026**: 179 days, 10 symbols (`data/raw/oos`). Untouched until `results/oos_plan.md` was written.

Rebuild everything: `scripts/run_all.sh` (about 10 minutes). Tests: `python3 -m pytest tests`.
R = profit in multiples of the risk taken, after a 0.02% round-trip cost.
Ranges are day-clustered bootstrap 95% intervals.

## Audit changes (Oct 5 2026)

1. **Removed a look-ahead.** The "typical 30-minute range" that sizes taps, stops, targets
   and S/R tolerance was a median over the whole period, so each day used future data.
   It is now the median of the previous 20 days (`node_claims.typical_range`); the first
   5 days of each period are skipped. Several Jul-Sep results weakened (below).
2. One shared bootstrap (`scripts/stats.py`) instead of four slightly different copies.
3. `pull_oos.py` reserves each call's credits under a lock, so parallel calls can no
   longer overshoot the budget floor (the Oct 1 pull went 4 credits over).
4. `tests/test_studies.py`: no-look-ahead checks for the trailing range, S/R levels and
   swing points, and stop/target checks for the trade simulators on synthetic bars.
5. `scripts/run_all.sh` rebuilds every table for both periods in order.

Note on H3: after the fix it "passes" out of sample by 0.002R (-0.042 vs -0.044), because
the pre-registered rule had no minimum margin. That is not evidence; treat H3 as failed.

## What held up

| Idea | Jul-Sep | Oct-Jun | Read |
|---|---|---|---|
| Named nodes hold on a touch like any strike | 49% vs 50% | 47% vs 49% | Nodes are not better support/resistance by themselves |
| Gamma regime doesn't change the next hour's range | no effect | no effect | Positive vs negative gamma did not predict volatility |
| Edge fades (buy floor / sell ceiling, target midpoint) lose | -0.05R | -0.10R | Reliable loss, range below 0 |
| Standalone OTE loses | -0.16R | -0.21R | Reliable loss, range below 0 |
| Rolling floors/ceilings don't predict the close | 43-53% | 47-51% | Coin flip |

## Win-rate filters that repeated in both periods (still not profitable)

| Filter on Rug + Reverse Rug | Jul-Sep | Oct-Jun |
|---|---|---|
| All trades | 46% win, -0.07R (96) | 35% win, -0.07R (348) |
| Level at price S/R | 51% win, +0.09R (41) | 40% win, -0.04R (122) |
| King 2-of-3 aligned (SPXW/SPY/QQQ) | 49% win, +0.16R (49) | 41% win, +0.01R (163) |
| King not aligned | 43% win, -0.30R (47) | 30% win, -0.15R (185) |

Both filters raise the win rate by roughly 5-10 points in both periods, but no version
has an average R reliably above zero. Their best use is as **skip rules** (avoid
non-aligned or non-S/R Rug trades), to be checked forward on new days, not as signals.
King alignment no longer helps the control trades in Jul-Sep after the audit.

## What failed out of sample (`results/tables/oos/verdict.csv`)

Negative King (H1), Rug at S/R (H2), King between entry and target (H3), 15:00 King pull
(H4), Rug / Reverse Rug profitable on the indexes (H5).

## How many things were tried

Roughly 50 comparisons across these studies (node claims, 9 trade features, S/R, OTE,
Trinity, roll states, edge sign/size, H1-H5). At that count, a few "significant" splits
are expected by chance; only results that repeat in both periods are listed above as
repeating, and none is yet a tested edge.

## Not yet tested

- **Front-expiry (0DTE) maps.** Every test used the 5-expiration net. Skylit's own guide
  says day trades should read the front column only. Cost: 5 credits per map time for
  SPXW/SPY/QQQ together.
- Forward tracking of the two skip rules above on live days.
