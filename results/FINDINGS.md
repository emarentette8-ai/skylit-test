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

## All filters on all setups, plus supply/demand (Oct 9 2026)

`scripts/filters.py` tags every trade from every study (rug, non-King rug x3 targets,
OTE, edge fade, King rejection, twin nodes) with nine filters, including new
supply/demand zones (`scripts/supply_demand.py`). Rule fixed before the run: a filter
works if it raises both win rate and mean R in both periods with >= 20 trades.
Tables: `results/tables/oos/filters_eval.csv`, `filters_combos.csv`.

- 14 filter/setup pairs passed. Random filters pass ~10 on average. With one random
  draw per underlying trade (the non-King rug targets share entries), random filters
  reach >= 14 passes in 26.5% of runs, so the passes are consistent with chance
  (`scripts/bias_checks.py`; an earlier version drew per family and gave 9.5%).
- Supply/demand fired on 3-10% of trades and did not help consistently (it passed
  only for OTE; pooled across setups it lowered the Oct-Jun win rate, 43% vs 47%).
- Stacking raises win rates but not reliably R. Best stacks: Rug + S/R + negative
  King 63% / +0.48R (19 trades) and 42% / +0.13R (59); King rejection + negative King
  + Trinity 61% / -0.04R (18) and 63% / -0.06R (64). No stack has an R interval above 0.
- Both periods were used to choose, so none of this is confirmed; it needs new data.

## Bias checks (Oct 9 2026, `results/tables/oos/bias_checks.txt`)

- **Outliers:** the best stack (Rug + S/R + negative King) has median R +0.03 and
  -1.06, and its mean falls to +0.03 / -0.02 without its 3 best trades. The profit is
  three trades, not a pattern. QQQ is 39-53% of its trades.
- **Direction:** some filters pick more longs (Trinity: 59-61% longs vs 44-48%), but
  most uplifts appear within longs and within shorts, and pooled longs did no better
  than shorts in either period, so market drift is not the main driver.
- **Fills:** entering at the next bar's open instead of the signal close changes
  results by <= 0.02R; no trade would already be past its stop.
- **Universe:** Jul-Sep's 50 tickers split 23 up / 27 down (median -2%); Oct-Jun's 10
  rose (median +9.5%). Both lists were chosen recently (hindsight selection).
- **Not modelled:** stop slippage (stops fill exactly), option spreads (cost is a
  0.02% share cost), and map latency (maps every 30 minutes).

## What losing trades have in common (Oct 9 2026, `results/tables/oos/losers_summary.txt`)

All setups pooled: 5,836 trades (58% losers) and 5,031 (57%). A factor counts if the
same bucket has the highest loss rate in both periods, >= 3 points above the rest,
>= 100 trades. Shuffled outcomes give 0.3 such factors on average; real: 2.

- **Tight stops** (< 0.4 typical 30-minute ranges): 68% / 76% losers, -0.29R / -0.67R.
  Almost all are OTE trades on small swing legs. Part is cost (a fixed 0.02% is a big
  share of a small risk: gross -0.06R vs net -0.29R in Jul-Sep), but in Oct-Jun they
  lose even before costs (-0.28R gross).
- Not in an OTE zone: 54% vs 48-50% losers, but the OTE-zone group is small (77/108).
- 82-83% of losers exit at the stop; 13-14% drift to the close.
- Not consistent: hour, weekday, side, index vs stock, with/against the day's move,
  move already done, day range so far.

## No-entry rules from the loser analysis (Oct 9 2026, `results/tables/oos/no_entry.csv`)

Skip: entries at/after 15:00; index maps where the King holds a top-quarter share of
total |gamma| (cut-off 0.23, set on Jul-Sep); stops < 0.4 typical range; trading
toward an unfilled gap. All setups pooled:

| | Jul-Sep | Oct-Jun |
|---|---|---|
| All trades | 44.5% win, -0.10R (5,836) | 46.6% win, -0.12R (5,031) |
| All rules | 45.4% win, -0.06R (3,470) | 47.5% win, -0.08R (3,178) |
| Same number removed at random | -0.10R | -0.12R |

The rules remove losers better than chance (+0.05R in both periods) but nothing turns
positive; no setup is above 0 in both periods. The King-share rule alone did not help
(R unchanged, index win rate lower). Three of the four rules came from these same
trades, so the improvement is in-sample.

## Not yet tested

- **Front-expiry (0DTE) maps.** Every test used the 5-expiration net. Skylit's own guide
  says day trades should read the front column only. Cost: 5 credits per map time for
  SPXW/SPY/QQQ together.
- Forward tracking of the two skip rules above on live days.
