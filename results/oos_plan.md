# Out-of-sample plan (written before the Oct 2025 - Jun 2026 data was analyzed)

Data: `data/raw/oos` from `scripts/pull_oos.py` — Heatseeker gamma replay every
30 min 10:00-15:00 ET (nearest 5 expirations, 92 strikes) and Atlas 1-minute
bars for SPY, QQQ, SPXW, IWM, NVDA, AAPL, MSFT, AMZN, META, TSLA.
Run every study unchanged with `SKYLIT_DATA=oos`; outputs go to `results/tables/oos/`.
No rule, threshold or definition is changed after looking at the new data.

The Jul-Sep 2026 findings to confirm or reject:

| # | Hypothesis (from Jul-Sep) | Passes if, on Oct 2025 - Jun 2026 |
|---|---|---|
| H1 | Rug-family trades (setups + controls, SPY/QQQ/SPXW) do better when the King is negative | mean R (net) for negative-King trades > 0 with day-bootstrap 95% CI above 0, and higher than positive-King trades |
| H2 | The Rug works better when the ceiling sits at price S/R | mean R (net) for Rug-at-S/R trades > 0 with CI above 0 |
| H3 | Rug + Reverse Rug do better when the King is between entry and target than behind the entry | mean R (net) "between" > "behind (not the level)" |
| H4 | From the 15:00 map, the close ends nearer the King more often than a mirror level | toward_king > toward_mirror at hour 15 |
| H5 | The Rug and Reverse Rug on the indexes are profitable (Skylit docs' backtest claim) | mean R (net) > 0 with CI above 0 |

Null results also confirmed from Jul-Sep, expected to stay null: node touch hold
rates ~ ordinary strikes; no positive-vs-negative regime effect on next-hour range.

Five hypotheses are tested, so a single pass at the 95% level is weak evidence on
its own; we report all five whatever the outcome.

## Result (added after the run; nothing above was changed)

179 trading days, Oct 13 2025 - Jun 30 2026. The pull itself used 9,973 credits
(balance 82,322 -> 72,349); 10,004 including earlier test calls this session. Full table:
`results/tables/oos/verdict.csv` (`scripts/oos_verdict.py`).

| # | Jul-Sep 2026 | Oct 2025 - Jun 2026 | Verdict |
|---|---|---|---|
| H1 negative King | +0.30R (n=88), CI +0.06..+0.54 | -0.08R (n=261), CI -0.25..+0.09 | fail |
| H2 Rug at S/R | +0.47R (n=25) | +0.05R (n=57), CI -0.34..+0.46 | fail |
| H3 King between vs behind | +0.47R vs -0.56R | +0.02R vs +0.02R | fail |
| H4 15:00 King pull | 47.3% vs 42.6% mirror | 44.1% vs 48.9% mirror | fail |
| H5 Rug / Reverse Rug | +0.14R / +0.24R | -0.11R / -0.06R | fail |

The expected nulls held: node touches hold about as often as ordinary strikes,
and gamma regime does not change the next hour's range.

## Audit note (Oct 5 2026)

A look-ahead in the typical-range scaling was removed (see `results/FINDINGS.md`) and
`verdict.csv` was rebuilt. All five still fail out of sample. H3 now shows "passed" by
0.002R (-0.042 vs -0.044) only because its rule had no minimum margin; it is treated as failed.
