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
