Skylit data collected for the watchlist backtest
Data: Skylit (https://skylit.ai). Credit "Data: Skylit" if you share it (https://www.skylit.ai/api-terms#attribution).

gamma_snapshots.parquet
  Heatseeker replay (/v1/historical, metric=gamma), every 30 minutes 10:00-15:00 ET, 64 trading days (Jul 1 - Sep 30, 2026), 50 tickers.
  Each snapshot nets the nearest 5 expirations per strike and lists 92 strikes around spot (Skylit's defaults).
  net_gamma is rounded to whole units. node_type is Skylit's classification (king, gatekeeper, pika, barney, significant, normal).
  Each snapshot is the latest one at or before the listed time (measured lag: median 1 second).

price_bars_1min.parquet
  Atlas one-minute bars, regular session, same tickers and dates. SPXW rows use the SPX index price.
  CRWD prices before Jul 2, 2026 are not split-adjusted (4-for-1); intraday comparisons are unaffected.

live_maps_2026-10-01.json
  Live Heatseeker maps used for the Oct 1 live scan and OTE check.

Times are US Eastern.

Reading the Parquet files:
  Python:  import pandas as pd; df = pd.read_parquet("gamma_snapshots.parquet")
  R:       arrow::read_parquet("gamma_snapshots.parquet")
  DuckDB:  SELECT * FROM 'gamma_snapshots.parquet' LIMIT 10;
  Excel cannot open Parquet directly; use one of the above to export a filtered CSV.

