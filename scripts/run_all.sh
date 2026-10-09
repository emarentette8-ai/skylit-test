#!/usr/bin/env bash
# Rebuild every study table for both data periods, then score the out-of-sample plan.
#   Jul-Sep 2026      -> results/tables/
#   Oct 2025-Jun 2026 -> results/tables/oos/   (SKYLIT_DATA=oos)
set -euo pipefail
cd "$(dirname "$0")"
for set in "" oos; do
  echo "== ${set:-watchlist}"
  for s in node_claims rug confluence trade_features ote trinity edges king_reject rug_targets twin_nodes; do
    echo "-- $s"
    SKYLIT_DATA=$set python3 "$s.py" > /dev/null
  done
  echo "-- filters flags"
  SKYLIT_DATA=$set python3 filters.py flags > /dev/null
done
python3 filters.py report > /dev/null
python3 bias_checks.py > /dev/null
python3 losers.py > /dev/null
python3 oos_verdict.py
