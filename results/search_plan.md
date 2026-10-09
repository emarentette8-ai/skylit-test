# Strategy search plan (written before the search ran, Oct 9 2026)

Goal: find a combination of setup and entry conditions with positive mean net R,
without fitting noise.

## Data split (by entry date)
- Design A: Jul 1 - Sep 30 2026 (`data/raw/watchlist`)
- Design B: Oct 13 2025 - Jan 31 2026 (`data/raw/oos`)
- Holdout:  Feb 1 - Jun 30 2026 (`data/raw/oos`). Not used during the search. Earlier
  studies did include these months in their totals, so the holdout is "unused by this
  search", not "never seen".

## Fixed rule (from the user)
Single-stock trades require the indexes to point the same way: the King is on the
trade's side of price on at least 2 of SPXW / SPY / QQQ (latest map at entry).

## Search
- Trades: every setup in `filters.py`'s pooled set (rug, OTE, edge fade, King
  rejection, twin nodes).
- Candidate conditions: keep-only or exclude one bucket of: setup, side, index vs
  stock, hour, stop distance, with/against the day's move, move done since the open,
  day range so far, King share of the map (indexes), index price direction (>= 2 of 3
  indexes moving the trade's way since the open), and the nine filters.
- Greedy forward selection: at each step add the condition that gives the highest
  mean net R on Design A+B, accepted only if it raises mean R in BOTH Design A and
  Design B, leaves >= 300 design trades and >= 100 in each design period, and gains
  >= 0.01R. At most 6 conditions.
- The number of candidate conditions evaluated is reported.

## Holdout test (one look)
The final rule set is applied once to the holdout. Reported: trades, win rate, mean
net R with a day-clustered 95% interval, share of positive days. Pass = mean R > 0
with the interval above 0. Whatever comes out is reported; the rules are not changed
afterwards.
