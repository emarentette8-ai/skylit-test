#!/usr/bin/env python3
"""Rank a watchlist's single-expiry gamma maps by how clean and actionable they look.

Pulls the live map for one expiration (default: this Friday) via /v1/heatmap, 10
symbols per call (1 credit each), and scores each map on the Skylit Academy's
reading rules. This is a screen for readable structure, NOT a tested signal:
the backtests in results/FINDINGS.md found no edge in these patterns.

Per map:
  king       largest |node|; distance from spot
  dominance  |King| / |largest node outside the King's own cluster (+-0.6%)|
  clutter    named-size nodes (>= 30% of the King) within +-4% of spot
  range      floor = largest |node| below spot, ceiling = largest above; position
             0 (at floor) .. 1 (at ceiling)
  pattern    rug (positive ceiling above, negative below), reverse rug (mirror),
             twin nodes (two near-equal nodes >= 1.5% apart, small nodes between)
Score = clean (dominance, low clutter) + structure (pattern, price at an edge)
        + reachable King (within 4%).

Usage: python3 scripts/weekly_scan.py 2026-10-16 SYM1,SYM2,...
Writes results/scans/scan_<expiry>_<time>.csv.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "results" / "scans"


def fetch(symbols, expiry):
    key = os.environ.get("SKYLIT_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    maps, credits = {}, None
    for i in range(0, len(symbols), 10):
        q = urllib.parse.urlencode(dict(symbols=",".join(symbols[i:i + 10]), expirations=expiry))
        req = urllib.request.Request(f"https://api.skylit.ai/v1/heatmap?{q}", headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            credits = r.headers.get("X-Credits-Remaining")
            for s in json.load(r)["data"]["symbols"]:
                maps[s["symbol"]] = s
    return maps, credits


def score(m):
    spot = m["spot"]
    s = pd.DataFrame(m["strikes"])
    if s.empty or s.value.abs().max() == 0:
        return None
    s["a"] = s.value.abs()
    k = s.loc[s.a.idxmax()]
    outside = s[(s.strike - k.strike).abs() / spot > 0.006]
    second = outside.a.max() if len(outside) else 0
    dominance = k.a / second if second else 9.9
    near = s[(s.strike - spot).abs() / spot <= 0.04]
    clutter = int((near.a >= 0.3 * k.a).sum())
    below, above = s[s.strike < spot], s[s.strike > spot]
    if below.empty or above.empty:
        return None
    fl, ce = below.loc[below.a.idxmax()], above.loc[above.a.idxmax()]
    pos = (spot - fl.strike) / (ce.strike - fl.strike)
    # rug / reverse rug within 3%
    up = above[above.strike <= spot * 1.03]
    dn = below[below.strike >= spot * 0.97]
    pattern = ""
    if len(up) and len(dn):
        u_top, d_top = up.loc[up.a.idxmax()], dn.loc[dn.a.idxmax()]
        if u_top.value > 0 and d_top.value < 0 and dn.value.sum() < 0 and u_top.a >= 0.3 * k.a:
            pattern = f"rug (ceiling +{u_top.strike:g}, negative below {d_top.strike:g})"
        elif d_top.value > 0 and u_top.value < 0 and up.value.sum() < 0 and d_top.a >= 0.3 * k.a:
            pattern = f"reverse rug (floor +{d_top.strike:g}, negative above {u_top.strike:g})"
    # twin nodes with a gap
    twin = ""
    b2 = s[(s.strike - k.strike).abs() / spot >= 0.015]
    if len(b2):
        b = b2.loc[b2.a.idxmax()]
        lo, hi = sorted((k.strike, b.strike))
        inner = s[(s.strike > lo * 1.006) & (s.strike < hi / 1.006)]
        if b.a / k.a >= 0.7 and lo < spot < hi and (inner.empty or inner.a.max() <= 0.5 * b.a):
            twin = f"twin nodes {k.strike:g}/{b.strike:g} ({b.a / k.a:.0%})"
    kd = 100 * (k.strike - spot) / spot
    edge = min(pos, 1 - pos)
    sc = (min(dominance, 3) / 3 * 3 + max(0, 4 - clutter) / 4 * 2 + (2 if pattern else 0) + (2 if twin else 0)
          + (1.5 if edge <= 0.15 else 0) + (1 if abs(kd) <= 4 else 0))
    return dict(symbol=m["symbol"], spot=round(spot, 2), chg_pct=round(m.get("priceChangePercent", np.nan), 2),
                king=k.strike, king_sign="+" if k.value > 0 else "-", king_dist_pct=round(kd, 2),
                king_value=round(k.value), dominance=round(dominance, 2), clutter=clutter,
                floor=f"{fl.strike:g} ({'+' if fl.value > 0 else '-'}{fl.a / k.a:.0%})",
                ceiling=f"{ce.strike:g} ({'+' if ce.value > 0 else '-'}{ce.a / k.a:.0%})",
                range_pos=round(pos, 2), pattern=pattern, twin=twin, score=round(sc, 2))


def main():
    expiry, symbols = sys.argv[1], sys.argv[2].split(",")
    maps, credits = fetch(symbols, expiry)
    rows = [r for r in (score(m) for m in maps.values()) if r]
    t = pd.DataFrame(rows).sort_values("score", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    t.to_csv(OUT / f"scan_{expiry}_{stamp}.csv", index=False)
    missing = sorted(set(symbols) - set(maps))
    pd.set_option("display.width", 250, "display.max_rows", 100, "display.max_colwidth", 60)
    print(t.to_string(index=False))
    print(f"\nno map for: {missing}  | credits left {credits}")


if __name__ == "__main__":
    main()
