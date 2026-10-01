#!/usr/bin/env python3
"""Pull out-of-sample Skylit data (Oct 2025 - Jun 2026) for re-testing the studies.

Matches data/raw/watchlist: Heatseeker gamma replay (nearest 5 expirations,
92 strikes) every 30 minutes 10:00-15:00 ET, plus Atlas one-minute bars.
Ten symbols share each replay call (5 credits per call).

Spending is capped: the run stops before the account balance drops below
--floor. Re-running resumes; finished days are skipped.

Usage: python3 scripts/pull_oos.py --floor 72353
Writes data/raw/oos/{gamma_snapshots,price_bars_1min}.parquet.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "oos"
PARTS = OUT / "parts"
ET = ZoneInfo("America/New_York")
SYMBOLS = ["SPY", "QQQ", "SPXW", "IWM", "NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA"]
ATLAS = {"SPXW": "SPX"}
TIMES = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)][:-1]   # 10:00..15:00
START, END = date(2025, 10, 2), date(2026, 6, 30)


class Budget(Exception):
    pass


class Client:
    def __init__(self, floor):
        self.floor, self.balance = floor, None
        key = os.environ.get("SKYLIT_API_KEY")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}

    def get(self, url, params, cost):
        if self.balance is not None and self.balance - cost < self.floor:
            raise Budget(f"balance {self.balance} would drop below floor {self.floor}")
        req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=self.headers)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    left = r.headers.get("X-Credits-Remaining")
                    if left is not None:
                        self.balance = int(left)
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt * 2)
                    continue
                raise RuntimeError(f"{e.code} {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2 ** attempt * 2)
        raise RuntimeError(f"gave up on {url} {params}")


def trading_days(c):
    # daily bars are stamped at 00:00 UTC, so read the date in UTC, not ET
    t0 = int(datetime.combine(START - timedelta(days=1), datetime.min.time(), ET).timestamp())
    t1 = int(datetime.combine(END + timedelta(days=1), datetime.min.time(), ET).timestamp())
    d = c.get("https://atlas-api.skylit.ai/v1/history",
              dict(symbol="SPY", resolution="D", **{"from": t0, "to": t1}), 1)
    days = {datetime.fromtimestamp(t, ZoneInfo("UTC")).date() for t in d["t"]}
    return sorted(x for x in days if START <= x <= END)


def pull_bars(c, days):
    path = OUT / "price_bars_1min.parquet"
    if path.exists():
        return
    frames = []
    for sym in SYMBOLS:
        for i in range(0, len(days), 60):
            chunk = days[i:i + 60]
            t0 = int(datetime.combine(chunk[0], datetime.min.time(), ET).timestamp())
            t1 = int(datetime.combine(chunk[-1] + timedelta(days=1), datetime.min.time(), ET).timestamp())
            d = c.get("https://atlas-api.skylit.ai/v1/history",
                      dict(symbol=ATLAS.get(sym, sym), resolution="1", **{"from": t0, "to": t1}), 1)
            if d.get("s") != "ok":
                print(f"bars {sym} {chunk[0]}: {d.get('s')}")
                continue
            df = pd.DataFrame(dict(t=d["t"], open=d["o"], high=d["h"], low=d["l"], close=d["c"]))
            df["time_et"] = (pd.to_datetime(df.t, unit="s", utc=True)
                             .dt.tz_convert(ET).dt.tz_localize(None))
            df["symbol"] = sym
            frames.append(df.drop(columns="t"))
        print(f"bars {sym} done, balance {c.balance}", flush=True)
    b = pd.concat(frames)
    b = b[(b.time_et.dt.time >= pd.Timestamp("09:30").time()) &
          (b.time_et.dt.time < pd.Timestamp("16:00").time())]
    b = b.drop_duplicates(["symbol", "time_et"]).sort_values(["symbol", "time_et"])
    b["symbol"] = b.symbol.astype("category")
    b[["time_et", "symbol", "open", "high", "low", "close"]].to_parquet(path, index=False)


def pull_snapshot(c, day, hhmm):
    at_et = datetime.combine(day, datetime.strptime(hhmm, "%H:%M").time(), ET)
    at = at_et.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    d = c.get("https://api.skylit.ai/v1/historical",
              dict(symbols=",".join(SYMBOLS), at=at, metric="gamma"), 5)
    rows = []
    for s in d["data"]["symbols"]:
        for k in s["strikes"]:
            rows.append((at_et.replace(tzinfo=None), s["symbol"], s["spot"], k["strike"],
                         round(k["value"]), k["nodeType"]))
    return rows


def pull_maps(c, days):
    PARTS.mkdir(parents=True, exist_ok=True)
    for day in sorted(days, reverse=True):          # newest first: a budget stop leaves a contiguous block
        part = PARTS / f"{day}.parquet"
        if part.exists():
            continue
        with ThreadPoolExecutor(2) as ex:            # Skylit allows 2 historical calls in flight
            rows = [r for rs in ex.map(lambda t: pull_snapshot(c, day, t), TIMES) for r in rs]
        pd.DataFrame(rows, columns=["time_et", "symbol", "spot", "strike", "net_gamma", "node_type"]
                     ).to_parquet(part, index=False)
        print(f"maps {day} done, balance {c.balance}", flush=True)


def combine():
    parts = sorted(PARTS.glob("*.parquet"))
    if not parts:
        return
    g = pd.concat(pd.read_parquet(p) for p in parts).sort_values(["time_et", "symbol", "strike"])
    for col in ("symbol", "node_type"):
        g[col] = g[col].astype("category")
    g.to_parquet(OUT / "gamma_snapshots.parquet", index=False)
    print(f"combined {len(parts)} days, {len(g):,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, required=True, help="stop before balance drops below this")
    c = Client(ap.parse_args().floor)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        days = trading_days(c)
        print(f"{len(days)} trading days {days[0]} -> {days[-1]}, balance {c.balance}", flush=True)
        pull_bars(c, days)
        pull_maps(c, days)
    except Budget as e:
        print(f"stopped: {e}")
    finally:
        combine()
        print(f"final balance {c.balance}")


if __name__ == "__main__":
    main()
