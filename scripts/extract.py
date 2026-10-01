#!/usr/bin/env python3
"""Flatten Skylit Heatseeker gamma map exports into CSV + a Markdown summary.

Usage: python3 scripts/extract.py data/raw/live_maps_2026-10-01.json
Writes to data/processed/<input-stem>_{strikes,summary}.csv and _summary.md.
"""
import csv
import json
import sys
from pathlib import Path

KEY_NODES = ("king", "gatekeeper", "pika", "barney")


def snapshots(doc):
    """Yield (snapshot_label, heatmap) for every heatmaps_* section."""
    for key, val in doc.items():
        if not key.startswith("heatmaps_"):
            continue
        maps = val.values() if isinstance(val, dict) else val
        for m in maps:
            yield key.removeprefix("heatmaps_"), m


def fmt_levels(strikes, node_type):
    return " ".join(f"{s['strike']:g}" for s in strikes if s["nodeType"] == node_type)


def summarize(snap, m):
    strikes = m["strikes"]
    spot = m["spot"]
    king = next((s for s in strikes if s["nodeType"] == "king"), None)
    total = sum(s["value"] for s in strikes)
    above = [s for s in strikes if s["strike"] > spot]
    below = [s for s in strikes if s["strike"] <= spot]
    top_pos = max(strikes, key=lambda s: s["value"])
    top_neg = min(strikes, key=lambda s: s["value"])
    return {
        "snapshot": snap,
        "symbol": m["symbol"],
        "asOf": m["asOf"],
        "spot": round(spot, 3),
        "priceChangePercent": round(m["priceChangePercent"], 2),
        "netGamma": round(total),
        "king": king["strike"] if king else "",
        "kingValue": round(king["value"]) if king else "",
        "kingDistPct": round((king["strike"] - spot) / spot * 100, 2) if king else "",
        "largestPositive": top_pos["strike"],
        "largestNegative": top_neg["strike"],
        "gammaAboveSpot": round(sum(s["value"] for s in above)),
        "gammaBelowSpot": round(sum(s["value"] for s in below)),
        "gatekeepers": fmt_levels(strikes, "gatekeeper"),
        "pikas": fmt_levels(strikes, "pika"),
        "barneys": fmt_levels(strikes, "barney"),
        "nearestExpiration": m["expirations"][0] if m["expirations"] else "",
    }


def main(path):
    src = Path(path)
    doc = json.loads(src.read_text())
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    strike_rows, summary_rows = [], []
    for snap, m in snapshots(doc):
        for s in m["strikes"]:
            strike_rows.append({
                "snapshot": snap, "symbol": m["symbol"], "asOf": m["asOf"],
                "spot": m["spot"], "strike": s["strike"], "value": s["value"],
                "nodeType": s["nodeType"], "velocityPct": s["velocityPct"],
            })
        summary_rows.append(summarize(snap, m))

    for name, rows in (("strikes", strike_rows), ("summary", summary_rows)):
        with open(out / f"{stem}_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    cols = ["snapshot", "symbol", "spot", "priceChangePercent", "king",
            "kingDistPct", "gatekeepers", "pikas", "barneys"]
    lines = [f"# {stem}", "", doc.get("note", ""), "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in summary_rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    (out / f"{stem}_summary.md").write_text("\n".join(lines) + "\n")
    print(f"{len(summary_rows)} maps, {len(strike_rows)} strike rows -> {out}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/live_maps_2026-10-01.json")
