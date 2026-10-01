"""Render reports/strike_crossing_report.html from results/report_data.json.

The page is self-contained: data embedded as JSON, charts drawn as inline SVG by a
small script (hover tooltips, light/dark tokens). Re-run after any analysis change:
    python3 -m study.report_data && python3 -m study.report
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("report_template.html")


def main():
    data = json.load(open(ROOT / "results" / "report_data.json"))
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "reports" / "strike_crossing_report.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print("wrote", out, f"{len(html)/1024:.0f} KB")


if __name__ == "__main__":
    main()
