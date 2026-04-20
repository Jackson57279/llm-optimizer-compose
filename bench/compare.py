"""Render bench/results/*.json into bench/results/SUMMARY.md.

Picks the latest run per alias (by file mtime). Column-aligned markdown table,
suitable for pasting into a PR or the project README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt(v, prec: int = 2, unit: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}{unit}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        print(f"Not a directory: {args.results_dir}", file=sys.stderr)
        return 2

    latest: dict[str, tuple[float, Path]] = {}
    for p in sorted(args.results_dir.glob("*.json")):
        if p.name == "SUMMARY.md":
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            alias = payload["aggregate"]["alias"]
        except Exception:  # noqa: BLE001
            continue
        mtime = p.stat().st_mtime
        if alias not in latest or mtime > latest[alias][0]:
            latest[alias] = (mtime, p)

    if not latest:
        print("No benchmark results found.", file=sys.stderr)
        return 1

    rows = []
    for alias, (_, path) in sorted(latest.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        agg = payload["aggregate"]
        rows.append(
            {
                "alias": alias,
                "ok": f"{agg['n_ok']}/{agg['n_prompts']}",
                "tps_mean": agg.get("tps_mean"),
                "tps_median": agg.get("tps_median"),
                "ttft_mean": agg.get("ttft_mean_s"),
                "ttft_p95": agg.get("ttft_p95_s"),
                "total_tokens": agg.get("total_completion_tokens"),
                "file": path.name,
            }
        )

    lines = [
        "# miniforge benchmark summary",
        "",
        f"Source: `{args.results_dir}`",
        "",
        "| Alias | OK | Mean TPS | Median TPS | TTFT mean | TTFT p95 | Total tokens | Source file |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {alias} | {ok} | {tm} | {tmd} | {tf} | {tp} | {tt} | `{f}` |".format(
                alias=r["alias"],
                ok=r["ok"],
                tm=_fmt(r["tps_mean"]),
                tmd=_fmt(r["tps_median"]),
                tf=_fmt(r["ttft_mean"], unit="s"),
                tp=_fmt(r["ttft_p95"], unit="s"),
                tt=_fmt(r["total_tokens"], 0),
                f=r["file"],
            )
        )
    lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    for line in lines[3:]:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
