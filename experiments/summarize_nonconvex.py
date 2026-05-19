

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


METRICS = [
    "path_safe_rate",
    "terminal_feasible_rate",
    "target_hit_rate",
    "mean_outside_fraction",
    "mean_path_length",
    "mean_smoothness",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.paths]
    methods = [key for key in records[0].keys() if key != "config"]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        summary[method] = {}
        for metric in METRICS:
            vals = [float(record[method][metric]) for record in records]
            summary[method][metric] = {
                "mean": mean(vals),
                "std": pstdev(vals) if len(vals) > 1 else 0.0,
            }

    payload = {
        "num_runs": len(records),
        "inputs": [str(path) for path in args.paths],
        "summary": summary,
    }
    text = json.dumps(payload, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()

