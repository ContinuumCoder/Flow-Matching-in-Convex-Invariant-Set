

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


SKIP_TOP_LEVEL = {"config"}


def flatten(prefix: str, value: object, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, out)
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, float]] = []
    for path in args.paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        flat: dict[str, float] = {}
        for key, value in payload.items():
            if key not in SKIP_TOP_LEVEL:
                flatten(key, value, flat)
        rows.append(flat)

    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        summary[key] = {
            "mean": mean(vals),
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    payload = {
        "num_runs": len(rows),
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

