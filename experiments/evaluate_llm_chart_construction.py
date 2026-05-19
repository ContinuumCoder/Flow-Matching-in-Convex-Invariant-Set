

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_PHASES = ("approach", "grasp", "lift", "move", "place", "release")
REQUIRED_GEOMETRY = (
    "overlap_radius",
    "stack_lift_z",
    "stack_place_z",
    "stack_close_dwell_s",
    "stack_place_dwell_s",
    "stack_release_dwell_s",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chart_text(chart: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in chart.get("charts", []):
        parts.extend(
            str(item.get(key, ""))
            for key in ("name", "goal", "entry_condition", "exit_condition", "action_template")
        )
    return " ".join(parts).lower()


def phase_hits(chart: dict[str, Any]) -> dict[str, bool]:
    text = chart_text(chart)
    return {phase: phase in text for phase in REQUIRED_PHASES}


def ordered_edge_score(chart: dict[str, Any]) -> tuple[int, int]:
    names = [str(item.get("name", "")).lower() for item in chart.get("charts", [])]
    positions: list[int] = []
    for phase in REQUIRED_PHASES:
        try:
            positions.append(next(i for i, name in enumerate(names) if phase in name))
        except StopIteration:
            return 0, len(REQUIRED_PHASES) - 1
    correct = sum(int(a < b) for a, b in zip(positions, positions[1:]))
    return correct, len(REQUIRED_PHASES) - 1


def geometry_hits(chart: dict[str, Any]) -> dict[str, bool]:
    params = chart.get("global_parameters", {})
    hits: dict[str, bool] = {}
    for key in REQUIRED_GEOMETRY:
        value = params.get(key)
        hits[key] = isinstance(value, (int, float)) and float(value) > 0.0
    return hits


def summarize_chart(path: Path) -> dict[str, Any]:
    chart = read_json(path)
    phases = phase_hits(chart)
    geometry = geometry_hits(chart)
    ordered, ordered_total = ordered_edge_score(chart)
    return {
        "path": str(path),
        "chart_count": len(chart.get("charts", [])),
        "edge_count": len(chart.get("edges", [])),
        "topology_hits": phases,
        "topology_score": sum(phases.values()),
        "topology_total": len(REQUIRED_PHASES),
        "ordered_edges": ordered,
        "ordered_edges_total": ordered_total,
        "geometry_hits": geometry,
        "geometry_score": sum(geometry.values()),
        "geometry_total": len(REQUIRED_GEOMETRY),
        "executable_atlas": all(phases.values()) and all(geometry.values()) and ordered == ordered_total,
    }


def method_row(
    name: str,
    source: str,
    topology_score: str,
    geometry_score: str,
    executable: str,
    method_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "source": source,
        "topology": topology_score,
        "geometry": geometry_score,
        "executable": executable,
        "stack_success": None,
        "cube2_lift_success": None,
        "policy_ms_per_env_step": None,
    }
    if method_metrics is not None:
        row.update(
            {
                "stack_success": method_metrics.get("success_rate"),
                "cube2_lift_success": method_metrics.get("cube2_lift_rate"),
                "policy_ms_per_env_step": method_metrics.get("policy_compute_ms_per_env_step"),
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-chart", type=Path, required=True)
    parser.add_argument("--refined-chart", type=Path, required=True)
    parser.add_argument("--stack-metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = summarize_chart(args.raw_chart)
    refined = summarize_chart(args.refined_chart)
    methods = read_json(args.stack_metrics)["methods"]

    raw_topology = f"{raw['topology_score']}/{raw['topology_total']}"
    raw_geometry = f"{raw['geometry_score']}/{raw['geometry_total']}"
    refined_topology = f"{refined['topology_score']}/{refined['topology_total']}"
    refined_geometry = f"{refined['geometry_score']}/{refined['geometry_total']}"

    rows = [
        method_row(
            "Direct horizon generator",
            "no chart atlas",
            "0/6",
            "--",
            "yes",
            methods.get("direct_gen"),
        ),
        method_row(
            "DeepSeek raw chart",
            "LLM only",
            raw_topology,
            raw_geometry,
            "no",
            None,
        ),
        method_row(
            "DeepSeek + FMCS validator (ours)",
            "LLM topology + geometric validator",
            refined_topology,
            refined_geometry,
            "yes",
            methods.get("oc_fmcs"),
        ),
        method_row(
            "Same atlas + MPPI-FSM",
            "chart-conditioned sampling ablation",
            refined_topology,
            refined_geometry,
            "yes",
            methods.get("mppi_fsm"),
        ),
    ]

    summary = {
        "raw_chart": raw,
        "refined_chart": refined,
        "stack_metrics": str(args.stack_metrics),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
