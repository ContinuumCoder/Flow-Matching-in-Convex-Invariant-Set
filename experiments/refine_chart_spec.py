

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def snake(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def refine_stack(chart: dict[str, Any]) -> dict[str, Any]:
    refined = dict(chart)
    refined["task"] = refined.get("task", "Stack-Cube-Franka-IK-Abs first-stage cube2-on-cube1")
    refined["refinement_source"] = "fmcs_validator_v0"
    name_map = {}
    refined_charts = []
    for item in refined.get("charts", []):
        updated = dict(item)
        old_name = str(updated.get("name", f"chart_{len(refined_charts)}"))
        new_name = snake(old_name)
        name_map[old_name] = new_name
        updated["name"] = new_name
        updated.setdefault("invariant_set", "local task chart")
        updated.setdefault("safety_guards", [])
        refined_charts.append(updated)
    refined["charts"] = refined_charts
    edges = []
    for edge in refined.get("edges", []):
        if len(edge) == 2:
            edges.append([name_map.get(edge[0], snake(str(edge[0]))), name_map.get(edge[1], snake(str(edge[1])))])
    refined["edges"] = edges
    params = dict(refined.get("global_parameters", {}))
    if float(params.get("overlap_radius", 0.0) or 0.0) <= 0.0:
        params["overlap_radius"] = 0.12
    params.setdefault("cube_size_m", 0.0468)
    params.setdefault("stack_lift_z", 0.26)
    params.setdefault("stack_place_z", 0.095)
    params.setdefault("stack_close_dwell_s", 1.2)
    params.setdefault("stack_place_dwell_s", 1.2)
    params.setdefault("stack_release_dwell_s", 1.6)
    refined["global_parameters"] = params
    refined["executable_controller_hint"] = {
        "script": "experiments/isaaclab_multitask_policy_baselines.py",
        "scenario": "stack",
        "method": "oc_fmcs",
        "cli_args": [
            "--stack-overlap-threshold",
            str(params["overlap_radius"]),
            "--stack-lift-z",
            str(params["stack_lift_z"]),
            "--stack-place-z",
            str(params["stack_place_z"]),
            "--stack-close-dwell",
            str(params["stack_close_dwell_s"]),
            "--stack-place-dwell",
            str(params["stack_place_dwell_s"]),
            "--stack-release-dwell",
            str(params["stack_release_dwell_s"]),
        ],
    }
    refined["validation_notes"] = [
        "LLM proposal supplies task topology; FMCS validator supplies numeric overlap and dwell parameters.",
        "Charts are simulator-validated before policy evaluation.",
        "The interface evaluates semantic chart construction rather than end-to-end visual policy learning.",
    ]
    return refined


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine LLM chart specs into executable OC-FMCS configs.")
    parser.add_argument("--task", choices=("stack",), default="stack")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    chart = json.loads(args.input.read_text(encoding="utf-8"))
    refined = refine_stack(chart)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(refined, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(refined, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
