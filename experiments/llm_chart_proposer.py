

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STACK_TASK = """\
Task: Stack-Cube-Franka-IK-Abs first stage.
Scene: a Franka gripper, a table, cube_1 as lower support, cube_2 as the object
to pick and place on cube_1, and cube_3 ignored for the first stage.
Available observations: end-effector pose, gripper opening, cube_1 pose,
cube_2 pose, cube_3 pose.
Action interface: absolute end-effector target position, target quaternion,
and binary gripper command where +1 opens and -1 closes.
Goal: lift cube_2, place it on cube_1, release without knocking it away.
"""


DRAWER_TASK = """\
Task: Open-Drawer-Franka-IK-Abs.
Scene: a Franka gripper, a cabinet with a drawer handle, and a table.
Available observations: end-effector pose, gripper opening, handle pose,
drawer joint position.
Action interface: absolute end-effector target position, target quaternion,
and binary gripper command where +1 opens and -1 closes.
Goal: approach the handle, establish contact, grasp, pull in the opening
direction, and avoid losing handle contact.
"""


def heuristic_stack_chart() -> dict[str, Any]:
    return {
        "task": "Stack-Cube-Franka-IK-Abs first-stage cube2-on-cube1",
        "prior_source": "heuristic_fallback",
        "required_observations": ["eef_pose", "gripper_gap", "cube_1_pose", "cube_2_pose"],
        "charts": [
            {
                "name": "approach_cube2",
                "goal": "move above cube_2 with open gripper",
                "invariant_set": "eef_xy within workspace; eef_z above table; gripper open",
                "entry_condition": "episode_start",
                "exit_condition": "||eef_pos - (cube2_pos + [0,0,0.14])|| < overlap_radius",
                "action_template": "target_pos = cube2_anchor + [0,0,0.14]; gripper = +1",
                "safety_guards": ["z >= table_height + 0.03", "stay inside robot workspace"],
            },
            {
                "name": "descend_to_grasp",
                "goal": "align fingers around cube_2",
                "invariant_set": "eef remains in local vertical prism over cube_2",
                "entry_condition": "inside approach_cube2 overlap",
                "exit_condition": "||eef_pos - (cube2_pos + [0,0,0.018])|| < overlap_radius",
                "action_template": "target_pos = cube2_anchor + [0,0,0.018]; gripper = +1",
                "safety_guards": ["slow vertical descent", "do not sweep laterally near objects"],
            },
            {
                "name": "close_and_lift",
                "goal": "close gripper and lift cube_2 above table",
                "invariant_set": "cube_2 remains near end-effector; gripper closed",
                "entry_condition": "inside grasp overlap",
                "exit_condition": "cube2_z - cube2_start_z > 0.05",
                "action_template": "hold grasp pose, close, then target_pos = cube2_anchor + [0,0,0.26]",
                "safety_guards": ["maintain gripper closed", "abort if cube_2 slips far from eef"],
            },
            {
                "name": "transport_above_cube1",
                "goal": "move cube_2 over cube_1 without contact",
                "invariant_set": "cube_2 lifted; eef path stays above tabletop and away from cube_3",
                "entry_condition": "cube_2 lifted",
                "exit_condition": "||eef_pos - (cube1_pos + [0,0,0.20])|| < overlap_radius",
                "action_template": "target_pos = cube1_anchor + [0,0,0.20]; gripper = -1",
                "safety_guards": ["keep cube_2 above cube_1", "avoid lateral collision with cube_3"],
            },
            {
                "name": "place_and_release",
                "goal": "lower cube_2 onto cube_1 and open gripper",
                "invariant_set": "cube_2 xy aligned with cube_1; controlled vertical release",
                "entry_condition": "inside transport overlap",
                "exit_condition": "xy_error(cube2,cube1) < 0.05 and |height_delta - cube_size| < 0.025",
                "action_template": "target_pos = cube1_anchor + [0,0,0.095]; release after dwell",
                "safety_guards": ["release only after dwell", "retreat vertically after opening"],
            },
        ],
        "edges": [
            ["approach_cube2", "descend_to_grasp"],
            ["descend_to_grasp", "close_and_lift"],
            ["close_and_lift", "transport_above_cube1"],
            ["transport_above_cube1", "place_and_release"],
        ],
        "global_parameters": {
            "overlap_radius": 0.12,
            "cube_size_m": 0.0468,
            "close_dwell_s": 1.2,
            "place_dwell_s": 1.2,
            "release_dwell_s": 1.6,
        },
    }


def heuristic_drawer_chart() -> dict[str, Any]:
    return {
        "task": "Open-Drawer-Franka-IK-Abs",
        "prior_source": "heuristic_fallback",
        "required_observations": ["eef_pose", "gripper_gap", "handle_pose", "drawer_joint"],
        "charts": [
            {
                "name": "approach_handle",
                "goal": "move in front of handle with open gripper",
                "entry_condition": "episode_start",
                "exit_condition": "eef reaches handle approach overlap",
                "action_template": "target_pos = handle_pos + [-0.10,0,0]; gripper = +1",
                "invariant_set": "eef stays in front of cabinet face",
                "safety_guards": ["avoid cabinet collision"],
            },
            {
                "name": "handle_contact",
                "goal": "move to handle contact pose",
                "entry_condition": "inside approach overlap",
                "exit_condition": "eef near handle frame",
                "action_template": "target_pos = handle_pos; gripper = +1",
                "invariant_set": "local handle prism",
                "safety_guards": ["small approach speed"],
            },
            {
                "name": "grasp_handle",
                "goal": "close gripper on handle",
                "entry_condition": "inside handle contact overlap",
                "exit_condition": "gripper closed and eef near handle",
                "action_template": "target_pos = handle_pos + [0.025,0,0]; gripper = -1",
                "invariant_set": "maintain contact around handle",
                "safety_guards": ["do not pull before closure"],
            },
            {
                "name": "pull_drawer",
                "goal": "pull handle along drawer opening direction",
                "entry_condition": "handle grasped",
                "exit_condition": "drawer_joint > 0.25",
                "action_template": "target_pos = handle_pos + [-0.30,0,0]; gripper = -1",
                "invariant_set": "pull direction aligned with drawer joint",
                "safety_guards": ["maintain gripper closed", "limit lateral force"],
            },
        ],
        "edges": [["approach_handle", "handle_contact"], ["handle_contact", "grasp_handle"], ["grasp_handle", "pull_drawer"]],
        "global_parameters": {"overlap_radius": 0.025, "pull_success_m": 0.25},
    }


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = first_balanced_json(text)
        if candidate is None:
            raise
        return json.loads(candidate)


def first_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def validate_chart(chart: dict[str, Any]) -> None:
    required = ["task", "required_observations", "charts", "edges", "global_parameters"]
    missing = [key for key in required if key not in chart]
    if missing:
        raise ValueError(f"missing top-level keys: {missing}")
    if not isinstance(chart["charts"], list) or not chart["charts"]:
        raise ValueError("charts must be a non-empty list")
    for idx, chart_item in enumerate(chart["charts"]):
        needed = ["name", "goal", "entry_condition", "exit_condition", "action_template", "safety_guards"]
        missing = [key for key in needed if key not in chart_item]
        if missing:
            raise ValueError(f"chart {idx} missing keys: {missing}")


def build_prompt(task_text: str) -> str:
    return f"""You design chart graphs for safe robot generative policies.
Return valid JSON only. No Markdown.

The JSON schema:
{{
  "task": "...",
  "prior_source": "llm_proposed",
  "required_observations": ["..."],
  "charts": [
    {{
      "name": "...",
      "goal": "...",
      "invariant_set": "...",
      "entry_condition": "...",
      "exit_condition": "...",
      "action_template": "...",
      "safety_guards": ["..."]
    }}
  ],
  "edges": [["chart_a", "chart_b"]],
  "global_parameters": {{"overlap_radius": 0.0}}
}}

Design a minimal but practical overlapping chart/FSM policy for:
{task_text}
"""


def run_deepseek_text(task_text: str, model_path: str, max_new_tokens: int) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    messages = [{"role": "user", "content": build_prompt(task_text)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose OC-FMCS robot chart/FSM JSON specs.")
    parser.add_argument("--task", choices=("stack", "drawer"), default="stack")
    parser.add_argument("--backend", choices=("heuristic", "deepseek"), default="heuristic")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--raw-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/llm_chart_proposals/chart.json"))
    args = parser.parse_args()

    task_text = STACK_TASK if args.task == "stack" else DRAWER_TASK
    if args.backend == "heuristic":
        chart = heuristic_stack_chart() if args.task == "stack" else heuristic_drawer_chart()
    else:
        model_path = args.model_path
        if model_path is None:
            model_path = str(Path.home() / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots")
            snapshots = sorted(Path(model_path).glob("*"))
            if not snapshots:
                raise FileNotFoundError(f"no DeepSeek snapshot under {model_path}")
            model_path = str(snapshots[-1])
        raw_text = run_deepseek_text(task_text, model_path, args.max_new_tokens)
        if args.raw_out is not None:
            args.raw_out.parent.mkdir(parents=True, exist_ok=True)
            args.raw_out.write_text(raw_text, encoding="utf-8")
        chart = extract_json(raw_text)
        chart.setdefault("prior_source", "llm_proposed")

    validate_chart(chart)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(chart, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(chart, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
