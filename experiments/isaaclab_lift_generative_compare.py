

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_METHODS = "direct_gen,phase_oc_feedback"
VALID_METHODS = {
    "direct_gen",
    "direct_gen_projected",
    "phase_oc_gen",
    "phase_oc_gen_projected",
    "phase_oc_feedback",
    "phase_oc_feedback_projected",
}
PHASE_NAMES = {
    0: "rest",
    1: "approach_above_object",
    2: "approach_object",
    3: "grasp_object",
    4: "lift_object",
}


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ.setdefault("OMNI_KIT_DISABLE_TELEMETRY", "1")


from isaaclab.app import AppLauncher


def parse_vec3(text: str) -> tuple[float, float, float]:
    values = [float(part) for part in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return (values[0], values[1], values[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lift-Cube generative policy comparison.")
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--fallback-task", type=str, default="Isaac-Lift-Cube-Franka-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument(
        "--methods",
        type=str,
        default=DEFAULT_METHODS,
        help="Comma-separated method names.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--disable-fabric", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results/isaaclab_lift_generative_compare"))
    parser.add_argument("--video-length", type=int, default=160)
    parser.add_argument("--camera-eye", type=parse_vec3, default=(1.35, 1.00, 1.10))
    parser.add_argument("--camera-target", type=parse_vec3, default=(0.0, 0.0, 0.42))
    parser.add_argument("--pregrasp-z-offset", type=float, default=0.20)
    parser.add_argument("--grasp-z-offset", type=float, default=0.09)
    parser.add_argument("--lift-z-offset", type=float, default=0.42)
    parser.add_argument("--feedback-above-offset", type=float, default=0.10)
    parser.add_argument("--feedback-position-threshold", type=float, default=0.025)
    parser.add_argument(
        "--store-phase-trace",
        action="store_true",
        help="Store every phase state in metrics.json. Disabled by default to keep metrics compact.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["auto", "ik_abs", "ik_rel", "joint_proxy"],
        default="auto",
        help="Action interpretation. auto uses ik_rel/ik_abs for IK task names and joint_proxy otherwise.",
    )
    parser.add_argument("--max-ee-delta", type=float, default=0.045)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.headless is None:
        args.headless = True
    args.enable_cameras = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


def unpack_reset(reset_result: Any) -> Any:
    return reset_result[0] if isinstance(reset_result, tuple) else reset_result


def unpack_step(step_result: Any) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if len(step_result) == 5:
        return step_result
    if len(step_result) == 4:
        obs, reward, done, extras = step_result
        return obs, reward, done, torch.zeros_like(done, dtype=torch.bool), extras
    raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")


def policy_tensor(obs: Any) -> torch.Tensor:
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    if isinstance(obs, torch.Tensor):
        return obs
    raise RuntimeError(f"Cannot find policy observation in {type(obs)}")


def lift_object_and_target(obs: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    policy = policy_tensor(obs)
    if policy.shape[-1] < 24:
        return None, None
    object_pos = policy[..., 18:21]
    target_pos = policy[..., 21:24]
    return object_pos, target_pos


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    while w.ndim < a.ndim:
        w = w.unsqueeze(-1)
    return (1.0 - w) * a + w * b


def clamp_workspace(pos: torch.Tensor, around: torch.Tensor, radius: float, z_min: float, z_max: float) -> torch.Tensor:
    delta = pos[..., :2] - around[..., :2]
    norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.minimum(torch.ones_like(norm), torch.tensor(radius, device=pos.device) / norm)
    xy = around[..., :2] + delta * scale
    z = pos[..., 2:3].clamp(z_min, z_max)
    return torch.cat([xy, z], dim=-1)


def make_ik_abs_action(
    method: str,
    obs: Any,
    step: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    ee_quat_ref: torch.Tensor | None,
) -> torch.Tensor:
    object_pos, target_pos = lift_object_and_target(obs)
    if object_pos is None or target_pos is None:
        raise RuntimeError("IK absolute control needs Lift-Cube object and target positions in the observation.")
    n = object_pos.shape[0]
    tau = torch.full((n,), step / max(horizon - 1, 1), device=device)
    pregrasp = object_pos + torch.tensor([0.0, 0.0, args_cli.pregrasp_z_offset], device=device)
    grasp = object_pos + torch.tensor([0.0, 0.0, args_cli.grasp_z_offset], device=device)
    lift = object_pos + torch.tensor([0.0, 0.0, args_cli.lift_z_offset], device=device)
    place = target_pos + torch.tensor([0.0, 0.0, 0.03], device=device)

    if method == "direct_gen":


        w = smoothstep(tau)
        pos = lerp(pregrasp, place, w)
        grip = 1.0 - 2.0 * smoothstep((tau - 0.18) / 0.65)
    elif method == "direct_gen_projected":
        w = smoothstep(tau)
        raw = lerp(pregrasp, place, w)
        pos = clamp_workspace(raw, object_pos, radius=0.45, z_min=0.06, z_max=0.55)
        grip = 1.0 - 2.0 * smoothstep((tau - 0.24) / 0.52)
    elif method in {"phase_oc_gen", "phase_oc_gen_projected"}:
        pos = torch.empty_like(object_pos)
        grip = torch.ones(n, device=device)
        phase0 = tau < 0.24
        phase1 = (tau >= 0.24) & (tau < 0.40)
        phase2 = (tau >= 0.40) & (tau < 0.52)
        phase3 = (tau >= 0.52) & (tau < 0.74)
        phase4 = tau >= 0.74
        pos[phase0] = pregrasp[phase0]
        pos[phase1] = lerp(pregrasp, grasp, smoothstep((tau - 0.24) / 0.16))[phase1]
        pos[phase2] = grasp[phase2]
        pos[phase3] = lerp(grasp, lift, smoothstep((tau - 0.52) / 0.22))[phase3]
        pos[phase4] = lerp(lift, place, smoothstep((tau - 0.74) / 0.26))[phase4]
        grip[phase2 | phase3 | phase4] = -1.0
        if method == "phase_oc_gen_projected":
            near_object = phase0 | phase1 | phase2 | phase3
            pos[near_object] = clamp_workspace(pos[near_object], object_pos[near_object], 0.12, 0.03, 0.50)
            pos[phase4] = clamp_workspace(pos[phase4], target_pos[phase4], 0.18, 0.18, 0.65)
    else:
        raise ValueError(f"Unknown method: {method}")

    if ee_quat_ref is not None:
        quat = ee_quat_ref.to(device=device, dtype=object_pos.dtype)
    else:
        quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).repeat(n, 1)
    action = torch.zeros((n, action_dim), device=device)
    if action_dim < 8:
        raise RuntimeError(f"IK absolute mode expected action dim >= 8, got {action_dim}")
    action[:, 0:3] = pos
    action[:, 3:7] = quat
    action[:, 7] = grip
    return action


def make_joint_proxy_action(
    method: str,
    obs: Any,
    step: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    n = policy_tensor(obs).shape[0]
    tau = torch.full((n, 1), step / max(horizon - 1, 1), device=device)
    act_ids = torch.arange(action_dim, device=device, dtype=torch.float32)[None, :]
    action = torch.zeros((n, action_dim), device=device)
    if method.startswith("direct"):
        action[:, :] = 0.22 * torch.sin(2.0 * math.pi * tau + 0.7 * act_ids)
        if action_dim >= 8:
            action[:, 7] = 1.0 - 2.0 * smoothstep((tau[:, 0] - 0.20) / 0.65)
    else:
        phase = tau[:, 0]
        action[:, 0] = -0.20
        action[:, 1] = 0.18
        action[:, 2] = 0.14 * torch.sin(math.pi * phase)
        action[:, 3:7] = 0.10 * torch.sin(3.0 * math.pi * phase[:, None] + 0.5 * act_ids[:, 3:7])
        if action_dim >= 8:
            action[:, 7] = torch.where(phase > 0.40, -torch.ones_like(phase), torch.ones_like(phase))
    if "projected" in method:
        action = action.clamp(-0.35, 0.35)
        if action_dim >= 8:
            action[:, 7] = torch.sign(action[:, 7]).clamp(-1.0, 1.0)
    return action


def make_ik_rel_action(
    method: str,
    obs: Any,
    step: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    ee_pos_curr: torch.Tensor,
) -> torch.Tensor:
    object_pos, target_pos = lift_object_and_target(obs)
    if object_pos is None or target_pos is None:
        raise RuntimeError("IK relative control needs Lift-Cube object and target positions in the observation.")
    n = object_pos.shape[0]
    tau = torch.full((n,), step / max(horizon - 1, 1), device=device)
    pregrasp = object_pos + torch.tensor([0.0, 0.0, args_cli.pregrasp_z_offset], device=device)
    grasp = object_pos + torch.tensor([0.0, 0.0, args_cli.grasp_z_offset], device=device)
    lift = object_pos + torch.tensor([0.0, 0.0, args_cli.lift_z_offset], device=device)
    place = target_pos + torch.tensor([0.0, 0.0, 0.03], device=device)

    if method == "direct_gen":
        desired = lerp(pregrasp, place, smoothstep(tau))
        grip = 1.0 - 2.0 * smoothstep((tau - 0.18) / 0.65)
    elif method == "direct_gen_projected":
        raw = lerp(pregrasp, place, smoothstep(tau))
        desired = clamp_workspace(raw, object_pos, radius=0.45, z_min=0.08, z_max=0.65)
        grip = 1.0 - 2.0 * smoothstep((tau - 0.24) / 0.52)
    elif method in {"phase_oc_gen", "phase_oc_gen_projected"}:
        desired = torch.empty_like(object_pos)
        grip = torch.ones(n, device=device)
        phase0 = tau < 0.24
        phase1 = (tau >= 0.24) & (tau < 0.42)
        phase2 = (tau >= 0.42) & (tau < 0.56)
        phase3 = (tau >= 0.56) & (tau < 0.78)
        phase4 = tau >= 0.78
        desired[phase0] = pregrasp[phase0]
        desired[phase1] = lerp(pregrasp, grasp, smoothstep((tau - 0.24) / 0.18))[phase1]
        desired[phase2] = grasp[phase2]
        desired[phase3] = lerp(grasp, lift, smoothstep((tau - 0.56) / 0.22))[phase3]
        desired[phase4] = lerp(lift, place, smoothstep((tau - 0.78) / 0.22))[phase4]
        grip[phase2 | phase3 | phase4] = -1.0
        if method == "phase_oc_gen_projected":
            near_object = phase0 | phase1 | phase2 | phase3
            desired[near_object] = clamp_workspace(desired[near_object], object_pos[near_object], 0.12, 0.06, 0.58)
            desired[phase4] = clamp_workspace(desired[phase4], target_pos[phase4], 0.18, 0.16, 0.70)
    else:
        raise ValueError(f"Unknown method: {method}")

    delta = (desired - ee_pos_curr).clamp(-args_cli.max_ee_delta, args_cli.max_ee_delta)
    action = torch.zeros((n, action_dim), device=device)
    if action_dim < 7:
        raise RuntimeError(f"IK relative lift mode expected action dim >= 7, got {action_dim}")
    action[:, 0:3] = delta
    action[:, 3:6] = 0.0
    action[:, 6] = grip
    return action


def make_action(
    method: str,
    task_name: str,
    obs: Any,
    step: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    ee_quat_ref: torch.Tensor | None,
    ee_pos_curr: torch.Tensor | None,
) -> torch.Tensor:
    mode = args_cli.control_mode
    if mode == "auto":
        if "IK-Rel" in task_name:
            mode = "ik_rel"
        elif "IK" in task_name:
            mode = "ik_abs"
        else:
            mode = "joint_proxy"
    if mode == "ik_abs":
        return make_ik_abs_action(method, obs, step, horizon, action_dim, device, ee_quat_ref)
    if mode == "ik_rel":
        if ee_pos_curr is None:
            raise RuntimeError("IK relative mode requires current end-effector position.")
        return make_ik_rel_action(method, obs, step, horizon, action_dim, device, ee_pos_curr)
    return make_joint_proxy_action(method, obs, step, horizon, action_dim, device)


def tensor_to_list(value: torch.Tensor | None) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in value.detach().cpu().reshape(-1)]


def summarize_phase_trace(trace: list[int]) -> dict[str, Any]:
    if not trace:
        return {}
    counts = {PHASE_NAMES.get(state, str(state)): trace.count(state) for state in sorted(set(trace))}
    transitions = []
    previous = trace[0]
    for step, state in enumerate(trace[1:], start=1):
        if state != previous:
            transitions.append(
                {
                    "step": step,
                    "from": PHASE_NAMES.get(previous, str(previous)),
                    "to": PHASE_NAMES.get(state, str(state)),
                }
            )
            previous = state
    return {
        "final_state": PHASE_NAMES.get(trace[-1], str(trace[-1])),
        "final_state_id": trace[-1],
        "state_counts": counts,
        "transitions": transitions,
    }


class PhaseFeedbackController:

    REST = 0
    APPROACH_ABOVE_OBJECT = 1
    APPROACH_OBJECT = 2
    GRASP_OBJECT = 3
    LIFT_OBJECT = 4

    def __init__(self, num_envs: int, dt: float, device: torch.device):
        self.dt = float(dt)
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wait = torch.zeros(num_envs, device=device)

    def make_action(
        self,
        obs: Any,
        action_dim: int,
        device: torch.device,
        ee_pos_curr: torch.Tensor,
        ee_quat_curr: torch.Tensor,
        projected: bool,
    ) -> torch.Tensor:
        object_pos, target_pos = lift_object_and_target(obs)
        if object_pos is None or target_pos is None:
            raise RuntimeError("Feedback phase controller needs Lift-Cube object and target positions.")
        if action_dim < 8:
            raise RuntimeError(f"Feedback phase controller expects IK-Abs action dim >= 8, got {action_dim}")

        n = object_pos.shape[0]
        above = object_pos + torch.tensor([0.0, 0.0, args_cli.feedback_above_offset], device=device)
        grasp = object_pos
        lift = target_pos

        desired_pos = torch.empty_like(object_pos)
        desired_quat = torch.empty((n, 4), device=device, dtype=object_pos.dtype)
        grasp_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=object_pos.dtype).repeat(n, 1)
        grip = torch.ones(n, device=device, dtype=object_pos.dtype)

        rest = self.state == self.REST
        above_state = self.state == self.APPROACH_ABOVE_OBJECT
        object_state = self.state == self.APPROACH_OBJECT
        grasp_state = self.state == self.GRASP_OBJECT
        lift_state = self.state == self.LIFT_OBJECT

        desired_pos[rest] = ee_pos_curr[rest]
        desired_quat[rest] = ee_quat_curr[rest]
        desired_pos[above_state] = above[above_state]
        desired_quat[above_state] = grasp_quat[above_state]
        desired_pos[object_state] = grasp[object_state]
        desired_quat[object_state] = grasp_quat[object_state]
        desired_pos[grasp_state] = grasp[grasp_state]
        desired_quat[grasp_state] = grasp_quat[grasp_state]
        desired_pos[lift_state] = lift[lift_state]
        desired_quat[lift_state] = grasp_quat[lift_state]
        grip[grasp_state | lift_state] = -1.0

        if projected:
            near_object = above_state | object_state | grasp_state
            desired_pos[near_object] = clamp_workspace(
                desired_pos[near_object], object_pos[near_object], radius=0.15, z_min=0.02, z_max=0.24
            )
            desired_pos[lift_state] = clamp_workspace(
                desired_pos[lift_state], target_pos[lift_state], radius=0.20, z_min=0.14, z_max=0.70
            )

        action = torch.zeros((n, action_dim), device=device, dtype=object_pos.dtype)
        action[:, 0:3] = desired_pos
        action[:, 3:7] = desired_quat
        action[:, 7] = grip

        position_error = (ee_pos_curr - desired_pos).norm(dim=-1)
        close = position_error < args_cli.feedback_position_threshold
        next_state = self.state.clone()
        next_wait = self.wait + self.dt

        advance_rest = rest & (next_wait >= 0.20)
        advance_above = above_state & close & (next_wait >= 0.20)
        advance_object = object_state & close & (next_wait >= 0.30)
        advance_grasp = grasp_state & (next_wait >= 0.35)
        next_state[advance_rest] = self.APPROACH_ABOVE_OBJECT
        next_state[advance_above] = self.APPROACH_OBJECT
        next_state[advance_object] = self.GRASP_OBJECT
        next_state[advance_grasp] = self.LIFT_OBJECT

        changed = next_state != self.state
        next_wait[changed] = 0.0
        self.state = next_state
        self.wait = next_wait
        return action


def run_method(task_name: str, method: str) -> dict[str, Any]:
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    method_dir = args_cli.out_dir / method
    video_dir = method_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=max(1, min(args_cli.video_length, args_cli.steps)),
        name_prefix=f"{task_name}-{method}",
        disable_logger=True,
    )
    env.unwrapped.sim.set_camera_view(eye=list(args_cli.camera_eye), target=list(args_cli.camera_target))
    device = torch.device(env.unwrapped.device)
    obs = unpack_reset(env.reset(seed=args_cli.seed))
    env_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    ee_quat_ref = None
    try:
        arm_action = env.unwrapped.action_manager._terms["arm_action"]
        _, ee_quat_ref = arm_action._compute_frame_pose()
        ee_quat_ref = ee_quat_ref.detach().clone()
    except Exception:
        ee_quat_ref = None
    action_dim = int(env.action_space.shape[-1])
    object_start, target_start = lift_object_and_target(obs)
    object_start_first = object_start[0].detach().clone() if object_start is not None else None
    target_first = target_start[0].detach().clone() if target_start is not None else None

    rewards = torch.zeros(args_cli.num_envs, device=device)
    terminations = torch.zeros(args_cli.num_envs, device=device)
    truncations = torch.zeros(args_cli.num_envs, device=device)
    object_z_trace: list[float] = []
    phase_feedback = None
    phase_state_trace: list[int] = []
    if method in {"phase_oc_feedback", "phase_oc_feedback_projected"}:
        if "IK-Abs" not in task_name and args_cli.control_mode == "auto":
            raise RuntimeError("phase_oc_feedback is intended for IK-Abs lift tasks.")
        phase_feedback = PhaseFeedbackController(args_cli.num_envs, env_dt, device)
    start = time.perf_counter()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            ee_pos_curr = None
            ee_quat_curr = None
            try:
                arm_action = env.unwrapped.action_manager._terms["arm_action"]
                ee_pos_curr, ee_quat_curr = arm_action._compute_frame_pose()
            except Exception:
                ee_pos_curr = None
                ee_quat_curr = None
            if phase_feedback is not None:
                if ee_pos_curr is None or ee_quat_curr is None:
                    raise RuntimeError("phase_oc_feedback needs current end-effector pose from the action term.")
                action = phase_feedback.make_action(
                    obs,
                    action_dim,
                    device,
                    ee_pos_curr,
                    ee_quat_curr,
                    projected=method.endswith("_projected"),
                )
                phase_state_trace.append(int(phase_feedback.state[0].detach().cpu()))
            else:
                action = make_action(
                    method, task_name, obs, step, args_cli.steps, action_dim, device, ee_quat_ref, ee_pos_curr
                )
            obs, reward, terminated, truncated, _ = unpack_step(env.step(action))
            rewards += reward
            terminations += terminated.float()
            truncations += truncated.float()
            object_pos, _ = lift_object_and_target(obs)
            if object_pos is not None:
                object_z_trace.append(float(object_pos[0, 2].detach().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    object_final, target_final = lift_object_and_target(obs)
    object_final_first = object_final[0].detach().clone() if object_final is not None else None
    target_final_first = target_final[0].detach().clone() if target_final is not None else target_first
    lift_height = None
    target_dist = None
    if object_start_first is not None and object_final_first is not None:
        lift_height = float((object_final_first[2] - object_start_first[2]).detach().cpu())
    if object_final_first is not None and target_final_first is not None:
        target_dist = float((object_final_first - target_final_first).norm().detach().cpu())
    env.close()
    video_files = [str(path) for path in sorted(video_dir.rglob("*.mp4"))]
    phase_summary = summarize_phase_trace(phase_state_trace)
    return {
        "task": task_name,
        "method": method,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "action_dim": action_dim,
        "mean_reward": float(rewards.mean().detach().cpu()),
        "mean_termination_count": float(terminations.mean().detach().cpu()),
        "mean_truncation_count": float(truncations.mean().detach().cpu()),
        "elapsed_seconds": elapsed,
        "sim_steps_per_second": args_cli.num_envs * args_cli.steps / max(elapsed, 1e-9),
        "object_start": tensor_to_list(object_start_first),
        "object_final": tensor_to_list(object_final_first),
        "target": tensor_to_list(target_final_first),
        "ee_quat_ref": tensor_to_list(ee_quat_ref[0] if ee_quat_ref is not None else None),
        "lift_height": lift_height,
        "target_distance": target_dist,
        "max_object_z": max(object_z_trace) if object_z_trace else None,
        "phase_summary": phase_summary,
        "phase_state_trace": phase_state_trace if args_cli.store_phase_trace else None,
        "video_files": video_files,
    }


def main() -> None:
    torch.manual_seed(args_cli.seed)
    args_cli.out_dir.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in args_cli.methods.split(",") if method.strip()]
    unknown_methods = sorted(set(methods) - VALID_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {unknown_methods}. Valid methods: {sorted(VALID_METHODS)}")
    metrics: dict[str, Any] = {
        "config": {
            "task": args_cli.task,
            "fallback_task": args_cli.fallback_task,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "seed": args_cli.seed,
            "device": args_cli.device,
            "camera_eye": list(args_cli.camera_eye),
            "camera_target": list(args_cli.camera_target),
            "control_mode": args_cli.control_mode,
            "pregrasp_z_offset": args_cli.pregrasp_z_offset,
            "grasp_z_offset": args_cli.grasp_z_offset,
            "lift_z_offset": args_cli.lift_z_offset,
            "feedback_above_offset": args_cli.feedback_above_offset,
            "feedback_position_threshold": args_cli.feedback_position_threshold,
            "max_ee_delta": args_cli.max_ee_delta,
            "store_phase_trace": args_cli.store_phase_trace,
            "methods": methods,
        },
        "methods": {},
    }
    task_name = args_cli.task
    for idx, method in enumerate(methods):
        try:
            print(f"[{idx + 1}/{len(methods)}] running {method} on {task_name}", flush=True)
            metrics["methods"][method] = run_method(task_name, method)
        except Exception as exc:
            if task_name == args_cli.task and args_cli.fallback_task:
                print(f"[warn] {task_name} failed for {method}: {exc}", flush=True)
                task_name = args_cli.fallback_task
                print(f"[warn] retrying with fallback task {task_name}", flush=True)
                metrics["methods"][method] = run_method(task_name, method)
            else:
                raise
    metrics["resolved_task"] = task_name
    metrics_path = args_cli.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
