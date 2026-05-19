

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "YES")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ.setdefault("OMNI_KIT_DISABLE_TELEMETRY", "1")


from isaaclab.app import AppLauncher


def parse_vec3(text: str) -> tuple[float, float, float]:
    values = [float(part) for part in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return values[0], values[1], values[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stack-Cube qualitative direct vs OC-FMCS comparison.")
    parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--method", choices=("direct_gen", "oc_fmcs"), required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--video-length", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--disable-fabric", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results/isaaclab_stack_visual_compare"))
    parser.add_argument("--camera-eye", type=parse_vec3, default=(1.05, -1.25, 0.85))
    parser.add_argument("--camera-target", type=parse_vec3, default=(0.50, 0.00, 0.12))
    parser.add_argument("--position-threshold", type=float, default=0.025)
    parser.add_argument("--stack-overlap-threshold", type=float, default=0.12)
    parser.add_argument("--stack-above-z", type=float, default=0.14)
    parser.add_argument("--stack-grasp-z", type=float, default=0.018)
    parser.add_argument("--stack-lift-z", type=float, default=0.26)
    parser.add_argument("--stack-above-place-z", type=float, default=0.20)
    parser.add_argument("--stack-place-z", type=float, default=0.095)
    parser.add_argument("--stack-retreat-z", type=float, default=0.20)
    parser.add_argument("--stack-close-dwell", type=float, default=1.20)
    parser.add_argument("--stack-place-dwell", type=float, default=1.20)
    parser.add_argument("--stack-release-dwell", type=float, default=1.60)
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


PHASE_NAMES = {
    0: "rest",
    1: "above_cube2",
    2: "grasp_cube2",
    3: "close_gripper",
    4: "lift_cube2",
    5: "above_cube1",
    6: "place_cube2",
    7: "release",
    8: "retreat",
}


def unpack_reset(reset_result: Any) -> Any:
    return reset_result[0] if isinstance(reset_result, tuple) else reset_result


def unpack_step(step_result: Any) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if len(step_result) == 5:
        return step_result
    obs, reward, done, extras = step_result
    return obs, reward, done, torch.zeros_like(done, dtype=torch.bool), extras


def sanitize_action(action: torch.Tensor) -> torch.Tensor:
    action = action.clone()
    action[:, 0] = action[:, 0].clamp(0.05, 1.15)
    action[:, 1] = action[:, 1].clamp(-0.65, 0.65)
    action[:, 2] = action[:, 2].clamp(0.00, 1.25)
    quat = action[:, 3:7]
    fallback = torch.tensor([0.0, 1.0, 0.0, 0.0], device=action.device, dtype=action.dtype)
    norm = quat.norm(dim=-1, keepdim=True)
    action[:, 3:7] = torch.where(norm > 1e-6, quat / norm.clamp_min(1e-6), fallback.repeat(action.shape[0], 1))
    action[:, 7] = action[:, 7].clamp(-1.0, 1.0)
    return action


def ee_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    frame = env.unwrapped.scene["ee_frame"]
    pos = frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    quat = frame.data.target_quat_w[..., 0, :].clone()
    return pos, quat


def cube_pos(env, name: str) -> torch.Tensor:
    cube = env.unwrapped.scene[name]
    return cube.data.root_pos_w.clone() - env.unwrapped.scene.env_origins


def fixed_grasp_quat(n: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=dtype).repeat(n, 1)


class StackController:
    def __init__(self, num_envs: int, dt: float, device: torch.device):
        self.num_envs = num_envs
        self.dt = float(dt)
        self.device = device
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wait = torch.zeros(num_envs, device=device)
        self.c1_anchor: torch.Tensor | None = None
        self.c2_anchor: torch.Tensor | None = None

    def action(self, env) -> torch.Tensor:
        ee_pos, ee_quat = ee_pose(env)
        c1 = cube_pos(env, "cube_1")
        c2 = cube_pos(env, "cube_2")
        if self.c1_anchor is None or self.c2_anchor is None:
            self.c1_anchor = c1.detach().clone()
            self.c2_anchor = c2.detach().clone()
        c1_anchor = self.c1_anchor
        c2_anchor = self.c2_anchor
        n = c1.shape[0]
        quat = fixed_grasp_quat(n, self.device, c1.dtype)
        above_c2 = c2_anchor + torch.tensor([0.0, 0.0, args_cli.stack_above_z], device=self.device)
        grasp_c2 = c2_anchor + torch.tensor([0.0, 0.0, args_cli.stack_grasp_z], device=self.device)
        lift_c2 = c2_anchor + torch.tensor([0.0, 0.0, args_cli.stack_lift_z], device=self.device)
        above_c1 = c1_anchor + torch.tensor([0.0, 0.0, args_cli.stack_above_place_z], device=self.device)
        place_c1 = c1_anchor + torch.tensor([0.0, 0.0, args_cli.stack_place_z], device=self.device)
        retreat = c1_anchor + torch.tensor([0.0, 0.0, args_cli.stack_retreat_z], device=self.device)

        desired_pos = torch.empty_like(c1)
        desired_quat = quat.clone()
        grip = torch.ones(n, device=self.device)
        targets = [ee_pos, above_c2, grasp_c2, grasp_c2, lift_c2, above_c1, place_c1, place_c1, retreat]
        grips = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
        for sid, pos in enumerate(targets):
            mask = self.state == sid
            desired_pos[mask] = pos[mask]
            grip[mask] = grips[sid]
        desired_quat[self.state == 0] = ee_quat[self.state == 0]

        close = (ee_pos - desired_pos).norm(dim=-1) < max(
            args_cli.position_threshold, args_cli.stack_overlap_threshold
        )
        schedule = [
            0.20,
            0.45,
            0.35,
            args_cli.stack_close_dwell,
            0.55,
            0.55,
            args_cli.stack_place_dwell,
            args_cli.stack_release_dwell,
            999.0,
        ]
        next_state = self.state.clone()
        next_wait = self.wait + self.dt
        for sid, wait in enumerate(schedule[:-1]):
            mask = self.state == sid
            ready = next_wait >= wait if sid == 0 else close & (next_wait >= wait)
            next_state[mask & ready] = sid + 1
        changed = next_state != self.state
        next_wait[changed] = 0.0
        self.state = next_state
        self.wait = next_wait

        return sanitize_action(torch.cat([desired_pos, desired_quat, grip.unsqueeze(-1)], dim=-1))


def direct_action(env, step: int, steps: int, initial_ee_pos: torch.Tensor) -> torch.Tensor:
    c1 = cube_pos(env, "cube_1")
    n = c1.shape[0]
    tau = torch.full((n, 1), step / max(steps - 1, 1), device=c1.device)
    w = tau * tau * (3.0 - 2.0 * tau)
    pos = (1.0 - w) * initial_ee_pos + w * (c1 + torch.tensor([0.0, 0.0, 0.08], device=c1.device))
    quat = fixed_grasp_quat(n, c1.device, c1.dtype)
    grip = 1.0 - 2.0 * ((tau - 0.35) / 0.40).clamp(0.0, 1.0)
    return sanitize_action(torch.cat([pos, quat, grip], dim=-1))


def stack_metrics(start_c2: torch.Tensor, max_c2_z: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> dict[str, float]:
    d12 = c2 - c1
    stack1 = (d12[:, :2].norm(dim=-1) < 0.05) & ((d12[:, 2] - 0.0468).abs() < 0.025)
    lift2 = (max_c2_z - start_c2[:, 2]) > 0.05
    return {
        "success_rate": float(stack1.float().mean().cpu()),
        "cube2_lift_rate": float(lift2.float().mean().cpu()),
        "mean_cube2_xy_error_to_cube1": float(d12[:, :2].norm(dim=-1).mean().cpu()),
        "mean_cube2_height_over_cube1": float(d12[:, 2].mean().cpu()),
        "mean_max_cube2_lift": float((max_c2_z - start_c2[:, 2]).mean().cpu()),
    }


def summarize_phase(trace: list[int]) -> dict[str, Any]:
    transitions = []
    if trace:
        last = trace[0]
        for step, state in enumerate(trace[1:], start=1):
            if state != last:
                transitions.append({"step": step, "from": PHASE_NAMES.get(last, str(last)), "to": PHASE_NAMES.get(state, str(state))})
                last = state
    return {
        "final_state": PHASE_NAMES.get(trace[-1], str(trace[-1])) if trace else None,
        "transitions": transitions,
        "counts": {PHASE_NAMES.get(s, str(s)): trace.count(s) for s in sorted(set(trace))},
    }


def main() -> None:
    torch.manual_seed(args_cli.seed)
    random.seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    args_cli.out_dir.mkdir(parents=True, exist_ok=True)
    method_dir = args_cli.out_dir / args_cli.method
    video_dir = method_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=max(1, min(args_cli.video_length, args_cli.steps)),
        name_prefix=f"{args_cli.task}-{args_cli.method}",
        disable_logger=True,
    )
    env.unwrapped.sim.set_camera_view(eye=list(args_cli.camera_eye), target=list(args_cli.camera_target))
    device = torch.device(env.unwrapped.device)
    obs = unpack_reset(env.reset(seed=args_cli.seed))
    dt = float(env_cfg.sim.dt * env_cfg.decimation)
    controller = StackController(args_cli.num_envs, dt, device)
    initial_ee_pos, _initial_ee_quat = ee_pose(env)
    start_c2 = cube_pos(env, "cube_2").detach().clone()
    max_c2_z = start_c2[:, 2].clone()
    phase_trace = []
    start = time.perf_counter()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            if args_cli.method == "direct_gen":
                action = direct_action(env, step, args_cli.steps, initial_ee_pos)
            else:
                action = controller.action(env)
                phase_trace.append(int(controller.state[0].detach().cpu()))
            obs, _reward, _terminated, _truncated, _ = unpack_step(env.step(action))
            c2 = cube_pos(env, "cube_2").detach()
            max_c2_z = torch.maximum(max_c2_z, c2[:, 2])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    c1 = cube_pos(env, "cube_1").detach()
    c2 = cube_pos(env, "cube_2").detach()
    env.close()
    videos = [str(path) for path in sorted(video_dir.rglob("*.mp4"))]
    metrics = {
        "config": {
            "task": args_cli.task,
            "method": args_cli.method,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "seed": args_cli.seed,
            "camera_eye": list(args_cli.camera_eye),
            "camera_target": list(args_cli.camera_target),
            "stack_overlap_threshold": args_cli.stack_overlap_threshold,
            "stack_place_z": args_cli.stack_place_z,
            "stack_lift_z": args_cli.stack_lift_z,
            "stack_close_dwell": args_cli.stack_close_dwell,
            "stack_place_dwell": args_cli.stack_place_dwell,
            "stack_release_dwell": args_cli.stack_release_dwell,
        },
        "method": args_cli.method,
        "phase_summary": summarize_phase(phase_trace),
        "elapsed_seconds": elapsed,
        "sim_steps_per_second": args_cli.num_envs * args_cli.steps / max(elapsed, 1e-9),
        "video_files": videos,
    }
    metrics.update(stack_metrics(start_c2, max_c2_z, c1, c2))
    metrics_path = method_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
