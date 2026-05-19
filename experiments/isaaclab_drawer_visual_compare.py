

from __future__ import annotations

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Drawer qualitative direct vs OC-FMCS comparison.")
    parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-IK-Abs-v0")
    parser.add_argument("--methods", type=str, default="direct_gen,oc_fmcs")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--video-length", type=int, default=480)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--disable-fabric", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results/isaaclab_drawer_visual_compare"))
    parser.add_argument("--camera-eye", type=parse_vec3, default=(-0.35, -1.10, 1.05))
    parser.add_argument("--camera-target", type=parse_vec3, default=(0.55, 0.00, 0.48))
    parser.add_argument("--position-threshold", type=float, default=0.025)
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
    1: "approach_infront_handle",
    2: "approach_handle",
    3: "grasp_handle",
    4: "open_drawer",
    5: "release_handle",
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


def handle_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    frame = env.unwrapped.scene["cabinet_frame"]
    pos = frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    quat = frame.data.target_quat_w[..., 0, :].clone()
    return pos, quat


def drawer_joint(env) -> torch.Tensor:
    cabinet = env.unwrapped.scene["cabinet"]
    joint_ids, _ = cabinet.find_joints(["drawer_top_joint"])
    return cabinet.data.joint_pos[:, joint_ids[0]]


class DrawerController:
    REST = 0
    APPROACH_INFRONT_HANDLE = 1
    APPROACH_HANDLE = 2
    GRASP_HANDLE = 3
    OPEN_DRAWER = 4
    RELEASE_HANDLE = 5

    def __init__(self, num_envs: int, dt: float, device: torch.device):
        self.num_envs = num_envs
        self.dt = float(dt)
        self.device = device
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wait = torch.zeros(num_envs, device=device)

    def action(self, env) -> torch.Tensor:
        ee_pos, ee_quat = ee_pose(env)
        handle_pos, handle_quat = handle_pose(env)
        n = handle_pos.shape[0]
        approach_pos = handle_pos + torch.tensor([-0.10, 0.0, 0.0], device=self.device)
        grasp_pos = handle_pos + torch.tensor([0.025, 0.0, 0.0], device=self.device)
        pull_pos = handle_pos + torch.tensor([-0.30, 0.0, 0.0], device=self.device)

        desired_pos = torch.empty_like(handle_pos)
        desired_quat = torch.empty_like(handle_quat)
        grip = torch.ones(n, device=self.device)
        for sid, pos, quat, gripper in [
            (self.REST, ee_pos, ee_quat, 1.0),
            (self.APPROACH_INFRONT_HANDLE, approach_pos, handle_quat, 1.0),
            (self.APPROACH_HANDLE, handle_pos, handle_quat, 1.0),
            (self.GRASP_HANDLE, grasp_pos, handle_quat, -1.0),
            (self.OPEN_DRAWER, pull_pos, handle_quat, -1.0),
            (self.RELEASE_HANDLE, ee_pos, ee_quat, -1.0),
        ]:
            mask = self.state == sid
            desired_pos[mask] = pos[mask]
            desired_quat[mask] = quat[mask]
            grip[mask] = gripper

        position_error = (ee_pos - desired_pos).norm(dim=-1)
        close = position_error < args_cli.position_threshold
        next_state = self.state.clone()
        next_wait = self.wait + self.dt
        waits = [0.50, 0.40, 0.35, 0.35, 2.00, 999.0]
        for sid, wait in enumerate(waits[:-1]):
            mask = self.state == sid
            ready = next_wait >= wait if sid == self.REST else close & (next_wait >= wait)
            next_state[mask & ready] = sid + 1
        changed = next_state != self.state
        next_wait[changed] = 0.0
        self.state = next_state
        self.wait = next_wait

        return sanitize_action(torch.cat([desired_pos, desired_quat, grip.unsqueeze(-1)], dim=-1))


def direct_action(env, step: int, steps: int, initial_ee_pos: torch.Tensor, _initial_ee_quat: torch.Tensor) -> torch.Tensor:
    handle_pos, handle_quat = handle_pose(env)
    n = handle_pos.shape[0]
    tau = torch.full((n, 1), step / max(steps - 1, 1), device=handle_pos.device)
    w = tau * tau * (3.0 - 2.0 * tau)
    final_pos = handle_pos + torch.tensor([-0.35, 0.0, 0.0], device=handle_pos.device)
    pos = (1.0 - w) * initial_ee_pos + w * final_pos
    grip = 1.0 - 2.0 * ((tau - 0.45) / 0.35).clamp(0.0, 1.0)
    return sanitize_action(torch.cat([pos, handle_quat, grip], dim=-1))


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


def run_method(method: str) -> dict[str, Any]:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    method_dir = args_cli.out_dir / method
    video_dir = method_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=max(1, min(args_cli.video_length, args_cli.steps)),
        name_prefix=f"{args_cli.task}-{method}",
        disable_logger=True,
    )
    env.unwrapped.sim.set_camera_view(eye=list(args_cli.camera_eye), target=list(args_cli.camera_target))
    device = torch.device(env.unwrapped.device)
    obs = unpack_reset(env.reset(seed=args_cli.seed))
    dt = float(env_cfg.sim.dt * env_cfg.decimation)
    controller = DrawerController(args_cli.num_envs, dt, device)
    initial_ee_pos, initial_ee_quat = ee_pose(env)
    rewards = torch.zeros(args_cli.num_envs, device=device)
    max_drawer = drawer_joint(env).detach().clone()
    drawer_trace = []
    phase_trace = []
    start = time.perf_counter()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            if method == "direct_gen":
                action = direct_action(env, step, args_cli.steps, initial_ee_pos, initial_ee_quat)
            elif method == "oc_fmcs":
                action = controller.action(env)
                phase_trace.append(int(controller.state[0].detach().cpu()))
            else:
                raise ValueError(method)
            obs, reward, _terminated, _truncated, _ = unpack_step(env.step(action))
            rewards += reward
            drawer = drawer_joint(env).detach()
            max_drawer = torch.maximum(max_drawer, drawer)
            drawer_trace.append(float(drawer[0].cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final_drawer = drawer_joint(env).detach()
    env.close()
    videos = [str(path) for path in sorted(video_dir.rglob("*.mp4"))]
    return {
        "method": method,
        "mean_reward": float(rewards.mean().detach().cpu()),
        "success_rate": float((max_drawer > 0.25).float().mean().detach().cpu()),
        "mean_final_drawer_pos": float(final_drawer.mean().cpu()),
        "mean_max_drawer_pos": float(max_drawer.mean().cpu()),
        "first_env_drawer_trace": drawer_trace,
        "phase_summary": summarize_phase(phase_trace),
        "elapsed_seconds": elapsed,
        "sim_steps_per_second": args_cli.num_envs * args_cli.steps / max(elapsed, 1e-9),
        "video_files": videos,
    }


def main() -> None:
    torch.manual_seed(args_cli.seed)
    args_cli.out_dir.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in args_cli.methods.split(",") if method.strip()]
    metrics = {
        "config": {
            "task": args_cli.task,
            "methods": methods,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "seed": args_cli.seed,
            "camera_eye": list(args_cli.camera_eye),
            "camera_target": list(args_cli.camera_target),
        },
        "methods": {},
    }
    for method in methods:
        print(f"[drawer-visual] {method}", flush=True)
        metrics["methods"][method] = run_method(method)
    metrics_path = args_cli.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
