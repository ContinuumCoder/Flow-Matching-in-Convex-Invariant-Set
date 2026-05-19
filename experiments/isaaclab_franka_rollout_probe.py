

from __future__ import annotations

import argparse
import json
import math
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
    parts = [float(part) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats, e.g. 1.6,1.0,1.3")
    return (parts[0], parts[1], parts[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded Isaac Lab rollout probe.")
    parser.add_argument("--task", type=str, default="Isaac-Reach-Franka-v0")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--policy", choices=["zero", "random", "sine"], default="zero")
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--disable-fabric", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/isaaclab_franka_probe/metrics.json"))
    parser.add_argument("--video", action="store_true", help="Record an RGB rollout video with gym RecordVideo.")
    parser.add_argument("--video-length", type=int, default=120, help="Maximum recorded video length in env steps.")
    parser.add_argument("--video-folder", type=Path, default=None, help="Directory for recorded mp4 files.")
    parser.add_argument(
        "--camera-eye",
        type=parse_vec3,
        default=None,
        help="Optional viewer camera eye as x,y,z. Defaults to a close robot-task view when --video is set.",
    )
    parser.add_argument(
        "--camera-target",
        type=parse_vec3,
        default=None,
        help="Optional viewer camera target as x,y,z. Defaults to a close robot-task view when --video is set.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.headless is None:
        args.headless = True
    if args.video:
        args.enable_cameras = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


def tensor_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "mean": float(value.float().mean().detach().cpu()) if value.numel() else 0.0,
            "std": float(value.float().std(unbiased=False).detach().cpu()) if value.numel() else 0.0,
        }
    if isinstance(value, dict):
        return {str(k): tensor_summary(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_summary(v) for v in value]
    return str(type(value))


def make_action(policy: str, shape: tuple[int, ...], device: torch.device, step: int, scale: float) -> torch.Tensor:
    if policy == "zero":
        return torch.zeros(shape, device=device)
    if policy == "random":
        return scale * (2.0 * torch.rand(shape, device=device) - 1.0)
    if policy == "sine":
        env_ids = torch.arange(shape[0], device=device, dtype=torch.float32)[:, None]
        act_ids = torch.arange(shape[1], device=device, dtype=torch.float32)[None, :]
        phase = 0.07 * step + 0.37 * env_ids + 0.61 * act_ids
        return scale * torch.sin(phase)
    raise ValueError(f"Unknown policy: {policy}")


def unpack_step(step_result: Any) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if len(step_result) == 5:
        obs, reward, terminated, truncated, extras = step_result
    elif len(step_result) == 4:
        obs, reward, done, extras = step_result
        terminated = done
        truncated = torch.zeros_like(done, dtype=torch.bool)
    else:
        raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")
    return obs, reward, terminated, truncated, extras


def list_video_files(video_folder: Path | None) -> list[str]:
    if video_folder is None or not video_folder.exists():
        return []
    return [str(path) for path in sorted(video_folder.rglob("*.mp4"))]


def main() -> None:
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    video_folder = args_cli.video_folder or (args_cli.out.parent / "videos")
    if args_cli.video:
        video_folder.mkdir(parents=True, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step == 0,
            video_length=max(1, min(args_cli.video_length, args_cli.steps)),
            name_prefix=f"{args_cli.task}-{args_cli.policy}",
            disable_logger=True,
        )
    camera_eye = args_cli.camera_eye
    camera_target = args_cli.camera_target
    if args_cli.video and camera_eye is None:
        camera_eye = (1.45, 1.10, 1.20)
    if args_cli.video and camera_target is None:
        camera_target = (0.0, 0.0, 0.45)
    if camera_eye is not None and camera_target is not None:
        env.unwrapped.sim.set_camera_view(eye=list(camera_eye), target=list(camera_target))
    device = torch.device(env.unwrapped.device)
    reset_result = env.reset(seed=args_cli.seed)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    action_shape = tuple(env.action_space.shape)
    reward_sum = torch.zeros(args_cli.num_envs, device=device)
    termination_count = torch.zeros(args_cli.num_envs, device=device)
    truncation_count = torch.zeros(args_cli.num_envs, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if hasattr(torch, "cuda") and device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    last_extras: dict[str, Any] = {}
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = make_action(args_cli.policy, action_shape, device, step, args_cli.action_scale)
            obs, reward, terminated, truncated, extras = unpack_step(env.step(actions))
            reward_sum += reward
            termination_count += terminated.float()
            truncation_count += truncated.float()
            last_extras = extras if isinstance(extras, dict) else {"extras": extras}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    metrics = {
        "task": args_cli.task,
        "policy": args_cli.policy,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "device": str(device),
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "action_shape": list(action_shape),
        "mean_reward_per_env": float(reward_sum.mean().detach().cpu()),
        "std_reward_per_env": float(reward_sum.std(unbiased=False).detach().cpu()),
        "mean_termination_count": float(termination_count.mean().detach().cpu()),
        "mean_truncation_count": float(truncation_count.mean().detach().cpu()),
        "elapsed_seconds": elapsed,
        "sim_steps_per_second": args_cli.num_envs * args_cli.steps / max(elapsed, 1e-9),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
        ),
        "obs_summary": tensor_summary(obs),
        "extras_keys": sorted(str(k) for k in last_extras.keys()),
        "video_enabled": bool(args_cli.video),
        "video_folder": str(video_folder) if args_cli.video else None,
        "video_length": max(1, min(args_cli.video_length, args_cli.steps)) if args_cli.video else 0,
        "camera_eye": list(camera_eye) if camera_eye is not None else None,
        "camera_target": list(camera_target) if camera_target is not None else None,
    }

    env.close()
    if args_cli.video:
        metrics["video_files"] = list_video_files(video_folder)

    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    args_cli.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {args_cli.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
