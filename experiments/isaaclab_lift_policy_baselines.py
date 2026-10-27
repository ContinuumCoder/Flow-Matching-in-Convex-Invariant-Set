

from __future__ import annotations

import argparse
import copy
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
    values = [float(part) for part in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return (values[0], values[1], values[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lift-Cube policy baseline sweep.")
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--num-envs", type=int, default=48)
    parser.add_argument("--demo-episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--eval-seed", type=int, default=2027)
    parser.add_argument("--disable-fabric", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/isaaclab_lift_policy_baselines/metrics.json"))
    parser.add_argument(
        "--eval-methods",
        type=str,
        default="direct_gen,direct_gen_projected,bc_mlp,chunk_mlp,diffusion_policy,oc_fmcs",
        help="Comma-separated methods to evaluate.",
    )
    parser.add_argument("--chunk-horizon", type=int, default=16)
    parser.add_argument("--bc-train-steps", type=int, default=1200)
    parser.add_argument("--chunk-train-steps", type=int, default=1200)
    parser.add_argument("--diffusion-train-steps", type=int, default=1600)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument(
        "--diffusion-schedule",
        choices=["cosine", "linear_legacy"],
        default="cosine",
        help="linear_legacy reproduces the original linspace(1e-4, 0.02, T) betas.",
    )
    parser.add_argument(
        "--diffusion-pred",
        choices=["x0", "eps"],
        default="x0",
        help="Denoiser target. With a small MLP on 128-D chunks, eps-prediction has to reproduce the full "
        "noise vector and stays far from converged at this training budget; x0 does not.",
    )
    parser.add_argument(
        "--diffusion-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip the predicted clean chunk to the normalized demo range at each denoising step.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.995, help="EMA for diffusion weights; 0 disables.")
    parser.add_argument(
        "--exec-steps",
        type=int,
        default=8,
        help="Execute this many actions of a sampled diffusion chunk before resampling "
        "(1 = resample every control step, the original behavior).",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--camera-eye", type=parse_vec3, default=(1.10, -0.55, 0.65))
    parser.add_argument("--camera-target", type=parse_vec3, default=(0.47, 0.07, 0.12))
    parser.add_argument("--feedback-above-offset", type=float, default=0.10)
    parser.add_argument("--feedback-position-threshold", type=float, default=0.025)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.headless is None:
        args.headless = True
    args.enable_cameras = False
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


VALID_METHODS = {
    "direct_gen",
    "direct_gen_projected",
    "bc_mlp",
    "chunk_mlp",
    "diffusion_policy",
    "phase_bc_mlp",
    "phase_chunk_mlp",
    "phase_diffusion_policy",
    "oc_fmcs",
}
NUM_PHASES = 5
PHASE_NAMES = {
    0: "rest",
    1: "approach_above_object",
    2: "approach_object",
    3: "grasp_object",
    4: "lift_object",
}


def log(message: str) -> None:
    print(f"[lift-baselines] {message}", flush=True)


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


def lift_object_and_target(obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    policy = policy_tensor(obs)
    if policy.shape[-1] < 24:
        raise RuntimeError(f"Lift-Cube policy observation too short: {policy.shape}")
    return policy[..., 18:21], policy[..., 21:24]


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


def sanitize_ik_abs_action(action: torch.Tensor) -> torch.Tensor:
    action = action.clone()
    action[:, 0] = action[:, 0].clamp(0.15, 0.85)
    action[:, 1] = action[:, 1].clamp(-0.45, 0.45)
    action[:, 2] = action[:, 2].clamp(0.00, 0.75)
    quat = action[:, 3:7]
    fallback = torch.tensor([0.0, 1.0, 0.0, 0.0], device=action.device, dtype=action.dtype)
    norm = quat.norm(dim=-1, keepdim=True)
    quat = torch.where(norm > 1e-6, quat / norm.clamp_min(1e-6), fallback.repeat(action.shape[0], 1))
    action[:, 3:7] = quat
    action[:, 7] = action[:, 7].clamp(-1.0, 1.0)
    return action


def direct_ik_abs_action(method: str, obs: Any, step: int, horizon: int, action_dim: int, device: torch.device):
    object_pos, target_pos = lift_object_and_target(obs)
    n = object_pos.shape[0]
    tau = torch.full((n,), step / max(horizon - 1, 1), device=device)
    pregrasp = object_pos + torch.tensor([0.0, 0.0, 0.20], device=device)
    place = target_pos + torch.tensor([0.0, 0.0, 0.03], device=device)
    raw = lerp(pregrasp, place, smoothstep(tau))
    if method == "direct_gen_projected":
        pos = clamp_workspace(raw, object_pos, radius=0.45, z_min=0.06, z_max=0.55)
        grip = 1.0 - 2.0 * smoothstep((tau - 0.24) / 0.52)
    else:
        pos = raw
        grip = 1.0 - 2.0 * smoothstep((tau - 0.18) / 0.65)
    action = torch.zeros((n, action_dim), device=device)
    action[:, 0:3] = pos
    action[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).repeat(n, 1)
    action[:, 7] = grip
    return sanitize_ik_abs_action(action)


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

    def reset(self):
        self.state.zero_()
        self.wait.zero_()

    def make_action(
        self,
        obs: Any,
        action_dim: int,
        device: torch.device,
        ee_pos_curr: torch.Tensor,
        ee_quat_curr: torch.Tensor,
    ) -> torch.Tensor:
        object_pos, target_pos = lift_object_and_target(obs)
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

        action = torch.zeros((n, action_dim), device=device, dtype=object_pos.dtype)
        action[:, 0:3] = desired_pos
        action[:, 3:7] = desired_quat
        action[:, 7] = grip

        position_error = (ee_pos_curr - desired_pos).norm(dim=-1)
        close = position_error < args_cli.feedback_position_threshold
        next_state = self.state.clone()
        next_wait = self.wait + self.dt
        next_state[rest & (next_wait >= 0.20)] = self.APPROACH_ABOVE_OBJECT
        next_state[above_state & close & (next_wait >= 0.20)] = self.APPROACH_OBJECT
        next_state[object_state & close & (next_wait >= 0.30)] = self.GRASP_OBJECT
        next_state[grasp_state & (next_wait >= 0.35)] = self.LIFT_OBJECT
        changed = next_state != self.state
        next_wait[changed] = 0.0
        self.state = next_state
        self.wait = next_wait
        return sanitize_ik_abs_action(action)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden_dim), nn.SiLU()]
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionChunkMLP(nn.Module):
    def __init__(self, obs_dim: int, chunk_dim: int, hidden_dim: int, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        self.net = MLP(obs_dim + chunk_dim + time_dim, chunk_dim, hidden_dim, depth=4)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device, dtype=torch.float32) * (-1.0)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, obs: torch.Tensor, noisy_chunk: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, noisy_chunk, self.time_embedding(t)], dim=-1))


def ema_copy(model: nn.Module) -> nn.Module | None:
    if args_cli.ema_decay <= 0:
        return None
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(ema: nn.Module | None, model: nn.Module) -> None:
    if ema is None:
        return
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p, 1.0 - args_cli.ema_decay)


class ChunkPlan:
    def __init__(self, num_envs: int, horizon: int, action_dim: int, exec_steps: int, device: torch.device):
        self.exec_steps = max(1, min(exec_steps, horizon))
        self.buf = torch.zeros(num_envs, horizon, action_dim, device=device)
        self.idx = torch.full((num_envs,), self.exec_steps, dtype=torch.long, device=device)
        self.last_phase: torch.Tensor | None = None

    def need_replan(self, phase: torch.Tensor | None = None) -> torch.Tensor:
        need = self.idx >= self.exec_steps
        if phase is not None:
            if self.last_phase is not None:
                need = need | (phase != self.last_phase)
            self.last_phase = phase.clone()
        return need

    def update(self, need: torch.Tensor, chunk: torch.Tensor) -> None:
        self.buf[need] = chunk
        self.idx[need] = 0

    def pop(self) -> torch.Tensor:
        action = self.buf[torch.arange(self.buf.shape[0], device=self.buf.device), self.idx]
        self.idx += 1
        return action


def get_ee_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    arm_action = env.unwrapped.action_manager._terms["arm_action"]
    return arm_action._compute_frame_pose()


def make_env():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.unwrapped.sim.set_camera_view(eye=list(args_cli.camera_eye), target=list(args_cli.camera_target))
    except Exception:
        pass
    return env, env_cfg


@torch.inference_mode()
def collect_oc_fmcs_demos(env, env_dt: float, action_dim: int, device: torch.device) -> dict[str, torch.Tensor]:
    obs_episodes = []
    action_episodes = []
    phase_episodes = []
    controller = PhaseFeedbackController(args_cli.num_envs, env_dt, device)
    for episode in range(args_cli.demo_episodes):
        episode_start = time.perf_counter()
        obs = unpack_reset(env.reset(seed=args_cli.seed + episode))
        controller.reset()
        obs_steps = []
        action_steps = []
        phase_steps = []
        for _step in range(args_cli.steps):
            with torch.inference_mode():
                ee_pos, ee_quat = get_ee_pose(env)
                action = controller.make_action(obs, action_dim, device, ee_pos, ee_quat)
                obs_steps.append(policy_tensor(obs).detach().clone())
                action_steps.append(action.detach().clone())
                phase_steps.append(controller.state.detach().clone())
                obs, _reward, _terminated, _truncated, _ = unpack_step(env.step(action))
        obs_episodes.append(torch.stack(obs_steps, dim=1))
        action_episodes.append(torch.stack(action_steps, dim=1))
        phase_episodes.append(torch.stack(phase_steps, dim=1))
        log(f"collected demo episode {episode + 1}/{args_cli.demo_episodes} in {time.perf_counter() - episode_start:.2f}s")
    return {
        "obs": torch.cat(obs_episodes, dim=0),
        "actions": torch.cat(action_episodes, dim=0),
        "phases": torch.cat(phase_episodes, dim=0),
    }


def normalize_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-4)
    return mean, std


def phase_conditioned_obs(obs: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    phase_oh = F.one_hot(phases.long(), num_classes=NUM_PHASES).to(obs.dtype)
    return torch.cat([obs, phase_oh], dim=-1)


def sample_flat_indices(total: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, total, (batch_size,), device=device)


def train_bc(obs: torch.Tensor, actions: torch.Tensor, obs_mean, obs_std, action_mean, action_std, name: str):
    stage_start = time.perf_counter()
    device = obs.device
    obs_dim = obs.shape[-1]
    action_dim = actions.shape[-1]
    model = MLP(obs_dim, action_dim, args_cli.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr)
    obs_flat = obs.reshape(-1, obs_dim)
    action_flat = actions.reshape(-1, action_dim)
    total = obs_flat.shape[0]
    losses = []
    for _ in range(args_cli.bc_train_steps):
        idx = sample_flat_indices(total, args_cli.batch_size, device)
        pred = model((obs_flat[idx] - obs_mean) / obs_std)
        target = (action_flat[idx] - action_mean) / action_std
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    log(f"trained {name} for {args_cli.bc_train_steps} steps in {time.perf_counter() - stage_start:.2f}s")
    return model.eval(), losses


def build_chunk_dataset(obs: torch.Tensor, actions: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    episodes, steps, obs_dim = obs.shape
    action_dim = actions.shape[-1]
    obs_windows = []
    chunks = []
    for t in range(steps):
        end = min(t + horizon, steps)
        chunk = actions[:, t:end]
        if end - t < horizon:
            pad = actions[:, end - 1 : end].repeat(1, horizon - (end - t), 1)
            chunk = torch.cat([chunk, pad], dim=1)
        obs_windows.append(obs[:, t])
        chunks.append(chunk.reshape(episodes, horizon * action_dim))
    return torch.cat(obs_windows, dim=0), torch.cat(chunks, dim=0)


def train_chunk_mlp(chunk_obs, chunks, obs_mean, obs_std, chunk_mean, chunk_std, name: str):
    stage_start = time.perf_counter()
    device = chunk_obs.device
    model = MLP(chunk_obs.shape[-1], chunks.shape[-1], args_cli.hidden_dim, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr)
    total = chunk_obs.shape[0]
    losses = []
    for _ in range(args_cli.chunk_train_steps):
        idx = sample_flat_indices(total, args_cli.batch_size, device)
        pred = model((chunk_obs[idx] - obs_mean) / obs_std)
        target = (chunks[idx] - chunk_mean) / chunk_std
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    log(f"trained {name} for {args_cli.chunk_train_steps} steps in {time.perf_counter() - stage_start:.2f}s")
    return model.eval(), losses


def diffusion_schedule(num_steps: int, device: torch.device):
    if args_cli.diffusion_schedule == "linear_legacy":
        betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    else:
        s = 0.008
        grid = torch.arange(num_steps + 1, device=device, dtype=torch.float64) / num_steps
        f = torch.cos((grid + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1.0 - f[1:] / f[:-1]).clamp(max=0.999).float()
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def train_diffusion(chunk_obs, chunks, obs_mean, obs_std, chunk_mean, chunk_std, name: str):
    stage_start = time.perf_counter()
    device = chunk_obs.device
    chunk_dim = chunks.shape[-1]
    model = DiffusionChunkMLP(chunk_obs.shape[-1], chunk_dim, args_cli.hidden_dim).to(device)
    ema = ema_copy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr)
    _betas, _alphas, alpha_bar = diffusion_schedule(args_cli.diffusion_steps, device)
    total = chunk_obs.shape[0]
    losses = []
    norm_chunks = (chunks - chunk_mean) / chunk_std
    norm_obs = (chunk_obs - obs_mean) / obs_std
    for _ in range(args_cli.diffusion_train_steps):
        idx = sample_flat_indices(total, args_cli.batch_size, device)
        t = torch.randint(0, args_cli.diffusion_steps, (args_cli.batch_size,), device=device)
        noise = torch.randn_like(norm_chunks[idx])
        ab = alpha_bar[t].unsqueeze(-1)
        noisy = ab.sqrt() * norm_chunks[idx] + (1.0 - ab).sqrt() * noise
        pred = model(norm_obs[idx], noisy, t)
        loss = F.mse_loss(pred, norm_chunks[idx] if args_cli.diffusion_pred == "x0" else noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema_update(ema, model)
        losses.append(float(loss.detach().cpu()))
    log(
        f"trained {name} for {args_cli.diffusion_train_steps} steps "
        f"with {args_cli.diffusion_steps} denoising steps ({args_cli.diffusion_schedule} schedule, "
        f"alpha_bar_T={float(alpha_bar[-1]):.4f}) in {time.perf_counter() - stage_start:.2f}s"
    )
    return (ema if ema is not None else model).eval(), losses


@torch.inference_mode()
def sample_diffusion_chunk(model, obs, obs_mean, obs_std, stats: dict[str, torch.Tensor], action_dim) -> torch.Tensor:
    device = obs.device
    n = obs.shape[0]
    chunk_dim = args_cli.chunk_horizon * action_dim
    betas, alphas, alpha_bar = diffusion_schedule(args_cli.diffusion_steps, device)
    legacy = args_cli.diffusion_schedule == "linear_legacy"
    x = torch.randn(n, chunk_dim, device=device)
    norm_obs = (obs - obs_mean) / obs_std
    for t_int in reversed(range(args_cli.diffusion_steps)):
        t = torch.full((n,), t_int, device=device, dtype=torch.long)
        pred = model(norm_obs, x, t)
        beta = betas[t_int]
        ab = alpha_bar[t_int]
        ab_prev = alpha_bar[t_int - 1] if t_int > 0 else torch.ones_like(ab)
        x0 = pred if args_cli.diffusion_pred == "x0" else (x - (1.0 - ab).sqrt() * pred) / ab.sqrt()
        if args_cli.diffusion_clip:
            x0 = torch.maximum(torch.minimum(x0, stats["chunk_hi"]), stats["chunk_lo"])
        mean = (ab_prev.sqrt() * beta / (1.0 - ab)) * x0 + (alphas[t_int].sqrt() * (1.0 - ab_prev) / (1.0 - ab)) * x
        if t_int > 0:
            var = beta if legacy else beta * (1.0 - ab_prev) / (1.0 - ab)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean
    chunk = x * stats["chunk_std"] + stats["chunk_mean"]
    return chunk.reshape(n, args_cli.chunk_horizon, action_dim)


@torch.inference_mode()
def evaluate_method(env, env_dt, method: str, models: dict[str, Any], stats: dict[str, torch.Tensor]):
    device = torch.device(env.unwrapped.device)
    action_dim = int(env.action_space.shape[-1])
    obs = unpack_reset(env.reset(seed=args_cli.eval_seed))
    controller = PhaseFeedbackController(args_cli.num_envs, env_dt, device)
    object_start, target_start = lift_object_and_target(obs)
    object_start = object_start.detach().clone()
    target_last = target_start.detach().clone()
    rewards = torch.zeros(args_cli.num_envs, device=device)
    terminations = torch.zeros(args_cli.num_envs, device=device)
    truncations = torch.zeros(args_cli.num_envs, device=device)
    max_z = object_start[:, 2].detach().clone()
    plan = ChunkPlan(args_cli.num_envs, args_cli.chunk_horizon, action_dim, args_cli.exec_steps, device)
    policy_compute = 0.0
    start = time.perf_counter()
    for step in range(args_cli.steps):
        policy_obs = policy_tensor(obs)
        policy_start = time.perf_counter()
        if method in {"direct_gen", "direct_gen_projected"}:
            action = direct_ik_abs_action(method, obs, step, args_cli.steps, action_dim, device)
        elif method == "oc_fmcs":
            ee_pos, ee_quat = get_ee_pose(env)
            action = controller.make_action(obs, action_dim, device, ee_pos, ee_quat)
        elif method == "bc_mlp":
            norm_obs = (policy_obs - stats["obs_mean"]) / stats["obs_std"]
            action = models["bc"](norm_obs) * stats["action_std"] + stats["action_mean"]
            action = sanitize_ik_abs_action(action)
        elif method == "chunk_mlp":
            norm_obs = (policy_obs - stats["obs_mean"]) / stats["obs_std"]
            chunk = models["chunk"](norm_obs) * stats["chunk_std"] + stats["chunk_mean"]
            action = sanitize_ik_abs_action(chunk.reshape(args_cli.num_envs, args_cli.chunk_horizon, action_dim)[:, 0])
        elif method == "diffusion_policy":
            need = plan.need_replan()
            if need.any():
                plan.update(
                    need,
                    sample_diffusion_chunk(
                        models["diffusion"], policy_obs[need], stats["obs_mean"], stats["obs_std"], stats, action_dim
                    ),
                )
            action = sanitize_ik_abs_action(plan.pop())
        elif method in {"phase_bc_mlp", "phase_chunk_mlp", "phase_diffusion_policy"}:
            phase_now = controller.state.clone()
            phase_obs = torch.cat(
                [
                    policy_obs,
                    F.one_hot(controller.state.long(), num_classes=NUM_PHASES).to(policy_obs.dtype),
                ],
                dim=-1,
            )
            ee_pos, ee_quat = get_ee_pose(env)
            _ = controller.make_action(obs, action_dim, device, ee_pos, ee_quat)
            if method == "phase_bc_mlp":
                norm_obs = (phase_obs - stats["phase_obs_mean"]) / stats["phase_obs_std"]
                action = models["phase_bc"](norm_obs) * stats["action_std"] + stats["action_mean"]
                action = sanitize_ik_abs_action(action)
            elif method == "phase_chunk_mlp":
                norm_obs = (phase_obs - stats["phase_obs_mean"]) / stats["phase_obs_std"]
                chunk = models["phase_chunk"](norm_obs) * stats["chunk_std"] + stats["chunk_mean"]
                action = sanitize_ik_abs_action(chunk.reshape(args_cli.num_envs, args_cli.chunk_horizon, action_dim)[:, 0])
            else:
                need = plan.need_replan(phase_now)
                if need.any():
                    plan.update(
                        need,
                        sample_diffusion_chunk(
                            models["phase_diffusion"],
                            phase_obs[need],
                            stats["phase_obs_mean"],
                            stats["phase_obs_std"],
                            stats,
                            action_dim,
                        ),
                    )
                action = sanitize_ik_abs_action(plan.pop())
        else:
            raise ValueError(method)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        policy_compute += time.perf_counter() - policy_start
        obs, reward, terminated, truncated, _ = unpack_step(env.step(action))
        rewards += reward
        terminations += terminated.float()
        truncations += truncated.float()
        object_pos, target_last = lift_object_and_target(obs)
        max_z = torch.maximum(max_z, object_pos[:, 2])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    object_final, target_final = lift_object_and_target(obs)
    lift_height = object_final[:, 2] - object_start[:, 2]
    target_distance = (object_final - target_final).norm(dim=-1)
    lifted = max_z > object_start[:, 2] + 0.20
    success = lifted & (target_distance < 0.05)
    dropped_after_lift = lifted & (object_final[:, 2] < object_start[:, 2] + 0.10)
    return {
        "method": method,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "mean_reward": float(rewards.mean().detach().cpu()),
        "success_rate": float(success.float().mean().detach().cpu()),
        "lift_success_rate": float(lifted.float().mean().detach().cpu()),
        "drop_after_lift_rate": float(dropped_after_lift.float().mean().detach().cpu()),
        "mean_lift_height": float(lift_height.mean().detach().cpu()),
        "mean_max_object_z": float(max_z.mean().detach().cpu()),
        "mean_target_distance": float(target_distance.mean().detach().cpu()),
        "median_target_distance": float(target_distance.median().detach().cpu()),
        "mean_termination_count": float(terminations.mean().detach().cpu()),
        "mean_truncation_count": float(truncations.mean().detach().cpu()),
        "elapsed_seconds": elapsed,
        "sim_steps_per_second": args_cli.num_envs * args_cli.steps / max(elapsed, 1e-9),
        "policy_compute_seconds": policy_compute,
        "policy_compute_ms_per_env_step": 1000.0 * policy_compute / max(args_cli.num_envs * args_cli.steps, 1),
    }


def mean_tail(values: list[float], k: int = 50) -> float | None:
    if not values:
        return None
    tail = values[-min(k, len(values)) :]
    return float(sum(tail) / len(tail))


def main() -> None:
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    methods = [m.strip() for m in args_cli.eval_methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - VALID_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {sorted(VALID_METHODS)}")
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)

    env, env_cfg = make_env()
    device = torch.device(env.unwrapped.device)
    action_dim = int(env.action_space.shape[-1])
    env_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    if action_dim < 8:
        raise RuntimeError(f"This baseline script expects IK-Abs action dim >= 8, got {action_dim}")

    log(
        f"start task={args_cli.task} envs={args_cli.num_envs} steps={args_cli.steps} "
        f"device={device} methods={','.join(methods)}"
    )
    collect_start = time.perf_counter()
    demos = collect_oc_fmcs_demos(env, env_dt, action_dim, device)
    collect_seconds = time.perf_counter() - collect_start
    log(f"collected OC-FMCS demos in {collect_seconds:.2f}s")
    obs = demos["obs"]
    actions = demos["actions"]
    obs_mean, obs_std = normalize_stats(obs)
    action_mean, action_std = normalize_stats(actions)
    chunk_obs, chunks = build_chunk_dataset(obs, actions, args_cli.chunk_horizon)
    chunk_mean, chunk_std = normalize_stats(chunks.reshape(-1, args_cli.chunk_horizon * action_dim))

    train_start = time.perf_counter()
    log("training learned baselines")
    bc_model, bc_losses = train_bc(obs, actions, obs_mean, obs_std, action_mean, action_std, "bc_mlp")
    chunk_model, chunk_losses = train_chunk_mlp(chunk_obs, chunks, obs_mean, obs_std, chunk_mean, chunk_std, "chunk_mlp")
    diffusion_model, diffusion_losses = train_diffusion(
        chunk_obs,
        chunks,
        obs_mean,
        obs_std,
        chunk_mean,
        chunk_std,
        "diffusion_policy",
    )
    phase_bc_losses: list[float] = []
    phase_chunk_losses: list[float] = []
    phase_diffusion_losses: list[float] = []
    phase_bc_model = None
    phase_chunk_model = None
    phase_diffusion_model = None
    phase_obs_mean = None
    phase_obs_std = None
    if any(method.startswith("phase_") for method in methods):
        log("training phase-conditioned learned baselines")
        phase_obs = phase_conditioned_obs(obs, demos["phases"])
        phase_obs_mean, phase_obs_std = normalize_stats(phase_obs)
        phase_chunk_obs, _phase_chunks = build_chunk_dataset(phase_obs, actions, args_cli.chunk_horizon)
        phase_bc_model, phase_bc_losses = train_bc(
            phase_obs,
            actions,
            phase_obs_mean,
            phase_obs_std,
            action_mean,
            action_std,
            "phase_bc_mlp",
        )
        phase_chunk_model, phase_chunk_losses = train_chunk_mlp(
            phase_chunk_obs,
            chunks,
            phase_obs_mean,
            phase_obs_std,
            chunk_mean,
            chunk_std,
            "phase_chunk_mlp",
        )
        phase_diffusion_model, phase_diffusion_losses = train_diffusion(
            phase_chunk_obs,
            chunks,
            phase_obs_mean,
            phase_obs_std,
            chunk_mean,
            chunk_std,
            "phase_diffusion_policy",
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_start
    log(f"trained all learned baselines in {train_seconds:.2f}s")

    models = {"bc": bc_model, "chunk": chunk_model, "diffusion": diffusion_model}
    if phase_bc_model is not None:
        models.update(
            {
                "phase_bc": phase_bc_model,
                "phase_chunk": phase_chunk_model,
                "phase_diffusion": phase_diffusion_model,
            }
        )
    stats = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "chunk_mean": chunk_mean,
        "chunk_std": chunk_std,
    }
    norm_chunks = (chunks - chunk_mean) / chunk_std
    stats["chunk_lo"], stats["chunk_hi"] = norm_chunks.min(dim=0).values, norm_chunks.max(dim=0).values
    if phase_obs_mean is not None and phase_obs_std is not None:
        stats.update({"phase_obs_mean": phase_obs_mean, "phase_obs_std": phase_obs_std})
    results = {}
    for method in methods:
        print(f"[eval] {method}", flush=True)
        results[method] = evaluate_method(env, env_dt, method, models, stats)
    env.close()

    metrics = {
        "config": {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "demo_episodes": args_cli.demo_episodes,
            "steps": args_cli.steps,
            "seed": args_cli.seed,
            "eval_seed": args_cli.eval_seed,
            "chunk_horizon": args_cli.chunk_horizon,
            "bc_train_steps": args_cli.bc_train_steps,
            "chunk_train_steps": args_cli.chunk_train_steps,
            "diffusion_train_steps": args_cli.diffusion_train_steps,
            "diffusion_steps": args_cli.diffusion_steps,
            "diffusion_schedule": args_cli.diffusion_schedule,
            "diffusion_pred": args_cli.diffusion_pred,
            "diffusion_clip": args_cli.diffusion_clip,
            "ema_decay": args_cli.ema_decay,
            "exec_steps": args_cli.exec_steps,
            "batch_size": args_cli.batch_size,
            "hidden_dim": args_cli.hidden_dim,
            "lr": args_cli.lr,
            "methods": methods,
        },
        "data": {
            "demo_transitions": int(obs.numel() // obs.shape[-1]),
            "obs_dim": int(obs.shape[-1]),
            "action_dim": action_dim,
            "collect_seconds": collect_seconds,
            "train_seconds": train_seconds,
            "bc_loss_tail": mean_tail(bc_losses),
            "chunk_loss_tail": mean_tail(chunk_losses),
            "diffusion_loss_tail": mean_tail(diffusion_losses),
            "phase_bc_loss_tail": mean_tail(phase_bc_losses),
            "phase_chunk_loss_tail": mean_tail(phase_chunk_losses),
            "phase_diffusion_loss_tail": mean_tail(phase_diffusion_losses),
        },
        "methods": results,
    }
    args_cli.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {args_cli.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
