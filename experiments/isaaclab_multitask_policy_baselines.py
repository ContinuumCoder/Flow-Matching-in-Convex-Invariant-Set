

from __future__ import annotations

import argparse
import copy
import json
import math
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


SCENARIO_TASKS = {
    "reach": "Isaac-Reach-Franka-IK-Abs-v0",
    "drawer": "Isaac-Open-Drawer-Franka-IK-Abs-v0",
    "stack": "Isaac-Stack-Cube-Franka-IK-Abs-v0",
    "soft_lift": "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
}
DEFAULT_METHODS = (
    "direct_gen,bc_mlp,chunk_mlp,diffusion_policy,"
    "phase_bc_mlp,phase_chunk_mlp,phase_diffusion_policy,oc_fmcs"
)
NUM_PHASES = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-task Isaac Lab policy baseline sweep.")
    parser.add_argument("--scenario", choices=sorted(SCENARIO_TASKS), required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--eval-seed", type=int, default=2027)
    parser.add_argument("--disable-fabric", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--eval-methods", type=str, default=DEFAULT_METHODS)
    parser.add_argument("--fresh-env-per-method", action="store_true")
    parser.add_argument("--chunk-horizon", type=int, default=16)
    parser.add_argument("--bc-train-steps", type=int, default=1200)
    parser.add_argument("--chunk-train-steps", type=int, default=1200)
    parser.add_argument("--act-train-steps", type=int, default=1200)
    parser.add_argument("--act-layers", type=int, default=2)
    parser.add_argument("--act-heads", type=int, default=4)
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
        help="Denoiser target. With a width-256 MLP on 112-128-D chunks, eps-prediction has to reproduce "
        "the full noise vector and stays far from converged at this training budget; x0 does not.",
    )
    parser.add_argument(
        "--diffusion-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip the predicted clean chunk to the normalized demo range at each denoising step.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.995, help="EMA for diffusion/FMCS weights; 0 disables.")
    parser.add_argument(
        "--exec-steps",
        type=int,
        default=8,
        help="Receding horizon for generative chunk policies: execute this many actions of a sampled "
        "chunk before resampling (1 = resample every control step, the original behavior).",
    )
    parser.add_argument(
        "--demo-episodes",
        type=int,
        default=1,
        help="Controller rollouts used as demonstrations (each adds num_envs trajectories; seeds seed..seed+E-1).",
    )
    parser.add_argument(
        "--dump-demos",
        type=Path,
        default=None,
        help="Save training demos plus eval-seed expert rollouts to this .pt file and exit (offline diagnosis).",
    )
    parser.add_argument("--fm-train-steps", type=int, default=1600)
    parser.add_argument(
        "--fm-param",
        choices=["x", "v"],
        default="x",
        help="x: the network predicts the chunk endpoint z_hat and the field is (z_hat - x) / (1 - t), the "
        "paper's u_t(x|z) = c_t (z - x); v: the network regresses the velocity z - x0 directly.",
    )
    parser.add_argument(
        "--fm-clip-endpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --fm-param x, clamp z_hat to the active chart box (an O(D) projection, the analogue of "
        "DDPM x0 clipping); every Euler iterate is then a convex combination of points in the chart.",
    )
    parser.add_argument("--fm-ode-steps", type=int, default=10)
    parser.add_argument("--fm-chart-margin", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--position-threshold", type=float, default=0.025)
    parser.add_argument("--stack-overlap-threshold", type=float, default=0.20)
    parser.add_argument("--stack-above-z", type=float, default=0.14)
    parser.add_argument("--stack-grasp-z", type=float, default=0.018)
    parser.add_argument("--stack-lift-z", type=float, default=0.24)
    parser.add_argument("--stack-above-place-z", type=float, default=0.20)
    parser.add_argument("--stack-place-z", type=float, default=0.090)
    parser.add_argument("--stack-retreat-z", type=float, default=0.20)
    parser.add_argument("--stack-close-dwell", type=float, default=0.80)
    parser.add_argument("--stack-place-dwell", type=float, default=0.50)
    parser.add_argument("--stack-release-dwell", type=float, default=0.80)
    parser.add_argument("--stack-track-current-cubes", action="store_true")
    parser.add_argument("--mppi-num-samples", type=int, default=64)
    parser.add_argument("--mppi-noise-pos", type=float, default=0.035)
    parser.add_argument("--mppi-noise-grip", type=float, default=0.20)
    parser.add_argument("--mppi-temperature", type=float, default=0.02)
    parser.add_argument("--cem-elite-frac", type=float, default=0.20)
    parser.add_argument(
        "--stub-dex-retargeting",
        action="store_true",
        help="Replace dex_retargeting.retargeting_config with a placeholder before loading task configs "
        "(for hosts whose pinocchio wheel was built against NumPy 1.x but NumPy 2.x is installed).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.task is None:
        args.task = SCENARIO_TASKS[args.scenario]
    if args.out is None:
        args.out = Path(f"results/isaaclab_multitask_policy_baselines/{args.scenario}/metrics.json")
    if args.headless is None:
        args.headless = True
    args.enable_cameras = False
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


if args_cli.stub_dex_retargeting:
    import sys
    import types

    _dex_pkg = types.ModuleType("dex_retargeting")
    _dex_pkg.__path__ = []
    _dex_cfg = types.ModuleType("dex_retargeting.retargeting_config")
    _dex_cfg.RetargetingConfig = type("RetargetingConfig", (), {})
    sys.modules.setdefault("dex_retargeting", _dex_pkg)
    sys.modules.setdefault("dex_retargeting.retargeting_config", _dex_cfg)

import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F

import isaaclab_tasks
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg


def log(message: str) -> None:
    print(f"[multitask-baselines] {message}", flush=True)


def unpack_reset(reset_result: Any) -> Any:
    return reset_result[0] if isinstance(reset_result, tuple) else reset_result


def unpack_step(step_result: Any) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if len(step_result) == 5:
        return step_result
    if len(step_result) == 4:
        obs, reward, done, extras = step_result
        return obs, reward, done, torch.zeros_like(done, dtype=torch.bool), extras
    raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")


def policy_obs(obs: Any) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs
    if not isinstance(obs, dict) or "policy" not in obs:
        raise RuntimeError(f"Cannot find policy observation in {type(obs)}")
    policy = obs["policy"]
    if isinstance(policy, torch.Tensor):
        return policy
    if isinstance(policy, dict):
        pieces = []
        for key in sorted(policy.keys()):
            value = policy[key]
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                pieces.append(value.reshape(value.shape[0], -1))
        if pieces:
            return torch.cat(pieces, dim=-1)
    raise RuntimeError(f"Cannot flatten policy observation: {type(policy)}")


def phase_conditioned_obs(obs_tensor: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    phase_oh = F.one_hot(phases.long().clamp(0, NUM_PHASES - 1), num_classes=NUM_PHASES).to(obs_tensor.dtype)
    return torch.cat([obs_tensor, phase_oh], dim=-1)


def normalize_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(dim=0), flat.std(dim=0).clamp_min(1e-4)


def sample_indices(total: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, total, (batch_size,), device=device)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(last, hidden_dim), nn.SiLU()])
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


class FlowChunkMLP(DiffusionChunkMLP):
    def __init__(self, obs_dim: int, chunk_dim: int, hidden_dim: int, time_dim: int = 32):
        super().__init__(obs_dim, chunk_dim, hidden_dim, time_dim)
        if args_cli.fm_param == "v":
            nn.init.zeros_(self.net.net[-1].weight)
            nn.init.zeros_(self.net.net[-1].bias)

    def forward(self, obs: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return super().forward(obs, x, t * 1000.0)


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


class ACTChunkTransformer(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_proj = MLP(obs_dim, hidden_dim, hidden_dim, depth=2)
        self.query = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        memory = self.obs_proj(obs).unsqueeze(1)
        query = (self.query + self.pos).unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(query, memory)
        return self.out(decoded).reshape(batch, self.horizon * self.action_dim)


def train_bc(obs: torch.Tensor, actions: torch.Tensor, obs_mean, obs_std, action_mean, action_std, name: str):
    start = time.perf_counter()
    device = obs.device
    obs_dim = obs.shape[-1]
    action_dim = actions.shape[-1]
    model = MLP(obs_dim, action_dim, args_cli.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr)
    obs_flat = obs.reshape(-1, obs_dim)
    action_flat = actions.reshape(-1, action_dim)
    losses = []
    for _ in range(args_cli.bc_train_steps):
        idx = sample_indices(obs_flat.shape[0], args_cli.batch_size, device)
        pred = model((obs_flat[idx] - obs_mean) / obs_std)
        target = (action_flat[idx] - action_mean) / action_std
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    log(f"trained {name} in {time.perf_counter() - start:.2f}s")
    return model.eval(), losses


def build_chunk_dataset(obs: torch.Tensor, actions: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    episodes, steps, _obs_dim = obs.shape
    action_dim = actions.shape[-1]
    obs_rows = []
    chunk_rows = []
    for t in range(steps):
        end = min(t + horizon, steps)
        chunk = actions[:, t:end]
        if end - t < horizon:
            chunk = torch.cat([chunk, actions[:, end - 1 : end].repeat(1, horizon - (end - t), 1)], dim=1)
        obs_rows.append(obs[:, t])
        chunk_rows.append(chunk.reshape(episodes, horizon * action_dim))
    return torch.cat(obs_rows, dim=0), torch.cat(chunk_rows, dim=0)


def train_chunk(chunk_obs, chunks, obs_mean, obs_std, chunk_mean, chunk_std, name: str):
    start = time.perf_counter()
    device = chunk_obs.device
    model = MLP(chunk_obs.shape[-1], chunks.shape[-1], args_cli.hidden_dim, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr)
    losses = []
    for _ in range(args_cli.chunk_train_steps):
        idx = sample_indices(chunk_obs.shape[0], args_cli.batch_size, device)
        pred = model((chunk_obs[idx] - obs_mean) / obs_std)
        target = (chunks[idx] - chunk_mean) / chunk_std
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    log(f"trained {name} in {time.perf_counter() - start:.2f}s")
    return model.eval(), losses


def train_act(chunk_obs, chunks, obs_mean, obs_std, chunk_mean, chunk_std, action_dim: int, name: str):
    start = time.perf_counter()
    device = chunk_obs.device
    model = ACTChunkTransformer(
        chunk_obs.shape[-1],
        action_dim,
        args_cli.chunk_horizon,
        args_cli.hidden_dim,
        args_cli.act_layers,
        args_cli.act_heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr)
    losses = []
    for _ in range(args_cli.act_train_steps):
        idx = sample_indices(chunk_obs.shape[0], args_cli.batch_size, device)
        pred = model((chunk_obs[idx] - obs_mean) / obs_std)
        target = (chunks[idx] - chunk_mean) / chunk_std
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    log(f"trained {name} in {time.perf_counter() - start:.2f}s")
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
    start = time.perf_counter()
    device = chunk_obs.device
    chunk_dim = chunks.shape[-1]
    model = DiffusionChunkMLP(chunk_obs.shape[-1], chunk_dim, args_cli.hidden_dim).to(device)
    ema = ema_copy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr)
    _betas, _alphas, alpha_bar = diffusion_schedule(args_cli.diffusion_steps, device)
    norm_chunks = (chunks - chunk_mean) / chunk_std
    norm_obs = (chunk_obs - obs_mean) / obs_std
    losses = []
    for _ in range(args_cli.diffusion_train_steps):
        idx = sample_indices(chunk_obs.shape[0], args_cli.batch_size, device)
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
        f"trained {name} in {time.perf_counter() - start:.2f}s "
        f"({args_cli.diffusion_train_steps} train steps, {args_cli.diffusion_steps} denoise steps, "
        f"{args_cli.diffusion_schedule} schedule, alpha_bar_T={float(alpha_bar[-1]):.4f}, "
        f"{args_cli.diffusion_pred}-prediction)"
    )
    return (ema if ema is not None else model).eval(), losses


@torch.inference_mode()
def sample_diffusion_chunk(model, obs, obs_mean, obs_std, stats: dict[str, torch.Tensor], action_dim: int) -> torch.Tensor:
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


def chart_boxes(norm_chunks: torch.Tensor, charts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lo = norm_chunks.min(dim=0).values.repeat(NUM_PHASES, 1)
    hi = norm_chunks.max(dim=0).values.repeat(NUM_PHASES, 1)
    for k in range(NUM_PHASES):
        mask = charts == k
        if mask.any():
            lo[k] = norm_chunks[mask].min(dim=0).values
            hi[k] = norm_chunks[mask].max(dim=0).values
    return lo - args_cli.fm_chart_margin, hi + args_cli.fm_chart_margin


def chart_prior(lo: torch.Tensor, hi: torch.Tensor, charts: torch.Tensor) -> torch.Tensor:
    eps = torch.randn(charts.shape[0], lo.shape[-1], device=lo.device)
    return torch.maximum(torch.minimum(eps, hi[charts]), lo[charts])


def train_fmcs(chunk_obs, chunks, charts, obs_mean, obs_std, chunk_mean, chunk_std, lo, hi, name: str):
    start = time.perf_counter()
    device = chunk_obs.device
    model = FlowChunkMLP(chunk_obs.shape[-1], chunks.shape[-1], args_cli.hidden_dim).to(device)
    ema = ema_copy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr)
    norm_chunks = (chunks - chunk_mean) / chunk_std
    norm_obs = (chunk_obs - obs_mean) / obs_std
    losses = []
    for _ in range(args_cli.fm_train_steps):
        idx = sample_indices(chunk_obs.shape[0], args_cli.batch_size, device)
        z = norm_chunks[idx]
        x0 = chart_prior(lo, hi, charts[idx])
        t = torch.rand(args_cli.batch_size, 1, device=device)
        xt = (1.0 - t) * x0 + t * z
        pred = model(norm_obs[idx], xt, t.squeeze(-1))
        loss = F.mse_loss(pred, z if args_cli.fm_param == "x" else z - x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema_update(ema, model)
        losses.append(float(loss.detach().cpu()))
    log(f"trained {name} in {time.perf_counter() - start:.2f}s ({args_cli.fm_train_steps} train steps)")
    return (ema if ema is not None else model).eval(), losses


@torch.inference_mode()
def sample_fmcs_chunk(model, phase_obs, charts, stats: dict[str, torch.Tensor], action_dim: int) -> torch.Tensor:
    n = phase_obs.shape[0]
    norm_obs = (phase_obs - stats["phase_obs_mean"]) / stats["phase_obs_std"]
    lo, hi = stats["chart_lo"][charts], stats["chart_hi"][charts]
    x = chart_prior(stats["chart_lo"], stats["chart_hi"], charts)
    dt = 1.0 / args_cli.fm_ode_steps
    for i in range(args_cli.fm_ode_steps):
        t = torch.full((n,), i * dt, device=x.device)
        out = model(norm_obs, x, t)
        if args_cli.fm_param == "x":
            if args_cli.fm_clip_endpoint:
                out = torch.maximum(torch.minimum(out, hi), lo)
            x = x + dt * (out - x) / (1.0 - i * dt)
        else:
            x = x + dt * out
    chunk = x * stats["chunk_std"] + stats["chunk_mean"]
    return chunk.reshape(n, args_cli.chunk_horizon, action_dim)


def mean_tail(values: list[float], k: int = 50) -> float | None:
    if not values:
        return None
    tail = values[-min(k, len(values)) :]
    return float(sum(tail) / len(tail))


def sanitize_action(action: torch.Tensor, scenario: str) -> torch.Tensor:
    action = action.clone()
    if action.shape[-1] >= 3:
        action[:, 0] = action[:, 0].clamp(0.05, 1.15)
        action[:, 1] = action[:, 1].clamp(-0.65, 0.65)
        action[:, 2] = action[:, 2].clamp(0.00, 0.85 if scenario != "drawer" else 1.25)
    if action.shape[-1] >= 7:
        quat = action[:, 3:7]
        fallback = torch.tensor([0.0, 1.0, 0.0, 0.0], device=action.device, dtype=action.dtype)
        norm = quat.norm(dim=-1, keepdim=True)
        action[:, 3:7] = torch.where(norm > 1e-6, quat / norm.clamp_min(1e-6), fallback.repeat(action.shape[0], 1))
    if action.shape[-1] >= 8:
        action[:, 7] = action[:, 7].clamp(-1.0, 1.0)
    return action


def ee_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    scene = env.unwrapped.scene
    device = torch.device(env.unwrapped.device)
    if "ee_frame" in scene.keys():
        frame = scene["ee_frame"]
        pos = frame.data.target_pos_w[..., 0, :].clone() - scene.env_origins
        quat = frame.data.target_quat_w[..., 0, :].clone()
        return pos.to(device), quat.to(device)
    arm_action = env.unwrapped.action_manager._terms["arm_action"]
    return arm_action._compute_frame_pose()


def reach_target(env) -> tuple[torch.Tensor, torch.Tensor]:
    cmd = env.unwrapped.command_manager.get_command("ee_pose")
    return cmd[..., :3], cmd[..., 3:7]


def drawer_handle_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    frame = env.unwrapped.scene["cabinet_frame"]
    pos = frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    quat = frame.data.target_quat_w[..., 0, :].clone()
    return pos, quat


def drawer_joint(env) -> torch.Tensor:
    cabinet = env.unwrapped.scene["cabinet"]
    joint_ids, _ = cabinet.find_joints(["drawer_top_joint"])
    return cabinet.data.joint_pos[:, joint_ids[0]]


def cube_pos(env, name: str) -> torch.Tensor:
    return env.unwrapped.scene[name].data.root_pos_w - env.unwrapped.scene.env_origins


def soft_object_centroid(env) -> torch.Tensor:
    return env.unwrapped.scene["object"].data.nodal_pos_w.mean(dim=1) - env.unwrapped.scene.env_origins


def gripper_gap(env) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    joint_ids, _ = robot.find_joints(["panda_finger.*"])
    if len(joint_ids) < 2:
        return torch.zeros(env.unwrapped.num_envs, device=robot.data.joint_pos.device)
    return robot.data.joint_pos[:, joint_ids[0]] + robot.data.joint_pos[:, joint_ids[1]]


def lift_target(env) -> torch.Tensor:
    return env.unwrapped.command_manager.get_command("object_pose")[..., :3]


def fixed_grasp_quat(n: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=dtype).repeat(n, 1)


class ScenarioController:
    def __init__(self, env, scenario: str):
        self.env = env
        self.scenario = scenario
        self.device = torch.device(env.unwrapped.device)
        self.num_envs = env.unwrapped.num_envs
        self.dt = float(env.unwrapped.step_dt)
        self.state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.wait = torch.zeros(self.num_envs, device=self.device)
        self.stack_c1_anchor: torch.Tensor | None = None
        self.stack_c2_anchor: torch.Tensor | None = None

    def reset(self) -> None:
        self.state.zero_()
        self.wait.zero_()
        self.stack_c1_anchor = None
        self.stack_c2_anchor = None

    def _advance(self, desired_pos: torch.Tensor, ee_pos: torch.Tensor, schedule: list[float]) -> None:
        threshold = (
            max(args_cli.position_threshold, args_cli.stack_overlap_threshold)
            if self.scenario == "stack"
            else args_cli.position_threshold
        )
        close = (ee_pos - desired_pos).norm(dim=-1) < threshold
        next_state = self.state.clone()
        next_wait = self.wait + self.dt
        for state_id, wait_time in enumerate(schedule[:-1]):
            mask = self.state == state_id
            if state_id == 0:
                ready = next_wait >= wait_time
            else:
                ready = close & (next_wait >= wait_time)
            next_state[mask & ready] = state_id + 1
        changed = next_state != self.state
        next_wait[changed] = 0.0
        self.state = next_state
        self.wait = next_wait

    @torch.inference_mode()
    def action(self, obs: Any) -> torch.Tensor:
        if self.scenario == "reach":
            return self._reach_action()
        if self.scenario == "drawer":
            return self._drawer_action()
        if self.scenario == "stack":
            return self._stack_action()
        if self.scenario == "soft_lift":
            return self._soft_lift_action()
        raise ValueError(self.scenario)

    def _reach_action(self) -> torch.Tensor:
        ee_pos, ee_quat = ee_pose(self.env)
        target_pos, target_quat = reach_target(self.env)
        mid_pos = target_pos + torch.tensor([0.0, 0.0, 0.08], device=self.device)
        desired_pos = torch.where((self.state == 1).unsqueeze(-1), mid_pos, target_pos)
        desired_pos[self.state == 0] = ee_pos[self.state == 0]
        desired_quat = target_quat.clone()
        desired_quat[self.state == 0] = ee_quat[self.state == 0]
        self._advance(desired_pos, ee_pos, [0.10, 0.05, 999.0])
        return sanitize_action(torch.cat([desired_pos, desired_quat], dim=-1), self.scenario)

    def _drawer_action(self) -> torch.Tensor:
        ee_pos, ee_quat = ee_pose(self.env)
        handle_pos, handle_quat = drawer_handle_pose(self.env)
        n = handle_pos.shape[0]
        approach_offset = torch.tensor([-0.10, 0.0, 0.0], device=self.device).repeat(n, 1)
        grasp_offset = torch.tensor([0.025, 0.0, 0.0], device=self.device).repeat(n, 1)
        pull_offset = torch.tensor([-0.30, 0.0, 0.0], device=self.device).repeat(n, 1)


        approach_pos, approach_quat = handle_pos + approach_offset, handle_quat
        grasp_pos, grasp_quat = handle_pos + grasp_offset, handle_quat
        pull_pos, pull_quat = handle_pos + pull_offset, handle_quat
        desired_pos = torch.empty_like(handle_pos)
        desired_quat = torch.empty_like(handle_quat)
        grip = torch.ones(n, device=self.device)
        for sid, pos, quat, gripper in [
            (0, ee_pos, ee_quat, 1.0),
            (1, approach_pos, approach_quat, 1.0),
            (2, handle_pos, handle_quat, 1.0),
            (3, grasp_pos, grasp_quat, -1.0),
            (4, pull_pos, pull_quat, -1.0),
            (5, ee_pos, ee_quat, -1.0),
        ]:
            mask = self.state == sid
            desired_pos[mask] = pos[mask]
            desired_quat[mask] = quat[mask]
            grip[mask] = gripper
        self._advance(desired_pos, ee_pos, [0.50, 0.40, 0.35, 0.35, 2.00, 999.0])
        return sanitize_action(torch.cat([desired_pos, desired_quat, grip.unsqueeze(-1)], dim=-1), self.scenario)

    def _stack_action(self) -> torch.Tensor:
        ee_pos, ee_quat = ee_pose(self.env)
        c1 = cube_pos(self.env, "cube_1")
        c2 = cube_pos(self.env, "cube_2")
        if self.stack_c1_anchor is None or self.stack_c2_anchor is None:
            self.stack_c1_anchor = c1.detach().clone()
            self.stack_c2_anchor = c2.detach().clone()
        c1_anchor = self.stack_c1_anchor
        c2_anchor = self.stack_c2_anchor
        c1_place_ref = c1 if args_cli.stack_track_current_cubes else c1_anchor
        c2_pick_ref = c2 if args_cli.stack_track_current_cubes else c2_anchor
        n = c1.shape[0]
        quat = fixed_grasp_quat(n, self.device, c1.dtype)
        above_c2 = c2_pick_ref + torch.tensor([0.0, 0.0, args_cli.stack_above_z], device=self.device)
        grasp_c2 = c2_pick_ref + torch.tensor([0.0, 0.0, args_cli.stack_grasp_z], device=self.device)
        lift_c2 = c2_anchor + torch.tensor([0.0, 0.0, args_cli.stack_lift_z], device=self.device)
        above_c1 = c1_place_ref + torch.tensor([0.0, 0.0, args_cli.stack_above_place_z], device=self.device)
        place_c1 = c1_place_ref + torch.tensor([0.0, 0.0, args_cli.stack_place_z], device=self.device)
        retreat = c1_place_ref + torch.tensor([0.0, 0.0, args_cli.stack_retreat_z], device=self.device)
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
        self._advance(
            desired_pos,
            ee_pos,
            [
                0.20,
                0.45,
                0.35,
                args_cli.stack_close_dwell,
                0.55,
                0.55,
                args_cli.stack_place_dwell,
                args_cli.stack_release_dwell,
                999.0,
            ],
        )
        return sanitize_action(torch.cat([desired_pos, desired_quat, grip.unsqueeze(-1)], dim=-1), self.scenario)

    def _soft_lift_action(self) -> torch.Tensor:
        ee_pos, ee_quat = ee_pose(self.env)
        obj = soft_object_centroid(self.env)
        target = lift_target(self.env)
        n = obj.shape[0]
        quat = fixed_grasp_quat(n, self.device, obj.dtype)
        above = obj + torch.tensor([0.0, 0.0, 0.12], device=self.device)
        grasp = obj + torch.tensor([0.0, 0.0, 0.01], device=self.device)
        desired_pos = torch.empty_like(obj)
        desired_quat = quat.clone()
        grip = torch.ones(n, device=self.device)
        targets = [ee_pos, above, grasp, grasp, target]
        grips = [1.0, 1.0, 1.0, -1.0, -1.0]
        for sid, pos in enumerate(targets):
            mask = self.state == sid
            desired_pos[mask] = pos[mask]
            grip[mask] = grips[sid]
        desired_quat[self.state == 0] = ee_quat[self.state == 0]
        self._advance(desired_pos, ee_pos, [0.20, 0.30, 0.35, 0.35, 999.0])
        return sanitize_action(torch.cat([desired_pos, desired_quat, grip.unsqueeze(-1)], dim=-1), self.scenario)


def direct_action(env, scenario: str, step: int, steps: int, action_dim: int, initial_ee_pos: torch.Tensor, initial_ee_quat: torch.Tensor):
    device = torch.device(env.unwrapped.device)
    n = env.unwrapped.num_envs
    tau = torch.full((n, 1), step / max(steps - 1, 1), device=device)
    w = tau * tau * (3.0 - 2.0 * tau)
    grip = torch.ones((n, 1), device=device)
    quat = initial_ee_quat.clone()
    if scenario == "reach":
        target_pos, target_quat = reach_target(env)
        pos = (1.0 - w) * initial_ee_pos + w * target_pos
        quat = target_quat
        action = torch.cat([pos, quat], dim=-1)
    elif scenario == "drawer":
        handle_pos, handle_quat = drawer_handle_pose(env)
        final_pos, quat = combine_frame_transforms(
            handle_pos, handle_quat, torch.tensor([-0.35, 0.0, 0.0], device=device).repeat(n, 1)
        )
        pos = (1.0 - w) * initial_ee_pos + w * final_pos
        grip = 1.0 - 2.0 * ((tau - 0.45) / 0.35).clamp(0.0, 1.0)
        action = torch.cat([pos, quat, grip], dim=-1)
    elif scenario == "stack":
        c1 = cube_pos(env, "cube_1")
        pos = (1.0 - w) * initial_ee_pos + w * (c1 + torch.tensor([0.0, 0.0, 0.08], device=device))
        quat = fixed_grasp_quat(n, device, pos.dtype)
        grip = 1.0 - 2.0 * ((tau - 0.35) / 0.40).clamp(0.0, 1.0)
        action = torch.cat([pos, quat, grip], dim=-1)
    elif scenario == "soft_lift":
        target = lift_target(env)
        pos = (1.0 - w) * initial_ee_pos + w * target
        quat = fixed_grasp_quat(n, device, target.dtype)
        grip = 1.0 - 2.0 * ((tau - 0.35) / 0.40).clamp(0.0, 1.0)
        action = torch.cat([pos, quat, grip], dim=-1)
    else:
        raise ValueError(scenario)
    if action.shape[-1] != action_dim:
        if action.shape[-1] < action_dim:
            action = torch.cat([action, torch.zeros((n, action_dim - action.shape[-1]), device=device)], dim=-1)
        else:
            action = action[:, :action_dim]
    return sanitize_action(action, scenario)


def sampling_chart_action(nominal_action: torch.Tensor, scenario: str, mode: str) -> torch.Tensor:

    n, action_dim = nominal_action.shape
    k = max(2, args_cli.mppi_num_samples)
    candidates = nominal_action.unsqueeze(1).repeat(1, k, 1)
    if action_dim >= 3:
        candidates[:, :, :3] += args_cli.mppi_noise_pos * torch.randn(
            n, k, 3, device=nominal_action.device, dtype=nominal_action.dtype
        )
    if action_dim >= 8:
        candidates[:, :, 7] += args_cli.mppi_noise_grip * torch.randn(
            n, k, device=nominal_action.device, dtype=nominal_action.dtype
        )
    candidates = sanitize_action(candidates.reshape(n * k, action_dim), scenario).reshape(n, k, action_dim)
    pos_cost = torch.zeros(n, k, device=nominal_action.device, dtype=nominal_action.dtype)
    if action_dim >= 3:
        pos_cost = ((candidates[:, :, :3] - nominal_action[:, None, :3]) ** 2).sum(dim=-1)
    quat_cost = torch.zeros_like(pos_cost)
    if action_dim >= 7:
        quat_cost = 0.05 * ((candidates[:, :, 3:7] - nominal_action[:, None, 3:7]) ** 2).sum(dim=-1)
    grip_cost = torch.zeros_like(pos_cost)
    if action_dim >= 8:
        grip_cost = 0.25 * ((candidates[:, :, 7] - nominal_action[:, None, 7]) ** 2)
    cost = pos_cost + quat_cost + grip_cost
    if mode == "mppi_fsm":
        weights = torch.softmax(-cost / max(args_cli.mppi_temperature, 1e-6), dim=1)
        action = (weights.unsqueeze(-1) * candidates).sum(dim=1)
    elif mode == "cem_fsm":
        elite = max(1, int(round(k * args_cli.cem_elite_frac)))
        idx = torch.topk(cost, elite, largest=False, dim=1).indices
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, action_dim)
        action = torch.gather(candidates, 1, gather_idx).mean(dim=1)
    else:
        raise ValueError(mode)
    return sanitize_action(action, scenario)


def global_prior_target(env, scenario: str) -> tuple[torch.Tensor, torch.Tensor | None]:

    device = torch.device(env.unwrapped.device)
    if scenario == "reach":
        target_pos, _ = reach_target(env)
        return target_pos, None
    if scenario == "drawer":
        handle_pos, handle_quat = drawer_handle_pose(env)
        final_pos, _ = combine_frame_transforms(
            handle_pos,
            handle_quat,
            torch.tensor([-0.35, 0.0, 0.0], device=device).repeat(handle_pos.shape[0], 1),
        )
        return final_pos, None
    if scenario == "stack":
        c1 = cube_pos(env, "cube_1")
        return c1 + torch.tensor([0.0, 0.0, 0.08], device=device), None
    if scenario == "soft_lift":
        return lift_target(env), None
    raise ValueError(scenario)


def sampling_direct_action(env, nominal_action: torch.Tensor, scenario: str, mode: str) -> torch.Tensor:

    n, action_dim = nominal_action.shape
    k = max(2, args_cli.mppi_num_samples)
    candidates = nominal_action.unsqueeze(1).repeat(1, k, 1)
    if action_dim >= 3:
        candidates[:, :, :3] += args_cli.mppi_noise_pos * torch.randn(
            n, k, 3, device=nominal_action.device, dtype=nominal_action.dtype
        )
    if action_dim >= 8:
        candidates[:, :, 7] += args_cli.mppi_noise_grip * torch.randn(
            n, k, device=nominal_action.device, dtype=nominal_action.dtype
        )
    candidates = sanitize_action(candidates.reshape(n * k, action_dim), scenario).reshape(n, k, action_dim)

    prior_cost = torch.zeros(n, k, device=nominal_action.device, dtype=nominal_action.dtype)
    if action_dim >= 3:
        prior_cost = prior_cost + ((candidates[:, :, :3] - nominal_action[:, None, :3]) ** 2).sum(dim=-1)
        global_target, _ = global_prior_target(env, scenario)
        task_cost = ((candidates[:, :, :3] - global_target[:, None, :3]) ** 2).sum(dim=-1)
    else:
        task_cost = torch.zeros_like(prior_cost)
    if action_dim >= 7:
        prior_cost = prior_cost + 0.05 * ((candidates[:, :, 3:7] - nominal_action[:, None, 3:7]) ** 2).sum(dim=-1)
    if action_dim >= 8:
        prior_cost = prior_cost + 0.25 * ((candidates[:, :, 7] - nominal_action[:, None, 7]) ** 2)
    cost = task_cost + 0.35 * prior_cost
    if mode == "mppi_direct":
        weights = torch.softmax(-cost / max(args_cli.mppi_temperature, 1e-6), dim=1)
        action = (weights.unsqueeze(-1) * candidates).sum(dim=1)
    elif mode == "cem_direct":
        elite = max(1, int(round(k * args_cli.cem_elite_frac)))
        idx = torch.topk(cost, elite, largest=False, dim=1).indices
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, action_dim)
        action = torch.gather(candidates, 1, gather_idx).mean(dim=1)
    else:
        raise ValueError(mode)
    return sanitize_action(action, scenario)


def scenario_snapshot(env, scenario: str) -> dict[str, torch.Tensor]:
    if scenario == "reach":
        ee_pos_curr, _ = ee_pose(env)
        target_pos, _ = reach_target(env)
        return {"ee_pos": ee_pos_curr, "target_pos": target_pos}
    if scenario == "drawer":
        return {"drawer_pos": drawer_joint(env)}
    if scenario == "stack":
        c1 = cube_pos(env, "cube_1")
        c2 = cube_pos(env, "cube_2")
        c3 = cube_pos(env, "cube_3")
        return {"cube1": c1, "cube2": c2, "cube3": c3}
    if scenario == "soft_lift":
        obj = soft_object_centroid(env)
        return {"object_pos": obj, "target_pos": lift_target(env)}
    raise ValueError(scenario)


def scenario_metrics(start: dict[str, torch.Tensor], final: dict[str, torch.Tensor], max_track: dict[str, torch.Tensor], scenario: str) -> dict[str, float]:
    if scenario == "reach":
        err = (final["ee_pos"] - final["target_pos"]).norm(dim=-1)
        return {"success_rate": float((err < 0.04).float().mean().cpu()), "mean_error": float(err.mean().cpu())}
    if scenario == "drawer":
        final_pos = final["drawer_pos"]
        max_pos = max_track["drawer_pos"]
        return {
            "success_rate": float((max_pos > 0.25).float().mean().cpu()),
            "mean_final_drawer_pos": float(final_pos.mean().cpu()),
            "mean_max_drawer_pos": float(max_pos.mean().cpu()),
        }
    if scenario == "stack":
        c1, c2, c3 = final["cube1"], final["cube2"], final["cube3"]
        d12 = c2 - c1
        d23 = c3 - c2
        stack1 = (d12[:, :2].norm(dim=-1) < 0.05) & ((d12[:, 2] - 0.0468).abs() < 0.025)
        stack2 = stack1 & (d23[:, :2].norm(dim=-1) < 0.05) & ((d23[:, 2] - 0.0468).abs() < 0.025)
        max_cube2_lift = max_track["cube2_z"] - start["cube2"][:, 2]
        lift2 = max_cube2_lift > 0.05
        return {
            "success_rate": float(stack1.float().mean().cpu()),
            "full_stack_rate": float(stack2.float().mean().cpu()),
            "cube2_lift_rate": float(lift2.float().mean().cpu()),
            "mean_max_cube2_lift": float(max_cube2_lift.mean().cpu()),
            "mean_cube2_xy_error_to_cube1": float(d12[:, :2].norm(dim=-1).mean().cpu()),
            "mean_cube2_height_over_cube1": float(d12[:, 2].mean().cpu()),
            "mean_min_ee_cube2_distance": float(max_track["min_ee_cube2_dist"].mean().cpu()),
            "mean_min_gripper_gap": float(max_track["min_gripper_gap"].mean().cpu()),
        }
    if scenario == "soft_lift":
        lift = final["object_pos"][:, 2] - start["object_pos"][:, 2]
        max_lift = max_track["object_z"] - start["object_pos"][:, 2]
        target_error = (final["object_pos"] - final["target_pos"]).norm(dim=-1)
        return {
            "success_rate": float((max_lift > 0.10).float().mean().cpu()),
            "mean_lift_height": float(lift.mean().cpu()),
            "mean_max_lift_height": float(max_lift.mean().cpu()),
            "mean_target_distance": float(target_error.mean().cpu()),
        }
    raise ValueError(scenario)


def update_max_track(max_track: dict[str, torch.Tensor], snap: dict[str, torch.Tensor], scenario: str, env=None) -> None:
    if scenario == "drawer":
        max_track["drawer_pos"] = torch.maximum(max_track["drawer_pos"], snap["drawer_pos"])
    elif scenario == "stack":
        max_track["cube2_z"] = torch.maximum(max_track["cube2_z"], snap["cube2"][:, 2])
        if env is not None:
            ee_pos_curr, _ = ee_pose(env)
            ee_cube2_dist = (ee_pos_curr - snap["cube2"]).norm(dim=-1)
            max_track["min_ee_cube2_dist"] = torch.minimum(max_track["min_ee_cube2_dist"], ee_cube2_dist)
            max_track["min_gripper_gap"] = torch.minimum(max_track["min_gripper_gap"], gripper_gap(env))
    elif scenario == "soft_lift":
        max_track["object_z"] = torch.maximum(max_track["object_z"], snap["object_pos"][:, 2])


@torch.inference_mode()
def collect_demos(env, action_dim: int, device: torch.device, seed: int | None = None, episodes: int | None = None):
    seed = args_cli.seed if seed is None else seed
    episodes = args_cli.demo_episodes if episodes is None else episodes
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    episode_data = []
    start = time.perf_counter()
    for episode in range(episodes):
        obs = unpack_reset(env.reset(seed=seed + episode))
        controller = ScenarioController(env, args_cli.scenario)
        obs_steps = []
        action_steps = []
        phase_steps = []
        for _ in range(args_cli.steps):
            action = controller.action(obs)
            obs_steps.append(policy_obs(obs).detach().clone())
            action_steps.append(action.detach().clone())
            phase_steps.append(controller.state.detach().clone())
            obs, _reward, _terminated, _truncated, _ = unpack_step(env.step(action))
        episode_data.append(
            (torch.stack(obs_steps, dim=1), torch.stack(action_steps, dim=1), torch.stack(phase_steps, dim=1))
        )
    log(f"collected {episodes} OC-FMCS demo episode(s) (seed {seed}) in {time.perf_counter() - start:.2f}s")
    return {
        "obs": torch.cat([e[0] for e in episode_data], dim=0),
        "actions": torch.cat([e[1] for e in episode_data], dim=0),
        "phases": torch.cat([e[2] for e in episode_data], dim=0),
    }


@torch.inference_mode()
def evaluate(env, method: str, models: dict[str, Any], stats: dict[str, torch.Tensor]):
    device = torch.device(env.unwrapped.device)
    action_dim = int(env.action_space.shape[-1])
    random.seed(args_cli.eval_seed)
    torch.manual_seed(args_cli.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args_cli.eval_seed)
    obs = unpack_reset(env.reset(seed=args_cli.eval_seed))
    controller = ScenarioController(env, args_cli.scenario)
    initial_ee_pos, initial_ee_quat = ee_pose(env)
    snap0 = scenario_snapshot(env, args_cli.scenario)
    max_track = {k: v.clone() for k, v in snap0.items()}
    if args_cli.scenario == "stack":
        max_track["cube2_z"] = snap0["cube2"][:, 2].clone()
        ee_pos_curr, _ = ee_pose(env)
        max_track["min_ee_cube2_dist"] = (ee_pos_curr - snap0["cube2"]).norm(dim=-1)
        max_track["min_gripper_gap"] = gripper_gap(env)
    if args_cli.scenario == "soft_lift":
        max_track["object_z"] = snap0["object_pos"][:, 2].clone()
    rewards = torch.zeros(env.unwrapped.num_envs, device=device)
    terminations = torch.zeros(env.unwrapped.num_envs, device=device)
    controller_state_counts = torch.zeros(16, device=device)
    plan = ChunkPlan(env.unwrapped.num_envs, args_cli.chunk_horizon, action_dim, args_cli.exec_steps, device)
    prev_action: torch.Tensor | None = None
    ee_hist = [ee_pose(env)[0].clone()]
    cmd_jitter = torch.zeros(env.unwrapped.num_envs, device=device)
    grip_flips = torch.zeros(env.unwrapped.num_envs, device=device)
    ee_accel = torch.zeros(env.unwrapped.num_envs, device=device)
    policy_compute = 0.0
    start = time.perf_counter()
    for step in range(args_cli.steps):
        obs_now = policy_obs(obs)
        policy_start = time.perf_counter()
        if method == "direct_gen":
            action = direct_action(env, args_cli.scenario, step, args_cli.steps, action_dim, initial_ee_pos, initial_ee_quat)
        elif method in {"mppi_direct", "cem_direct"}:
            nominal = direct_action(env, args_cli.scenario, step, args_cli.steps, action_dim, initial_ee_pos, initial_ee_quat)
            action = sampling_direct_action(env, nominal, args_cli.scenario, method)
        elif method == "oc_fmcs":
            action = controller.action(obs)
        elif method in {"mppi_fsm", "cem_fsm"}:
            nominal = controller.action(obs)
            action = sampling_chart_action(nominal, args_cli.scenario, method)
        elif method == "bc_mlp":
            action = models["bc"]((obs_now - stats["obs_mean"]) / stats["obs_std"]) * stats["action_std"] + stats["action_mean"]
            action = sanitize_action(action, args_cli.scenario)
        elif method == "chunk_mlp":
            chunk = models["chunk"]((obs_now - stats["obs_mean"]) / stats["obs_std"]) * stats["chunk_std"] + stats["chunk_mean"]
            action = sanitize_action(chunk.reshape(env.unwrapped.num_envs, args_cli.chunk_horizon, action_dim)[:, 0], args_cli.scenario)
        elif method == "act_chunk":
            chunk = models["act"]((obs_now - stats["obs_mean"]) / stats["obs_std"]) * stats["chunk_std"] + stats["chunk_mean"]
            action = sanitize_action(chunk.reshape(env.unwrapped.num_envs, args_cli.chunk_horizon, action_dim)[:, 0], args_cli.scenario)
        elif method == "diffusion_policy":
            need = plan.need_replan()
            if need.any():
                plan.update(
                    need,
                    sample_diffusion_chunk(
                        models["diffusion"], obs_now[need], stats["obs_mean"], stats["obs_std"], stats, action_dim
                    ),
                )
            action = sanitize_action(plan.pop(), args_cli.scenario)
        elif method == "oc_fmcs_learned":
            _chart_command = controller.action(obs)
            need = plan.need_replan(controller.state)
            if need.any():
                charts = controller.state[need].clamp(0, NUM_PHASES - 1)
                plan.update(
                    need,
                    sample_fmcs_chunk(
                        models["oc_fmcs_learned"], phase_conditioned_obs(obs_now[need], charts), charts, stats, action_dim
                    ),
                )
            action = sanitize_action(plan.pop(), args_cli.scenario)
        elif method in {"phase_bc_mlp", "phase_chunk_mlp", "phase_act_chunk", "phase_diffusion_policy"}:
            _teacher_action = controller.action(obs)
            phase_obs = phase_conditioned_obs(obs_now, controller.state)
            if method == "phase_bc_mlp":
                action = models["phase_bc"]((phase_obs - stats["phase_obs_mean"]) / stats["phase_obs_std"])
                action = action * stats["action_std"] + stats["action_mean"]
            elif method == "phase_chunk_mlp":
                chunk = models["phase_chunk"]((phase_obs - stats["phase_obs_mean"]) / stats["phase_obs_std"])
                chunk = chunk * stats["chunk_std"] + stats["chunk_mean"]
                action = chunk.reshape(env.unwrapped.num_envs, args_cli.chunk_horizon, action_dim)[:, 0]
            elif method == "phase_act_chunk":
                chunk = models["phase_act"]((phase_obs - stats["phase_obs_mean"]) / stats["phase_obs_std"])
                chunk = chunk * stats["chunk_std"] + stats["chunk_mean"]
                action = chunk.reshape(env.unwrapped.num_envs, args_cli.chunk_horizon, action_dim)[:, 0]
            else:
                need = plan.need_replan(controller.state)
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
                action = plan.pop()
            action = sanitize_action(action, args_cli.scenario)
        else:
            raise ValueError(method)
        if args_cli.scenario == "stack" and method in {
            "oc_fmcs",
            "oc_fmcs_learned",
            "mppi_fsm",
            "cem_fsm",
            "phase_bc_mlp",
            "phase_chunk_mlp",
            "phase_act_chunk",
            "phase_diffusion_policy",
        }:
            controller_state_counts += torch.bincount(controller.state.clamp(0, 15), minlength=16).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        policy_compute += time.perf_counter() - policy_start
        if prev_action is not None:
            cmd_jitter += (action[:, :3] - prev_action[:, :3]).norm(dim=-1)
            if action_dim >= 8:
                grip_flips += ((action[:, 7] < 0) != (prev_action[:, 7] < 0)).float()
        prev_action = action.clone()
        obs, reward, terminated, _truncated, _ = unpack_step(env.step(action))
        rewards += reward
        terminations += terminated.float()
        ee_hist = (ee_hist + [ee_pose(env)[0].clone()])[-3:]
        if len(ee_hist) == 3:
            ee_accel += (ee_hist[2] - 2.0 * ee_hist[1] + ee_hist[0]).norm(dim=-1) / float(env.unwrapped.step_dt) ** 2
        update_max_track(max_track, scenario_snapshot(env, args_cli.scenario), args_cli.scenario, env)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final_snap = scenario_snapshot(env, args_cli.scenario)
    task_metrics = scenario_metrics(snap0, final_snap, max_track, args_cli.scenario)
    if args_cli.scenario == "stack" and controller_state_counts.sum() > 0:
        counts = controller_state_counts.detach().cpu()
        task_metrics["controller_final_state_mean"] = float(controller.state.float().mean().detach().cpu())
        task_metrics["controller_state_counts"] = {
            str(i): float(counts[i].item()) for i in range(len(counts)) if counts[i].item() > 0
        }
    task_metrics.update(
        {
            "method": method,
            "mean_cmd_pos_jitter_m_per_step": float(cmd_jitter.mean().cpu()) / max(args_cli.steps - 1, 1),
            "mean_gripper_flips": float(grip_flips.mean().cpu()),
            "mean_ee_accel_m_s2": float(ee_accel.mean().cpu()) / max(args_cli.steps - 1, 1),
            "mean_reward": float(rewards.mean().detach().cpu()),
            "mean_termination_count": float(terminations.mean().detach().cpu()),
            "elapsed_seconds": elapsed,
            "sim_steps_per_second": env.unwrapped.num_envs * args_cli.steps / max(elapsed, 1e-9),
            "policy_compute_seconds": policy_compute,
            "policy_compute_ms_per_env_step": 1000.0
            * policy_compute
            / max(env.unwrapped.num_envs * args_cli.steps, 1),
        }
    )
    return task_metrics


def make_env():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    return gym.make(args_cli.task, cfg=env_cfg), env_cfg


def main() -> None:
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args_cli.eval_methods.split(",") if m.strip()]
    valid_methods = {
        "direct_gen",
        "bc_mlp",
        "chunk_mlp",
        "act_chunk",
        "diffusion_policy",
        "phase_bc_mlp",
        "phase_chunk_mlp",
        "phase_act_chunk",
        "phase_diffusion_policy",
        "mppi_direct",
        "cem_direct",
        "mppi_fsm",
        "cem_fsm",
        "oc_fmcs",
        "oc_fmcs_learned",
    }
    unknown = sorted(set(methods) - valid_methods)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {sorted(valid_methods)}")
    env, env_cfg = make_env()
    device = torch.device(env.unwrapped.device)
    action_dim = int(env.action_space.shape[-1])
    log(f"start scenario={args_cli.scenario} task={args_cli.task} envs={args_cli.num_envs} steps={args_cli.steps}")

    demos = collect_demos(env, action_dim, device)
    if args_cli.dump_demos is not None:
        eval_demos = collect_demos(env, action_dim, device, seed=args_cli.eval_seed, episodes=1)
        args_cli.dump_demos.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train": {k: v.cpu() for k, v in demos.items()},
                "eval": {k: v.cpu() for k, v in eval_demos.items()},
                "scenario": args_cli.scenario,
                "action_dim": action_dim,
                "env_dt": float(env_cfg.sim.dt * env_cfg.decimation),
            },
            args_cli.dump_demos,
        )
        log(f"dumped demos to {args_cli.dump_demos}")
        env.close()
        return
    obs = demos["obs"]
    actions = demos["actions"]
    phases = demos["phases"]
    obs_mean, obs_std = normalize_stats(obs)
    action_mean, action_std = normalize_stats(actions)
    chunk_obs, chunks = build_chunk_dataset(obs, actions, args_cli.chunk_horizon)
    chunk_mean, chunk_std = normalize_stats(chunks)
    phase_obs = phase_conditioned_obs(obs, phases)
    phase_obs_mean, phase_obs_std = normalize_stats(phase_obs)
    phase_chunk_obs, _ = build_chunk_dataset(phase_obs, actions, args_cli.chunk_horizon)
    chunk_charts = phases.transpose(0, 1).reshape(-1).long().clamp(0, NUM_PHASES - 1)
    norm_chunks = (chunks - chunk_mean) / chunk_std
    chunk_lo, chunk_hi = norm_chunks.min(dim=0).values, norm_chunks.max(dim=0).values
    chart_lo, chart_hi = chart_boxes(norm_chunks, chunk_charts)

    train_start = time.perf_counter()
    bc_model, bc_losses = train_bc(obs, actions, obs_mean, obs_std, action_mean, action_std, "bc_mlp")
    chunk_model, chunk_losses = train_chunk(chunk_obs, chunks, obs_mean, obs_std, chunk_mean, chunk_std, "chunk_mlp")
    act_model = None
    act_losses: list[float] = []
    if "act_chunk" in methods:
        act_model, act_losses = train_act(chunk_obs, chunks, obs_mean, obs_std, chunk_mean, chunk_std, action_dim, "act_chunk")
    diffusion_model = None
    diffusion_losses: list[float] = []
    if "diffusion_policy" in methods:
        diffusion_model, diffusion_losses = train_diffusion(
            chunk_obs, chunks, obs_mean, obs_std, chunk_mean, chunk_std, "diffusion_policy"
        )
    phase_bc_model, phase_bc_losses = train_bc(
        phase_obs, actions, phase_obs_mean, phase_obs_std, action_mean, action_std, "phase_bc_mlp"
    )
    phase_chunk_model, phase_chunk_losses = train_chunk(
        phase_chunk_obs, chunks, phase_obs_mean, phase_obs_std, chunk_mean, chunk_std, "phase_chunk_mlp"
    )
    phase_act_model = None
    phase_act_losses: list[float] = []
    if "phase_act_chunk" in methods:
        phase_act_model, phase_act_losses = train_act(
            phase_chunk_obs,
            chunks,
            phase_obs_mean,
            phase_obs_std,
            chunk_mean,
            chunk_std,
            action_dim,
            "phase_act_chunk",
        )
    phase_diffusion_model = None
    phase_diffusion_losses: list[float] = []
    if "phase_diffusion_policy" in methods:
        phase_diffusion_model, phase_diffusion_losses = train_diffusion(
            phase_chunk_obs, chunks, phase_obs_mean, phase_obs_std, chunk_mean, chunk_std, "phase_diffusion_policy"
        )
    fmcs_model = None
    fmcs_losses: list[float] = []
    if "oc_fmcs_learned" in methods:
        fmcs_model, fmcs_losses = train_fmcs(
            phase_chunk_obs,
            chunks,
            chunk_charts,
            phase_obs_mean,
            phase_obs_std,
            chunk_mean,
            chunk_std,
            chart_lo,
            chart_hi,
            "oc_fmcs_learned",
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_start
    models = {"bc": bc_model, "chunk": chunk_model, "phase_bc": phase_bc_model, "phase_chunk": phase_chunk_model}
    if act_model is not None:
        models["act"] = act_model
    if diffusion_model is not None:
        models["diffusion"] = diffusion_model
    if phase_act_model is not None:
        models["phase_act"] = phase_act_model
    if phase_diffusion_model is not None:
        models["phase_diffusion"] = phase_diffusion_model
    if fmcs_model is not None:
        models["oc_fmcs_learned"] = fmcs_model
    stats = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "chunk_mean": chunk_mean,
        "chunk_std": chunk_std,
        "phase_obs_mean": phase_obs_mean,
        "phase_obs_std": phase_obs_std,
        "chunk_lo": chunk_lo,
        "chunk_hi": chunk_hi,
        "chart_lo": chart_lo,
        "chart_hi": chart_hi,
    }

    results = {}
    if args_cli.fresh_env_per_method:
        env.close()
        for method in methods:
            log(f"eval {method} in a fresh env")
            eval_env, _ = make_env()
            results[method] = evaluate(eval_env, method, models, stats)
            eval_env.close()
    else:
        for method in methods:
            log(f"eval {method}")
            results[method] = evaluate(env, method, models, stats)
        env.close()

    metrics = {
        "config": {
            "scenario": args_cli.scenario,
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "seed": args_cli.seed,
            "eval_seed": args_cli.eval_seed,
            "demo_episodes": args_cli.demo_episodes,
            "fresh_env_per_method": args_cli.fresh_env_per_method,
            "chunk_horizon": args_cli.chunk_horizon,
            "bc_train_steps": args_cli.bc_train_steps,
            "chunk_train_steps": args_cli.chunk_train_steps,
            "act_train_steps": args_cli.act_train_steps,
            "act_layers": args_cli.act_layers,
            "act_heads": args_cli.act_heads,
            "diffusion_train_steps": args_cli.diffusion_train_steps,
            "diffusion_steps": args_cli.diffusion_steps,
            "diffusion_schedule": args_cli.diffusion_schedule,
            "diffusion_pred": args_cli.diffusion_pred,
            "diffusion_clip": args_cli.diffusion_clip,
            "ema_decay": args_cli.ema_decay,
            "exec_steps": args_cli.exec_steps,
            "fm_train_steps": args_cli.fm_train_steps,
            "fm_param": args_cli.fm_param,
            "fm_clip_endpoint": args_cli.fm_clip_endpoint,
            "fm_ode_steps": args_cli.fm_ode_steps,
            "fm_chart_margin": args_cli.fm_chart_margin,
            "num_phases": NUM_PHASES,
            "batch_size": args_cli.batch_size,
            "hidden_dim": args_cli.hidden_dim,
            "lr": args_cli.lr,
            "position_threshold": args_cli.position_threshold,
            "stack_overlap_threshold": args_cli.stack_overlap_threshold,
            "stack_above_z": args_cli.stack_above_z,
            "stack_grasp_z": args_cli.stack_grasp_z,
            "stack_lift_z": args_cli.stack_lift_z,
            "stack_above_place_z": args_cli.stack_above_place_z,
            "stack_place_z": args_cli.stack_place_z,
            "stack_retreat_z": args_cli.stack_retreat_z,
            "stack_close_dwell": args_cli.stack_close_dwell,
            "stack_place_dwell": args_cli.stack_place_dwell,
            "stack_release_dwell": args_cli.stack_release_dwell,
            "stack_track_current_cubes": args_cli.stack_track_current_cubes,
            "mppi_num_samples": args_cli.mppi_num_samples,
            "mppi_noise_pos": args_cli.mppi_noise_pos,
            "mppi_noise_grip": args_cli.mppi_noise_grip,
            "mppi_temperature": args_cli.mppi_temperature,
            "cem_elite_frac": args_cli.cem_elite_frac,
            "methods": methods,
            "env_dt": float(env_cfg.sim.dt * env_cfg.decimation),
        },
        "data": {
            "demo_transitions": int(obs.numel() // obs.shape[-1]),
            "obs_dim": int(obs.shape[-1]),
            "action_dim": action_dim,
            "train_seconds": train_seconds,
            "bc_loss_tail": mean_tail(bc_losses),
            "chunk_loss_tail": mean_tail(chunk_losses),
            "act_loss_tail": mean_tail(act_losses),
            "diffusion_loss_tail": mean_tail(diffusion_losses),
            "phase_bc_loss_tail": mean_tail(phase_bc_losses),
            "phase_chunk_loss_tail": mean_tail(phase_chunk_losses),
            "phase_act_loss_tail": mean_tail(phase_act_losses),
            "phase_diffusion_loss_tail": mean_tail(phase_diffusion_losses),
            "oc_fmcs_learned_loss_tail": mean_tail(fmcs_losses),
        },
        "methods": results,
    }
    args_cli.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    log(f"wrote {args_cli.out}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
