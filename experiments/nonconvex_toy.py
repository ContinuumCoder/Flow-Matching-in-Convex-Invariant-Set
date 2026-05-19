

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


Tensor = torch.Tensor


@dataclass(frozen=True)
class Box:
    name: str
    low: tuple[float, float]
    high: tuple[float, float]

    def low_tensor(self, device: torch.device) -> Tensor:
        return torch.tensor(self.low, device=device, dtype=torch.float32)

    def high_tensor(self, device: torch.device) -> Tensor:
        return torch.tensor(self.high, device=device, dtype=torch.float32)

    def sample(self, n: int, device: torch.device, shrink: float = 0.0) -> Tensor:
        low = self.low_tensor(device) + shrink
        high = self.high_tensor(device) - shrink
        return low + (high - low) * torch.rand(n, 2, device=device)

    def contains(self, x: Tensor, tol: float = 1e-6) -> Tensor:
        low = self.low_tensor(x.device) - tol
        high = self.high_tensor(x.device) + tol
        return ((x >= low) & (x <= high)).all(dim=-1)

    def project(self, x: Tensor) -> Tensor:
        low = self.low_tensor(x.device)
        high = self.high_tensor(x.device)
        return torch.minimum(torch.maximum(x, low), high)

    def as_patch_args(self) -> tuple[float, float, float, float]:
        return (
            self.low[0],
            self.low[1],
            self.high[0] - self.low[0],
            self.high[1] - self.low[1],
        )


def build_boxes() -> tuple[Box, Box, Box, Box, Box]:
    left = Box("left", (-3.0, -1.45), (-1.0, 1.45))
    corridor = Box("corridor", (-1.2, -0.35), (1.2, 0.35))
    right = Box("right", (1.0, -1.45), (3.0, 1.45))
    left_overlap = Box("left_overlap", (-1.2, -0.30), (-1.0, 0.30))
    right_overlap = Box("right_overlap", (1.0, -0.30), (1.2, 0.30))
    return left, corridor, right, left_overlap, right_overlap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_features(t: Tensor) -> Tensor:
    freqs = torch.tensor([1.0, 2.0, 4.0, 8.0], device=t.device)
    angles = 2.0 * math.pi * t * freqs
    return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=-1)


class VectorField(nn.Module):
    def __init__(self, hidden: int = 128, zero_init: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        if zero_init:
            final = self.net[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t[:, None]
        inp = torch.cat([x, time_features(t)], dim=-1)
        return self.net(inp)


def sample_source(n: int, device: torch.device) -> Tensor:
    left, *_ = build_boxes()
    modes = torch.where(torch.rand(n, 1, device=device) < 0.5, -0.95, 0.95)
    x = torch.empty(n, 2, device=device)
    x[:, 0] = -2.55 + 0.30 * torch.randn(n, device=device)
    x[:, 1] = modes[:, 0] + 0.16 * torch.randn(n, device=device)
    return left.project(x)


def sample_target(n: int, device: torch.device) -> Tensor:
    _, _, right, *_ = build_boxes()
    modes = torch.where(torch.rand(n, 1, device=device) < 0.5, -0.95, 0.95)
    x = torch.empty(n, 2, device=device)
    x[:, 0] = 2.55 + 0.30 * torch.randn(n, device=device)
    x[:, 1] = modes[:, 0] + 0.16 * torch.randn(n, device=device)
    return right.project(x)


def make_pair_sampler(
    source_sampler: Callable[[int, torch.device], Tensor],
    target_sampler: Callable[[int, torch.device], Tensor],
) -> Callable[[int, torch.device], tuple[Tensor, Tensor]]:
    def sample(n: int, device: torch.device) -> tuple[Tensor, Tensor]:
        return source_sampler(n, device), target_sampler(n, device)

    return sample


def train_field(
    name: str,
    pair_sampler: Callable[[int, torch.device], tuple[Tensor, Tensor]],
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    hidden: int,
    zero_init: bool,
    log_every: int,
    chart: Box | None = None,
    guard_weight: float = 0.0,
    guard_margin: float = 0.0,
    guard_dt: float = 1.0 / 48.0,
) -> tuple[VectorField, list[float]]:
    model = VectorField(hidden=hidden, zero_init=zero_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    losses: list[float] = []
    for step in range(1, steps + 1):
        x0, z = pair_sampler(batch_size, device)
        t = torch.rand(batch_size, 1, device=device) * 0.995
        xt = (1.0 - t) * x0 + t * z
        ut = z - x0
        pred = model(xt, t)
        fm_loss = ((pred - ut) ** 2).mean()
        loss = fm_loss
        if chart is not None and guard_weight > 0.0:
            low = chart.low_tensor(device) + guard_margin
            high = chart.high_tensor(device) - guard_margin
            guarded_next = xt + guard_dt * pred
            lower_violation = torch.relu(low - guarded_next)
            upper_violation = torch.relu(guarded_next - high)
            guard_loss = (lower_violation.square() + upper_violation.square()).mean()
            loss = loss + guard_weight * guard_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            losses.append(float(loss.detach().cpu()))
            print(f"[{name}] step={step:05d} loss={losses[-1]:.6f}", flush=True)
    return model, losses


@torch.no_grad()
def integrate(
    model: VectorField,
    x0: Tensor,
    ode_steps: int,
    projector: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    xs = [x0]
    x = x0
    dt = 1.0 / ode_steps
    for i in range(ode_steps):
        t = torch.full((x.shape[0], 1), i / ode_steps, device=x.device)
        x = x + dt * model(x, t)
        if projector is not None:
            x = projector(x)
        xs.append(x)
    return torch.stack(xs, dim=1)


@torch.no_grad()
def integrate_oc(
    models: list[VectorField],
    x0: Tensor,
    ode_steps: int,
    chart_projectors: list[Callable[[Tensor], Tensor]] | None = None,
) -> Tensor:
    full_path = []
    x = x0
    for idx, model in enumerate(models):
        projector = None if chart_projectors is None else chart_projectors[idx]
        if projector is not None:
            x = projector(x)
        seg = integrate(model, x, ode_steps, projector=projector)
        if idx > 0:
            seg = seg[:, 1:]
        full_path.append(seg)
        x = seg[:, -1]
    return torch.cat(full_path, dim=1)


def union_contains(x: Tensor, boxes: list[Box], tol: float = 1e-6) -> Tensor:
    inside = torch.zeros(x.shape[:-1], device=x.device, dtype=torch.bool)
    flat = x.reshape(-1, 2)
    inside_flat = torch.zeros(flat.shape[0], device=x.device, dtype=torch.bool)
    for box in boxes:
        inside_flat |= box.contains(flat, tol=tol)
    return inside_flat.reshape(x.shape[:-1])


def project_to_union(x: Tensor, boxes: list[Box]) -> Tensor:
    candidates = torch.stack([box.project(x) for box in boxes], dim=0)
    d2 = ((candidates - x[None, :, :]) ** 2).sum(dim=-1)
    best = d2.argmin(dim=0)
    return candidates[best, torch.arange(x.shape[0], device=x.device)]


def compute_metrics(path: Tensor, boxes: list[Box]) -> dict[str, float]:
    inside = union_contains(path, boxes)
    final = path[:, -1]
    _, _, right, *_ = build_boxes()
    hit = right.contains(final) & (final[:, 0] > 1.65) & (final[:, 1].abs() > 0.45)
    diffs = path[:, 1:] - path[:, :-1]
    length = diffs.norm(dim=-1).sum(dim=1)
    if path.shape[1] > 2:
        accel = path[:, 2:] - 2.0 * path[:, 1:-1] + path[:, :-2]
        smooth = (accel.norm(dim=-1) ** 2).mean(dim=1)
    else:
        smooth = torch.zeros(path.shape[0], device=path.device)
    return {
        "path_safe_rate": float(inside.all(dim=1).float().mean().cpu()),
        "terminal_feasible_rate": float(inside[:, -1].float().mean().cpu()),
        "target_hit_rate": float(hit.float().mean().cpu()),
        "mean_outside_fraction": float((~inside).float().mean().cpu()),
        "mean_path_length": float(length.mean().cpu()),
        "mean_smoothness": float(smooth.mean().cpu()),
    }


def plot_paths(paths: dict[str, Tensor], boxes: list[Box], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(paths), figsize=(5.2 * len(paths), 4.4), sharex=True, sharey=True)
    if len(paths) == 1:
        axes = [axes]
    for ax, (name, path) in zip(axes, paths.items()):
        for box in boxes:
            x, y, w, h = box.as_patch_args()
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor="#e9f5ef",
                    edgecolor="#2f7d58",
                    alpha=0.75,
                    linewidth=1.0,
                )
            )
        p = path.detach().cpu().numpy()
        for i in range(min(96, p.shape[0])):
            ax.plot(p[i, :, 0], p[i, :, 1], color="#3655a3", alpha=0.26, linewidth=1.0)
        ax.scatter(p[:96, 0, 0], p[:96, 0, 1], s=7, color="#2f7d58", alpha=0.65, label="start")
        ax.scatter(p[:96, -1, 0], p[:96, -1, 1], s=7, color="#b23b3b", alpha=0.65, label="end")
        ax.set_title(name)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.35, alpha=0.35)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_losses(losses: dict[str, list[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, vals in losses.items():
        ax.plot(vals, label=name)
    ax.set_xlabel("logged update")
    ax.set_ylabel("flow matching loss")
    ax.set_yscale("log")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--ode-steps", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("results/nonconvex_toy"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-zero-init", action="store_true")
    parser.add_argument("--guard-weight", type=float, default=0.0)
    parser.add_argument("--guard-margin", type=float, default=0.02)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}", flush=True)

    left, corridor, right, left_overlap, right_overlap = build_boxes()
    union_boxes = [left, corridor, right]

    def sample_left_overlap(n: int, device_: torch.device) -> Tensor:
        return left_overlap.sample(n, device_, shrink=0.02)

    def sample_right_overlap(n: int, device_: torch.device) -> Tensor:
        return right_overlap.sample(n, device_, shrink=0.02)

    zero_init = not args.no_zero_init
    direct, direct_losses = train_field(
        "direct_cfm",
        make_pair_sampler(sample_source, sample_target),
        device,
        args.steps,
        args.batch_size,
        args.lr,
        args.hidden,
        zero_init,
        args.log_every,
    )
    stage0, loss0 = train_field(
        "oc_stage_left_to_overlap",
        make_pair_sampler(sample_source, sample_left_overlap),
        device,
        args.steps,
        args.batch_size,
        args.lr,
        args.hidden,
        zero_init,
        args.log_every,
        chart=left,
        guard_weight=args.guard_weight,
        guard_margin=args.guard_margin,
        guard_dt=1.0 / args.ode_steps,
    )
    stage1, loss1 = train_field(
        "oc_stage_corridor",
        make_pair_sampler(sample_left_overlap, sample_right_overlap),
        device,
        args.steps,
        args.batch_size,
        args.lr,
        args.hidden,
        zero_init,
        args.log_every,
        chart=corridor,
        guard_weight=args.guard_weight,
        guard_margin=args.guard_margin,
        guard_dt=1.0 / args.ode_steps,
    )
    stage2, loss2 = train_field(
        "oc_stage_overlap_to_right",
        make_pair_sampler(sample_right_overlap, sample_target),
        device,
        args.steps,
        args.batch_size,
        args.lr,
        args.hidden,
        zero_init,
        args.log_every,
        chart=right,
        guard_weight=args.guard_weight,
        guard_margin=args.guard_margin,
        guard_dt=1.0 / args.ode_steps,
    )

    x_eval = sample_source(args.eval_samples, device)
    paths = {
        "direct_cfm": integrate(direct, x_eval, args.ode_steps),
        "direct_projected": integrate(
            direct,
            x_eval,
            args.ode_steps,
            projector=lambda x: project_to_union(x, union_boxes),
        ),
        "oc_fmcs": integrate_oc([stage0, stage1, stage2], x_eval, args.ode_steps),
        "oc_fmcs_chart_projected": integrate_oc(
            [stage0, stage1, stage2],
            x_eval,
            args.ode_steps,
            chart_projectors=[left.project, corridor.project, right.project],
        ),
    }

    metrics = {
        name: compute_metrics(path, union_boxes)
        for name, path in paths.items()
    }
    metrics["config"] = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_samples": args.eval_samples,
        "ode_steps": args.ode_steps,
        "hidden": args.hidden,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "zero_init": zero_init,
        "guard_weight": args.guard_weight,
        "guard_margin": args.guard_margin,
    }

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_paths(paths, union_boxes, args.out_dir / "paths.png")
    plot_losses(
        {
            "direct": direct_losses,
            "stage0": loss0,
            "stage1": loss1,
            "stage2": loss2,
        },
        args.out_dir / "losses.png",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
