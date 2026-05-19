

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

    def sample(self, n: int, device: torch.device) -> Tensor:
        low = self.low_tensor(device)
        high = self.high_tensor(device)
        return low + (high - low) * torch.rand(n, 2, device=device)

    def contains(self, q: Tensor, tol: float = 1e-6) -> Tensor:
        low = self.low_tensor(q.device) - tol
        high = self.high_tensor(q.device) + tol
        return ((q >= low) & (q <= high)).all(dim=-1)

    def project(self, q: Tensor) -> Tensor:
        low = self.low_tensor(q.device)
        high = self.high_tensor(q.device)
        return torch.minimum(torch.maximum(q, low), high)

    def patch_args(self) -> tuple[float, float, float, float]:
        return (
            self.low[0],
            self.low[1],
            self.high[0] - self.low[0],
            self.high[1] - self.low[1],
        )


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    chart: Box
    source: Box
    target: Box


@dataclass(frozen=True)
class ArmTask:
    link_lengths: tuple[float, float]
    obstacle_center: tuple[float, float]
    obstacle_radius: float
    link_radius: float
    joint_low: tuple[float, float]
    joint_high: tuple[float, float]
    source_core: Box
    target_core: Box
    segments: list[SegmentSpec]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_features(t: Tensor) -> Tensor:
    freqs = torch.tensor([1.0, 2.0, 4.0, 8.0], device=t.device)
    angles = 2.0 * math.pi * t * freqs
    return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=-1)


class DirectField(nn.Module):
    def __init__(self, hidden: int = 160, zero_init: bool = True) -> None:
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

    def forward(self, q: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([q, time_features(t)], dim=-1))


class SharedChartField(nn.Module):
    def __init__(
        self,
        num_segments: int,
        hidden: int = 160,
        embed_dim: int = 16,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_segments, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(11 + embed_dim, hidden),
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

    def forward(self, q: Tensor, t: Tensor, segment_id: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t[:, None]
        emb = self.embedding(segment_id.long())
        return self.net(torch.cat([q, time_features(t), emb], dim=-1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_box(name: str, center: tuple[float, float], half_width: tuple[float, float]) -> Box:
    return Box(
        name=name,
        low=(center[0] - half_width[0], center[1] - half_width[1]),
        high=(center[0] + half_width[0], center[1] + half_width[1]),
    )


def build_arm_task(core_half_width: float = 0.025, chart_margin: float = 0.03) -> ArmTask:
    centers = [
        (-0.41, -1.48),
        (-0.41, -0.90),
        (-0.41, -0.30),
        (-0.41, 0.30),
        (-0.41, 0.90),
        (-0.57, 1.41),
        (-0.90, 2.05),
        (-1.40, 2.05),
        (-1.40, 2.70),
        (-1.40, 3.00),
        (-0.75, 3.00),
    ]
    cores = [
        make_box(f"core_{idx}", center, (core_half_width, core_half_width))
        for idx, center in enumerate(centers)
    ]
    segments: list[SegmentSpec] = []
    for idx, (start, end) in enumerate(zip(centers[:-1], centers[1:])):
        low = (min(start[0], end[0]) - chart_margin, min(start[1], end[1]) - chart_margin)
        high = (max(start[0], end[0]) + chart_margin, max(start[1], end[1]) + chart_margin)
        segments.append(
            SegmentSpec(
                name=f"chart_{idx:02d}",
                chart=Box(f"chart_{idx:02d}", low, high),
                source=cores[idx],
                target=cores[idx + 1],
            )
        )
    return ArmTask(
        link_lengths=(1.0, 0.8),
        obstacle_center=(0.55, 0.15),
        obstacle_radius=0.22,
        link_radius=0.025,
        joint_low=(-math.pi, -math.pi),
        joint_high=(math.pi, math.pi),
        source_core=cores[0],
        target_core=cores[-1],
        segments=segments,
    )


def forward_kinematics(q: Tensor, link_lengths: tuple[float, float]) -> tuple[Tensor, Tensor, Tensor]:
    l1, l2 = link_lengths
    p0 = torch.zeros(*q.shape[:-1], 2, device=q.device, dtype=q.dtype)
    p1 = torch.stack([l1 * torch.cos(q[..., 0]), l1 * torch.sin(q[..., 0])], dim=-1)
    p2 = p1 + torch.stack(
        [
            l2 * torch.cos(q[..., 0] + q[..., 1]),
            l2 * torch.sin(q[..., 0] + q[..., 1]),
        ],
        dim=-1,
    )
    return p0, p1, p2


def point_segment_distance(point: Tensor, a: Tensor, b: Tensor) -> Tensor:
    v = b - a
    denom = (v.square().sum(dim=-1)).clamp_min(1e-9)
    tau = ((point - a) * v).sum(dim=-1) / denom
    tau = tau.clamp(0.0, 1.0)
    closest = a + tau[..., None] * v
    return (point - closest).norm(dim=-1)


def arm_clearance(q: Tensor, task: ArmTask) -> Tensor:
    p0, p1, p2 = forward_kinematics(q, task.link_lengths)
    center = torch.tensor(task.obstacle_center, device=q.device, dtype=q.dtype)
    inflated = task.obstacle_radius + task.link_radius
    d1 = point_segment_distance(center, p0, p1) - inflated
    d2 = point_segment_distance(center, p1, p2) - inflated
    return torch.minimum(d1, d2)


def validate_charts(task: ArmTask, device: torch.device, grid: int = 17) -> dict[str, float]:
    mins: dict[str, float] = {}
    for segment in task.segments:
        q1 = torch.linspace(segment.chart.low[0], segment.chart.high[0], grid, device=device)
        q2 = torch.linspace(segment.chart.low[1], segment.chart.high[1], grid, device=device)
        mesh = torch.stack(torch.meshgrid(q1, q2, indexing="ij"), dim=-1).reshape(-1, 2)
        mins[segment.name] = float(arm_clearance(mesh, task).min().cpu())
    return mins


def union_contains(q: Tensor, boxes: list[Box], tol: float = 1e-6) -> Tensor:
    flat = q.reshape(-1, 2)
    inside = torch.zeros(flat.shape[0], device=q.device, dtype=torch.bool)
    for box in boxes:
        inside |= box.contains(flat, tol=tol)
    return inside.reshape(q.shape[:-1])


def project_to_union(q: Tensor, boxes: list[Box]) -> Tensor:
    candidates = torch.stack([box.project(q) for box in boxes], dim=0)
    d2 = ((candidates - q[None, :, :]) ** 2).sum(dim=-1)
    best = d2.argmin(dim=0)
    return candidates[best, torch.arange(q.shape[0], device=q.device)]


def train_direct(
    task: ArmTask,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    hidden: int,
    log_every: int,
) -> tuple[DirectField, list[float], float]:
    model = DirectField(hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    losses: list[float] = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        q0 = task.source_core.sample(batch_size, device)
        z = task.target_core.sample(batch_size, device)
        t = torch.rand(batch_size, 1, device=device) * 0.995
        qt = (1.0 - t) * q0 + t * z
        ut = z - q0
        loss = ((model(qt, t) - ut) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            losses.append(float(loss.detach().cpu()))
            print(f"[direct] step={step:05d} loss={losses[-1]:.6f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return model, losses, time.perf_counter() - start


def train_shared(
    segments: list[SegmentSpec],
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    hidden: int,
    embed_dim: int,
    log_every: int,
) -> tuple[SharedChartField, list[float], float]:
    model = SharedChartField(len(segments), hidden=hidden, embed_dim=embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    losses: list[float] = []
    per_segment = max(1, batch_size // len(segments))
    start = time.perf_counter()
    for step in range(1, steps + 1):
        qs: list[Tensor] = []
        zs: list[Tensor] = []
        ids: list[Tensor] = []
        for seg_id, segment in enumerate(segments):
            qs.append(segment.source.sample(per_segment, device))
            zs.append(segment.target.sample(per_segment, device))
            ids.append(torch.full((per_segment,), seg_id, device=device, dtype=torch.long))
        q0 = torch.cat(qs, dim=0)
        z = torch.cat(zs, dim=0)
        segment_id = torch.cat(ids, dim=0)
        t = torch.rand(q0.shape[0], 1, device=device) * 0.995
        qt = (1.0 - t) * q0 + t * z
        ut = z - q0
        loss = ((model(qt, t, segment_id) - ut) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            losses.append(float(loss.detach().cpu()))
            print(f"[shared_oc] step={step:05d} loss={losses[-1]:.6f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return model, losses, time.perf_counter() - start


@torch.no_grad()
def integrate_direct(
    model: DirectField,
    q0: Tensor,
    ode_steps: int,
    projector: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    qs = [q0]
    q = q0
    dt = 1.0 / ode_steps
    for idx in range(ode_steps):
        t = torch.full((q.shape[0], 1), idx / ode_steps, device=q.device)
        q = q + dt * model(q, t)
        if projector is not None:
            q = projector(q)
        qs.append(q)
    return torch.stack(qs, dim=1)


@torch.no_grad()
def integrate_shared(
    model: SharedChartField,
    q0: Tensor,
    ode_steps: int,
    segments: list[SegmentSpec],
    chart_project: bool,
) -> Tensor:
    full_path = []
    q = q0
    for seg_id, segment in enumerate(segments):
        if chart_project:
            q = segment.chart.project(q)
        segment_ids = torch.full((q.shape[0],), seg_id, device=q.device, dtype=torch.long)
        qs = [q]
        dt = 1.0 / ode_steps
        for idx in range(ode_steps):
            t = torch.full((q.shape[0], 1), idx / ode_steps, device=q.device)
            q = q + dt * model(q, t, segment_ids)
            if chart_project:
                q = segment.chart.project(q)
            qs.append(q)
        seg_path = torch.stack(qs, dim=1)
        if seg_id > 0:
            seg_path = seg_path[:, 1:]
        full_path.append(seg_path)
    return torch.cat(full_path, dim=1)


def compute_metrics(path: Tensor, task: ArmTask, chart_boxes: list[Box]) -> dict[str, float]:
    corridor_inside = union_contains(path, chart_boxes)
    clearance = arm_clearance(path, task)
    final = path[:, -1]
    target_hit = task.target_core.contains(final, tol=0.04)
    joint_low = torch.tensor(task.joint_low, device=path.device)
    joint_high = torch.tensor(task.joint_high, device=path.device)
    joint_valid = ((path >= joint_low) & (path <= joint_high)).all(dim=-1)
    _, _, ee = forward_kinematics(final, task.link_lengths)
    target_center = (
        task.target_core.low_tensor(path.device) + task.target_core.high_tensor(path.device)
    ) / 2.0
    _, _, target_ee = forward_kinematics(target_center[None, :], task.link_lengths)
    ee_error = (ee - target_ee).norm(dim=-1)
    diffs = path[:, 1:] - path[:, :-1]
    length = diffs.norm(dim=-1).sum(dim=1)
    accel = path[:, 2:] - 2.0 * path[:, 1:-1] + path[:, :-2]
    smoothness = (accel.norm(dim=-1) ** 2).mean(dim=1)
    min_clearance = clearance.min(dim=1).values
    return {
        "corridor_path_safe_rate": float(corridor_inside.all(dim=1).float().mean().cpu()),
        "collision_free_path_rate": float((clearance > 0.0).all(dim=1).float().mean().cpu()),
        "terminal_collision_free_rate": float((clearance[:, -1] > 0.0).float().mean().cpu()),
        "target_hit_rate": float(target_hit.float().mean().cpu()),
        "joint_limit_path_valid_rate": float(joint_valid.all(dim=1).float().mean().cpu()),
        "mean_outside_corridor_fraction": float((~corridor_inside).float().mean().cpu()),
        "mean_collision_fraction": float((clearance <= 0.0).float().mean().cpu()),
        "mean_min_clearance": float(min_clearance.mean().cpu()),
        "p05_min_clearance": float(torch.quantile(min_clearance, 0.05).cpu()),
        "mean_end_effector_error": float(ee_error.mean().cpu()),
        "mean_joint_path_length": float(length.mean().cpu()),
        "mean_joint_smoothness": float(smoothness.mean().cpu()),
    }


def plot_joint_paths(
    paths: dict[str, Tensor],
    task: ArmTask,
    chart_boxes: list[Box],
    out_path: Path,
    grid: int = 170,
) -> None:
    fig, axes = plt.subplots(1, len(paths), figsize=(5.0 * len(paths), 4.2), sharex=True, sharey=True)
    if len(paths) == 1:
        axes = [axes]
    q1 = torch.linspace(-1.65, -0.25, grid)
    q2 = torch.linspace(-1.7, 3.15, grid)
    mesh = torch.stack(torch.meshgrid(q1, q2, indexing="ij"), dim=-1).reshape(-1, 2)
    clearance = arm_clearance(mesh, task).reshape(grid, grid).cpu().numpy()
    source_center = (
        task.source_core.low_tensor(torch.device("cpu"))
        + task.source_core.high_tensor(torch.device("cpu"))
    ) / 2.0
    target_center = (
        task.target_core.low_tensor(torch.device("cpu"))
        + task.target_core.high_tensor(torch.device("cpu"))
    ) / 2.0
    for ax, (name, path) in zip(axes, paths.items()):
        ax.contourf(
            q1.numpy(),
            q2.numpy(),
            (clearance <= 0.0).T,
            levels=[0.5, 1.5],
            colors=["#e3b2ae"],
            alpha=0.55,
        )
        ax.contour(
            q1.numpy(),
            q2.numpy(),
            clearance.T,
            levels=[0.0],
            colors=["#9b2f2f"],
            linewidths=0.8,
        )
        for box in chart_boxes:
            x, y, w, h = box.patch_args()
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor="#e9f5ef",
                    edgecolor="#2f7d58",
                    alpha=0.52,
                    linewidth=0.8,
                )
            )
        for core, color, label in [
            (task.source_core, "#2f7d58", "start set"),
            (task.target_core, "#b23b3b", "goal set"),
        ]:
            x, y, w, h = core.patch_args()
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.18,
                    linewidth=1.2,
                    label=label,
                )
            )
        arr = path.detach().cpu()
        safe_count = 0
        for idx in range(min(96, arr.shape[0])):
            is_safe = bool((arm_clearance(arr[idx], task) > 0.0).all())
            safe_count += int(is_safe)
            color = "#3655a3" if is_safe else "#b23b3b"
            ax.plot(arr[idx, :, 0], arr[idx, :, 1], color=color, alpha=0.22, linewidth=1.0)
        ax.scatter(arr[:96, 0, 0], arr[:96, 0, 1], s=7, color="#2f7d58", alpha=0.7)
        ax.scatter(arr[:96, -1, 0], arr[:96, -1, 1], s=7, color="#b23b3b", alpha=0.7)
        ax.scatter(
            [float(source_center[0])],
            [float(source_center[1])],
            marker="o",
            s=42,
            color="#2f7d58",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.scatter(
            [float(target_center[0])],
            [float(target_center[1])],
            marker="*",
            s=105,
            color="#b23b3b",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.text(float(source_center[0]) - 0.18, float(source_center[1]) - 0.14, "start", fontsize=7)
        ax.text(float(target_center[0]) - 0.13, float(target_center[1]) - 0.17, "goal", fontsize=7)
        ax.text(-0.98, 2.48, "collision set", color="#8c2d2d", fontsize=7)
        ax.text(-1.43, 2.86, "OC chart corridor", color="#2f7d58", fontsize=7)
        ax.set_title(f"{name}\nsafe paths: {safe_count}/{min(96, arr.shape[0])}", fontsize=11)
        ax.set_xlabel("q1")
        ax.grid(True, linewidth=0.35, alpha=0.35)
    axes[0].set_ylabel("q2")
    handles = [
        plt.Line2D([0], [0], color="#3655a3", linewidth=1.5, label="collision-free path"),
        plt.Line2D([0], [0], color="#b23b3b", linewidth=1.5, label="colliding path"),
        plt.Line2D([0], [0], color="#2f7d58", linewidth=1.5, label="convex chart"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#b23b3b", markersize=10, label="goal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, framealpha=0.92)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_workspace_paths(paths: dict[str, Tensor], task: ArmTask, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(paths), figsize=(4.5 * len(paths), 4.4), sharex=True, sharey=True)
    if len(paths) == 1:
        axes = [axes]
    source_center = (
        task.source_core.low_tensor(torch.device("cpu"))
        + task.source_core.high_tensor(torch.device("cpu"))
    ) / 2.0
    target_center = (
        task.target_core.low_tensor(torch.device("cpu"))
        + task.target_core.high_tensor(torch.device("cpu"))
    ) / 2.0
    _, _, source_ee = forward_kinematics(source_center[None, :], task.link_lengths)
    _, _, target_ee = forward_kinematics(target_center[None, :], task.link_lengths)
    target_ee_xy = target_ee[0].detach().cpu()
    for ax, (name, path) in zip(axes, paths.items()):
        ax.add_patch(plt.Circle(task.obstacle_center, task.obstacle_radius, color="#b23b3b", alpha=0.22))
        ax.add_patch(
            plt.Circle(
                task.obstacle_center,
                task.obstacle_radius + task.link_radius,
                fill=False,
                edgecolor="#9b2f2f",
                linestyle="--",
                linewidth=0.9,
                alpha=0.9,
            )
        )
        ax.scatter(
            [float(target_ee_xy[0])],
            [float(target_ee_xy[1])],
            marker="*",
            s=120,
            color="#b23b3b",
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )
        ax.add_patch(
            plt.Circle(
                (float(target_ee_xy[0]), float(target_ee_xy[1])),
                0.09,
                fill=False,
                edgecolor="#b23b3b",
                linewidth=1.0,
                alpha=0.85,
            )
        )
        arr = path.detach().cpu()
        sample_ids = torch.linspace(0, arr.shape[1] - 1, 9).long()
        safe_count = 0
        for traj_idx in range(min(12, arr.shape[0])):
            p0, p1, p2 = forward_kinematics(arr[traj_idx, sample_ids], task.link_lengths)
            safe = bool((arm_clearance(arr[traj_idx], task) > 0.0).all())
            safe_count += int(safe)
            color = "#3655a3" if safe else "#b23b3b"
            for step_idx in range(sample_ids.shape[0]):
                xs = [p0[step_idx, 0], p1[step_idx, 0], p2[step_idx, 0]]
                ys = [p0[step_idx, 1], p1[step_idx, 1], p2[step_idx, 1]]
                alpha = 0.17 if safe else 0.22
                ax.plot(xs, ys, color=color, alpha=alpha, linewidth=1.05)
        p0, p1, p2 = forward_kinematics(arr[0, [0, -1]], task.link_lengths)
        ax.plot([p0[0, 0], p1[0, 0], p2[0, 0]], [p0[0, 1], p1[0, 1], p2[0, 1]], color="#2f7d58", linewidth=2.0)
        ax.plot([p0[1, 0], p1[1, 0], p2[1, 0]], [p0[1, 1], p1[1, 1], p2[1, 1]], color="#b23b3b", linewidth=2.0)
        ee_trace = forward_kinematics(arr[:48].reshape(-1, 2), task.link_lengths)[2].reshape(
            min(48, arr.shape[0]), arr.shape[1], 2
        )
        for idx in range(ee_trace.shape[0]):
            safe = bool((arm_clearance(arr[idx], task) > 0.0).all())
            color = "#3655a3" if safe else "#b23b3b"
            ax.plot(ee_trace[idx, :, 0], ee_trace[idx, :, 1], color=color, alpha=0.12, linewidth=0.8)
        ax.scatter([0.0], [0.0], s=18, color="#333333")
        ax.text(task.obstacle_center[0] - 0.18, task.obstacle_center[1] + 0.34, "obstacle", color="#8c2d2d", fontsize=8)
        ax.text(float(target_ee_xy[0]) - 0.15, float(target_ee_xy[1]) + 0.12, "EE goal", color="#8c2d2d", fontsize=8)
        ax.text(float(source_ee[0, 0]) - 0.18, float(source_ee[0, 1]) - 0.15, "start", color="#2f7d58", fontsize=8)
        ax.set_title(f"{name}\nsafe rollouts: {safe_count}/{min(12, arr.shape[0])}", fontsize=11)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.35, 1.55)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.35, alpha=0.35)
    handles = [
        plt.Line2D([0], [0], color="#3655a3", linewidth=1.5, label="collision-free rollout"),
        plt.Line2D([0], [0], color="#b23b3b", linewidth=1.5, label="colliding rollout / goal posture"),
        plt.Line2D([0], [0], color="#2f7d58", linewidth=1.5, label="start posture"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#b23b3b", markersize=10, label="end-effector goal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, framealpha=0.92)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_losses(losses: dict[str, list[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
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
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=1600)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--direct-ode-steps", type=int, default=96)
    parser.add_argument("--segment-ode-steps", type=int, default=28)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--log-every", type=int, default=350)
    parser.add_argument("--out-dir", type=Path, default=Path("results/planar_arm_ocfmcs"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"device={device}", flush=True)

    task = build_arm_task()
    chart_boxes = [segment.chart for segment in task.segments]
    chart_clearances = validate_charts(task, device)
    print("min chart clearance:", chart_clearances, flush=True)

    direct, direct_losses, direct_train_seconds = train_direct(
        task,
        device,
        args.steps,
        args.batch_size,
        args.lr,
        args.hidden,
        args.log_every,
    )
    shared, shared_losses, shared_train_seconds = train_shared(
        task.segments,
        device,
        args.steps,
        args.batch_size,
        args.lr,
        args.hidden,
        args.embed_dim,
        args.log_every,
    )

    q_eval = task.source_core.sample(args.eval_samples, device)
    eval_times: dict[str, float] = {}
    paths: dict[str, Tensor] = {}
    runners = {
        "direct_cfm": lambda: integrate_direct(direct, q_eval, args.direct_ode_steps),
        "direct_projected": lambda: integrate_direct(
            direct,
            q_eval,
            args.direct_ode_steps,
            projector=lambda q: project_to_union(q, chart_boxes),
        ),
        "shared_ocfmcs": lambda: integrate_shared(
            shared,
            q_eval,
            args.segment_ode_steps,
            task.segments,
            chart_project=False,
        ),
        "shared_ocfmcs_chart_projected": lambda: integrate_shared(
            shared,
            q_eval,
            args.segment_ode_steps,
            task.segments,
            chart_project=True,
        ),
    }
    for name, runner in runners.items():
        start = time.perf_counter()
        paths[name] = runner()
        if device.type == "cuda":
            torch.cuda.synchronize()
        eval_times[name] = time.perf_counter() - start

    metrics = {name: compute_metrics(path, task, chart_boxes) for name, path in paths.items()}
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
    )
    metrics["efficiency"] = {
        "direct_train_seconds": direct_train_seconds,
        "shared_train_seconds": shared_train_seconds,
        "direct_params": count_params(direct),
        "shared_params": count_params(shared),
        "peak_cuda_memory_mb": peak_memory_mb,
        "eval_seconds": eval_times,
        "eval_samples_per_second": {
            name: args.eval_samples / max(seconds, 1e-9) for name, seconds in eval_times.items()
        },
    }
    metrics["chart_validation"] = chart_clearances
    metrics["config"] = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_samples": args.eval_samples,
        "direct_ode_steps": args.direct_ode_steps,
        "segment_ode_steps": args.segment_ode_steps,
        "num_segments": len(task.segments),
        "hidden": args.hidden,
        "embed_dim": args.embed_dim,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "link_lengths": task.link_lengths,
        "obstacle_center": task.obstacle_center,
        "obstacle_radius": task.obstacle_radius,
    }

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_joint_paths(paths, task, chart_boxes, args.out_dir / "joint_paths.png")
    plot_workspace_paths(paths, task, args.out_dir / "workspace_paths.png")
    plot_losses({"direct": direct_losses, "shared_oc": shared_losses}, args.out_dir / "losses.png")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
