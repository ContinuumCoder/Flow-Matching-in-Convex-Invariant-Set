

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
    source: Callable[[int, torch.device], Tensor]
    target: Callable[[int, torch.device], Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_maze() -> dict[str, Box]:
    return {
        "left_room": Box("left_room", (-4.0, -1.8), (-2.4, 1.8)),
        "right_room": Box("right_room", (2.4, -1.8), (4.0, 1.8)),
        "upper_passage": Box("upper_passage", (-2.65, 0.82), (2.65, 1.8)),
        "lower_passage": Box("lower_passage", (-2.65, -1.8), (2.65, -0.82)),
        "left_upper_overlap": Box("left_upper_overlap", (-2.65, 0.95), (-2.4, 1.55)),
        "right_upper_overlap": Box("right_upper_overlap", (2.4, 0.95), (2.65, 1.55)),
        "left_lower_overlap": Box("left_lower_overlap", (-2.65, -1.55), (-2.4, -0.95)),
        "right_lower_overlap": Box("right_lower_overlap", (2.4, -1.55), (2.65, -0.95)),
        "source_core": Box("source_core", (-3.75, -0.35), (-3.25, 0.35)),
        "target_core": Box("target_core", (3.25, -0.35), (3.75, 0.35)),
    }


def sample_from(box: Box, shrink: float = 0.0) -> Callable[[int, torch.device], Tensor]:
    def sampler(n: int, device: torch.device) -> Tensor:
        return box.sample(n, device, shrink=shrink)

    return sampler


def make_segments(boxes: dict[str, Box]) -> list[SegmentSpec]:
    return [
        SegmentSpec(
            "left_to_upper_overlap",
            boxes["left_room"],
            sample_from(boxes["source_core"]),
            sample_from(boxes["left_upper_overlap"], shrink=0.02),
        ),
        SegmentSpec(
            "upper_passage",
            boxes["upper_passage"],
            sample_from(boxes["left_upper_overlap"], shrink=0.02),
            sample_from(boxes["right_upper_overlap"], shrink=0.02),
        ),
        SegmentSpec(
            "upper_overlap_to_right",
            boxes["right_room"],
            sample_from(boxes["right_upper_overlap"], shrink=0.02),
            sample_from(boxes["target_core"]),
        ),
        SegmentSpec(
            "left_to_lower_overlap",
            boxes["left_room"],
            sample_from(boxes["source_core"]),
            sample_from(boxes["left_lower_overlap"], shrink=0.02),
        ),
        SegmentSpec(
            "lower_passage",
            boxes["lower_passage"],
            sample_from(boxes["left_lower_overlap"], shrink=0.02),
            sample_from(boxes["right_lower_overlap"], shrink=0.02),
        ),
        SegmentSpec(
            "lower_overlap_to_right",
            boxes["right_room"],
            sample_from(boxes["right_lower_overlap"], shrink=0.02),
            sample_from(boxes["target_core"]),
        ),
    ]


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

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x, time_features(t)], dim=-1))


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

    def forward(self, x: Tensor, t: Tensor, segment_id: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t[:, None]
        emb = self.embedding(segment_id.long())
        return self.net(torch.cat([x, time_features(t), emb], dim=-1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sample_source(n: int, device: torch.device, boxes: dict[str, Box]) -> Tensor:
    return boxes["source_core"].sample(n, device)


def sample_target(n: int, device: torch.device, boxes: dict[str, Box]) -> Tensor:
    return boxes["target_core"].sample(n, device)


def train_direct(
    boxes: dict[str, Box],
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
        x0 = sample_source(batch_size, device, boxes)
        z = sample_target(batch_size, device, boxes)
        t = torch.rand(batch_size, 1, device=device) * 0.995
        xt = (1.0 - t) * x0 + t * z
        ut = z - x0
        loss = ((model(xt, t) - ut) ** 2).mean()
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
    model = SharedChartField(
        num_segments=len(segments),
        hidden=hidden,
        embed_dim=embed_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    losses: list[float] = []
    start = time.perf_counter()
    per_segment = max(1, batch_size // len(segments))
    for step in range(1, steps + 1):
        xs: list[Tensor] = []
        zs: list[Tensor] = []
        ids: list[Tensor] = []
        for seg_id, segment in enumerate(segments):
            xs.append(segment.source(per_segment, device))
            zs.append(segment.target(per_segment, device))
            ids.append(torch.full((per_segment,), seg_id, device=device, dtype=torch.long))
        x0 = torch.cat(xs, dim=0)
        z = torch.cat(zs, dim=0)
        segment_id = torch.cat(ids, dim=0)
        t = torch.rand(x0.shape[0], 1, device=device) * 0.995
        xt = (1.0 - t) * x0 + t * z
        ut = z - x0
        loss = ((model(xt, t, segment_id) - ut) ** 2).mean()
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
def integrate_shared_route(
    model: SharedChartField,
    x0: Tensor,
    route_ids: Tensor,
    ode_steps: int,
    segments: list[SegmentSpec],
    chart_project: bool,
) -> Tensor:
    route_top = [0, 1, 2]
    route_lower = [3, 4, 5]
    full_path = []
    x = x0
    for stage in range(3):
        segment_id = torch.where(
            route_ids == 0,
            torch.full_like(route_ids, route_top[stage]),
            torch.full_like(route_ids, route_lower[stage]),
        )
        if chart_project:
            x_next = x.clone()
            for sid in sorted(segment_id.unique().tolist()):
                mask = segment_id == sid
                x_next[mask] = segments[sid].chart.project(x_next[mask])
            x = x_next
        xs = [x]
        dt = 1.0 / ode_steps
        for i in range(ode_steps):
            t = torch.full((x.shape[0], 1), i / ode_steps, device=x.device)
            x = x + dt * model(x, t, segment_id)
            if chart_project:
                x_next = x.clone()
                for sid in sorted(segment_id.unique().tolist()):
                    mask = segment_id == sid
                    x_next[mask] = segments[sid].chart.project(x_next[mask])
                x = x_next
            xs.append(x)
        seg_path = torch.stack(xs, dim=1)
        if stage > 0:
            seg_path = seg_path[:, 1:]
        full_path.append(seg_path)
    return torch.cat(full_path, dim=1)


def union_contains(x: Tensor, boxes: list[Box], tol: float = 1e-6) -> Tensor:
    flat = x.reshape(-1, 2)
    inside = torch.zeros(flat.shape[0], device=x.device, dtype=torch.bool)
    for box in boxes:
        inside |= box.contains(flat, tol=tol)
    return inside.reshape(x.shape[:-1])


def project_to_union(x: Tensor, boxes: list[Box]) -> Tensor:
    candidates = torch.stack([box.project(x) for box in boxes], dim=0)
    d2 = ((candidates - x[None, :, :]) ** 2).sum(dim=-1)
    best = d2.argmin(dim=0)
    return candidates[best, torch.arange(x.shape[0], device=x.device)]


def compute_metrics(
    path: Tensor,
    route_ids: Tensor,
    union_boxes: list[Box],
    target_core: Box,
) -> dict[str, float]:
    inside = union_contains(path, union_boxes)
    final = path[:, -1]
    hit = target_core.contains(final, tol=0.08)
    diffs = path[:, 1:] - path[:, :-1]
    length = diffs.norm(dim=-1).sum(dim=1)
    accel = path[:, 2:] - 2.0 * path[:, 1:-1] + path[:, :-2]
    smooth = (accel.norm(dim=-1) ** 2).mean(dim=1)
    uses_upper = path[:, :, 1].max(dim=1).values > 0.75
    uses_lower = path[:, :, 1].min(dim=1).values < -0.75
    expected_upper = route_ids == 0
    route_adherence = torch.where(expected_upper, uses_upper, uses_lower)
    ambiguous = uses_upper & uses_lower
    middle_cross = ((path[:, :, 0].abs() < 2.3) & (path[:, :, 1].abs() < 0.75)).any(dim=1)
    return {
        "path_safe_rate": float(inside.all(dim=1).float().mean().cpu()),
        "terminal_feasible_rate": float(inside[:, -1].float().mean().cpu()),
        "target_hit_rate": float(hit.float().mean().cpu()),
        "route_adherence_rate": float(route_adherence.float().mean().cpu()),
        "ambiguous_two_passage_rate": float(ambiguous.float().mean().cpu()),
        "middle_obstacle_cross_rate": float(middle_cross.float().mean().cpu()),
        "mean_outside_fraction": float((~inside).float().mean().cpu()),
        "mean_path_length": float(length.mean().cpu()),
        "mean_smoothness": float(smooth.mean().cpu()),
    }


def plot_paths(paths: dict[str, Tensor], boxes: list[Box], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(paths), figsize=(5.0 * len(paths), 4.2), sharex=True, sharey=True)
    if len(paths) == 1:
        axes = [axes]
    for ax, (name, path) in zip(axes, paths.items()):
        for box in boxes:
            x, y, w, h = box.patch_args()
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor="#edf7f1",
                    edgecolor="#2f7d58",
                    alpha=0.75,
                    linewidth=0.9,
                )
            )
        arr = path.detach().cpu()
        for i in range(min(96, arr.shape[0])):
            color = "#3260a8" if arr[i, :, 1].max() > 0.75 else "#8c4fb3"
            ax.plot(arr[i, :, 0], arr[i, :, 1], color=color, alpha=0.23, linewidth=1.0)
        ax.scatter(arr[:96, 0, 0], arr[:96, 0, 1], s=7, color="#2f7d58", alpha=0.7)
        ax.scatter(arr[:96, -1, 0], arr[:96, -1, 1], s=7, color="#b23b3b", alpha=0.7)
        ax.set_title(name)
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.35, alpha=0.35)
    fig.tight_layout()
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
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1536)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--ode-steps", type=int, default=56)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("results/maze_shared_ocfmcs"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"device={device}", flush=True)

    boxes = build_maze()
    segments = make_segments(boxes)
    union_boxes = [
        boxes["left_room"],
        boxes["upper_passage"],
        boxes["lower_passage"],
        boxes["right_room"],
    ]

    direct, direct_losses, direct_train_seconds = train_direct(
        boxes,
        device,
        args.steps,
        args.batch_size,
        args.lr,
        args.hidden,
        args.log_every,
    )
    shared, shared_losses, shared_train_seconds = train_shared(
        segments,
        device,
        args.steps,
        args.batch_size,
        args.lr,
        args.hidden,
        args.embed_dim,
        args.log_every,
    )

    x_eval = sample_source(args.eval_samples, device, boxes)
    route_ids = torch.randint(0, 2, (args.eval_samples,), device=device)

    eval_times: dict[str, float] = {}
    paths: dict[str, Tensor] = {}
    for name, runner in {
        "direct_cfm": lambda: integrate_direct(direct, x_eval, args.ode_steps),
        "direct_projected": lambda: integrate_direct(
            direct,
            x_eval,
            args.ode_steps,
            projector=lambda x: project_to_union(x, union_boxes),
        ),
        "shared_ocfmcs": lambda: integrate_shared_route(
            shared,
            x_eval,
            route_ids,
            args.ode_steps,
            segments,
            chart_project=False,
        ),
        "shared_ocfmcs_chart_projected": lambda: integrate_shared_route(
            shared,
            x_eval,
            route_ids,
            args.ode_steps,
            segments,
            chart_project=True,
        ),
    }.items():
        start = time.perf_counter()
        paths[name] = runner()
        if device.type == "cuda":
            torch.cuda.synchronize()
        eval_times[name] = time.perf_counter() - start

    metrics = {
        name: compute_metrics(path, route_ids, union_boxes, boxes["target_core"])
        for name, path in paths.items()
    }
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if device.type == "cuda"
        else 0.0
    )
    metrics["efficiency"] = {
        "direct_train_seconds": direct_train_seconds,
        "shared_train_seconds": shared_train_seconds,
        "direct_params": count_params(direct),
        "shared_params": count_params(shared),
        "peak_cuda_memory_mb": peak_memory_mb,
        "eval_seconds": eval_times,
        "eval_samples_per_second": {
            name: args.eval_samples / max(seconds, 1e-9)
            for name, seconds in eval_times.items()
        },
    }
    metrics["config"] = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_samples": args.eval_samples,
        "ode_steps": args.ode_steps,
        "hidden": args.hidden,
        "embed_dim": args.embed_dim,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "num_segments": len(segments),
    }

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_paths(paths, union_boxes, args.out_dir / "paths.png")
    plot_losses({"direct": direct_losses, "shared_oc": shared_losses}, args.out_dir / "losses.png")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

