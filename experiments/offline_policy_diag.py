from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dump", type=Path)
    p.add_argument("--episodes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--train-steps", type=int, nargs="+", default=[1800])
    p.add_argument("--variants", type=str, default="chunk,ddpm_legacy,ddpm_eps,ddpm_x0,cfm,fmcs")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument("--fm-ode-steps", type=int, default=10)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--chart-margin", type=float, default=0.10)
    p.add_argument("--num-phases", type=int, default=9)
    p.add_argument("--eval-stride", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


args = parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, i, o, h, depth=3):
        super().__init__()
        layers, last = [], i
        for _ in range(depth):
            layers += [nn.Linear(last, h), nn.SiLU()]
            last = h
        layers.append(nn.Linear(last, o))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Denoiser(nn.Module):
    def __init__(self, obs_dim, chunk_dim, h, time_dim=32, zero_out=False, t_scale=1.0):
        super().__init__()
        self.time_dim, self.t_scale = time_dim, t_scale
        self.net = MLP(obs_dim + chunk_dim + time_dim, chunk_dim, h, depth=4)
        if zero_out:
            nn.init.zeros_(self.net.net[-1].weight)
            nn.init.zeros_(self.net.net[-1].bias)

    def forward(self, obs, x, t):
        half = self.time_dim // 2
        freqs = torch.exp(-torch.linspace(0, math.log(10000), half, device=t.device))
        a = (t.float() * self.t_scale).unsqueeze(-1) * freqs
        return self.net(torch.cat([obs, x, torch.sin(a), torch.cos(a)], dim=-1))


def stats(x):
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(0), flat.std(0).clamp_min(1e-4)


def chunk_rows(obs, actions, phases, horizon):
    e, steps, _ = obs.shape
    o, c, p = [], [], []
    for t in range(steps):
        end = min(t + horizon, steps)
        ch = actions[:, t:end]
        if end - t < horizon:
            ch = torch.cat([ch, actions[:, end - 1 : end].repeat(1, horizon - (end - t), 1)], dim=1)
        o.append(obs[:, t])
        c.append(ch.reshape(e, -1))
        p.append(phases[:, t])
    return torch.cat(o), torch.cat(c), torch.cat(p)


def cosine_schedule(T):
    s = 0.008
    g = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((g + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(max=0.999).float()
    return betas.to(dev)


def linear_schedule(T):
    return torch.linspace(1e-4, 0.02, T, device=dev)


def train_loop(model, step_fn, steps, ema_decay):
    ema = copy.deepcopy(model) if ema_decay > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for _ in range(steps):
        loss = step_fn(model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(p, 1 - ema_decay)
    return (ema if ema is not None else model).eval(), float(loss.detach())


def run_variant(name, steps, tr, ev):
    phase_cond = name.startswith("fmcs") or name.startswith("phase_")
    obs_in, ev_obs_in = (tr["pobs"], ev["pobs"]) if phase_cond else (tr["obs"], ev["obs"])
    name = name.removeprefix("phase_")
    D, n = tr["z"].shape[1], tr["z"].shape[0]
    B = args.batch_size

    if name == "chunk":
        model = MLP(obs_in.shape[1], D, args.hidden_dim, depth=4).to(dev)

        def step(m):
            idx = torch.randint(0, n, (B,), device=dev)
            return F.mse_loss(m(obs_in[idx]), tr["z"][idx])

        model, loss = train_loop(model, step, steps, ema_decay=0.0)
        with torch.no_grad():
            return model(ev_obs_in), loss

    if name.startswith("ddpm"):
        legacy = name == "ddpm_legacy"
        x0_pred = name == "ddpm_x0"
        T = args.diffusion_steps
        betas = linear_schedule(T) if legacy else cosine_schedule(T)
        alphas = 1 - betas
        ab = torch.cumprod(alphas, 0)
        model = Denoiser(obs_in.shape[1], D, args.hidden_dim).to(dev)

        def step(m):
            idx = torch.randint(0, n, (B,), device=dev)
            t = torch.randint(0, T, (B,), device=dev)
            eps = torch.randn(B, D, device=dev)
            a = ab[t].unsqueeze(-1)
            xt = a.sqrt() * tr["z"][idx] + (1 - a).sqrt() * eps
            return F.mse_loss(m(obs_in[idx], xt, t), tr["z"][idx] if x0_pred else eps)

        model, loss = train_loop(model, step, steps, ema_decay=0.0 if legacy else args.ema_decay)
        with torch.no_grad():
            m = ev_obs_in.shape[0]
            x = torch.randn(m, D, device=dev)
            for ti in reversed(range(T)):
                t = torch.full((m,), ti, device=dev)
                out = model(ev_obs_in, x, t)
                a, a_prev = ab[ti], (ab[ti - 1] if ti > 0 else torch.ones_like(ab[ti]))
                x0 = out if x0_pred else (x - (1 - a).sqrt() * out) / a.sqrt()
                if not legacy:
                    x0 = torch.maximum(torch.minimum(x0, tr["hi"]), tr["lo"])
                mean = (a_prev.sqrt() * betas[ti] / (1 - a)) * x0 + (alphas[ti].sqrt() * (1 - a_prev) / (1 - a)) * x
                if ti > 0:
                    var = betas[ti] if legacy else betas[ti] * (1 - a_prev) / (1 - a)
                    x = mean + var.sqrt() * torch.randn_like(x)
                else:
                    x = mean
            return x, loss

    if name in {"cfm", "fmcs", "cfm_x", "fmcs_x", "fmcs_xc"}:
        chart = name.startswith("fmcs")
        xpred = name.endswith("_x") or name.endswith("_xc")
        clamp_end = name.endswith("_xc")
        model = Denoiser(obs_in.shape[1], D, args.hidden_dim, zero_out=not xpred, t_scale=1000.0).to(dev)

        def prior(k):
            e = torch.randn(k.shape[0], D, device=dev)
            return torch.maximum(torch.minimum(e, tr["chi"][k]), tr["clo"][k]) if chart else e

        def step(m):
            idx = torch.randint(0, n, (B,), device=dev)
            z = tr["z"][idx]
            x0 = prior(tr["ph"][idx])
            t = torch.rand(B, 1, device=dev)
            out = m(obs_in[idx], (1 - t) * x0 + t * z, t.squeeze(-1))
            return F.mse_loss(out, z if xpred else z - x0)

        model, loss = train_loop(model, step, steps, ema_decay=args.ema_decay)
        with torch.no_grad():
            m = ev_obs_in.shape[0]
            x = prior(ev["ph"])
            dt = 1.0 / args.fm_ode_steps
            for i in range(args.fm_ode_steps):
                out = model(ev_obs_in, x, torch.full((m,), i * dt, device=dev))
                if clamp_end:
                    out = torch.maximum(torch.minimum(out, tr["chi"][ev["ph"]]), tr["clo"][ev["ph"]])
                x = x + dt * ((out - x) / (1.0 - i * dt) if xpred else out)
            return x, loss
    raise ValueError(name)


def main():
    data = torch.load(args.dump, map_location="cpu")
    A = data["action_dim"]
    H = args.horizon
    P = args.num_phases
    results = []
    for E in args.episodes:
        n_envs = data["eval"]["obs"].shape[0]
        sub = {k: v[: E * n_envs].to(dev) for k, v in data["train"].items()}
        if sub["obs"].shape[0] < E * n_envs:
            print(f"skip E={E}: dump has only {sub['obs'].shape[0] // n_envs} episodes")
            continue
        evd = {k: v.to(dev) for k, v in data["eval"].items()}
        obs_m, obs_s = stats(sub["obs"])
        po = torch.cat([sub["obs"], F.one_hot(sub["phases"].long().clamp(0, P - 1), P).float()], -1)
        pobs_m, pobs_s = stats(po)
        o, c, ph = chunk_rows(sub["obs"], sub["actions"], sub["phases"], H)
        po_rows, _, _ = chunk_rows(po, sub["actions"], sub["phases"], H)
        c_m, c_s = stats(c)
        z = (c - c_m) / c_s
        ph = ph.long().clamp(0, P - 1)
        clo = z.min(0).values.repeat(P, 1)
        chi = z.max(0).values.repeat(P, 1)
        for k in range(P):
            if (ph == k).any():
                clo[k], chi[k] = z[ph == k].min(0).values, z[ph == k].max(0).values
        tr = {
            "obs": (o - obs_m) / obs_s,
            "pobs": (po_rows - pobs_m) / pobs_s,
            "z": z,
            "ph": ph,
            "lo": z.min(0).values,
            "hi": z.max(0).values,
            "clo": clo - args.chart_margin,
            "chi": chi + args.chart_margin,
        }
        eo, ec, eph = chunk_rows(evd["obs"], evd["actions"], evd["phases"], H)
        epo = torch.cat([evd["obs"], F.one_hot(evd["phases"].long().clamp(0, P - 1), P).float()], -1)
        epo_rows, _, _ = chunk_rows(epo, evd["actions"], evd["phases"], H)
        keep = torch.arange(0, eo.shape[0], args.eval_stride, device=dev)
        ev = {
            "obs": ((eo - obs_m) / obs_s)[keep],
            "pobs": ((epo_rows - pobs_m) / pobs_s)[keep],
            "ph": eph.long().clamp(0, P - 1)[keep],
        }
        expert = ec[keep].reshape(-1, H, A)
        n_outlier = int((((eo - obs_m) / obs_s).abs() > 50).any(-1).sum())
        for steps in args.train_steps:
            for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
                torch.manual_seed(args.seed)
                t0 = time.perf_counter()
                pred_z, loss = run_variant(name, steps, tr, ev)
                pred = (pred_z * c_s + c_m).reshape(-1, H, A)
                first = (pred[:, 0, :3] - expert[:, 0, :3]).norm(dim=-1)
                chunk = (pred[:, :, :3] - expert[:, :, :3]).norm(dim=-1).mean(-1)
                row = {
                    "episodes": E,
                    "train_steps": steps,
                    "variant": name,
                    "loss": round(loss, 4),
                    "first_pos_err_mean": round(float(first.mean()), 4),
                    "first_pos_err_median": round(float(first.median()), 4),
                    "frac_first_err_lt_2cm": round(float((first < 0.02).float().mean()), 3),
                    "chunk_pos_err_mean": round(float(chunk.mean()), 4),
                    "eval_obs_outlier_rows": n_outlier,
                    "seconds": round(time.perf_counter() - t0, 1),
                }
                if A >= 8:
                    row["grip_sign_err"] = round(float(((pred[:, 0, 7] < 0) != (expert[:, 0, 7] < 0)).float().mean()), 3)
                results.append(row)
                print(json.dumps(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
