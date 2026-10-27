# Flow Matching in Convex Invariant Sets

This repository contains experiment code for Flow Matching in Convex Invariant
Sets (FMCS) and its overlapping-convex extension, OC-FMCS.

FMCS builds generative policies directly on feasible geometric support. OC-FMCS
extends this idea to non-convex domains by composing overlapping convex charts,
so semantic phases such as approach, grasp, lift, place, and release become a
validated geometric topology rather than an unconstrained action sequence.

## Contents

```text
experiments/   Experiment scripts and metric utilities
results/       Selected compact summaries; large run outputs are ignored
docs/          Reproduction and artifact notes
```

## Setup

The toy, maze, planar-arm, and local proxy scripts use a standard Python
environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Isaac Lab experiments require a separate Isaac Lab installation and should be
run with the Isaac Lab Python executable. Set:

```bash
export ISAACLAB_PYTHON=<isaac-lab-python>
```

## Core Experiments

Non-convex corridor:

```bash
python experiments/nonconvex_toy.py \
  --steps 300 \
  --eval-samples 256 \
  --out-dir results/nonconvex_toy_minimal
```

Multi-homotopy maze:

```bash
python experiments/maze_shared_ocfmcs.py \
  --steps 1800 \
  --eval-samples 2048 \
  --out-dir results/maze_shared
```

Planar collision-avoidance arm:

```bash
python experiments/planar_arm_ocfmcs.py \
  --steps 1400 \
  --batch-size 1600 \
  --eval-samples 2048 \
  --direct-ode-steps 96 \
  --segment-ode-steps 28 \
  --out-dir results/planar_arm_ocfmcs
```

## Robot Experiments

Isaac Lab scripts evaluate OC-FMCS and generative-policy baselines on Franka
manipulation tasks including Lift-Cube, Open-Drawer, and Stack-Cube:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
OMNI_KIT_DISABLE_TELEMETRY=1 \
xvfb-run -a "$ISAACLAB_PYTHON" \
  experiments/isaaclab_multitask_policy_baselines.py \
  --scenario stack \
  --num-envs 24 \
  --steps 240 \
  --device cuda:0 \
  --headless \
  --out results/isaaclab_stack_metrics/metrics.json
```

Qualitative comparison scripts render camera views with the robot, object,
obstacle or goal, and success or failure behavior visible in the same frame.

In these scripts `oc_fmcs` is the scripted chart controller (atlas, analytic
local field, IK). [docs/isaac_ddpm_fix/](docs/isaac_ddpm_fix/) documents a learned
variant (`oc_fmcs_learned`) and an upgraded DDPM baseline, with all metrics.

## Language-Conditioned Charts

FMCS can serve as a geometric topology layer for language or code-policy
interfaces. A high-level policy proposes semantic phases; the chart compiler
turns those phases into an executable, physically ordered atlas.

Relevant scripts:

- `experiments/llm_chart_proposer.py`
- `experiments/refine_chart_spec.py`
- `experiments/evaluate_llm_chart_construction.py`
- `experiments/capx_local_llm_proxy.py`
- `experiments/capx_fmcs_chart_proxy.py`

## Reproduction

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for experiment commands
and [docs/ARTIFACT_POLICY.md](docs/ARTIFACT_POLICY.md) for tracked artifact
conventions.
