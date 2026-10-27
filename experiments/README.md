# Experiments

This directory contains experiment and figure-generation scripts for FMCS and
OC-FMCS.

## Core Scripts

| Script | Purpose |
| --- | --- |
| `nonconvex_toy.py` | 2D overlapping-convex corridor benchmark |
| `maze_shared_ocfmcs.py` | Shared-vector-field multi-homotopy maze |
| `planar_arm_ocfmcs.py` | Collision-aware planar arm benchmark |
| `summarize_metrics.py` | Multi-seed summary utility |
| `render_robot_figures.py` | Robot comparison figure-sheet renderer |

## Robot Scripts

| Script | Purpose |
| --- | --- |
| `isaaclab_lift_generative_compare.py` | Lift-Cube qualitative comparison |
| `isaaclab_lift_policy_baselines.py` | Lift-Cube policy baseline metrics |
| `isaaclab_drawer_visual_compare.py` | Open-Drawer qualitative comparison |
| `isaaclab_multitask_policy_baselines.py` | Lift, drawer, stack, and reach metric runs |
| `isaaclab_stack_visual_compare.py` | Stack-Cube qualitative comparison |
| `isaaclab_franka_rollout_probe.py` | Isaac Lab environment and rollout probe |

## Language and CaP-X Scripts

| Script | Purpose |
| --- | --- |
| `llm_chart_proposer.py` | Generate semantic chart specifications |
| `refine_chart_spec.py` | Normalize chart proposals into executable phases |
| `evaluate_llm_chart_construction.py` | Evaluate chart validity and task coverage |
| `capx_local_llm_proxy.py` | OpenAI-compatible local language-model proxy |
| `capx_fmcs_chart_proxy.py` | OpenAI-compatible FMCS chart compiler proxy |

## Non-Convex Corridor

```bash
python experiments/nonconvex_toy.py \
  --steps 300 \
  --eval-samples 256 \
  --out-dir results/nonconvex_toy_minimal
```

The benchmark compares direct conditional flow matching, projected direct flow,
OC-FMCS, and chart-projected OC-FMCS on an overlapping-convex non-convex domain.

## Multi-Homotopy Maze

```bash
python experiments/maze_shared_ocfmcs.py \
  --steps 1800 \
  --eval-samples 2048 \
  --out-dir results/maze_shared
```

The maze benchmark tests a shared local vector field conditioned by chart and
segment identifiers across multiple feasible homotopy classes.

## Planar Arm

```bash
python experiments/planar_arm_ocfmcs.py \
  --steps 1400 \
  --batch-size 1600 \
  --eval-samples 2048 \
  --direct-ode-steps 96 \
  --segment-ode-steps 28 \
  --out-dir results/planar_arm_ocfmcs
```

The planar arm benchmark evaluates collision-free generation in a non-convex
configuration space using a chain of validated joint-space charts.

## Isaac Lab

Set the Isaac Lab Python executable before running robot experiments:

```bash
export ISAACLAB_PYTHON=<isaac-lab-python>
```

Stack-Cube metrics:

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

Stack-Cube qualitative comparison:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
OMNI_KIT_DISABLE_TELEMETRY=1 \
"$ISAACLAB_PYTHON" \
  experiments/isaaclab_stack_visual_compare.py \
  --method oc_fmcs \
  --steps 240 \
  --device cuda:0 \
  --headless \
  --video \
  --out-dir results/isaaclab_stack_visual_compare
```

Camera-based robot comparisons should show the robot, manipulated object,
target or obstacle, and outcome in the same view.

## LLM-Assisted Chart Construction

```bash
python experiments/llm_chart_proposer.py \
  --out results/llm_chart_proposals/deepseek_stack_chart_v2.json
```

```bash
python experiments/refine_chart_spec.py \
  results/llm_chart_proposals/deepseek_stack_chart_v2.json \
  --out results/llm_chart_proposals/deepseek_stack_chart_v2_refined.json
```

```bash
python experiments/evaluate_llm_chart_construction.py \
  --chart results/llm_chart_proposals/deepseek_stack_chart_v2.json \
  --refined results/llm_chart_proposals/deepseek_stack_chart_v2_refined.json \
  --out results/llm_chart_proposals/stack_llm_chart_construction_metrics.json
```

## CaP-X Interface

FMCS can be exposed as an OpenAI-compatible chart compiler service:

```bash
python experiments/capx_fmcs_chart_proxy.py --port 8112
```

A local Hugging Face model can be exposed through the same API shape:

```bash
export CAPX_LOCAL_MODEL_PATH=<local-model-or-hf-id>
python experiments/capx_local_llm_proxy.py --port 8110
```

Use the CaP-X client configuration to point to either proxy endpoint.
