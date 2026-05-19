# Reproducibility Notes

This document lists the main commands for reproducing the experiment families in
the repository. Commands write outputs under `results/`.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Isaac Lab experiments require an external Isaac Lab installation:

```bash
export ISAACLAB_PYTHON=<isaac-lab-python>
```

## Non-Convex Corridor

```bash
python experiments/nonconvex_toy.py \
  --steps 300 \
  --eval-samples 256 \
  --out-dir results/nonconvex_toy_minimal
```

```bash
python experiments/nonconvex_toy.py \
  --steps 1500 \
  --eval-samples 1024 \
  --out-dir results/nonconvex_toy
```

## Multi-Homotopy Maze

```bash
python experiments/maze_shared_ocfmcs.py \
  --steps 1800 \
  --eval-samples 2048 \
  --out-dir results/maze_shared
```

```bash
python experiments/summarize_metrics.py \
  results/maze_shared_seed_31/metrics.json \
  results/maze_shared_seed_37/metrics.json \
  results/maze_shared_seed_41/metrics.json \
  --out results/maze_shared_seed_sweep_summary.json
```

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

## Isaac Lab Stack-Cube

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
OMNI_KIT_DISABLE_TELEMETRY=1 \
xvfb-run -a "$ISAACLAB_PYTHON" \
  experiments/isaaclab_multitask_policy_baselines.py \
  --task stack \
  --num-envs 24 \
  --steps 240 \
  --device cuda:0 \
  --headless \
  --out-dir results/isaaclab_stack_metrics
```

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

Robot benchmark metrics should be paired with at least one rendered camera
sequence that makes the robot, object, goal, and outcome visually readable.

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

Run the FMCS chart proxy:

```bash
python experiments/capx_fmcs_chart_proxy.py --port 8112
```

Run a local language-model proxy:

```bash
export CAPX_LOCAL_MODEL_PATH=<local-model-or-hf-id>
python experiments/capx_local_llm_proxy.py --port 8110
```
