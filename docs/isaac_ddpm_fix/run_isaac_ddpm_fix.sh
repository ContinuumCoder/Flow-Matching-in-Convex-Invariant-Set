#!/usr/bin/env bash
# Reruns the Isaac Lab DDPM-baseline fix study whose metrics are in docs/isaac_ddpm_fix/results/.
# Usage, from the repository root:
#   ISAACLAB_PYTHON=<isaac-lab-python> bash docs/isaac_ddpm_fix/run_isaac_ddpm_fix.sh main|ablation
# Each run is one Isaac process (24 envs for Reach/Stack, 48 for Lift; ~2 GB GPU memory).
# Set EXTRA=--stub-dex-retargeting if Stack hangs while importing pinocchio (pinocchio built
# against NumPy 1.x under NumPy 2.x).
set -u
PY=${ISAACLAB_PYTHON:?set ISAACLAB_PYTHON to the Isaac Lab python executable}
OUT=${OUT:-results/isaac_ddpm_fix}
EXTRA=${EXTRA:-}
MULTI=experiments/isaaclab_multitask_policy_baselines.py
LIFT=experiments/isaaclab_lift_policy_baselines.py
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_DISABLE_TELEMETRY=1

# Original DDPM behavior: linear betas 1e-4..0.02 over 16 steps, eps-prediction, no x0 clip,
# no EMA, resample every control step.
LEGACY="--diffusion-schedule linear_legacy --diffusion-pred eps --no-diffusion-clip --ema-decay 0 --exec-steps 1"
COMMON="--num-envs 24 --seed 23 --eval-seed 2027 --chunk-horizon 16 --bc-train-steps 1200 \
  --chunk-train-steps 1200 --act-train-steps 1600 --diffusion-steps 16 --batch-size 512 --hidden-dim 256 \
  --lr 3e-4 --diffusion-train-steps 1800 --fm-train-steps 1800 --device cuda:0 --headless $EXTRA"
STACK="--scenario stack --steps 240 --stack-overlap-threshold 0.12 --stack-lift-z 0.26 --stack-place-z 0.095 \
  --stack-close-dwell 1.2 --stack-place-dwell 1.2 --stack-release-dwell 1.6"
REACH="--scenario reach --steps 120"
STACK_METHODS=chunk_mlp,diffusion_policy,phase_chunk_mlp,phase_diffusion_policy,oc_fmcs,oc_fmcs_learned
REACH_METHODS=bc_mlp,chunk_mlp,diffusion_policy,phase_diffusion_policy,oc_fmcs,oc_fmcs_learned
LIFT_METHODS=bc_mlp,chunk_mlp,diffusion_policy,phase_bc_mlp,phase_chunk_mlp,phase_diffusion_policy,oc_fmcs

run() {
  local name=$1 script=$2; shift 2
  mkdir -p "$OUT/$name"
  echo "[$(date +%T)] start $name"
  "$PY" "$script" "$@" --out "$OUT/$name/metrics.json" > "$OUT/$name.log" 2>&1
  echo "[$(date +%T)] done $name exit=$?"
}

case "${1:-main}" in
  main)
    run reach_legacy $MULTI $COMMON $REACH $LEGACY --eval-methods $REACH_METHODS
    run reach_fixed $MULTI $COMMON $REACH --eval-methods $REACH_METHODS
    run stack_24demos_legacy $MULTI $COMMON $STACK $LEGACY --eval-methods $STACK_METHODS
    run stack_24demos_fixed $MULTI $COMMON $STACK --eval-methods $STACK_METHODS
    run stack_192demos_fixed $MULTI $COMMON $STACK --demo-episodes 8 --eval-methods $STACK_METHODS
    # The Lift "before" run in results/ kept x0-prediction and changed only schedule, clip, EMA
    # and execution; add --diffusion-pred eps for the fully original sampler.
    run lift_legacy_schedule $LIFT --diffusion-schedule linear_legacy --no-diffusion-clip --ema-decay 0 \
      --exec-steps 1 --eval-methods $LIFT_METHODS --device cuda:0 --headless
    run lift_fixed $LIFT --eval-methods $LIFT_METHODS --device cuda:0 --headless
    ;;
  ablation)
    run reach_cosine_eps $MULTI $COMMON $REACH --diffusion-pred eps --eval-methods $REACH_METHODS
    run stack_24demos_cosine_only $MULTI $COMMON $STACK --diffusion-pred eps --no-diffusion-clip --ema-decay 0 \
      --exec-steps 1 --eval-methods diffusion_policy,phase_diffusion_policy
    run stack_24demos_cosine_eps $MULTI $COMMON $STACK --diffusion-pred eps --eval-methods $STACK_METHODS
    run stack_24demos_cosine_eps_6000steps $MULTI $COMMON $STACK --diffusion-pred eps \
      --diffusion-train-steps 6000 --eval-methods diffusion_policy,phase_diffusion_policy
    ;;
  *)
    echo "usage: $0 main|ablation" >&2
    exit 2
    ;;
esac
