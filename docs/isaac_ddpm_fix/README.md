# FMCS Isaac Lab update

This folder documents an update to the Isaac Lab Franka experiments. The full
write-up is at <https://continuumcoder.github.io/Flow-Matching-in-Convex-Invariant-Set/>
([`index.html`](index.html), Chinese version in [`index.zh.html`](index.zh.html)).

## What changed

- **DDPM baseline.** The baseline now uses a cosine noise schedule, predicts the
  clean action chunk, clips it to the demonstration range, keeps EMA weights and
  executes 8 actions of each 16-step chunk. The arm moves smoothly, and success
  rises as follows.

  | Task | Demos | DDPM before → upgraded | OC-FMCS |
  |---|---:|---:|---:|
  | Reach | 24 | 0% → 62.5% | 100% |
  | Lift | 192 | 0% → 83.3% | 97.9% |
  | Stack, stacked | 24 | 0% → 4.2–8.3% | 70.8–79.2% |
  | Stack, stacked | 192 | 41.7–50.0% | 62.5–66.7% |

- **Learned local field.** The new method `oc_fmcs_learned` generates every action
  with a learned FMCS field conditioned on the active chart. The chart atlas manages
  the handoffs. It reaches 45.8% on Reach and lifts the cube in 54.2–62.5% of Stack
  episodes.
- **Maintenance.** Stack has one phase per controller state. The Lift script runs on
  current Isaac Lab releases. `--stub-dex-retargeting` supports NumPy 2 environments.
  The README commands match the script flags. New motion-smoothness metrics and an
  offline evaluation script (`experiments/offline_policy_diag.py`) are included.

## Files

- `index.html`, `index.zh.html`: the write-up.
- `run_isaac_ddpm_fix.sh`: rerun script. `main` produces the main results and
  `ablation` runs each setting change on its own.
- `results/<run>/metrics.json`: full metrics of every run. `*_fixed` folders hold
  the upgraded runs and `*_legacy` folders the earlier configuration. The other
  folders change one setting at a time. `results/offline_diag/` holds the offline
  evaluation.
