# Isaac Lab 实验复查：oc_fmcs 与 DDPM 基线

完整说明是本目录的 `index.html`（单页网站，下载后用浏览器打开即可）。这里只列结论和文件。

## 结论

1. **三维实验里的 `oc_fmcs` 是手写控制器。** `isaaclab_multitask_policy_baselines.py` 中该分支为
   `action = controller.action(obs)`，即 `ScenarioController` 有限状态机加差分 IK，没有学习的速度场；
   所有学习型基线模仿的也是它生成的演示。新增的 `oc_fmcs_learned` 用学习的、按 chart 条件化的 FMCS 场
   生成全部动作：Reach 45.8%，Stack 叠放 4.2–8.3%，主要卡在抬升状态的门控上。
2. **DDPM 的 0% 是基线实现问题。** 16 步线性调度的 ᾱ_T = 0.851（采样却从纯噪声出发），宽 256 的 MLP
   学不会 ε-预测，以及每步重采样。改为余弦调度、x₀-预测、x₀ 裁剪、EMA、执行 8/16 步后：

   | 任务 | 演示数 | DDPM 修复前 → 修复后 | 状态机 oc_fmcs |
   |---|---:|---:|---:|
   | Reach | 24 | 0% → 62.5% | 100% |
   | Lift | 192 | 0% → 83.3% | 97.9% |
   | Stack 叠放 | 24 | 0% → 4.2–8.3% | 70.8–79.2% |
   | Stack 叠放 | 192 | — → 41.7–50.0% | 62.5–66.7% |

3. 另外修复：Stack 的 `NUM_PHASES` 8 → 9；Lift 脚本第二轮采集在新版 Isaac Lab 下报错卡死；
   README 中 Stack 命令的 `--task`/`--out-dir` 应为 `--scenario`/`--out`。

## 文件

- `index.html`：说明页面（问题、证据、结果、对论文的影响、复现方法）。
- `run_isaac_ddpm_fix.sh`：复现脚本，`main` 跑主结果，`ablation` 跑消融。
- `results/<run>/metrics.json`：每次运行的完整指标；`results/offline_diag/`：离线诊断结果。

| 目录 | 内容 |
|---|---|
| `reach_legacy`、`stack_24demos_legacy` | 原实现（线性调度、ε、无裁剪、无 EMA、每步重采样） |
| `lift_legacy_schedule` | Lift 旧调度与每步重采样（保留 x₀-预测） |
| `stack_24demos_cosine_only` | 只换余弦调度 |
| `reach_cosine_eps`、`stack_24demos_cosine_eps` | 余弦 + 裁剪 + EMA + 执行 8 步，仍用 ε |
| `stack_24demos_cosine_eps_6000steps` | 同上，训练 6000 步 |
| `reach_fixed`、`stack_24demos_fixed`、`stack_192demos_fixed`、`lift_fixed` | 修复后（当前默认参数） |
| `stack_*_fixed_noclamp` | DDPM 同修复后；oc_fmcs_learned 尚无终点裁剪 |

消融目录中的 `oc_fmcs_learned` 是早期的速度预测版本，以 `*_fixed` 为准。
