# FFSM 评估口径（统一版）

## 目标

让 `pretrain`、`finetune`、`eval` 的 `success rate` 使用同一把尺子，避免“看起来提升/下降，但其实阈值不同”的假象。

## 统一定义

在一个 episode 内，先定义：

- `episode_reward = sum(step_reward)`
- `episode_best_reward = max(step_reward) / act_steps`（与当前代码实现一致）

然后定义成功判定：

- `success = episode_best_reward >= best_reward_threshold_for_success`

最终：

- `success_rate = mean(success over episodes)`

## FFSM 当前统一阈值

- `best_reward_threshold_for_success = -0.1`

## 配置入口

### 1) Pretrain（已支持从配置读取）

在 FFSM 预训练配置中设置顶层字段：

```yaml
best_reward_threshold_for_success: -0.1
```

对应文件：

- `cfg/gym/pretrain/FFSMEnv6dof-v0/pre_reflow_mlp.yaml`
- `cfg/gym/pretrain/FFSMEnv6dof-v0/pre_diffusion_mlp.yaml`
- `cfg/gym/pretrain/FFSMEnv6dof-v0/pre_shortcut_mlp.yaml`

### 2) Finetune

在 `env.best_reward_threshold_for_success` 设置同样阈值：

```yaml
env:
  best_reward_threshold_for_success: -0.1
```

FFSM 新配置目录 `cfg/gym/finetune/FFSMEnv6dof-v0/` 已按该口径使用。

## 注意事项

1. 旧目录 `cfg/gym/finetune/FFSM-6dof-v0/` 与 `cfg/gym/eval/FFSM-6dof-v0/` 已归档到 `cfg/_legacy/`，不要再用于新实验。  
2. `success_rate` 只是阈值化指标，论文主结论仍应同时报告 `avg episode reward`、`avg best reward`、碰撞/末端误差等连续指标。
