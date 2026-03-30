# ReinFlow x FFSM 项目复盘（截至 2026-03-29）

## 1. 项目在做什么

这个仓库的主线是把 **离线预训练的生成式策略（ReFlow / Diffusion / Shortcut）** 用 **在线 RL（主要是 PPO 变体）** 做二阶段微调。  
你这条分支的核心目标是把 ReinFlow 落到 **FFSM 自定义浮动基空间机械臂任务** 上。

当前主入口是：

- 训练入口：`script/run.py`
- 预训练：`agent/pretrain/*`
- 微调：`agent/finetune/reinflow/*`
- 环境适配：`env/gymnasium_utils/*`
- FFSM 配置：`cfg/gym/pretrain/FFSMEnv6dof-v0`、`cfg/gym/finetune/FFSMEnv6dof-v0`

## 2. 你这条 FFSM 线已经做到的事情

结合代码、提交记录和日志，当前状态不是“没跑通”，而是“**链路跑通，但效果与评估定义还不稳**”：

1. 预训练和微调都可以完整跑完（有 checkpoint、日志、视频输出）。
2. FFSM 相关 wrapper、渲染、TB 写板问题已修过（12 月下旬连续提交）。
3. 你已经做了系统化消融矩阵（`exp1~exp9`）+ HLGauss 版本。

最近日志（`log/gym/...`）显示：

- `pre_reflow_mlp_ta1_td4` 最后 `avg episode reward ≈ -20`，`success rate = 0`（见“问题 1”）。
- `ft_ppo_reflow_mlp_ta1_td4_tdf4` 可稳定训练到 600 itr，最后回报约 `-34`，中途最好约 `-23.6`。
- `exp1/3/4/7` 这类“有预训练起点”的方案整体可用。
- `exp5/6/9`（从零训练或大模型从零）明显崩，回报在 `-80~-160` 区间。

## 3. 目前最核心的问题

### 问题 1：评估指标定义前后不一致，导致“success”可比性差

- 预训练测试里，Gym 任务成功阈值在代码里写死为 `3.0`（`agent/pretrain/train_agent.py`），对 FFSM 这种负奖励任务几乎必然是 0。
- 微调配置里又用 `best_reward_threshold_for_success: -0.1`（`cfg/gym/finetune/FFSMEnv6dof-v0/*`），因此 success 会很高。
- 结果是：预训练 success 和微调 success 不是同一把尺子，论文叙事会被这个问题卡住。

### 问题 2：配置分叉严重（旧 FFSM 名称与新名称并存）

当前同时存在：

- `FFSM-6dof-v0`（旧，已归档到 `cfg/_legacy`）
- `FFSMEnv6dof-v0`（新）

并且：

- 历史目录 `cfg/gym/finetune/FFSM-6dof-v0/` 和 `cfg/gym/eval/FFSM-6dof-v0/` 已归档到 `cfg/_legacy/`，仅保留追溯用途。

这会直接影响“几个月后回看是否能一键复现”。

### 问题 3：研究问题本身还没闭环

你的日记结论与代码现状一致：

- ReinFlow 在 FFSM 上“能训”，但效果尚未形成稳定优势。
- 专家策略/专家数据质量仍是瓶颈（你 2026-01-04 还发现 URDF 惯量问题）。
- 现在更像在做“参数工程 + 经验拼接”，创新点和故事线还没完全收束。

### 问题 4：训练资源开销风险较高

`train_ppo_flow_agent.py` 里 `repeat_samples=true` 时有 `duplicate_multiplier=10`，结合大 `batch_size` 和多 denoising step，显存压力会非常高（你在 `MyNote.md` 的判断是对的）。

## 4. 对你时间线的对应解释

- **2025-12-02**：预训练跑通但评估有问题  
  对应现在看到的：链路可运行，但评估定义和渲染链路当时不稳定。
- **2025-12-16**：FFSM 重构（样条 + 指数 cost），并试 PPO/SAC  
  对应现在“专家策略质量仍未稳定”的根因来源。
- **2025-12-26**：ReinFlow 效果一般、专家未完全搞定、开始大量调参  
  对应你后续 `exp1~exp9` 消融矩阵提交。
- **2026-01-04**：URDF 转动惯量问题  
  这是非常关键的数据/动力学一致性风险点，会污染专家数据和后续 RL 结论。

## 5. 建议的下一步（按优先级）

1. 先统一评估口径：把 pretrain/finetune 的 success 定义统一到同一套 FFSM 指标（建议用 `distance_to_target`、`collision_num`、最终轨迹误差阈值）。
2. 清理配置：保留 `FFSMEnv6dof-v0` 一套配置，旧 `FFSM-6dof-v0` 全部归档到 `cfg/_legacy/`。
3. 固定一个主 baseline：以 `ft_ppo_reflow_mlp` 和 `exp7` 为主，跑 3 个 seed，先拿稳定均值和方差。
4. 专家数据重建：基于修复后 URDF 重新导出专家轨迹和 `normalization.npz`，再复跑 pretrain + finetune。
5. 论文故事建议：把“从零训练失败、预训练初始化有效、探索噪声/critic 设计影响稳定性”作为主线，不强行做过多横向模型替换。

## 6. 现在可直接使用的命令（建议起点）

```bash
# 1) FFSM ReFlow 预训练
python3 script/run.py \
  --config-dir=cfg/gym/pretrain/FFSMEnv6dof-v0 \
  --config-name=pre_reflow_mlp

# 2) FFSM ReinFlow 微调（主 baseline）
python3 script/run.py \
  --config-dir=cfg/gym/finetune/FFSMEnv6dof-v0 \
  --config-name=ft_ppo_reflow_mlp

# 3) 你当时的消融矩阵批量脚本
bash start_experiments.sh
```

---

如果你愿意，我下一步可以直接帮你做一版“配置清理 PR”（只动配置与文档，不动算法），把这条线恢复到一眼可复现的状态。
