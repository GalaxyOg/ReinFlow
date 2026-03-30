# FFSM 三阶段流水线（单仓库版）

本文档只覆盖 FFSM 主线，不讨论其他 benchmark。

## 1. 目标与范围

本分支固定流程：

1. 用 `rl_zoo3` 训练 FFSM 专家（SAC/PPO）；
2. 导出专家轨迹为 `train.npz` / `normalization.npz`；
3. 在 `ReinFlow` 上做 `pre_reflow_mlp` 预训练；
4. 基于预训练权重做 `ft_ppo_reflow_mlp` 微调。

关键点：三阶段代码、配置、日志全部在 `ReinFlow` 内统一管理。

统一入口脚本：`script/ffsm_pipeline.py`

## 2. 前置条件

- 当前 conda 环境可直接 `import ffsm_env` 与 `import rl_zoo3`。
- `ReinFlow` 仓库安装为 editable：

```bash
cd /home/yh/Algo_test/ReinFlow
pip install -e .
```

专家超参文件已内置在仓库：

- `cfg/ffsm/expert/ffsm_sac_hyperparams.yml`
- `cfg/ffsm/expert/ffsm_ppo_hyperparams.yml`

## 3. 分阶段命令

### 3.1 专家训练

```bash
python script/ffsm_pipeline.py train-expert \
  --algo sac \
  --expert-device cuda:0
```

默认日志目录：`log/expert/rl_zoo3/`

### 3.2 导出数据集

```bash
python script/ffsm_pipeline.py export-dataset \
  --algo sac \
  --expert-device cuda:0 \
  --num-trajectories 20 \
  --max-steps 200
```

默认输出：`data/gym/FFSMEnv6dof-v0/`

### 3.3 预训练

```bash
python script/ffsm_pipeline.py pretrain \
  --pretrain-config pre_reflow_mlp \
  --pretrain-device cuda:0
```

### 3.4 微调

```bash
python script/ffsm_pipeline.py finetune \
  --finetune-config ft_ppo_reflow_mlp \
  --finetune-device cuda:0
```

不传 `--base-policy-path` 时自动选最新 pretrain checkpoint。

## 4. 一条命令跑全流程

```bash
python script/ffsm_pipeline.py all \
  --train-expert \
  --algo sac \
  --expert-device cuda:0 \
  --pretrain-device cuda:0 \
  --finetune-device cuda:0
```

## 5. 稳定性约定

- 默认 `wandb` 离线；默认 `finetune` 不存视频（减少渲染依赖）。
- `--dry-run` 可先验证命令与路径。
- `script/start_ffsm_three_stage_tmux_test.sh` 也已切换为单仓库路径。
