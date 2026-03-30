# FFSM 三阶段流水线（rl-zoo3 Expert -> ReFlow 预训练 -> RL 微调）

本文档只覆盖 FFSM 主线，不讨论其他 benchmark。

## 1. 目标与范围

本分支聚焦以下固定流程：

1. 在 `FFSM_Env` 中用 `rl-zoo3` 训练专家（SAC/PPO）。
2. 用专家策略回放采样，生成 `train.npz` 与 `normalization.npz`。
3. 在 `ReinFlow` 上执行 `pre_reflow_mlp` 预训练。
4. 以预训练权重为初始化，执行 `ft_ppo_reflow_mlp` 微调。

统一入口脚本：`script/ffsm_pipeline.py`

## 2. 前置条件

- `ReinFlow` 与 `FFSM_Env` 在同级目录（默认路径假设：`../FFSM_Env`）。
- 已安装 `FFSM_Env` 包（推荐）：

```bash
cd /home/yh/Algo_test/FFSM_Env
pip install -e .
```

- 已设置 ReinFlow 路径变量（可选但建议）：

```bash
cd /home/yh/Algo_test/ReinFlow
bash script/set_path.sh
```

## 3. 分阶段命令

### 3.1 训练专家（rl-zoo3）

```bash
python script/ffsm_pipeline.py train-expert \
  --algo sac \
  --expert-device cuda:0
```

### 3.2 导出专家数据集

```bash
python script/ffsm_pipeline.py export-dataset \
  --algo sac \
  --expert-device cuda:0 \
  --num-trajectories 20 \
  --max-steps 200
```

默认输出到：`data/gym/FFSMEnv6dof-v0/`

- `expert_raw.npz`
- `train.npz`
- `normalization.npz`

### 3.3 ReFlow 预训练

```bash
python script/ffsm_pipeline.py pretrain \
  --pretrain-config pre_reflow_mlp \
  --pretrain-device cuda:0
```

### 3.4 RL 微调

```bash
python script/ffsm_pipeline.py finetune \
  --finetune-config ft_ppo_reflow_mlp \
  --finetune-device cuda:0
```

不传 `--base-policy-path` 时会自动选最新 pretrain checkpoint。

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

- 默认 `wandb` 离线、`finetune` 关闭视频渲染，减少环境依赖导致的报错。
- 支持 `--dry-run` 打印即将执行的命令，用于先检查参数与路径。
- 预训练与微调阶段会自动注入 `REINFLOW_*` 环境变量兜底，避免未配置时直接崩溃。
