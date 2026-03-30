# FFSM 会话上下文与跨设备恢复

## 1. 说明

当前 AI 聊天会话本身（对话历史、临时推理过程）不会自动进入 git 仓库。  
本文件用于把“可复现的上下文”固化到仓库，便于在其他机器恢复工作。

## 2. 当前分支与关键提交

- 工作分支：`feat_ffsm_e2e_pipeline`
- 关键提交（按时间倒序）：
  - `95f0678`：澄清历史结果文档中的外部路径表述
  - `8a91967`：新增 FFSM 最小环境文件与单仓库文档
  - `34a9ba7`：新增仓库内 expert 超参数文件（SAC/PPO）
  - `13c05b8`：三阶段流程改为单仓库执行与统一日志

## 3. 单仓库目标口径

- `FFSM_Env` 作为独立 pip 包安装后可 `import ffsm_env` 即可。
- 三阶段训练代码、配置、日志统一在 `ReinFlow` 仓库内管理：
  - expert：`log/expert/rl_zoo3/`
  - pretrain：`log/gym/pretrain/`
  - finetune：`log/gym/finetune/`
  - tmux 总日志：`log/ffsm_tmux/`

## 4. 其他设备恢复步骤

```bash
git clone git@github.com:GalaxyOg/ReinFlow.git
cd ReinFlow
git checkout feat_ffsm_e2e_pipeline

conda env create -f installation/environment_ffsm_minimal.yml
conda activate rlzoo3

pip install -e .
pip install -e /path/to/FFSM_Env
```

自检：

```bash
python - <<'PY'
import ffsm_env, rl_zoo3
print("ffsm_env + rl_zoo3 import ok")
PY
```

## 5. 启动方式

- tmux 三阶段启动：`bash script/start_ffsm_three_stage_tmux_test.sh`
- 一体化流程脚本：`python script/ffsm_pipeline.py all --dry-run --train-expert`

> 注意：运行前必须 `conda activate rlzoo3`。
