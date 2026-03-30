# FFSM 最小环境复现（单仓库）

本说明只覆盖 FFSM 三阶段训练，不包含 robomimic / kitchen / d3il / furniture 依赖。

## 1. 创建 conda 环境

```bash
cd /home/yh/Algo_test/ReinFlow
conda env create -f installation/environment_ffsm_minimal.yml
conda activate rlzoo3
```

## 2. 安装 ReinFlow 与 FFSM 环境包

```bash
# 在 ReinFlow 仓库内
pip install -e .

# 安装 FFSM_Env（示例：本地源码 editable）
pip install -e /path/to/FFSM_Env
```

如果 FFSM_Env 已发布到私有/公开 pip，可替换为 `pip install <ffsm_env_package_name>`。

## 3. 快速自检

```bash
python - <<'PY'
import ffsm_env, rl_zoo3
print("ffsm_env + rl_zoo3 import ok")
PY
```

## 4. 运行三阶段（tmux）

```bash
bash script/start_ffsm_three_stage_tmux_test.sh
```

默认日志目录：

- 专家：`log/expert/rl_zoo3/`
- 预训练：`log/gym/pretrain/`
- 微调：`log/gym/finetune/`
- tmux 总日志：`log/ffsm_tmux/`

## 5. 说明

- 本最小环境文件不包含 robomimic 相关依赖。
- 如只需 CPU，可将 `installation/environment_ffsm_minimal.yml` 里的 `pytorch-cuda` 条目移除后重建环境。
