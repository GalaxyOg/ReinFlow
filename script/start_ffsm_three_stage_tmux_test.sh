#!/usr/bin/env bash
set -Eeuo pipefail

# FFSM 3-stage launcher
# Stage 1: rl-zoo3 expert training (from installed ffsm_env package)
# Stage 2: ReFlow pretraining (in ReinFlow)
# Stage 3: ReinFlow finetuning (in ReinFlow, from existing checkpoint)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${REINFLOW_DATA_DIR:-${REPO_ROOT}/data}"
LOG_ROOT="${REINFLOW_LOG_DIR:-${REPO_ROOT}/log}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${DATA_ROOT}/gym/FFSMEnv6dof-v0/train.npz}"
NORMALIZATION_PATH="${NORMALIZATION_PATH:-${DATA_ROOT}/gym/FFSMEnv6dof-v0/normalization.npz}"
EXPERT_LOG_ROOT="${EXPERT_LOG_ROOT:-${LOG_ROOT}/expert/rl_zoo3}"

# 设备分配（默认三张卡）
CUDA_EXPERT="${CUDA_EXPERT:-0}"
CUDA_PRETRAIN="${CUDA_PRETRAIN:-1}"
CUDA_FINETUNE="${CUDA_FINETUNE:-2}"

# 训练测试强度（默认是“能跑通优先”的小规模）
SEED="${SEED:-42}"
EXPERT_ALGO="${EXPERT_ALGO:-sac}"
EXPERT_CONF_FILE="${EXPERT_CONF_FILE:-${REPO_ROOT}/cfg/ffsm/expert/ffsm_${EXPERT_ALGO}_hyperparams.yml}"
LONG_TEST="${LONG_TEST:-0}"
EXPERT_TIMESTEPS_DEFAULT="50000"
PRETRAIN_EPOCHS_DEFAULT="10"
FINETUNE_ITR_DEFAULT="20"
if [[ "${LONG_TEST}" == "1" ]]; then
  EXPERT_TIMESTEPS_DEFAULT="500000"
  PRETRAIN_EPOCHS_DEFAULT="100"
  FINETUNE_ITR_DEFAULT="300"
fi
EXPERT_TIMESTEPS="${EXPERT_TIMESTEPS:-${EXPERT_TIMESTEPS_DEFAULT}}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-${PRETRAIN_EPOCHS_DEFAULT}}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
FINETUNE_ITR="${FINETUNE_ITR:-${FINETUNE_ITR_DEFAULT}}"
FINETUNE_STEPS="${FINETUNE_STEPS:-64}"
FINETUNE_N_ENVS="${FINETUNE_N_ENVS:-4}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-1024}"
FINETUNE_UPDATE_EPOCHS="${FINETUNE_UPDATE_EPOCHS:-2}"
FINETUNE_LOGPROB_BATCH_SIZE="${FINETUNE_LOGPROB_BATCH_SIZE:-10000}"
FINETUNE_LOGPROB_BATCH_STRICT="${FINETUNE_LOGPROB_BATCH_STRICT:-0}"

# tmux session 名称
SESSION_EXPERT="${SESSION_EXPERT:-ffsm_expert_test}"
SESSION_PRETRAIN="${SESSION_PRETRAIN:-ffsm_pretrain_test}"
SESSION_FINETUNE="${SESSION_FINETUNE:-ffsm_finetune_test}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-rlzoo3}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"
USE_XVFB="${USE_XVFB:-1}"
DRY_RUN="${DRY_RUN:-0}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-1}"
KEEP_SESSION_ON_EXIT="${KEEP_SESSION_ON_EXIT:-1}"
TMUX_LOG_DIR="${TMUX_LOG_DIR:-${REPO_ROOT}/log/ffsm_tmux}"

function require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] 未找到命令: $1" >&2
    exit 1
  }
}

function require_file() {
  local f="$1"
  [[ -f "${f}" ]] || {
    echo "[ERROR] 文件不存在: ${f}" >&2
    exit 1
  }
}

function require_dir() {
  local d="$1"
  [[ -d "${d}" ]] || {
    echo "[ERROR] 目录不存在: ${d}" >&2
    exit 1
  }
}

function require_positive_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] ${name} 必须是正整数，当前值: ${value}" >&2
    exit 1
  }
  (( value > 0 )) || {
    echo "[ERROR] ${name} 必须 > 0，当前值: ${value}" >&2
    exit 1
  }
}

function find_latest_pretrain_ckpt() {
  local log_root="$1"
  python - "$log_root" <<'PY'
from pathlib import Path
import sys
log_root = Path(sys.argv[1])
root = log_root / "gym" / "pretrain"
patterns = [
    "FFSMEnv6dof-v0_pre_reflow_mlp_ta*_td*_seed*/*/checkpoint/last.pt",
    "FFSMEnv6dof-v0_pre_reflow_mlp_ta*_td*_seed*/*/checkpoint/best.pt",
]
all_ckpts = []
for p in patterns:
    all_ckpts.extend(root.glob(p))
if not all_ckpts:
    raise SystemExit(1)
all_ckpts = sorted(all_ckpts, key=lambda x: x.stat().st_mtime, reverse=True)
print(str(all_ckpts[0]))
PY
}

function session_script_prefix() {
  cat <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
  _conda_sh=""
  for _p in "\$HOME/anaconda3/etc/profile.d/conda.sh" "\$HOME/miniconda3/etc/profile.d/conda.sh" "\$HOME/mambaforge/etc/profile.d/conda.sh"; do
    if [[ -f "\$_p" ]]; then
      _conda_sh="\$_p"
      break
    fi
  done
  if [[ -z "\$_conda_sh" ]] && command -v conda >/dev/null 2>&1; then
    _base="\$(conda info --base 2>/dev/null || true)"
    if [[ -n "\$_base" && -f "\$_base/etc/profile.d/conda.sh" ]]; then
      _conda_sh="\$_base/etc/profile.d/conda.sh"
    fi
  fi
  if [[ -z "\$_conda_sh" ]]; then
    echo "[ERROR] 找不到 conda.sh，无法激活环境 ${CONDA_ENV_NAME}" >&2
    exit 1
  fi
  source "\$_conda_sh"
  conda activate "${CONDA_ENV_NAME}"
  echo "[INFO] conda_env=\${CONDA_DEFAULT_ENV:-<none>}"
  python -c "import sys; print('[INFO] python_exe=' + sys.executable)"
fi
EOF
}

function activate_conda_in_main() {
  if [[ "${SKIP_CONDA_ACTIVATE}" == "1" ]]; then
    echo "[WARN] SKIP_CONDA_ACTIVATE=1，跳过主进程 conda 激活"
    return
  fi
  local conda_sh=""
  for p in "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/mambaforge/etc/profile.d/conda.sh"; do
    if [[ -f "${p}" ]]; then
      conda_sh="${p}"
      break
    fi
  done
  if [[ -z "${conda_sh}" ]] && command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
      conda_sh="${base}/etc/profile.d/conda.sh"
    fi
  fi
  if [[ -z "${conda_sh}" ]]; then
    echo "[ERROR] 找不到 conda.sh，无法激活环境 ${CONDA_ENV_NAME}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${conda_sh}"
  conda activate "${CONDA_ENV_NAME}"
  echo "[INFO] main_conda_env=${CONDA_DEFAULT_ENV:-<none>}"
  python -c "import sys; print('[INFO] main_python_exe=' + sys.executable)"
}

function validate_cuda_slot() {
  local stage="$1"
  local cuda_slot="$2"
  if ! CUDA_VISIBLE_DEVICES="${cuda_slot}" python - "${stage}" "${cuda_slot}" <<'PY'
import sys
import torch
stage = sys.argv[1]
slot = sys.argv[2]
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
if not ok:
    raise SystemExit(f"[ERROR] {stage}: CUDA_VISIBLE_DEVICES={slot} 不可用（torch.cuda 不可用）")
_ = torch.tensor([1.0], device="cuda:0")
print(f"[INFO] {stage}: CUDA_VISIBLE_DEVICES={slot} -> cuda:0 可用 ({torch.cuda.get_device_name(0)})")
PY
  then
    echo "[ERROR] ${stage} 的 GPU 配置不可用，请调整 CUDA_EXPERT/CUDA_PRETRAIN/CUDA_FINETUNE" >&2
    exit 1
  fi
}

function launch_tmux_session() {
  local session_name="$1"
  local body="$2"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local log_file="${TMUX_LOG_DIR}/${session_name}_${ts}.log"
  mkdir -p "${TMUX_LOG_DIR}"

  local runner
  runner="$(mktemp "/tmp/${session_name}.XXXX.sh")"
  {
    session_script_prefix
    echo ""
    echo "LOG_FILE='${log_file}'"
    echo "echo \"[INFO] session=${session_name}\" | tee -a \"\${LOG_FILE}\""
    echo "echo \"[INFO] start_time=\$(date '+%F %T')\" | tee -a \"\${LOG_FILE}\""
    echo "echo \"[INFO] pwd=\$(pwd)\" | tee -a \"\${LOG_FILE}\""
    echo "exec > >(tee -a \"\${LOG_FILE}\") 2>&1"
    echo "set +e"
    echo "${body}"
    echo "stage_status=\$?"
    echo "set -e"
    echo "echo \"[INFO] end_time=\$(date '+%F %T')\""
    echo "echo \"[INFO] exit_code=\${stage_status}\""
    echo "echo \"[INFO] log_file=\${LOG_FILE}\""
    echo "if [[ \"${KEEP_SESSION_ON_EXIT}\" == \"1\" ]]; then"
    echo "  echo \"[INFO] 任务已结束，保留会话用于排查。输入 exit 可关闭会话。\""
    echo "  exec bash"
    echo "else"
    echo "  exit \${stage_status}"
    echo "fi"
  } > "${runner}"
  chmod +x "${runner}"

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    tmux kill-session -t "${session_name}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] session=${session_name}"
    echo "[DRY_RUN] script=${runner}"
    echo "[DRY_RUN] log_file=${log_file}"
    sed -n '1,220p' "${runner}"
    return
  fi

  tmux new-session -d -s "${session_name}" "bash '${runner}'"
  echo "[OK] 启动 session: ${session_name} (log: ${log_file})"
}

function main() {
  require_cmd tmux
  require_cmd python
  require_dir "${REPO_ROOT}"
  activate_conda_in_main

  require_file "${TRAIN_DATASET_PATH}"
  require_file "${NORMALIZATION_PATH}"
  require_file "${EXPERT_CONF_FILE}"
  mkdir -p "${EXPERT_LOG_ROOT}"

  if ! python - <<'PY'
import importlib
importlib.import_module("ffsm_env")
importlib.import_module("rl_zoo3")
print("[INFO] python import check: ffsm_env + rl_zoo3 OK")
PY
  then
    echo "[ERROR] 未能导入 ffsm_env/rl_zoo3。请先安装 FFSM_Env 包并确认 rl_zoo3 可用。" >&2
    exit 1
  fi

  local expert_model_path
  expert_model_path="$(find "${EXPERT_LOG_ROOT}/${EXPERT_ALGO}" -maxdepth 3 -type f -name best_model.zip 2>/dev/null | sort | tail -n 1 || true)"

  local base_policy_path
  base_policy_path="${BASE_POLICY_PATH:-}"
  if [[ -z "${base_policy_path}" ]]; then
    if ! base_policy_path="$(find_latest_pretrain_ckpt "${LOG_ROOT}" 2>/dev/null)"; then
      echo "[WARN] 未自动找到最新 pretrain checkpoint，请设置 BASE_POLICY_PATH" >&2
      exit 1
    fi
  fi
  require_file "${base_policy_path}"

  require_positive_int "FINETUNE_N_ENVS" "${FINETUNE_N_ENVS}"
  require_positive_int "FINETUNE_LOGPROB_BATCH_SIZE" "${FINETUNE_LOGPROB_BATCH_SIZE}"

  local finetune_logprob_batch_size_effective="${FINETUNE_LOGPROB_BATCH_SIZE}"
  if (( finetune_logprob_batch_size_effective % FINETUNE_N_ENVS != 0 )); then
    if [[ "${FINETUNE_LOGPROB_BATCH_STRICT}" == "1" ]]; then
      echo "[ERROR] FINETUNE_LOGPROB_BATCH_SIZE(${FINETUNE_LOGPROB_BATCH_SIZE}) 不能被 FINETUNE_N_ENVS(${FINETUNE_N_ENVS}) 整除" >&2
      echo "[ERROR] 请手动设置可整除的 FINETUNE_LOGPROB_BATCH_SIZE，或将 FINETUNE_LOGPROB_BATCH_STRICT=0 允许自动对齐" >&2
      exit 1
    fi
    finetune_logprob_batch_size_effective="$(( (finetune_logprob_batch_size_effective / FINETUNE_N_ENVS + 1) * FINETUNE_N_ENVS ))"
    echo "[WARN] 自动对齐 train.logprob_batch_size: ${FINETUNE_LOGPROB_BATCH_SIZE} -> ${finetune_logprob_batch_size_effective} (n_envs=${FINETUNE_N_ENVS})"
  fi

  validate_cuda_slot "expert" "${CUDA_EXPERT}"
  validate_cuda_slot "pretrain" "${CUDA_PRETRAIN}"
  validate_cuda_slot "finetune" "${CUDA_FINETUNE}"

  local run_prefix=""
  if [[ "${USE_XVFB}" == "1" ]] && command -v xvfb-run >/dev/null 2>&1; then
    run_prefix="xvfb-run -a "
  fi

  echo "[INFO] REPO_ROOT=${REPO_ROOT}"
  echo "[INFO] EXPERT_LOG_ROOT=${EXPERT_LOG_ROOT}"
  echo "[INFO] EXPERT_CONF_FILE=${EXPERT_CONF_FILE}"
  echo "[INFO] TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH}"
  echo "[INFO] NORMALIZATION_PATH=${NORMALIZATION_PATH}"
  echo "[INFO] BASE_POLICY_PATH=${base_policy_path}"
  echo "[INFO] KEEP_SESSION_ON_EXIT=${KEEP_SESSION_ON_EXIT}"
  echo "[INFO] TMUX_LOG_DIR=${TMUX_LOG_DIR}"
  echo "[INFO] LONG_TEST=${LONG_TEST}"
  echo "[INFO] EXPERT_TIMESTEPS=${EXPERT_TIMESTEPS}"
  echo "[INFO] PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS}"
  echo "[INFO] FINETUNE_ITR=${FINETUNE_ITR}"
  echo "[INFO] FINETUNE_N_ENVS=${FINETUNE_N_ENVS}"
  echo "[INFO] FINETUNE_LOGPROB_BATCH_SIZE=${finetune_logprob_batch_size_effective}"
  if [[ -n "${expert_model_path}" ]]; then
    echo "[INFO] Latest existing expert model=${expert_model_path}"
  fi

  local expert_body
  expert_body="
cd '${REPO_ROOT}'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES='${CUDA_EXPERT}'
python -m rl_zoo3.train --algo ${EXPERT_ALGO} --env FFSMEnv6dof-v0 --verbose 0 -P --device cuda:0 --vec-env subproc --gym-packages ffsm_env --log-folder '${EXPERT_LOG_ROOT}' --conf-file '${EXPERT_CONF_FILE}' --seed ${SEED} -n ${EXPERT_TIMESTEPS}
"

  local pretrain_body
  pretrain_body="
cd '${REPO_ROOT}'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES='${CUDA_PRETRAIN}'
export REINFLOW_DIR='${REPO_ROOT}'
export REINFLOW_DATA_DIR='${DATA_ROOT}'
export REINFLOW_LOG_DIR='${LOG_ROOT}'
export REINFLOW_WANDB_ENTITY='${REINFLOW_WANDB_ENTITY:-local}'
${run_prefix}python script/run.py \\
  --config-dir=cfg/gym/pretrain/FFSMEnv6dof-v0 \\
  --config-name=pre_reflow_mlp \\
  train_dataset_path='${TRAIN_DATASET_PATH}' \\
  normalization_path='${NORMALIZATION_PATH}' \\
  use_d4rl_dataset=False \\
  device=cuda:0 \\
  sim_device=cuda:0 \\
  seed=${SEED} \\
  wandb.offline_mode=True \\
  batch_size=${PRETRAIN_BATCH_SIZE} \\
  train.batch_size=${PRETRAIN_BATCH_SIZE} \\
  train.n_epochs=${PRETRAIN_EPOCHS} \\
  train.test_freq=1
"

  local finetune_body
  finetune_body="
cd '${REPO_ROOT}'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES='${CUDA_FINETUNE}'
export REINFLOW_DIR='${REPO_ROOT}'
export REINFLOW_DATA_DIR='${DATA_ROOT}'
export REINFLOW_LOG_DIR='${LOG_ROOT}'
export REINFLOW_WANDB_ENTITY='${REINFLOW_WANDB_ENTITY:-local}'
${run_prefix}python script/run.py \\
  --config-dir=cfg/gym/finetune/FFSMEnv6dof-v0 \\
  --config-name=ft_ppo_reflow_mlp \\
  base_policy_path='${base_policy_path}' \\
  normalization_path='${NORMALIZATION_PATH}' \\
  device=cuda:0 \\
  sim_device=cuda:0 \\
  seed=${SEED} \\
  wandb.offline_mode=True \\
  env.save_video=False \\
  env.n_envs=${FINETUNE_N_ENVS} \\
  train.logprob_batch_size=${finetune_logprob_batch_size_effective} \\
  train.n_train_itr=${FINETUNE_ITR} \\
  train.n_steps=${FINETUNE_STEPS} \\
  train.batch_size=${FINETUNE_BATCH_SIZE} \\
  train.update_epochs=${FINETUNE_UPDATE_EPOCHS} \\
  train.repeat_samples=False \\
  train.val_freq=2 \\
  train.render.freq=100000
"

  launch_tmux_session "${SESSION_EXPERT}" "${expert_body}"
  sleep "${START_DELAY_SECONDS}"
  launch_tmux_session "${SESSION_PRETRAIN}" "${pretrain_body}"
  sleep "${START_DELAY_SECONDS}"
  launch_tmux_session "${SESSION_FINETUNE}" "${finetune_body}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi

  echo ""
  echo "[INFO] 已启动 3 个 session："
  echo "  - ${SESSION_EXPERT}   (GPU cuda:${CUDA_EXPERT})"
  echo "  - ${SESSION_PRETRAIN} (GPU cuda:${CUDA_PRETRAIN})"
  echo "  - ${SESSION_FINETUNE} (GPU cuda:${CUDA_FINETUNE})"
  echo ""
  echo "[INFO] 查看状态: tmux ls"
  echo "[INFO] 进入会话: tmux attach -t ${SESSION_PRETRAIN}"
  echo "[INFO] 终止会话: tmux kill-session -t ${SESSION_PRETRAIN}"
}

main "$@"
