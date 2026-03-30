#!/usr/bin/env python3
"""FFSM end-to-end pipeline runner for ReinFlow.

Stages:
1) Train expert policy with rl-zoo3 in FFSM_Env
2) Export expert rollouts to ReinFlow dataset format
3) Pretrain ReFlow policy with offline dataset
4) RL finetune from pretrained checkpoint
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("REINFLOW_DATA_DIR", REPO_ROOT / "data"))
DEFAULT_LOG_ROOT = Path(os.environ.get("REINFLOW_LOG_DIR", REPO_ROOT / "log"))
DEFAULT_FFSM_ENV_ROOT = (REPO_ROOT.parent / "FFSM_Env").resolve()


def _bool_str(flag: bool) -> str:
    return "True" if flag else "False"


def _reinflow_env() -> Dict[str, str]:
    return {
        "REINFLOW_DIR": os.environ.get("REINFLOW_DIR", str(REPO_ROOT)),
        "REINFLOW_DATA_DIR": os.environ.get("REINFLOW_DATA_DIR", str(DEFAULT_DATA_ROOT)),
        "REINFLOW_LOG_DIR": os.environ.get("REINFLOW_LOG_DIR", str(DEFAULT_LOG_ROOT)),
        "REINFLOW_WANDB_ENTITY": os.environ.get("REINFLOW_WANDB_ENTITY", "local"),
    }


def _ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} 不存在: {path}")


def _print_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    prefix = f"(cd {cwd} && " if cwd else ""
    suffix = ")" if cwd else ""
    print(f"[CMD] {prefix}{shlex.join(list(cmd))}{suffix}")


def _run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> None:
    _print_cmd(cmd, cwd=cwd)
    if dry_run:
        return

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, env=env, check=True)


def _merge_pythonpath(extra_path: Path, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = {} if base_env is None else dict(base_env)
    original = os.environ.get("PYTHONPATH", "")
    parts = [str(extra_path)]
    if original:
        parts.append(original)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _find_latest_expert_model(ffsm_env_root: Path, algo: str, env_id: str) -> Path:
    logs_dir = ffsm_env_root / "rl-zoo3" / "logs" / algo
    _ensure_exists(logs_dir, "rl-zoo3 日志目录")

    candidates = list(logs_dir.glob(f"{env_id}_*/best_model.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"未找到 expert model: {logs_dir}/{env_id}_*/best_model.zip"
        )
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _find_latest_pretrain_ckpt(log_root: Path) -> Path:
    pretrain_root = log_root / "gym" / "pretrain"
    _ensure_exists(pretrain_root, "pretrain 日志目录")

    patterns = [
        "FFSMEnv6dof-v0_pre_reflow_mlp_ta*_td*_seed*/*/checkpoint/last.pt",
        "FFSMEnv6dof-v0_pre_reflow_mlp_ta*_td*_seed*/*/checkpoint/best.pt",
    ]

    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(pretrain_root.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"未找到 FFSM pretrain checkpoint: {pretrain_root}"
        )
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


@dataclass
class DatasetOutput:
    raw_path: Path
    train_path: Path
    normalization_path: Path


def stage_train_expert(args: argparse.Namespace) -> Optional[Path]:
    ffsm_env_root = Path(args.ffsm_env_root).expanduser().resolve()
    rlzoo_dir = ffsm_env_root / "rl-zoo3"
    _ensure_exists(ffsm_env_root, "FFSM_Env 根目录")
    _ensure_exists(rlzoo_dir, "rl-zoo3 目录")

    conf_file = (
        Path(args.conf_file).expanduser().resolve()
        if args.conf_file
        else (rlzoo_dir / f"ffsm_{args.algo}_hyperparams.yml")
    )
    _ensure_exists(conf_file, "rl-zoo3 超参数文件")

    cmd = [
        sys.executable,
        "-m",
        "rl_zoo3.train",
        "--algo",
        args.algo,
        "--env",
        args.env_id,
        "--verbose",
        "0",
        "-P",
        "--device",
        args.expert_device,
        "--vec-env",
        "subproc",
        "--conf-file",
        str(conf_file),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.n_timesteps is not None:
        cmd.extend(["-n", str(args.n_timesteps)])

    env = _merge_pythonpath(ffsm_env_root)
    _run_cmd(cmd, cwd=rlzoo_dir, extra_env=env, dry_run=args.dry_run)

    if args.dry_run:
        return None

    latest = _find_latest_expert_model(ffsm_env_root, args.algo, args.env_id)
    print(f"[INFO] 最新 expert model: {latest}")
    return latest


def _load_sb3_model(algo: str, model_path: Path, device: str):
    algo = algo.lower()
    if algo == "sac":
        from stable_baselines3 import SAC

        return SAC.load(str(model_path), device=device)
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(model_path), device=device)
    raise ValueError(f"暂不支持的算法: {algo}")


def stage_export_dataset(args: argparse.Namespace) -> DatasetOutput:
    ffsm_env_root = Path(args.ffsm_env_root).expanduser().resolve()
    ffsm_pkg_root = (
        Path(args.ffsm_package_root).expanduser().resolve()
        if args.ffsm_package_root
        else ffsm_env_root
    )

    _ensure_exists(ffsm_env_root, "FFSM_Env 根目录")
    _ensure_exists(ffsm_pkg_root, "ffsm_env 包路径")

    if args.expert_model:
        expert_model = Path(args.expert_model).expanduser().resolve()
    else:
        if args.dry_run:
            try:
                expert_model = _find_latest_expert_model(ffsm_env_root, args.algo, args.env_id)
            except FileNotFoundError:
                expert_model = Path("<AUTO_LATEST_EXPERT_MODEL>")
        else:
            expert_model = _find_latest_expert_model(ffsm_env_root, args.algo, args.env_id)
    if not args.dry_run:
        _ensure_exists(expert_model, "expert model")

    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_path = (
        Path(args.output_raw).expanduser().resolve()
        if args.output_raw
        else (output_dir / "expert_raw.npz")
    )
    train_path = output_dir / "train.npz"
    normalization_path = output_dir / "normalization.npz"

    if args.dry_run:
        print(f"[INFO] expert model: {expert_model}")
        print(f"[INFO] raw dataset: {raw_path}")
        print(f"[INFO] train dataset: {train_path}")
        print(f"[INFO] normalization: {normalization_path}")
        return DatasetOutput(
            raw_path=raw_path,
            train_path=train_path,
            normalization_path=normalization_path,
        )

    if str(ffsm_pkg_root) not in sys.path:
        sys.path.insert(0, str(ffsm_pkg_root))

    try:
        import ffsm_env  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            f"无法导入 ffsm_env，请先安装 FFSM_Env 或检查路径: {ffsm_pkg_root}"
        ) from exc

    import gymnasium as gym

    env_kwargs = {}
    if args.frame_skip is not None:
        env_kwargs["frame_skip"] = args.frame_skip
    if args.spline_k is not None:
        env_kwargs["spline_K"] = args.spline_k

    model = _load_sb3_model(args.algo, expert_model, args.expert_device)
    env = gym.make(args.env_id, render_mode=None, **env_kwargs)

    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    traj_lengths: List[int] = []

    for epi in range(args.num_trajectories):
        obs, _ = env.reset(seed=args.seed + epi)
        done = False
        steps = 0

        episode_states: List[np.ndarray] = []
        episode_actions: List[np.ndarray] = []

        while not done and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=(not args.stochastic))
            next_obs, _, terminated, truncated, _ = env.step(action)

            episode_states.append(np.asarray(obs, dtype=np.float32))
            episode_actions.append(np.asarray(action, dtype=np.float32))

            obs = next_obs
            done = bool(terminated or truncated)
            steps += 1

        if steps == 0:
            continue

        all_states.append(np.stack(episode_states, axis=0))
        all_actions.append(np.stack(episode_actions, axis=0))
        traj_lengths.append(steps)

    env.close()

    if not traj_lengths:
        raise RuntimeError("未采集到任何轨迹，请检查 expert model 与环境配置")

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    traj_lengths_arr = np.asarray(traj_lengths, dtype=np.int32)

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(raw_path, states=states, actions=actions, traj_lengths=traj_lengths_arr)

    obs_min = states.min(axis=0)
    obs_max = states.max(axis=0)
    action_min = actions.min(axis=0)
    action_max = actions.max(axis=0)

    np.savez(
        normalization_path,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
    )

    states_norm = 2.0 * (states - obs_min) / (obs_max - obs_min + 1e-6) - 1.0
    actions_norm = 2.0 * (actions - action_min) / (action_max - action_min + 1e-6) - 1.0

    np.savez(
        train_path,
        states=states_norm.astype(np.float32),
        actions=actions_norm.astype(np.float32),
        traj_lengths=traj_lengths_arr,
    )

    print(f"[INFO] expert model: {expert_model}")
    print(f"[INFO] raw dataset: {raw_path}")
    print(f"[INFO] train dataset: {train_path}")
    print(f"[INFO] normalization: {normalization_path}")
    print(
        f"[INFO] trajectories={len(traj_lengths_arr)}, steps={int(traj_lengths_arr.sum())}, "
        f"obs_dim={states.shape[1]}, act_dim={actions.shape[1]}"
    )

    return DatasetOutput(
        raw_path=raw_path,
        train_path=train_path,
        normalization_path=normalization_path,
    )


def stage_pretrain(args: argparse.Namespace) -> Optional[Path]:
    train_dataset_path = Path(args.train_dataset_path).expanduser().resolve()
    normalization_path = Path(args.normalization_path).expanduser().resolve()
    if not args.dry_run:
        _ensure_exists(train_dataset_path, "train.npz")
        _ensure_exists(normalization_path, "normalization.npz")

    sim_device = args.pretrain_sim_device or args.pretrain_device

    cmd = [
        sys.executable,
        "script/run.py",
        "--config-dir=cfg/gym/pretrain/FFSMEnv6dof-v0",
        f"--config-name={args.pretrain_config}",
        f"train_dataset_path={train_dataset_path}",
        f"normalization_path={normalization_path}",
        "use_d4rl_dataset=False",
        f"device={args.pretrain_device}",
        f"sim_device={sim_device}",
        f"seed={args.seed}",
        f"wandb.offline_mode={_bool_str(not args.wandb_online)}",
    ]
    if args.pretrain_denoising_steps is not None:
        cmd.append(f"denoising_steps={args.pretrain_denoising_steps}")

    _run_cmd(cmd, cwd=REPO_ROOT, extra_env=_reinflow_env(), dry_run=args.dry_run)

    if args.dry_run:
        return None

    latest = _find_latest_pretrain_ckpt(Path(args.log_root).expanduser().resolve())
    print(f"[INFO] 最新 pretrain checkpoint: {latest}")
    return latest


def stage_finetune(args: argparse.Namespace) -> None:
    normalization_path = Path(args.normalization_path).expanduser().resolve()
    if not args.dry_run:
        _ensure_exists(normalization_path, "normalization.npz")

    if args.base_policy_path:
        base_policy_path = Path(args.base_policy_path).expanduser().resolve()
    else:
        if args.dry_run:
            try:
                base_policy_path = _find_latest_pretrain_ckpt(Path(args.log_root).expanduser().resolve())
            except FileNotFoundError:
                base_policy_path = Path("<AUTO_LATEST_PRETRAIN_CKPT>")
        else:
            base_policy_path = _find_latest_pretrain_ckpt(Path(args.log_root).expanduser().resolve())
    if not args.dry_run:
        _ensure_exists(base_policy_path, "base policy checkpoint")

    sim_device = args.finetune_sim_device or args.finetune_device

    cmd = [
        sys.executable,
        "script/run.py",
        "--config-dir=cfg/gym/finetune/FFSMEnv6dof-v0",
        f"--config-name={args.finetune_config}",
        f"base_policy_path={base_policy_path}",
        f"normalization_path={normalization_path}",
        f"device={args.finetune_device}",
        f"sim_device={sim_device}",
        f"seed={args.seed}",
        f"env.save_video={_bool_str(args.save_video)}",
        f"wandb.offline_mode={_bool_str(not args.wandb_online)}",
    ]
    if args.finetune_n_envs is not None:
        cmd.append(f"env.n_envs={args.finetune_n_envs}")

    _run_cmd(cmd, cwd=REPO_ROOT, extra_env=_reinflow_env(), dry_run=args.dry_run)


def stage_all(args: argparse.Namespace) -> None:
    expert_model_path: Optional[Path] = None
    if args.train_expert:
        expert_model_path = stage_train_expert(args)

    if args.expert_model:
        args.expert_model = str(Path(args.expert_model).expanduser().resolve())
    elif expert_model_path is not None:
        args.expert_model = str(expert_model_path)

    dataset_output = stage_export_dataset(args)
    args.train_dataset_path = str(dataset_output.train_path)
    args.normalization_path = str(dataset_output.normalization_path)

    pretrain_ckpt = stage_pretrain(args)
    if pretrain_ckpt is not None:
        args.base_policy_path = str(pretrain_ckpt)

    stage_finetune(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FFSM expert->pretrain->finetune 一体化脚本（面向 ReinFlow）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ffsm-env-root", default=str(DEFAULT_FFSM_ENV_ROOT), help="FFSM_Env 仓库根目录")
        p.add_argument("--env-id", default="FFSMEnv6dof-v0", help="Gym 环境 ID")
        p.add_argument("--algo", default="sac", choices=["sac", "ppo"], help="expert 算法")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际执行")

    p_train = subparsers.add_parser("train-expert", help="使用 rl-zoo3 训练 FFSM expert")
    add_shared_arguments(p_train)
    p_train.add_argument("--expert-device", default="cuda:0", help="expert 训练设备")
    p_train.add_argument("--conf-file", default=None, help="超参文件路径，不传则按算法自动选择")
    p_train.add_argument("--n-timesteps", type=int, default=None, help="覆盖训练总步数")

    p_export = subparsers.add_parser("export-dataset", help="从 expert 模型导出 FFSM 数据集")
    add_shared_arguments(p_export)
    p_export.add_argument("--expert-model", default=None, help="expert model.zip 路径，不传则自动找最新")
    p_export.add_argument("--ffsm-package-root", default=None, help="ffsm_env 包路径，不传则等于 --ffsm-env-root")
    p_export.add_argument("--expert-device", default="cpu", help="expert 推理设备")
    p_export.add_argument("--num-trajectories", type=int, default=20)
    p_export.add_argument("--max-steps", type=int, default=200)
    p_export.add_argument("--frame-skip", type=int, default=None)
    p_export.add_argument("--spline-k", type=int, default=None)
    p_export.add_argument("--stochastic", action="store_true", help="采样动作而不是确定性动作")
    p_export.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_ROOT / "gym" / "FFSMEnv6dof-v0"),
        help="输出目录（会写 train.npz 和 normalization.npz）",
    )
    p_export.add_argument("--output-raw", default=None, help="原始轨迹 npz 路径")

    p_pre = subparsers.add_parser("pretrain", help="运行 FFSM ReFlow 预训练")
    add_shared_arguments(p_pre)
    p_pre.add_argument("--pretrain-config", default="pre_reflow_mlp")
    p_pre.add_argument("--pretrain-device", default="cuda:0")
    p_pre.add_argument("--pretrain-sim-device", default=None)
    p_pre.add_argument("--pretrain-denoising-steps", type=int, default=None)
    p_pre.add_argument("--wandb-online", action="store_true", help="默认离线；加此参数启用在线 WandB")
    p_pre.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT), help="ReinFlow 日志根目录")
    p_pre.add_argument(
        "--train-dataset-path",
        default=str(DEFAULT_DATA_ROOT / "gym" / "FFSMEnv6dof-v0" / "train.npz"),
    )
    p_pre.add_argument(
        "--normalization-path",
        default=str(DEFAULT_DATA_ROOT / "gym" / "FFSMEnv6dof-v0" / "normalization.npz"),
    )

    p_ft = subparsers.add_parser("finetune", help="运行 FFSM ReinFlow 微调")
    add_shared_arguments(p_ft)
    p_ft.add_argument("--finetune-config", default="ft_ppo_reflow_mlp")
    p_ft.add_argument("--finetune-device", default="cuda:0")
    p_ft.add_argument("--finetune-sim-device", default=None)
    p_ft.add_argument("--finetune-n-envs", type=int, default=None)
    p_ft.add_argument("--base-policy-path", default=None, help="不传则自动找最新 pretrain checkpoint")
    p_ft.add_argument("--save-video", action="store_true", help="默认关闭，减少渲染依赖")
    p_ft.add_argument("--wandb-online", action="store_true")
    p_ft.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT), help="ReinFlow 日志根目录")
    p_ft.add_argument(
        "--normalization-path",
        default=str(DEFAULT_DATA_ROOT / "gym" / "FFSMEnv6dof-v0" / "normalization.npz"),
    )

    p_all = subparsers.add_parser("all", help="按顺序执行 expert->dataset->pretrain->finetune")
    add_shared_arguments(p_all)
    p_all.add_argument("--train-expert", action="store_true", help="先执行 rl-zoo3 expert 训练")
    p_all.add_argument("--conf-file", default=None)
    p_all.add_argument("--n-timesteps", type=int, default=None)
    p_all.add_argument("--expert-model", default=None)
    p_all.add_argument("--ffsm-package-root", default=None)
    p_all.add_argument("--expert-device", default="cuda:0")
    p_all.add_argument("--num-trajectories", type=int, default=20)
    p_all.add_argument("--max-steps", type=int, default=200)
    p_all.add_argument("--frame-skip", type=int, default=None)
    p_all.add_argument("--spline-k", type=int, default=None)
    p_all.add_argument("--stochastic", action="store_true")
    p_all.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_ROOT / "gym" / "FFSMEnv6dof-v0"),
        help="数据输出目录",
    )
    p_all.add_argument("--output-raw", default=None)
    p_all.add_argument("--pretrain-config", default="pre_reflow_mlp")
    p_all.add_argument("--pretrain-device", default="cuda:0")
    p_all.add_argument("--pretrain-sim-device", default=None)
    p_all.add_argument("--pretrain-denoising-steps", type=int, default=None)
    p_all.add_argument("--finetune-config", default="ft_ppo_reflow_mlp")
    p_all.add_argument("--finetune-device", default="cuda:0")
    p_all.add_argument("--finetune-sim-device", default=None)
    p_all.add_argument("--finetune-n-envs", type=int, default=None)
    p_all.add_argument("--base-policy-path", default=None)
    p_all.add_argument("--save-video", action="store_true")
    p_all.add_argument("--wandb-online", action="store_true")
    p_all.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT), help="ReinFlow 日志根目录")
    p_all.add_argument(
        "--train-dataset-path",
        default=str(DEFAULT_DATA_ROOT / "gym" / "FFSMEnv6dof-v0" / "train.npz"),
    )
    p_all.add_argument(
        "--normalization-path",
        default=str(DEFAULT_DATA_ROOT / "gym" / "FFSMEnv6dof-v0" / "normalization.npz"),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train-expert":
        stage_train_expert(args)
    elif args.command == "export-dataset":
        stage_export_dataset(args)
    elif args.command == "pretrain":
        stage_pretrain(args)
    elif args.command == "finetune":
        stage_finetune(args)
    elif args.command == "all":
        stage_all(args)
    else:
        raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
