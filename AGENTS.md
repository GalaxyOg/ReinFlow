# Repository Guidelines

## Project Structure & Module Organization
`ReinFlow` is organized by training stage and policy family.
- `agent/`: training and evaluation entry classes (`pretrain/`, `finetune/`, `eval/`, `dataset/`).
- `model/`: policy/value implementations (`flow/`, `diffusion/`, `gaussian/`, `common/`).
- `cfg/`: Hydra configs by benchmark and stage, e.g. `cfg/gym/pretrain/...`, `cfg/robomimic/finetune/...`, `cfg/*/eval/...`.
- `script/`: launch and utility scripts (`run.py`, render tests, dataset helpers).
- `data_process/` and `util/`: preprocessing and shared utilities.
- `docs/` and `installation/`: reproduction and environment setup notes.

## Build, Test, and Development Commands
- Install package in editable mode:
  ```bash
  pip install -e .
  ```
- Set repository paths/env vars (recommended once per machine):
  ```bash
  bash script/set_path.sh && source ~/.bashrc
  ```
- Run experiments through Hydra:
  ```bash
  python script/run.py --config-dir=cfg/gym/pretrain/walker2d-medium-v2 --config-name=pre_diffusion_mlp
  ```
- Headless GPU rendering (common on servers):
  ```bash
  xvfb-run -a python script/run.py --config-dir=cfg/gym/finetune/hopper-v2 --config-name=ft_ppo_reflow_mlp
  ```

## Coding Style & Naming Conventions
- Python 3.8+ with 4-space indentation; follow existing style in touched files.
- Use `snake_case` for functions/files/config keys, `PascalCase` for classes.
- Keep Hydra config names descriptive and stage-specific: `pre_*`, `ft_*`, `eval_*`.
- Prefer small, focused edits; avoid unrelated refactors in training-critical modules.
- No enforced formatter/linter is configured in `pyproject.toml`; keep imports clean and code readable.

## Testing Guidelines
- There is no centralized `pytest` suite yet; use targeted smoke tests.
- Validate environment/render setup with:
  ```bash
  python script/test_robomimic_render.py
  python script/test_d3il_render.py
  ```
- For algorithm changes, run at least one representative `script/run.py` config and confirm logs/checkpoints are written under `log/` or `outputs/`.
- Include exact command lines used for verification in your PR.

## Commit & Pull Request Guidelines
- Current history favors short, descriptive subjects (Chinese or English), sometimes with tags like `(feat)`/`(bugfix)`.
- Prefer one logical change per commit; subject should state what changed, e.g. `fix robomimic eval rendering`.
- PRs should include: purpose, key files/configs changed, reproduction command(s), and any metric or rendering evidence.
- Do not commit large artifacts (`data/`, `wandb/`, checkpoints, logs); these are ignored by default.
