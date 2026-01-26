#!/bin/bash

# Experiment Comparison Matrix
# ========================================================================================================
# Exp ID | GPU    | Name          | Type    | BC   | Key Changes vs Baseline                     | Goal
# --------------------------------------------------------------------------------------------------------
# Exp 1  | cuda:2 | N_Base_Real   | Normal  | No   | reward_scale_running=False, max_ep=200      | Honest Baseline
# Exp 2  | cuda:2 | N_Explore     | Normal  | No   | ent_coef=0.01, min_std=0.12, max_std=0.25   | Force Exploration
# Exp 7  | cuda:2 | H_Hybrid_Str  | HLGauss | 0.05 | ent=0.01, noise=[0.12, 0.25], from PT       | Strong Explore + Anchor
#
# Exp 3  | cuda:3 | H_Sigma       | HLGauss | No   | sigma=2.5, reward_scale_running=False       | Stabilize without BC
# Exp 4  | cuda:3 | H_Hybrid      | HLGauss | 0.05 | ent_coef=0.01, min_std=0.12                 | Anchor + Explore
# Exp 8  | cuda:3 | H_Sigma_Mid   | HLGauss | No   | sigma=1.5, ent=0.005, from PT               | Mid-Sigma Stability
#
# Exp 5  | cuda:4 | N_Large       | Normal  | No   | mlp=[1024], FROM SCRATCH (No PT)            | Capacity (Scratch)
# Exp 6  | cuda:4 | H_Large       | HLGauss | 0.05 | mlp=[1024], FROM SCRATCH (No PT)            | Cap + HLGauss (Scratch)
# Exp 9  | cuda:4 | N_Scratch     | Normal  | No   | mlp=[512], FROM SCRATCH (No PT)             | Scratch Baseline
# ========================================================================================================

# --- Kill potentially conflicting sessions from previous attempt ---
tmux kill-session -t exp5_n_large 2>/dev/null
tmux kill-session -t exp6_h_large 2>/dev/null

# --- GPU 2 Group ---
# Note: Exp 1 and 2 might already be running. If not, uncomment to restart.
# tmux new-session -d -s exp1_base_real ...
# tmux new-session -d -s exp2_explore ...

tmux new-session -d -s exp7_h_hybrid_strong
tmux send-keys -t exp7_h_hybrid_strong "conda activate rlzoo3" C-m
tmux send-keys -t exp7_h_hybrid_strong "cd /home/yh/Algo_test/ReinFlow" C-m
tmux send-keys -t exp7_h_hybrid_strong "xvfb-run -a python script/run.py --config-dir=cfg/gym/finetune/FFSMEnv6dof-v0 --config-name=ft_ppo_exp7_H_Hybrid_Strong" C-m
echo "Started Exp 7 on cuda:2."

# --- GPU 3 Group ---
# Note: Exp 3 and 4 might already be running.
# tmux new-session -d -s exp3_h_sigma ...
# tmux new-session -d -s exp4_h_hybrid ...

tmux new-session -d -s exp8_h_sigma_mid
tmux send-keys -t exp8_h_sigma_mid "conda activate rlzoo3" C-m
tmux send-keys -t exp8_h_sigma_mid "cd /home/yh/Algo_test/ReinFlow" C-m
tmux send-keys -t exp8_h_sigma_mid "xvfb-run -a python script/run.py --config-dir=cfg/gym/finetune/FFSMEnv6dof-v0 --config-name=ft_ppo_exp8_H_Sigma_Mid" C-m
echo "Started Exp 8 on cuda:3."

# --- GPU 4 Group (Restarted/New) ---
tmux new-session -d -s exp5_n_large
tmux send-keys -t exp5_n_large "conda activate rlzoo3" C-m
tmux send-keys -t exp5_n_large "cd /home/yh/Algo_test/ReinFlow" C-m
tmux send-keys -t exp5_n_large "xvfb-run -a python script/run.py --config-dir=cfg/gym/finetune/FFSMEnv6dof-v0 --config-name=ft_ppo_exp5_N_Large" C-m
echo "Restarted Exp 5 (Scratch) on cuda:4. Waiting 60s..."
sleep 60

tmux new-session -d -s exp6_h_large
tmux send-keys -t exp6_h_large "conda activate rlzoo3" C-m
tmux send-keys -t exp6_h_large "cd /home/yh/Algo_test/ReinFlow" C-m
tmux send-keys -t exp6_h_large "xvfb-run -a python script/run.py --config-dir=cfg/gym/finetune/FFSMEnv6dof-v0 --config-name=ft_ppo_exp6_H_Large" C-m
echo "Restarted Exp 6 (Scratch) on cuda:4. Waiting 60s..."
sleep 60

tmux new-session -d -s exp9_n_scratch
tmux send-keys -t exp9_n_scratch "conda activate rlzoo3" C-m
tmux send-keys -t exp9_n_scratch "cd /home/yh/Algo_test/ReinFlow" C-m
tmux send-keys -t exp9_n_scratch "xvfb-run -a python script/run.py --config-dir=cfg/gym/finetune/FFSMEnv6dof-v0 --config-name=ft_ppo_exp9_N_Scratch" C-m
echo "Started Exp 9 (Scratch Baseline) on cuda:4."

echo "Launched supplementary experiments (7, 8, 9) and restarted fixed experiments (5, 6)."
