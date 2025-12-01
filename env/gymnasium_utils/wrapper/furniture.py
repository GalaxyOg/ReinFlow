"""
Furniture wrapper adapted for gymnasium.
"""

import gymnasium as gym
import numpy as np
import torch
from collections import deque

from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
from furniture_bench.controllers.control_utils import proprioceptive_quat_to_6d_rotation
from env.gymnasium_utils.furniture_normalizer import LinearNormalizer
from .multi_step import repeated_space

import logging

log = logging.getLogger(__name__)


def stack_last_n_obs_dict(all_obs, n_steps):
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = {
        key: torch.zeros(
            list(all_obs[-1][key].shape)[0:1]
            + [n_steps]
            + list(all_obs[-1][key].shape)[1:],
            dtype=all_obs[-1][key].dtype,
        ).to(all_obs[-1][key].device)
        for key in all_obs[-1]
    }
    start_idx = -min(n_steps, len(all_obs))
    for key in all_obs[-1]:
        result[key][:, start_idx:] = torch.concatenate([obs[key][:, None] for obs in all_obs[start_idx:]], dim=1)
        if n_steps > len(all_obs):
            result[key][:start_idx] = result[key][start_idx]
    return result


class FurnitureRLSimEnvMultiStepWrapper(gym.Wrapper):
    env: FurnitureRLSimEnv

    def __init__(
        self,
        env: FurnitureRLSimEnv,
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=None,
        sparse_reward=False,
        reset_within_step=False,
        pass_full_observations=False,
        normalization_path=None,
        prev_action=False,
    ):
        assert not reset_within_step
        assert not pass_full_observations
        assert not prev_action

        super().__init__(env)
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.pass_full_observations = pass_full_observations

        self.sparse_reward = sparse_reward

        self.normalize = normalization_path is not None
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(
            torch.load(normalization_path, map_location=self.device, weights_only=True)
        )
        log.info(f"Loaded normalization from {normalization_path}")

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        obs = stack_last_n_obs_dict(self.obs, self.n_obs_steps)
        nobs = self.process_obs(obs)
        self.best_reward = torch.zeros(self.env.num_envs).to(self.device)
        self.done = list()
        if kwargs.get("return_info", False):
            return {"state": nobs}, {}
        return {"state": nobs}

    def reset_arg(self, options_list=None):
        return self.reset()

    def reset_one_arg(self, env_ind=None, options=None):
        return self.reset()

    def step(self, action: np.ndarray):
        action = torch.tensor(action, device=self.device)
        action = self.normalizer(action, "actions", forward=False)

        obs, sparse_reward, dense_reward, info = self._inner_step(action)
        if self.sparse_reward:
            reward = sparse_reward.clone().cpu().numpy()
        else:
            reward = dense_reward.clone().cpu().numpy()

        truncated = self.env.env_steps >= self.max_env_steps

        nobs: np.ndarray = self.process_obs(obs)
        truncated: np.ndarray = truncated.squeeze().cpu().numpy()

        return {"state": nobs}, reward, truncated, truncated, info

    def _inner_step(self, action_chunk: torch.Tensor):
        dense_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        sparse_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        for i in range(self.n_action_steps):
            obs, reward, done, info = self.env.step(action_chunk[:, i, :])
            self.obs.append(obs)

            sparse_reward += reward.squeeze()
            self.best_reward += reward.squeeze()
            dense_reward += self.best_reward

        obs = stack_last_n_obs_dict(self.obs, self.n_obs_steps)
        return obs, sparse_reward, dense_reward, info

    def process_obs(self, obs: torch.Tensor) -> np.ndarray:
        robot_state = obs["robot_state"]
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        parts_poses = obs["parts_poses"]
        obs = torch.cat([robot_state, parts_poses], dim=-1)

        nobs = self.normalizer(obs, "observations", forward=True)
        nobs = torch.clamp(nobs, -5, 5)
        return nobs.cpu().numpy()

    def seed(self, seed=None):
        """Forward seeding to the wrapped `FurnitureRLSimEnv` when possible.

        The async vector env sends a ``seed`` command to each worker. Ensure
        the wrapper responds by forwarding the call to the inner env if it
        supports ``seed``; otherwise fall back to seeding numpy.
        """
        if hasattr(self.env, "seed") and callable(getattr(self.env, "seed")):
            try:
                return self.env.seed(seed)
            except TypeError:
                return self.env.seed()
        else:
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed()
            return None
