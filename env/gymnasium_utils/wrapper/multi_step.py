"""
Multi-step wrapper adapted to gymnasium.
"""

import gymnasium as gym
from typing import Optional
from gymnasium import spaces
import numpy as np
from collections import defaultdict, deque


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max"):
    if method == "max":
        return np.max(data)
    elif method == "min":
        return np.min(data)
    elif method == "mean":
        return np.mean(data)
    elif method == "sum":
        return np.sum(data)
    else:
        raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        result[:start_idx] = result[start_idx]
    return result


class MultiStep(gym.Wrapper):
    def __init__(
        self,
        env,
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=None,
        reward_agg_method="sum",
        prev_action=True,
        reset_within_step=False,
        pass_full_observations=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(env)
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.prev_action = prev_action
        self.reset_within_step = reset_within_step
        self.pass_full_observations = pass_full_observations
        self.verbose = verbose

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: dict = None):
        options = options or {}
        obs = self.env.reset(seed=seed, options=options, return_info=return_info)
        if return_info:
            obs, info = obs
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        if self.prev_action:
            self.action = deque([self._single_action_space.sample()], maxlen=self.n_obs_steps)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        obs = self._get_obs(self.n_obs_steps)

        self.cnt = 0
        if return_info:
            return obs, info
        return obs

    def step(self, action):
        if action.ndim == 1:
            action = action[None]
        truncated = False
        terminated = False
        for act_step, act in enumerate(action):
            self.cnt += 1
            if terminated or truncated:
                break
            step_res = self.env.step(act)
            # Support both Gym (obs, reward, done, info) and Gymnasium
            # (obs, reward, terminated, truncated, info)
            if len(step_res) == 5:
                observation, reward, terminated, truncated, info = step_res
            else:
                observation, reward, done, info = step_res
                terminated = bool(done)
                truncated = False

            self.obs.append(observation)
            self.action.append(act)
            self.reward.append(reward)

            if "TimeLimit.truncated" not in info:
                # If environment didn't provide TimeLimit info, apply
                # our own max_episode_steps truncation logic.
                if not terminated and (self.max_episode_steps is not None) and self.cnt >= self.max_episode_steps:
                    truncated = True
            else:
                truncated = info["TimeLimit.truncated"]
                # If original env used a 4-tuple API, `terminated` was set
                # above from `done`.

            done = truncated or terminated
            self.done.append(done)
            self._add_info(info)
        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done_agg = aggregate(self.done, "max")
        info = dict_take_last_n(self.info, self.n_obs_steps)
        if self.pass_full_observations:
            info["full_obs"] = self._get_obs(act_step + 1)

        if self.reset_within_step and self.done[-1]:
            if truncated:
                info["final_obs"] = observation
            observation = self.reset()
            self.verbose and print("Reset env within wrapper.")

        self.reward = list()
        self.done = list()
        return observation, reward, terminated, truncated, info

    def _get_obs(self, n_steps=1):
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
            return result
        else:
            raise RuntimeError("Unsupported space type")

    def get_prev_action(self, n_steps=None):
        if n_steps is None:
            n_steps = self.n_obs_steps - 1
        assert len(self.action) > 0
        return stack_last_n_obs(self.action, n_steps)

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def seed(self, seed=None):
        """Forward seeding to the wrapped environment when possible.

        Some environments provide a ``seed`` method while others rely on
        ``reset(seed=...)``. The async vector worker sends a ``seed``
        command to each env, so wrappers must implement this method and
        forward it to the inner env to avoid AttributeError.
        """
        if hasattr(self.env, "seed") and callable(getattr(self.env, "seed")):
            try:
                return self.env.seed(seed)
            except TypeError:
                # Some envs implement seed() without arguments
                return self.env.seed()
        else:
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed()
            return None
