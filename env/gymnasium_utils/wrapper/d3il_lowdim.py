import numpy as np
import gymnasium as gym
from gymnasium import spaces


class D3ilLowdimWrapper(gym.Env):
    def __init__(self, env, normalization_path):
        self.env = env

        self.action_space = env.action_space
        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]

        self.observation_space = spaces.Dict()
        # gymnasium env.reset() returns (obs, info); handle both cases
        raw_obs = self.env.reset()
        if isinstance(raw_obs, (tuple, list)):
            obs_example = raw_obs[0]
        else:
            obs_example = raw_obs
        obs_example = np.asarray(obs_example)
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space["state"] = spaces.Box(low=low, high=high, shape=low.shape, dtype=low.dtype)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, *, seed=None, return_info: bool = False, options: dict = None):
        options = options or {}
        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)
            raw_reset = self.env.reset()
        else:
            raw_reset = self.env.reset()

        raw_obs = raw_reset[0] if isinstance(raw_reset, (tuple, list)) else raw_reset
        obs = self.normalize_obs(raw_obs)
        if return_info:
            return {"state": obs}, {}
        return {"state": obs}

    def normalize_obs(self, obs):
        return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormaliza_action(self, action):
        action = (action + 1) / 2
        return action * (self.action_max - self.action_min) + self.action_min

    def step(self, action):
        action = self.unnormaliza_action(action)
        step_res = self.env.step(action)
        if len(step_res) == 5:
            raw_obs, reward, terminated, truncated, info = step_res
        else:
            raw_obs, reward, done, info = step_res
            terminated = bool(done)
            truncated = False

        obs = self.normalize_obs(raw_obs)
        return {"state": obs}, reward, terminated, truncated, info

    def render(self, mode="rgb_array", width: int = 256, height: int = 256):
        return self.env.render(mode=mode, height=height, width=width, camera_name=self.render_camera_name)
