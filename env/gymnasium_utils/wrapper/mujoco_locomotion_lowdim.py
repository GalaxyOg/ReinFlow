import numpy as np
import gymnasium as gym
from gymnasium import spaces
import imageio


class MujocoLocomotionLowdimWrapper(gym.Env):
    def __init__(self, env, normalization_path):
        self.env = env
        self.video_writer = None

        self.action_space = env.action_space
        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]

        self.observation_space = spaces.Dict()
        # gymnasium env.reset() returns (obs, info); handle both cases
        raw_obs = self.env.reset()
        if isinstance(raw_obs, tuple) or isinstance(raw_obs, list):
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
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)
        raw_reset = self.env.reset()
        raw_obs = raw_reset[0] if isinstance(raw_reset, (tuple, list)) else raw_reset
        obs = self.normalize_obs(raw_obs)
        if return_info:
            return {"state": obs}, {}
        return {"state": obs}

    def normalize_obs(self, obs):
        return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormalize_action(self, action):
        action = (action + 1) / 2
        return action * (self.action_max - self.action_min) + self.action_min

    def step(self, action):
        raw_action = self.unnormalize_action(action)
        step_res = self.env.step(raw_action)
        if len(step_res) == 5:
            raw_obs, reward, terminated, truncated, info = step_res
        else:
            raw_obs, reward, done, info = step_res
            terminated = bool(done)
            truncated = False
        obs = self.normalize_obs(raw_obs)

        if self.video_writer is not None:
            video_img = self.env.render()
            self.video_writer.append_data(video_img)

        return {"state": obs}, reward, terminated, truncated, info

    def render(self):
        return self.env.render()
