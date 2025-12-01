import numpy as np
import gymnasium as gym
from gymnasium import spaces
import imageio


class RobomimicLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ],
        clamp_obs=False,
        init_state=None,
        render_camera_name="agentview",
    ):
        self.env = env
        self.init_state = init_state
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs

        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = gym.spaces.Box(low=low, high=high, shape=low.shape, dtype=low.dtype)
        self.obs_keys = low_dim_keys
        self.observation_space = spaces.Dict()
        obs_example_full = self.env.get_observation()
        obs_example = np.concatenate([obs_example_full[key] for key in self.obs_keys], axis=0)
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space["state"] = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

    def normalize_obs(self, obs):
        obs = 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self, raw_obs):
        obs = {"state": np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)}
        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        return obs

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
        if self.init_state is not None:
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_reset = self.env.reset()
            raw_obs = raw_reset[0] if isinstance(raw_reset, (tuple, list)) else raw_reset
        else:
            raw_reset = self.env.reset()
            raw_obs = raw_reset[0] if isinstance(raw_reset, (tuple, list)) else raw_reset
        obs = self.get_observation(raw_obs)
        if return_info:
            return obs, {}
        return obs

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        step_res = self.env.step(action)
        if len(step_res) == 5:
            raw_obs, reward, terminated, truncated, info = step_res
        else:
            raw_obs, reward, done, info = step_res
            terminated = bool(done)
            truncated = False
        obs = self.get_observation(raw_obs)

        # If the wrapped env provides a _check_success truncation, include it
        try:
            truncated = truncated or bool(self.env.env._check_success())
        except Exception:
            pass

        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array", width: int = 256, height: int = 256):
        return self.env.render(mode=mode, height=height, width=width, camera_name=self.render_camera_name)
