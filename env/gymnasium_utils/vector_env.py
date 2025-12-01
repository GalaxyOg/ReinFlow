from typing import Optional, Union, List

import numpy as np
import gymnasium as gym
from gymnasium.logger import deprecation
from gymnasium import spaces


def batch_space(space: spaces.Space, n: int) -> spaces.Space:
    """Create a batched version of a Gymnasium space with leading dim n.

    Supports Box, Dict and Tuple. This is a lightweight replacement for
    `gymnasium.vector.utils.spaces.batch_space` to avoid importing an internal
    utils module that may not be present in some gymnasium releases.
    """
    if isinstance(space, spaces.Box):
        low = np.repeat(np.expand_dims(space.low, 0), n, axis=0)
        high = np.repeat(np.expand_dims(space.high, 0), n, axis=0)
        return spaces.Box(low=low, high=high, dtype=space.dtype)
    elif isinstance(space, spaces.Dict):
        return spaces.Dict({k: batch_space(s, n) for k, s in space.spaces.items()})
    elif isinstance(space, spaces.Tuple):
        return spaces.Tuple(tuple(batch_space(s, n) for s in space.spaces))
    elif isinstance(space, spaces.Discrete):
        # For Discrete spaces, the batched space is still Discrete with same n_actions
        # because each environment in the vector chooses actions independently
        return space
    else:
        # Fallback: try to wrap as a Box-like array by repeating low/high if available
        try:
            low = getattr(space, "low", None)
            high = getattr(space, "high", None)
            if low is not None and high is not None:
                low = np.repeat(np.expand_dims(low, 0), n, axis=0)
                high = np.repeat(np.expand_dims(high, 0), n, axis=0)
                return spaces.Box(low=low, high=high, dtype=getattr(space, "dtype", np.float32))
        except Exception:
            pass
        raise TypeError(f"Unsupported space type for batching: {type(space)}")


__all__ = ["VectorEnv"]


class VectorEnv:
    def __init__(self, num_envs, observation_space, action_space):
        # VectorEnv inherits from object implicitly via Python's default behavior
        # Do not call super().__init__() as object.__init__() doesn't accept parameters
        
        self.num_envs = num_envs
        self.is_vector_env = True
        self.observation_space = batch_space(observation_space, n=num_envs)
        self.action_space = batch_space(action_space, n=num_envs)

        self.closed = False
        self.viewer = None

        self.single_observation_space = observation_space
        self.single_action_space = action_space

    def reset_async(self, seed: Optional[Union[int, List[int]]] = None, return_info: bool = False, options: Optional[dict] = None):
        pass

    def reset_wait(self, seed: Optional[Union[int, List[int]]] = None, return_info: bool = False, options: Optional[dict] = None):
        raise NotImplementedError()

    def reset(self, *, seed: Optional[Union[int, List[int]]] = None, return_info: bool = False, options: Optional[dict] = None):
        self.reset_async(seed=seed, return_info=return_info, options=options)
        return self.reset_wait(seed=seed, return_info=return_info, options=options)

    def step_async(self, actions):
        pass

    def step_wait(self, **kwargs):
        raise NotImplementedError()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def call_async(self, name, *args, **kwargs):
        pass

    def call_wait(self, **kwargs):
        raise NotImplementedError()

    def call(self, name, *args, **kwargs):
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def get_attr(self, name):
        return self.call(name)

    def set_attr(self, name, values):
        raise NotImplementedError()

    def close_extras(self, **kwargs):
        pass

    def close(self, **kwargs):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras(**kwargs)
        self.closed = True

    def seed(self, seed=None):
        deprecation("Function `env.seed(seed)` is marked as deprecated and will be removed in the future. Please use `env.reset(seed=seed)` instead in VectorEnvs.")

    def __del__(self):
        if not getattr(self, "closed", True):
            self.close()

    def __repr__(self):
        if getattr(self, "spec", None) is None:
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"


class VectorEnvWrapper(VectorEnv):
    def __init__(self, env):
        assert isinstance(env, VectorEnv)
        self.env = env

    def reset_async(self, **kwargs):
        return self.env.reset_async(**kwargs)

    def reset_wait(self, **kwargs):
        return self.env.reset_wait(**kwargs)

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        return self.env.step_wait()

    def close(self, **kwargs):
        return self.env.close(**kwargs)

    def close_extras(self, **kwargs):
        return self.env.close_extras(**kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __repr__(self):
        return f"<{self.__class__.__name__}, {self.env}>"
