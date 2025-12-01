from typing import List, Union, Optional

import numpy as np
from copy import deepcopy

import gymnasium as gym
from gymnasium import logger
from gymnasium.vector.utils import concatenate, iterate, create_empty_array


__all__ = ["SyncVectorEnv"]


class SyncVectorEnv:
    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        # Initialize VectorEnv attributes directly without calling super().__init__()
        # as VectorEnv inherits from object and doesn't use super().__init__()
        from env.gymnasium_utils.vector_env import batch_space
        
        self.num_envs = len(env_fns)
        self.is_vector_env = True
        
        # Set single observation/action spaces for the underlying environments
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        
        # Create batched spaces for the vector environment
        self.observation_space = batch_space(observation_space, n=len(env_fns))
        self.action_space = batch_space(action_space, n=len(env_fns))

        self._check_spaces()
        self.observations = create_empty_array(self.single_observation_space, n=self.num_envs, fn=np.zeros)
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminates = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncates = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seed=None):
        # VectorEnv doesn't have a seed method, so we implement it directly
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def reset(self, seed: Optional[Union[int, List[int]]] = None, return_info: bool = False, options: Optional[dict] = None):
        """Reset all environments and return observations"""
        return self.reset_wait(seed=seed, return_info=return_info, options=options)

    def reset_async(self, seed: Optional[Union[int, List[int]]] = None, return_info: bool = False, options: Optional[dict] = None):
        """Asynchronous reset - immediately returns after starting reset process"""
        # For sync vector env, reset is synchronous, so we just store the args for reset_wait
        self._reset_seed = seed
        self._reset_return_info = return_info
        self._reset_options = options

    def reset_wait(self, seed: Optional[Union[int, List[int]]] = None, return_info: bool = False, options: Optional[dict] = None):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminates[:] = False
        self._truncates[:] = False
        observations = []
        data_list = []
        for env, single_seed in zip(self.envs, seed):

            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options
            if return_info == True:
                kwargs["return_info"] = return_info

            if not return_info:
                observation = env.reset(**kwargs)
                observations.append(observation)
            else:
                observation, data = env.reset(**kwargs)
                observations.append(observation)
                data_list.append(data)

        self.observations = concatenate(self.single_observation_space, observations, self.observations)
        if not return_info:
            return deepcopy(self.observations) if self.copy else self.observations
        else:
            return (deepcopy(self.observations) if self.copy else self.observations), data_list

    def step_async(self, actions):
        self._actions = iterate(self.action_space, actions)

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[i],
                self._terminates[i],
                self._truncates[i],
                info,
            ) = env.step(action)
            observations.append(observation)
            infos.append(info)
        self.observations = concatenate(self.single_observation_space, observations, self.observations)

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminates),
            np.copy(self._truncates),
            infos,
        )

    def step(self, actions):
        """Synchronous step - takes actions, advances the environment, returns results"""
        self.step_async(actions)
        return self.step_wait()

    def call(self, name, *args, **kwargs):
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name, values):
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError("Values must be a list or tuple with length equal to the number of environments.")

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        [env.close() for env in self.envs]

    def _check_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError("Some environments have an observation space different from the single observation space.")

            if not (env.action_space == self.single_action_space):
                raise RuntimeError("Some environments have an action space different from the single action space.")

        else:
            return True
