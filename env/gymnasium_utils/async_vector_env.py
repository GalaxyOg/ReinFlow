"""
Async vector env adapted for gymnasium API.

Mostly copied from the original `gym_utils.async_vector_env` and updated imports
to use `gymnasium` (>=1.0.0) and its modules.
"""

from typing import Optional, Union, List

import numpy as np
import multiprocessing as mp
import time
import sys
from enum import Enum
from copy import deepcopy

import gymnasium as gym
from gymnasium import logger

from .vector_env import VectorEnv
from gymnasium.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
)
from gymnasium.vector.utils import (
    create_shared_memory,
    create_empty_array,
    write_to_shared_memory,
    read_from_shared_memory,
    concatenate,
    iterate,
    CloudpickleWrapper,
    clear_mpi_env_vars,
)


__all__ = ["AsyncVectorEnv"]


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorEnv(VectorEnv):
    def __init__(
        self,
        env_fns,
        dummy_env_fn=None,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
        delay_init=False,
    ):
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        if dummy_env_fn is None:
            dummy_env_fn = env_fns[0]
        dummy_env = dummy_env_fn()
        self.metadata = dummy_env.metadata

        self.n_envs = len(env_fns)
        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
        dummy_env.close()
        del dummy_env
        super().__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    self.single_observation_space, _obs_buffer, n=self.num_envs
                )
            except CustomSpaceError:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` "
                    "is incompatible with non-standard Gym observation spaces "
                    "(i.e. custom spaces inheriting from `gym.Space`), and is "
                    "only compatible with default Gym spaces (e.g. `Box`, "
                    "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
                    "if you use custom observation spaces."
                )
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = _worker_shared_memory if self.shared_memory else _worker
        target = worker or target
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()
                if delay_init:
                    time.sleep(0.1)

        self._state = AsyncState.DEFAULT

    def seed(self, seed=None):
        super().seed(seed=seed)
        self._assert_is_running()
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `seed` while waiting for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe, seed in zip(self.parent_pipes, seed):
            pipe.send(("seed", seed))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        self._assert_is_running()

        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
                self._state.value,
            )

        for pipe, single_seed in zip(self.parent_pipes, seed):
            single_kwargs = {}
            if single_seed is not None:
                single_kwargs["seed"] = single_seed
            if return_info:
                single_kwargs["return_info"] = return_info
            if options is not None:
                single_kwargs["options"] = options

            pipe.send(("reset", single_kwargs))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(
        self,
        timeout=None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `reset_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if return_info:
            results, infos = zip(*results)
            infos = list(infos)

            if not self.shared_memory:
                self.observations = concatenate(
                    self.single_observation_space, results, self.observations
                )

            return (
                deepcopy(self.observations) if self.copy else self.observations,
                infos,
            )
        else:
            if not self.shared_memory:
                self.observations = concatenate(
                    self.single_observation_space, results, self.observations
                )

            return deepcopy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        actions = iterate(self.action_space, actions)
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout=None):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `step_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        observations_list, rewards, terminates, truncates, infos = zip(*results)

        if not self.shared_memory:
            self.observations = concatenate(
                self.single_observation_space,
                observations_list,
                self.observations,
            )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(terminates, dtype=np.bool_),
            np.array(truncates, dtype=np.bool_),
            infos,
        )

    def call_async(self, name, *args, **kwargs):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting for a pending call to another method.",
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout=None):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def set_attr(self, name, values):
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError("Values must be a list or tuple with length equal to the number of environments.")

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `set_attr` while waiting for a pending call to another method.",
                self._state.value,
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def close_extras(self, timeout=None, terminate=False):
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    f"Calling `close` while waiting for a pending call to `{self._state.value}`."
                )
                function = getattr(self, f"{self._state.value}_wait")
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _check_spaces(self):
        self._assert_is_running()
        spaces = (self.single_observation_space, self.single_action_space)
        for pipe in self.parent_pipes:
            pipe.send(("_check_spaces", spaces))
        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        same_observation_spaces, same_action_spaces = zip(*results)
        if not all(same_observation_spaces):
            raise RuntimeError("Some environments have an observation space different from the single observation space.")
        if not all(same_action_spaces):
            raise RuntimeError("Some environments have an action space different from the single action space.")

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError("Trying to operate on AsyncVectorEnv after close().")

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        for _ in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(f"Received the following error from Worker-{index}: {exctype.__name__}: {value}")
            logger.error(f"Shutting down Worker-{index}.")
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

        logger.error("Raising the last exception back to the main process.")
        raise exctype(value)

    def __del__(self):
        if not getattr(self, "closed", True):
            self.close(terminate=True)

    # Convenience synchronous call helpers
    def call_sync(self, method_name, indices=None, **method_kwargs):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("_call_sync", (method_name, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def call_sync_arg(self, method_name, method_arg_name, method_arg_list, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for method_arg, remote in zip(method_arg_list, target_remotes):
            method_kwargs = {method_arg_name: method_arg}
            remote.send(("_call_sync", (method_name, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        if indices is None:
            indices = range(self.n_envs)
        return [self.parent_pipes[i] for i in indices]

    def reset_arg(self, options_list, **kwargs):
        results = self.call_sync_arg("reset", "options", options_list)
        obs = [result[0] for result in results]
        if isinstance(obs[0], np.ndarray):
            return np.stack(obs)
        else:
            assert isinstance(obs[0], dict)
            return obs

    def reset_one_arg(self, env_ind, options=None):
        obs, success = self.call_sync("reset", options=options, indices=[env_ind])[0]
        return obs

    def render(self, *args, **kwargs):
        return self.call("render", *args, **kwargs)


def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                if "return_info" in data and data["return_info"] == True:
                    observation, info = env.reset(**data)
                    pipe.send(((observation, info), True))
                else:
                    observation = env.reset(**data)
                    pipe.send((observation, True))

            elif command == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call_sync":
                function = getattr(env, data[0])
                pipe.send((function(**data[1]), True))
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(f"Trying to call function `{name}` with `_call`. Use `{name}` directly instead.")
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(**kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(((data[0] == env.observation_space, data[1] == env.action_space), True))
            else:
                raise RuntimeError(f"Received unknown command `{command}`.")
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                if "return_info" in data and data["return_info"] == True:
                    observation, info = env.reset(**data)
                    write_to_shared_memory(observation_space, index, observation, shared_memory)
                    pipe.send(((None, info), True))
                else:
                    observation = env.reset(**data)
                    write_to_shared_memory(observation_space, index, observation, shared_memory)
                    pipe.send((None, True))
            elif command == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call_sync":
                function = getattr(env, data[0])
                pipe.send((function(**data[1]), True))
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(f"Trying to call function `{name}` with `_call`. Use `{name}` directly instead.")
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(((data[0] == observation_space, data[1] == env.action_space), True))
            else:
                raise RuntimeError(f"Received unknown command `{command}`.")
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
