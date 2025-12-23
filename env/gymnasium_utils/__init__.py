# MIT License

import os
import json

try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)


def make_async(
    env_name: str,
    num_envs=1,
    asynchronous=True,
    wrappers=None,
    render=False,
    obs_dim=23,
    action_dim=7,
    env_type=None,
    max_episode_steps=None,
    # below for furniture only
    gpu_id=0,
    headless=True,
    record=False,
    normalization_path=None,
    furniture="one_leg",
    randomness="low",
    obs_steps=1,
    act_steps=8,
    sparse_reward=False,
    # below for robomimic only
    robomimic_env_cfg_path=None,
    use_image_obs=False,
    render_offscreen=False,
    reward_shaping=False,
    shape_meta=None,
    **kwargs,
):
    """Create a vectorized environment compatible with gymnasium.

    This mirrors the previous `gym_utils.make_async` but uses `gymnasium` imports
    and conforms to gymnasium's reset/step signatures.
    """

    # Fix for library conflict in this specific environment (Mesa vs Conda)
    # Ensure libffi.so.7 is preloaded for swrast_dri.so to prevent llvmpipe/swrast failure
    ffi_path = "/usr/lib/x86_64-linux-gnu/libffi.so.7"
    if os.path.exists(ffi_path):
        current_preload = os.environ.get("LD_PRELOAD", "")
        if ffi_path not in current_preload:
            os.environ["LD_PRELOAD"] = f"{current_preload}:{ffi_path}" if current_preload else ffi_path
            print(f"Updated LD_PRELOAD for workers: {os.environ['LD_PRELOAD']}")

    if env_type == "furniture":
        from furniture_bench.envs.observation import DEFAULT_STATE_OBS
        from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
        from env.gymnasium_utils.wrapper.furniture import FurnitureRLSimEnvMultiStepWrapper
        env = FurnitureRLSimEnv(
            act_rot_repr="rot_6d",
            action_type="pos",
            april_tags=False,
            concat_robot_state=True,
            ctrl_mode="diffik",
            obs_keys=DEFAULT_STATE_OBS,
            furniture=furniture,
            gpu_id=gpu_id,
            headless=headless,
            num_envs=num_envs,
            observation_space="state",
            randomness=randomness,
            max_env_steps=max_episode_steps,
            record=record,
            pos_scalar=1,
            rot_scalar=1,
            stiffness=1_000,
            damping=200,
        )
        env = FurnitureRLSimEnvMultiStepWrapper(
            env,
            n_obs_steps=obs_steps,
            n_action_steps=act_steps,
            prev_action=False,
            reset_within_step=False,
            pass_full_observations=False,
            normalization_path=normalization_path,
            sparse_reward=sparse_reward,
        )
        return env

    # avoid import error due incompatible gym versions
    from gymnasium import spaces
    from env.gymnasium_utils.async_vector_env import AsyncVectorEnv
    from env.gymnasium_utils.sync_vector_env import SyncVectorEnv
    from env.gymnasium_utils.wrapper import wrapper_dict

    # import the envs
    if robomimic_env_cfg_path is not None:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
    elif "avoiding" in env_name:
        import gym_avoiding
    else:
        pass

    import ffsm_env
    import gymnasium as gym
    make_ = gym.make

    def _make_env():
        if robomimic_env_cfg_path is not None:
            obs_modality_dict = {
                "low_dim": (
                    wrappers.robomimic_image.low_dim_keys
                    if "robomimic_image" in wrappers
                    else wrappers.robomimic_lowdim.low_dim_keys
                ),
                "rgb": (
                    wrappers.robomimic_image.image_keys
                    if "robomimic_image" in wrappers
                    else None
                ),
            }
            if obs_modality_dict["rgb"] is None:
                obs_modality_dict.pop("rgb")
            ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
            if render_offscreen or use_image_obs:
                os.environ["MUJOCO_GL"] = "egl"
            with open(robomimic_env_cfg_path, "r") as f:
                env_meta = json.load(f)
            env_meta["reward_shaping"] = reward_shaping

            print(f"Robomimic env_meta={env_meta}")
            print(f"""Robomimic env_name={env_meta["env_name"]}""")
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=render,
                render_offscreen=render_offscreen,
                use_image_obs=use_image_obs,
            )
            env.env.hard_reset = False
        else:
            # Only pass render parameter to environments that support it
            # Most gymnasium environments like CartPole don't accept render parameter
            if render_offscreen:
                kwargs["render_mode"] = "rgb_array"
            elif render:
                kwargs["render_mode"] = "human"

            # Let Mujoco/Gymnasium decide the best backend (glfw/egl/osmesa)
            # If render_mode is rgb_array, we need to ensure backend is set correctly
            if kwargs.get("render_mode") == "rgb_array":
                # Ensure correct drivers are found (swrast exists here)
                os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
                os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"
                os.environ["LIBGL_DRIVERS_PATH"] = "/usr/lib/x86_64-linux-gnu/dri"

                # Force EGL + Software Rendering (proven to work in pretrain if LD_PRELOAD is correct)
                # This works headless and avoids Xvfb/GLFW issues
                if "MUJOCO_GL" not in os.environ:
                    os.environ["MUJOCO_GL"] = "egl"
                
                if os.environ["MUJOCO_GL"] == "egl":
                    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

            if render and "kitchen" in env_name:
                kwargs["render"] = render
            if "Humanoid" in env_name:
                print(f"make humanoid!")
                env = make_("Humanoid-v3")
            else:
                print(f'Making gymnasium environment id={env_name}')
                env = make_(env_name, **kwargs)

        if wrappers is not None:
            for wrapper, args in wrappers.items():
                env = wrapper_dict[wrapper](env, **args)

        return env

    def dummy_env_fn():
        # import d4rl
        import gymnasium as gym
        import numpy as np
        import sys
        import os
        from env.gymnasium_utils.wrapper.multi_step import MultiStep
        
        # Ensure obs_dim is valid
        current_obs_dim = obs_dim
        if not current_obs_dim:
            current_obs_dim = 48  # Default for FFSMEnv6dof-v0
            
        env = gym.Env()

        if shape_meta is not None:
            observation_space = spaces.Dict()
            for key, value in shape_meta["obs"].items():
                shape = value["shape"]
                if key.endswith("rgb"):
                    min_value, max_value = -1, 1
                elif key.endswith("state"):
                    min_value, max_value = -1, 1
                else:
                    raise RuntimeError(f"Unsupported type {key}")
                observation_space[key] = spaces.Box(
                    low=min_value,
                    high=max_value,
                    shape=shape,
                    dtype=np.float32,
                )
        else:
            observation_space = spaces.Dict({
                "state": spaces.Box(
                    -1,
                    1,
                    shape=(current_obs_dim,),
                    dtype=np.float32,
                )
            })
        env.observation_space = observation_space
        env.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,), dtype=np.float32)
        env.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": 12,
        }
        # Safely extract n_obs_steps from wrappers which may be a dict
        n_obs_steps = obs_steps
        if isinstance(wrappers, dict):
            multi_cfg = wrappers.get("multi_step")
            if isinstance(multi_cfg, dict):
                try:
                    n_obs_steps = int(multi_cfg.get("n_obs_steps", n_obs_steps))
                except Exception:
                    n_obs_steps = obs_steps
        
        if n_obs_steps < 1:
            n_obs_steps = 1
            
        env = MultiStep(env=env, n_obs_steps=n_obs_steps)
        # print(f"DEBUG: dummy_env observation_space: {env.observation_space}", file=sys.stderr)
        return env

    env_fns = [_make_env for _ in range(num_envs)]
    
    # Define exports
    __all__ = [
        "AsyncVectorEnv",
        "SyncVectorEnv", 
        "VectorEnv",
        "VectorEnvWrapper",
        "make_async",
    ]
    
    if asynchronous:
        return AsyncVectorEnv(
            env_fns,
            dummy_env_fn=(dummy_env_fn if render or render_offscreen or use_image_obs else None),
            delay_init="avoiding" in env_name,
            shared_memory=False,
        )
    else:
        # Create a temporary environment to get observation_space and action_space
        temp_env = _make_env()
        observation_space = temp_env.observation_space
        action_space = temp_env.action_space
        temp_env.close()
        
        return SyncVectorEnv(
            env_fns,
            observation_space=observation_space,
            action_space=action_space
        )
