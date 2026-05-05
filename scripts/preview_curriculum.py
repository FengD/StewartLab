#!/usr/bin/env python3

"""Preview Stewart curriculum stages in Isaac Sim without training a policy.

This script shortens the curriculum duration and periodically resets the scene so
you can visually inspect whether object drops and wave disturbances ramp as expected.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_SOURCE_PATH = os.path.join(PROJECT_ROOT, "source", "stewart_test")
if PROJECT_SOURCE_PATH not in sys.path:
    sys.path.insert(0, PROJECT_SOURCE_PATH)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview Stewart task curriculum in Isaac Sim.")
parser.add_argument("--task", type=str, default="Template-Stewart-Wave-System-Direct-v0", help="Task id to preview.")
parser.add_argument("--num_envs", type=int, default=6, help="Number of environments to visualize.")
parser.add_argument("--steps", type=int, default=1200, help="Number of simulation steps to run.")
parser.add_argument(
    "--curriculum_steps",
    type=int,
    default=600,
    help="Compressed curriculum duration in env steps for preview.",
)
parser.add_argument(
    "--reset_interval",
    type=int,
    default=120,
    help="Reset all environments at this interval so new curriculum levels are sampled.",
)
parser.add_argument("--print_interval", type=int, default=60, help="Print curriculum/wave state at this interval.")
parser.add_argument("--preview_envs", type=int, default=6, help="Maximum number of env difficulty rows to print.")
parser.add_argument("--random_actions", action="store_true", help="Use random actions instead of zero actions.")
parser.add_argument("--real_time", action="store_true", help="Throttle stepping to environment real time.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import stewart_test.tasks  # noqa: F401


def _format_tuple(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{value:.3f}" for value in values) + ")"


def _lerp(start: float, end: float, progress: float) -> float:
    return float(start) + (float(end) - float(start)) * progress


def _lerp_pair(start: tuple[float, float], end: tuple[float, float], progress: float) -> tuple[float, float]:
    return (_lerp(start[0], end[0], progress), _lerp(start[1], end[1], progress))


def _lerp_triple(
    start: tuple[float, float, float], end: tuple[float, float, float], progress: float
) -> tuple[float, float, float]:
    return (
        _lerp(start[0], end[0], progress),
        _lerp(start[1], end[1], progress),
        _lerp(start[2], end[2], progress),
    )


def _axis_scale(starts: tuple[float, ...], ramp: float, progress: float) -> tuple[float, ...]:
    ramp = max(float(ramp), 1.0e-6)
    return tuple(max(0.0, min((progress - float(start)) / ramp, 1.0)) for start in starts)


def _print_curriculum_state(env, step: int) -> None:
    unwrapped = env.unwrapped
    cfg = unwrapped.cfg
    global_progress = unwrapped._get_curriculum_progress() if hasattr(unwrapped, "_get_curriculum_progress") else 1.0
    if hasattr(unwrapped, "_get_curriculum_progress_tensor"):
        progress_tensor = unwrapped._get_curriculum_progress_tensor().squeeze(-1).detach().cpu()
    else:
        progress_tensor = torch.ones(unwrapped.num_envs)

    if unwrapped.num_envs <= args_cli.preview_envs:
        env_indices = list(range(unwrapped.num_envs))
    else:
        env_indices = torch.linspace(0, unwrapped.num_envs - 1, args_cli.preview_envs).round().long().tolist()

    print(
        f"[preview] step={step:05d} global_progress={global_progress:.3f} "
        f"env_progress_range=({_format_tuple((float(progress_tensor.min()), float(progress_tensor.max())))[1:-1]})",
        flush=True,
    )

    for env_id in env_indices:
        progress = float(progress_tensor[env_id])
        spawn_radius = _lerp(cfg.object_spawn_radius_start, cfg.object_spawn_radius, progress)
        height_range = _lerp_pair(cfg.object_drop_height_range_start, cfg.object_drop_height_range, progress)
        down_velocity = _lerp(cfg.object_initial_down_velocity_start, cfg.object_initial_down_velocity, progress)
        spin_velocity = _lerp(cfg.object_spin_velocity_start, cfg.object_spin_velocity_end, progress)

        msg = (
            f"  env_{env_id:04d}: progress={progress:.3f} spawn_radius={spawn_radius:.3f} "
            f"height={_format_tuple(height_range)} down_vel={down_velocity:.3f} spin={spin_velocity:.3f}"
        )

        if hasattr(cfg, "wave_axis_start_progress"):
            pos_amp = _lerp_triple(cfg.wave_pos_amplitude_start, cfg.wave_pos_amplitude, progress)
            rot_amp = _lerp_triple(cfg.wave_rot_amplitude_start, cfg.wave_rot_amplitude, progress)
            freq_range = _lerp_pair(cfg.wave_frequency_range_start, cfg.wave_frequency_range, progress)
            scales = _axis_scale(cfg.wave_axis_start_progress, cfg.wave_axis_ramp_progress, progress)
            msg += (
                f" wave_pos_amp={_format_tuple(pos_amp)} wave_rot_amp={_format_tuple(rot_amp)} "
                f"freq={_format_tuple(freq_range)} axis_scale={_format_tuple(scales)}"
            )

        print(msg, flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, _agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.curriculum_duration_steps = args_cli.curriculum_steps
    env_cfg.enable_curriculum = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    del obs

    dt = env.unwrapped.step_dt
    action_shape = (env.unwrapped.num_envs, env.unwrapped.cfg.action_space)
    _print_curriculum_state(env, 0)

    for step in range(1, args_cli.steps + 1):
        start_time = time.time()

        if args_cli.random_actions:
            actions = 0.25 * torch.randn(action_shape, device=env.unwrapped.device)
        else:
            actions = torch.zeros(action_shape, device=env.unwrapped.device)

        env.step(actions)

        if args_cli.reset_interval > 0 and step % args_cli.reset_interval == 0:
            env.reset()
        if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
            _print_curriculum_state(env, step)

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if not simulation_app.is_running():
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
