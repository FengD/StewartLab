# Stewart Platform RL Tasks

## Project Goal

This project trains a Stewart platform in Isaac Lab to stabilize falling objects. The main control objective is to move the six prismatic sliders so the top disk can catch the object and drive it toward the disk center.

Three direct-RL tasks are provided:

| Task | Object | Base motion |
| --- | --- | --- |
| `Template-Stewart-Test-Direct-v0` | Ellipsoid | Fixed base |
| `Template-Stewart-Wave-System-Direct-v0` | Sphere, radius `0.05 m` | Random continuous 6-DOF wave base |
| `Template-Stewart-Wave-System-IMU-Direct-v0` | Sphere, radius `0.05 m` | Same as wave task, IMU-only disturbance observation |

Both tasks are compatible with `scripts/rsl_rl/train.py`.

## Stewart Platform Structure

The Stewart platform is a 6-DOF parallel mechanism. It contains a fixed lower structure, six extensible legs, and an upper moving disk. Each leg includes revolute joints and one prismatic slider joint.

The actuated joints are:

```text
Slider_13, Slider_18, Slider_17, Slider_14, Slider_16, Slider_15
```

The platform is configured as an Isaac Lab `Articulation`. Because this is a closed-chain mechanism, the environment avoids resetting all joint positions every episode. This prevents the solver from receiving inconsistent closed-loop joint states.

## Assets

The platform USD is stored in the project:

```text
assets/Stewart_full.usd
assets/Stewart/Stewart/Stewart.usd
assets/Stewart/Stewart/configuration/
```

`Stewart_full.usd` references the inner USD asset with a project-relative path. Keep these files together when moving the repository.

The default path is computed from the Python file location. Runtime override is supported:

```bash
STEWART_USD_PATH=/path/to/Stewart_full.usd python scripts/rsl_rl/train.py --task <TASK_NAME>
```

## Control

The policy outputs six actions. Each action controls one slider through direct effort:

```text
effort = action * action_scale
```

This avoids PhysX reduced-coordinate articulation drive-target limitations seen when using position targets with GPU direct API enabled.

The action is smoothed before application:

```text
processed_action = action_smoothing * action + (1 - action_smoothing) * previous_processed_action
```

## Observation

The policy observation is a vector (no images).

Fixed-base task (`Template-Stewart-Test-Direct-v0`) observation has **31** values:

```text
6  slider joint positions
6  slider joint velocities
3  top platform projected gravity
3  top platform angular velocity
3  object center position relative to top disk center
3  object linear velocity (world)
6  previous smoothed actions
1  curriculum progress
```

Wave-base task (`Template-Stewart-Wave-System-Direct-v0`) augments the above with commanded base-motion cues:

```text
3  wave/root projected gravity
6  commanded wave pose: x, y, z, roll, pitch, yaw
6  commanded wave velocity: vx, vy, vz, roll_rate, pitch_rate, yaw_rate
```

Total wave observation size: **46**.

IMU-only wave task (`Template-Stewart-Wave-System-IMU-Direct-v0`) replaces commanded wave pose/velocity with IMU-like root measurements:

```text
3  root projected gravity
3  root angular velocity in root frame (gyro)
3  root linear acceleration in root frame (accelerometer)
```

Total IMU-wave observation size: **40**.

The relative position is:

```text
object_rel_pos = object_center_world - platform_center_world
```

`platform_center_world` is based on body `UJ61` plus `platform_center_offset`.

## Rewards

The reward encourages:

- object center close to the top disk center,
- object motion toward the disk center after contact or near-contact,
- reasonable object height near the top disk,
- low object linear and angular velocity near the platform,
- moderate platform tilt,
- smooth slider motion and moderate action magnitude.

Object linear velocity is part of the policy observation for predictive catching. Other simulation-only quantities may
still be used for reward shaping.

### Wave task specific notes

- Rewards related to platform stability (flatness / angular velocity) are computed as **residuals w.r.t. wave root motion**.
- Residual orientation reference uses simulated root state (`robot.data.root_quat_w`) to avoid command/sim mismatch.
- Termination checks are also wave-aware to avoid false failures while the ball is still high.
- A predictive interception shaping term is used to encourage proactive catching (see `docs/WAVE_SYSTEM_TUNING.md`).
- Curriculum starts from easier drops / smaller disturbances and ramps toward the configured final task.
- Parallel envs use a per-env difficulty window, so a 1024-env run contains multiple adjacent curriculum levels at once.

## Wave System Task

`Template-Stewart-Wave-System-Direct-v0` adds a kinematic rectangular base below the Stewart platform. The base and Stewart root follow a continuous random 6-DOF sinusoidal trajectory:

```text
x, y, z translation
roll, pitch, yaw rotation
```

The system is lifted by `system_z_offset` to keep the base above the ground plane. The current random wave parameters are defined in `stewart_wave_system_env_cfg.py`:

```text
wave_pos_amplitude
wave_rot_amplitude
wave_frequency_range
wave_axis_start_progress
wave_axis_ramp_progress
```

## Training Commands

Fixed-base ellipsoid task:

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
python scripts/rsl_rl/train.py \
  --task Template-Stewart-Test-Direct-v0 \
  --num_envs 1024 \
  --headless
```

Wave-base sphere task:

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
python scripts/rsl_rl/train.py \
  --task Template-Stewart-Wave-System-Direct-v0 \
  --num_envs 1024 \
  --headless
```

Wave-base IMU-only task:

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
python scripts/rsl_rl/train.py \
  --task Template-Stewart-Wave-System-IMU-Direct-v0 \
  --num_envs 1024 \
  --headless
```

For debugging, use fewer environments and render:

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
python scripts/rsl_rl/train.py \
  --task Template-Stewart-Wave-System-Direct-v0 \
  --num_envs 16
```

## Troubleshooting

- `Failed to find an articulation`: USD reference path is likely broken.
- Platform barely moves: check `action_scale`, actuator effort limits, and PhysX GPU direct API warnings.
- Platform explodes or tunnels: reduce action scale, lower stiffness/effort, or avoid full joint-state reset.
- Object starts off target: reduce `object_spawn_radius`.
