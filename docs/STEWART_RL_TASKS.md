# Stewart Platform RL Tasks

## Project Goal

This project trains a Stewart platform in Isaac Lab to stabilize falling objects. The main control objective is to move the six prismatic sliders so the top disk can catch the object and drive it toward the disk center.

Two direct-RL tasks are provided:

| Task | Object | Base motion |
| --- | --- | --- |
| `Template-Stewart-Test-Direct-v0` | Ellipsoid | Fixed base |
| `Template-Stewart-Wave-System-Direct-v0` | Sphere, radius `0.05 m` | Random continuous 6-DOF wave base |

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

The policy observation has 27 values:

```text
6  slider joint positions
6  slider joint velocities
3  top platform projected gravity
3  top platform angular velocity
3  object center position relative to top disk center
6  previous smoothed actions
```

The object velocity is intentionally not included in the policy observation. The relative position can be obtained by vision in a real system, while linear and angular velocity are not assumed to be directly measurable.

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

Simulation-only values such as object velocity are used for reward shaping, but they are not part of the policy observation.

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
