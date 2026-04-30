# Stewart Platform Isaac Lab Tasks

This repository contains Isaac Lab direct-RL environments for training a Stewart platform to stabilize falling objects.
The active training backend is RSL-RL PPO.

## Environments

| Task | Description |
| --- | --- |
| `Template-Stewart-Test-Direct-v0` | Stewart platform stabilizes a falling ellipsoid on the top disk center. |
| `Template-Stewart-Wave-System-Direct-v0` | Stewart platform is mounted on a continuously moving 6-DOF rectangular base and stabilizes a falling sphere. |

## Assets

The Stewart platform asset is stored under `assets/` and uses project-relative USD references:

```text
assets/Stewart_full.usd
assets/Stewart/Stewart/Stewart.usd
assets/Stewart/Stewart/configuration/
```

The default asset path is resolved from the project root. You can override it at runtime:

```bash
STEWART_USD_PATH=/absolute/path/to/Stewart_full.usd python scripts/rsl_rl/train.py --task <TASK_NAME>
```

## Stewart Platform Model

The Stewart platform is a closed-chain 6-DOF parallel mechanism. It is controlled through six prismatic slider joints:

```text
Slider_13, Slider_18, Slider_17, Slider_14, Slider_16, Slider_15
```

The top disk reference body is `UJ61`. The observation uses the object center position relative to the top disk center. If the USD top disk origin changes, adjust `platform_center_offset` in `stewart_test_env_cfg.py`.

Implementation choices for stability:

- Direct slider effort control is used instead of PhysX drive targets.
- Full articulation joint-state reset is disabled by default to avoid breaking closed-chain constraints.
- Articulation self-collision is disabled.
- The six sliders start with a `0.10 m` extension to provide buffer travel.

## Observation And Action

The policy observation has 27 values:

```text
6  slider joint positions
6  slider joint velocities
3  top platform projected gravity
3  top platform angular velocity
3  object center position relative to top disk center
6  previous smoothed actions
```

The action has 6 values, one per slider. Actions are clipped to `[-1, 1]`, smoothed, and mapped to force:

```text
slider_force = action * action_scale
```

## Installation

Use the Python environment that can run Isaac Lab:

```bash
cd /home/ding/Documents/stewart_test/stewart_test
python -m pip install -e .
```

## Training

Train the ellipsoid task:

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
python scripts/rsl_rl/train.py \
  --task Template-Stewart-Test-Direct-v0 \
  --num_envs 1024 \
  --headless
```

Train the wave-base sphere task:

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
python scripts/rsl_rl/train.py \
  --task Template-Stewart-Wave-System-Direct-v0 \
  --num_envs 1024 \
  --headless
```

For visual debugging:

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
python scripts/rsl_rl/train.py \
  --task Template-Stewart-Wave-System-Direct-v0 \
  --num_envs 16
```

## Utilities

Inspect the USD model:

```bash
python scripts/load_usd_model.py --usd_path assets/Stewart_full.usd
```

List registered tasks:

```bash
python scripts/list_envs.py --keyword Stewart
```

Run random actions:

```bash
python scripts/random_agent.py --task Template-Stewart-Test-Direct-v0 --num_envs 16
```

## Notes

Warnings about `UsdPreviewSurface.mdl` or mesh collision fallback can appear with the imported CAD/USD asset. They do not necessarily stop training. A hard failure such as `Failed to find an articulation` usually means a USD reference path is broken.

## Documentation

- `docs/STEWART_RL_TASKS.md`: Stewart RL tasks, observations, actions, rewards, and training commands.
- `docs/USD_ASSET_CHECK.md`: USD reference inspection and repair workflow.
- `docs/TUNING_NOTES.md`: Tuning history and practical debugging notes.
