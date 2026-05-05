# Stewart Platform Model Structure (USD)

This document is a *human-readable* summary of the Stewart USD model structure as used by this repository.
It intentionally avoids dumping long loader logs. If you need to re-check the USD content, use:

- `scripts/load_usd_model.py`
- `scripts/check_usd.py`
- Isaac Sim UI (Physics Inspector)

## Key bodies / frames used by RL

- **Articulation root**: `base_link` (root of the articulation in the USD)
- **Top disk reference body**: `UJ61`
  - RL computes `platform_center_pos` from this body plus `platform_center_offset`
- **Top disk surface center offset**: configured in `stewart_test_env_cfg.py` as `platform_center_offset`

## Actuated joints (policy actions)

The RL policy controls six prismatic slider joints via direct effort:

```text
Slider_13, Slider_18, Slider_17, Slider_14, Slider_16, Slider_15
```

## High-level kinematic chain (per leg)

Each leg follows the pattern:

```text
base_link -> (revolute) -> bottom link -> (revolute) -> cylinder -> (prismatic slider) -> rod
        -> (revolute) -> piston -> (revolute) -> top link -> (revolute) -> upper joint group
```

The six legs connect into a closed-chain structure. Because of this, the RL environments avoid resetting the full joint
state every episode (see `reset_robot_joint_state` in `stewart_test_env_cfg.py`).

## Notes for task implementation

- **Top disk stability** is monitored via body `UJ61` (gravity projection, angular velocity, linear velocity).
- In the wave-base task, the articulation root pose is overwritten each step to follow a 6-DoF sinusoidal trajectory
  (see `stewart_wave_system_env.py::_apply_base_wave_motion`).
