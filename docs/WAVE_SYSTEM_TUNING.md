# Wave System Task Tuning Record

This document captures the key design decisions and tuning iterations for the wave-base task:

- Task id: `Template-Stewart-Wave-System-Direct-v0`
- Code: `source/stewart_test/stewart_test/tasks/direct/stewart_wave_system/`

Goal: **match the fixed-base task behavior** (catch + stabilize near disk center with smooth control) while the whole
Stewart system is mounted on a continuously moving 6-DoF wave base.

Current strategy: use an in-environment curriculum so the policy first learns to catch and stabilize near the disk
center, then gradually sees harder drops and wave disturbances.

## 1. Why the wave task is harder

The wave task is not just “add noise”. It changes the system dynamics:

- Stewart articulation **root pose/velocity is overwritten every step** to follow a random sinusoidal 6-DoF motion.
- The policy must do **feed-forward compensation** to reject base motion.
- Termination/reward terms that are correct for a fixed base may become **incorrect** because they penalize mandated base motion.

## 2. Observations (policy input)

### 2.1 Fixed-base task (`Template-Stewart-Test-Direct-v0`)

Current policy observation is **31D**:

```text
6   slider joint positions
6   slider joint velocities
3   top-disk projected gravity
3   top-disk angular velocity
3   ball position relative to top-disk center (world)
3   ball linear velocity (world)
6   previous smoothed actions
1   curriculum progress
```

Implementation: `stewart_test_env.py::_get_observations`.

Rationale:

- Ball relative xyz and linear velocity are assumed observable in deployment (vision / LiDAR fusion).
- Ball velocity enables **predictive catching** instead of reactive “wait until contact”.

### 2.2 Wave-base task (`Template-Stewart-Wave-System-Direct-v0`)

Wave task augments the base observation with **root motion cues** for feed-forward control.

Current policy observation is **46D**:

```text
31  base observation (same as fixed-base)
3   root projected gravity (commanded wave root)
6   commanded wave pose: x, y, z, roll, pitch, yaw
6   commanded wave velocity: vx, vy, vz, roll_rate, pitch_rate, yaw_rate
```

Implementation: `stewart_wave_system_env.py::_get_observations`.

The extra wave pose/velocity terms are important because translation disturbances are not observable from root
projected gravity alone. They give the policy feed-forward information similar to how legged locomotion policies use
terrain or command observations.

### 2.3 IMU-only wave task (`Template-Stewart-Wave-System-IMU-Direct-v0`)

For deployment realism, an IMU-only variant is provided:

```text
31  base observation (same as fixed-base)
3   root projected gravity (IMU gravity direction)
3   root angular velocity in root frame (gyro)
3   root linear acceleration in root frame (accelerometer)
```

Total IMU-wave observation size: **40**.

Implementation:

- task registration: `stewart_wave_system_imu/__init__.py`
- env logic: `stewart_wave_system_imu_env.py`
- cfg: `stewart_wave_system_imu_env_cfg.py`

This task removes direct commanded wave pose/velocity from observations and replaces them with simulated IMU signals.

## 3. Rewards: avoid penalizing base-mandated motion

### 3.1 Residual stabilization vs wave base

For wave task, “platform stability” rewards should not punish mandated base rotation. Therefore wave task computes:

- residual angular velocity: \( \omega_{disk} - \omega_{root} \)
- residual flatness: gravity projection in disk frame **relative to root frame**

Implementation: `stewart_wave_system_env.py::_get_rewards` (calls shared `compute_rewards` with residual quantities).

Recent update: residual orientation now uses **simulated root quaternion**
(`robot.data.root_quat_w`) instead of commanded cached quaternion to avoid command/sim skew.

### 3.2 Energy / jitter penalties are gated by phase

Pure always-on action penalties can create a local optimum: “do nothing”.

We gate penalties using `near_platform` (based on ball height relative to target):

- when the ball is far: allow large motions to intercept
- near the target: enforce smooth/energy-efficient stabilization

Implementation: `stewart_test_env.py::compute_rewards` (`penalty_gate`).

### 3.3 Predictive interception shaping

To avoid “wait for the ball”, we add an interception reward when the ball is still high and descending:

- predict impact time to the target height \(t \approx \Delta z / |v_z|\) (clamped)
- predict XY at that time
- reward small predicted XY error

Implementation: `stewart_test_env.py::compute_rewards` (`intercept` term).

Key cfg knobs (in `stewart_test_env_cfg.py`):

- `rew_scale_intercept`
- `intercept_z_gate`
- `intercept_time_max`
- `intercept_xy_sigma`

## 4. Termination: prevent false failures under base motion

Two fixed-base failure checks caused systematic early termination in wave mode:

1) `object_far` based on world-frame relative XY can trigger while the ball is still high because the disk center is moving.
2) `platform_tilted` based on absolute disk gravity projection can trigger because the wave base mandates tilt.

Wave task overrides `_get_dones` to:

- gate `object_far` until the ball is closer vertically (`object_far_rel_height_gate`)
- compute tilt using **residual tilt vs wave root**

Implementation: `stewart_wave_system_env.py::_get_dones`

## 5. Control bandwidth limiting (deployment realism)

To reduce high-frequency energy-wasting jitter:

- actions are smoothed (`action_smoothing`)
- per-step action delta is clamped (`max_action_delta`)
- optionally adaptive action delta:
  - ball high: allow larger action changes (catch phase)
  - ball near target: tighter changes (stabilize phase)

Implementation: `stewart_test_env.py::_pre_physics_step`

Key cfg knobs (in `stewart_test_env_cfg.py`):

- `max_action_delta`
- `adaptive_action_delta`, `max_action_delta_high`, `max_action_delta_low`, `adaptive_action_delta_switch_height`

## 6. Wave-task specific stability weights

Wave task needs stronger stabilization weights (residual flatness/ang vel/linear jitter/pose reg) than fixed-base.

Implementation: overrides in `stewart_wave_system_env_cfg.py`:

- `rew_scale_platform_flat`
- `rew_scale_platform_ang_vel`
- `rew_scale_platform_lin_residual`
- `rew_scale_disk_pose_reg`

### 6.1 Post-catch: persistence + “world hold” + calmer commands

Residual stabilization alone can still admit a poor local optimum after contact: **chase the ball with endless small motions** instead of absorbing the wave energetically and keeping the tray calm.

Additional shared shaping (implemented in `stewart_test_env.py::compute_rewards`, tuned in cfg):

| Term | Intent | Typical wave cfg overrides |
| --- | --- | --- |
| `on_disk` | Soft product gate on height × radial XY Gaussian (height weight ≈ **`exp(-1)` when \|Δz\|` = on_disk_height_band`**). Gives dense shaping on short failures so PPO learns persistence before perfection. | `rew_scale_on_disk`, `on_disk_height_band`, `on_disk_xy_sigma` |
| `world_center_hold` | Penalize **world-frame XY drift** of the top disk center versus a snapshot taken **after wave pose is applied on reset**. This encourages using **slider/extension** and Stewart geometry to cancel lateral base motion rather than rigidly translating the mechanism with the wave. Normalized by `world_center_hold_sigma` so tightening `sigma` increases penalty steepness without changing dimensional meaning. | `rew_scale_world_center_hold`, `world_center_hold_sigma` |
| Extra `action_rate` on disk | Add an additional action-chatter penalty gated by the same “on-disk” mask to suppress visible plate tremor after catch. | `rew_scale_action_rate_on_disk` |

Fixed-base task keeps:

- `rew_scale_world_center_hold = 0.0`
- `rew_scale_action_rate_on_disk = 0.0`

Episode length plateau debugging (wave): if **`Mean episode_length` stalls near ~100** while **`time_out≈0`**, almost all resets are **`failure`** (ground / `object_far` / tilt). Besides reward balance, wave cfg also overrides **`max_object_xy_dist`**, **`max_platform_tilt`**, and **`rew_scale_alive`** to reduce overly aggressive early exits and bias the learner toward surviving the full horizon.

## 7. Curriculum

The curriculum is configured directly in cfg classes and uses `common_step_counter / curriculum_duration_steps` as
global progress. Parallel environments do not all use exactly the same difficulty. When `curriculum_per_env=True`,
each reset samples a per-env progress from a trailing difficulty window:

```text
global_progress = common_step_counter / curriculum_duration_steps
env_progress in [global_progress - window, global_progress]
window = curriculum_env_progress_spread * (1 - global_progress)
```

This is closer to legged-locomotion terrain curricula: at any training moment, different envs cover adjacent difficulty
levels, and the window narrows as training approaches the final task.

It avoids runtime mesh swapping because PhysX collision geometry is fixed at scene creation.

Fixed-base curriculum knobs in `stewart_test_env_cfg.py`:

```text
enable_curriculum
curriculum_duration_steps
curriculum_per_env
curriculum_env_progress_spread
object_spawn_radius_start -> object_spawn_radius
object_drop_height_range_start -> object_drop_height_range
object_initial_down_velocity_start -> object_initial_down_velocity
object_spin_velocity_start -> object_spin_velocity_end
```

Wave curriculum adds staged disturbance axes in `stewart_wave_system_env_cfg.py`:

```text
wave_pos_amplitude_start -> wave_pos_amplitude
wave_rot_amplitude_start -> wave_rot_amplitude
wave_frequency_range_start -> wave_frequency_range
wave_axis_start_progress = (x, y, z, roll, pitch, yaw)
wave_axis_ramp_progress
```

Current wave order:

```text
0.00+: x translation
0.08+: y translation
0.16+: z translation
0.30+: roll
0.45+: pitch
0.60+: yaw
```

Recent wave curriculum tuning updates:

- wave task uses faster curriculum pacing (`curriculum_duration_steps=20000`, narrower spread) than fixed-base.
- non-zero start amplitudes are kept active in early training.
- amplitude computation uses:

```text
start_amp + (end_amp - start_amp) * (progress * axis_scale)
```

instead of globally multiplying by `axis_scale`, so early-stage disturbance is not accidentally zeroed out.
- additional training logs are exported:
  - `Curriculum/global_progress`
  - `Wave/mean_env_progress`
  - `Wave/mean_amp_xyz`, `Wave/mean_amp_rpy`
  - `Wave/axis_scale_x`, `Wave/axis_scale_yaw`

These metrics are intended to diagnose “curriculum looks stuck / no visible wave” issues quickly.

## 8. Play-mode behavior

`scripts/rsl_rl/play.py` defaults to **final-task difficulty** for evaluation:

- if env cfg has `enable_curriculum`, play disables it by default.
- use `--use_curriculum_in_play` to keep curriculum behavior during play.

This avoids confusion where play appears to have no wave motion because progress starts near zero.

For object-shape curriculum, the safe path is to train separate task assets or add explicit multi-object management
(spawn sphere, ellipsoid, and mesh objects up front, then reset only the active object for each env). The current
single-object environment ramps drop/spin difficulty but does not mutate collision mesh geometry at runtime.

## 9. Smoke test command

Suggested learning progression:

```text
1. fixed base + round ball
2. fixed base + ellipsoid
3. small wave + round ball
4. small wave + ellipsoid
5. larger wave + ellipsoid
```

The current code implements step 3 by using milder start amplitudes and curriculum ramping.

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
/home/ding/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Template-Stewart-Wave-System-Direct-v0 \
  --num_envs 2 \
  --headless \
  --max_iterations 1
```

Success criteria:

- environment initializes
- actor/critic shapes match observation dims (wave task: `in_features=46`)
- no early runtime errors in reward/done computation

For IMU task, actor input should be `in_features=40`.

