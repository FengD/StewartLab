# Stewart Centering Diagnostics (2026-05-05)

This note records the quick centering-diagnostic runs requested in the plan.
All runs use `seed=42` and short training for fast iteration.

## 1) Baseline and reward A/B

Task: `Template-Stewart-Test-Direct-v0`

| Run | Main change | Num envs | Iters | Mean reward | center | center_velocity | on_disk | action_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | none | 256 | 20 | 1625.21 | 36.43 | -1.76 | 12.34 | -0.0037 |
| rewardA | `rew_scale_center=16`, `rew_scale_center_velocity=4` | 256 | 20 | 1713.22 | 59.41 | -4.00 | 15.62 | -0.0045 |
| rewardB | `rew_scale_action=-0.0005`, `rew_scale_action_rate=-0.015` | 256 | 20 | 1631.59 | 31.66 | -2.69 | 12.40 | -0.0018 |

Conclusion (quick screen): rewardA produces the strongest center/on_disk gain; rewardB reduces penalty pressure but does not improve centering as much.

## 2) platform_center_offset check (Y-axis sweep)

Task: `Template-Stewart-Test-Direct-v0`

Short-run config: `num_envs=128`, `max_iterations=12`.

| Offset Y | Mean reward | center | center_velocity | on_disk | note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.200 | 1152.64 | 30.57 | -4.97 | 12.64 | one run with video crashed; rerun without video succeeded |
| 0.217 (current) | 947.34 | 30.49 | -3.76 | 12.15 | reference |
| 0.230 | 1011.44 | 22.48 | -3.32 | 8.93 | clearly worse center/on_disk |

Conclusion (quick screen): `0.230` is misaligned. `0.200` vs `0.217` is close/noisy under short runs; keep current `0.217` for now.

## 3) Bandwidth A/B

Task: `Template-Stewart-Test-Direct-v0`

Base for this section: current offset + rewardA scales.

| Run | Main change | Num envs | Iters | Mean reward | center | center_velocity | on_disk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bwA | `action_smoothing=0.12` | 128 | 12 | 1236.59 | 41.49 | -6.19 | 11.35 |
| bwB | bwA + `max_action_delta=0.09`, `high=0.18`, `low=0.06` | 128 | 12 | 1325.56 | 39.27 | -4.99 | 10.95 |

Conclusion (quick screen): lower smoothing helps exploration/response; adding delta headroom improves total learning signal in short runs.

## 4) Wave transfer verification

Task: `Template-Stewart-Wave-System-Direct-v0`

Short-run config: `num_envs=128`, `max_iterations=12`.

| Run | Fixed-base settings carried | Mean reward | center | center_velocity | on_disk | catch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wave_reference | old baseline settings | 557.00 | 17.72 | -5.07 | 6.00 | 5.59 |
| wave_transfer | rewardA + bwB settings | 632.22 | 27.76 | -7.06 | 8.15 | 7.85 |

Conclusion (quick screen): transferred settings improve wave small-disturbance early learning (`center`, `on_disk`, `catch`, and mean reward all higher).

## 5) Current working config kept in code

In `source/stewart_test/stewart_test/tasks/direct/stewart_test/stewart_test_env_cfg.py`:

- `platform_center_offset = (-0.054, 0.217, 0.2)`
- `rew_scale_center = 16.0`
- `rew_scale_center_velocity = 4.0`
- `action_smoothing = 0.12`
- `max_action_delta = 0.09`
- `max_action_delta_high = 0.18`
- `max_action_delta_low = 0.06`

## 6) Suggested next validation step

Promote quick-screen winners to stability validation:

1. Run at least 2-3 seeds for fixed-base and wave (`max_iterations` increased).
2. Add one evaluation pass on saved checkpoints with video for qualitative centering trajectory.
3. If `center_velocity` remains strongly negative, test hybrid reward:
   - keep `rew_scale_center=16`
   - reduce `rew_scale_center_velocity` from `4` to `3` while retaining bandwidth settings.
