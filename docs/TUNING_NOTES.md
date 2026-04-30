# Stewart RL Tuning Notes

本文档整理本项目从 Isaac Lab 模板环境到 Stewart 平台强化学习环境的主要调参过程。目标是记录每个改动背后的原因，便于后续继续调试、复现实验和提交代码。

## 1. 从 Cartpole 模板切换到 Stewart 平台

初始工程来自 Isaac Lab direct RL 模板，默认任务是 cartpole。第一步是将环境替换为 Stewart 平台：

- `robot_cfg` 从 `CARTPOLE_CFG` 改为 `ArticulationCfg + UsdFileCfg`。
- USD 路径改为项目资产 `assets/Stewart_full.usd`。
- 动作空间从 1 维改为 6 维，对应 6 个 slider。
- 使用的 slider 名称：

```text
Slider_13, Slider_18, Slider_17, Slider_14, Slider_16, Slider_15
```

Stewart 是闭链机构，和 cartpole 这种树状结构不一样，所以后续很多稳定性设置都围绕“不要破坏闭链约束状态”展开。

## 2. 解决闭链机构发散和穿模

最初训练环境启动后，平台会疯狂移动、穿模。后来确认 USD 单独在 Isaac Sim 中运行正常，因此问题主要来自 Isaac Lab reset 和执行器设置。

关键调整：

- 移除 `joint_pos={".*": 0.0}`，不再强制所有关节初始位置为 0。
- 默认关闭 `reset_robot_joint_state`，避免每个 episode 都对 36 个 DOF 调用 `write_joint_state_to_sim()`。
- 关闭 articulation self-collision，减少闭链内部碰撞和约束求解互相打架。
- 提高 articulation solver iteration：

```text
solver_position_iteration_count = 16
solver_velocity_iteration_count = 4
```

结论：闭链 USD 如果在 Isaac Sim 原生仿真中是稳定的，Isaac Lab 中应尽量避免重写完整关节状态。

## 3. 从 Position Target 改为 Direct Effort

曾经使用 `set_joint_position_target()` 控制 slider，但 Physics Inspector 报错：

```text
PxArticulationJointReducedCoordinate::setDriveTarget() is illegal ...
PxSceneFlag::eENABLE_DIRECT_GPU_API is enabled
```

这表示 GPU direct API 模式下，PhysX 不允许对 reduced-coordinate articulation 写 drive target。因此即使增加刚度或驱动力，平台也可能没有明显响应。

解决方式：

- actuator 从 `ImplicitActuatorCfg` 改为 `IdealPDActuatorCfg`。
- `stiffness = 0.0`，`damping = 0.0`。
- `_apply_action()` 改为：

```python
effort = action_scale * processed_actions
robot.set_joint_effort_target(effort, joint_ids=slider_dof_idx)
```

当前动作含义：

```text
action in [-1, 1]
slider_force = action * action_scale
```

## 4. 初始 Slider Extension

为了给 Stewart 平台接球时留出更明显的缓冲行程，增加了初始 slider 伸长量：

```text
SLIDER_INITIAL_EXTENSION = 0.10
```

这会让平台初始位置略微抬高，并在球接触时有更多可用行程。

## 5. 观测空间调整

最初 observation 包含椭球/小球线速度和角速度。后来考虑到真实系统中这些速度不一定能直接获得，因此从 policy observation 中移除。

当前 policy observation 为 27 维：

```text
6  slider joint positions
6  slider joint velocities
3  top platform projected gravity
3  top platform angular velocity
3  object center position relative to top disk center
6  previous smoothed actions
```

其中 object relative position 定义为：

```text
object_rel_pos = object_center_world - platform_center_world
```

`platform_center_world` 由 `UJ61` body 加 `platform_center_offset` 得到。当前 offset 为 `(0, 0, 0)`。

注意：object velocity 仍可用于 reward shaping，因为 reward 是训练期信号，不必完全等同于部署期可观测量。

## 6. 奖励调参

核心目标是把落体稳定到上圆盘中心。

奖励项主要包括：

- `center`：鼓励 object center 靠近圆盘中心。
- `center_velocity`：接近平台后，朝圆心运动给正奖励，向外运动扣分。
- `height`：鼓励物体停留在平台上方合理高度。
- `lin_vel` 和 `ang_vel`：接近平台后鼓励低速度、低旋转。
- `platform_flat`：限制平台过度倾斜。
- `action` 和 `slider_vel`：抑制过大动作和过快 slider 运动。
- `termination`：物体掉落、偏离过远或平台过度倾斜时惩罚。

调参经验：

- 如果策略只学会接住球，但不主动带回中心，应提高 `rew_scale_center` 和 `rew_scale_center_velocity`。
- 如果平台动作过激，应增大 `rew_scale_action` 的负权重或降低 `action_scale`。
- 如果平台几乎不动，应先检查 PhysX drive target 报错，再考虑提高 `action_scale`。

## 7. 落体初始化

固定底座任务使用椭球，wave-base 任务使用半径 `0.05 m` 的标准球。

落体位置按上圆盘中心附近圆盘随机采样：

```text
radius = object_spawn_radius * sqrt(rand)
angle = uniform(0, 2*pi)
xy = radius * [cos(angle), sin(angle)]
```

这样可以保证空间采样在圆盘面积上近似均匀，而不是在半径上过度集中。

为了减少过偏，逐步缩小了 `object_spawn_radius`。

## 8. Wave Base 环境

新增 `Template-Stewart-Wave-System-Direct-v0` 后，底部矩形平台和 Stewart root 同步做连续 6-DOF 正弦扰动：

```text
x, y, z
roll, pitch, yaw
```

初始版本扰动太小，后续增大到：

```text
wave_pos_amplitude = (0.08, 0.08, 0.04)
wave_rot_amplitude = (0.12, 0.12, 0.18)
wave_frequency_range = (0.35, 0.90)
```

为了避免地面以下穿模，增加：

```text
system_z_offset = 0.14
```

## 9. 当前建议的调参方向

如果 Stewart 动作不明显：

1. 先确认没有 `setDriveTarget()` / GPU direct API 报错。
2. 增大 `action_scale`。
3. 检查 actuator effort limit。
4. 降低 action smoothing，让动作更快进入系统。

如果系统发散或穿模：

1. 降低 `action_scale`。
2. 增大 action penalty。
3. 减小 wave amplitude 或 frequency。
4. 避免开启完整关节 reset。

如果球能接住但不回中心：

1. 提高 `rew_scale_center`。
2. 提高 `rew_scale_center_velocity`。
3. 缩小 center reward 的误差尺度。
4. 检查 `platform_center_offset` 是否准确。

## 10. Smoke Test

每次大改后建议运行：

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
/home/ding/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Template-Stewart-Wave-System-Direct-v0 \
  --num_envs 2 \
  --headless \
  --max_iterations 1
```

成功标准：

- 成功打印 Actor/Critic 网络。
- 完成 `Learning iteration 0/1`。
- 没有 `Failed to find an articulation`。
- 没有 PhysX `setDriveTarget()` illegal 报错。
