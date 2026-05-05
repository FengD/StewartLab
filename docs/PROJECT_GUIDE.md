# IsaacLab Stewart 平台项目指南

该仓库是一个专门用于训练 Stewart 平台控制的 Isaac Lab 工程，目标是通过强化学习控制 6 个 slider，使上圆盘接住落体并将其稳定在圆盘中心附近；同时提供 wave-base 版本以提升难度。

## 任务概览

请以 `docs/STEWART_RL_TASKS.md` 为唯一权威入口（任务名、观测/动作、奖励、训练/推理命令、资产说明都在其中）。

- `Template-Stewart-Test-Direct-v0`：固定底座，控制 Stewart 上圆盘接住并稳住物体
- `Template-Stewart-Wave-System-Direct-v0`：底座做连续随机 6DoF 运动，增加接住与稳球难度

## 代码位置

- 任务实现：
  - 固定底座：`source/stewart_test/stewart_test/tasks/direct/stewart_test/`
  - wave-base：`source/stewart_test/stewart_test/tasks/direct/stewart_wave_system/`
- 训练/推理脚本：`scripts/rsl_rl/`
- 资产：`assets/Stewart_full.usd`

## 常用调试

- 查看 USD 引用/路径：`scripts/check_usd.py`
- 关节/结构检查：`scripts/load_usd_model.py`
