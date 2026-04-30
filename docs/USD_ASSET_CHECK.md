# USD Asset Check Notes

本文档记录本项目中 Stewart 平台 USD 资产的检查过程，以及如何定位和修复 USD reference 断链导致的 Isaac Lab Articulation 加载失败问题。

## 背景

训练 `Template-Stewart-Wave-System-Direct-v0` 时曾出现以下错误：

```text
Could not open asset @/home/ding/Documents/stewart_test/Downloads/Stewart-Platform/Stewart/Stewart/Stewart.usd@
Failed to find an articulation when resolving '/World/envs/env_0/Robot'.
Please ensure that the prim has 'USD ArticulationRootAPI' applied.
```

这说明外层 `assets/Stewart_full.usd` 可以被打开，但它内部 reference 指向了不存在的位置。由于真正带有 articulation 结构的内层 USD 没有加载成功，Isaac Lab 最终无法在 `/World/envs/env_0/Robot` 下找到 ArticulationRootAPI。

## 快速检查文件

首先确认工程内资产文件存在：

```bash
cd /home/ding/Documents/stewart_test/stewart_test
ls -la assets
file assets/Stewart_full.usd
```

`Stewart_full.usd` 是二进制 USD crate，普通文本读取工具无法直接查看。可以先用 `strings` 快速找出可疑 reference：

```bash
python3 - <<'PY'
import subprocess

out = subprocess.check_output(["strings", "assets/Stewart_full.usd"], text=True, errors="ignore")
for line in out.splitlines():
    if any(s in line for s in ("Downloads", "Stewart", ".usd", "references", "payload")):
        print(line)
PY
```

如果输出中出现类似 `../../Downloads/...` 的路径，就要确认这个路径相对当前 USD 文件是否仍然有效。

## 使用 Isaac Sim 的 USD API 检查 Reference

系统 Python 通常没有 `pxr` 模块，应使用 Isaac Lab/Isaac Sim 环境启动后再导入 USD API：

```bash
PYTHONPATH=/home/ding/Documents/IsaacLab/source/isaaclab:$PYTHONPATH \
/home/ding/anaconda3/envs/env_isaaclab/bin/python - <<'PY'
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Sdf

path = "/home/ding/Documents/stewart_test/stewart_test/assets/Stewart_full.usd"
layer = Sdf.Layer.FindOrOpen(path)
print("layer", bool(layer))

if layer:
    text = layer.ExportToString()
    for line in text.splitlines():
        if "@" in line or "references" in line or "payload" in line:
            print(line)

app.close()
PY
```

本项目中检查到的旧 reference 是：

```text
prepend references = @../../Downloads/Stewart-Platform/Stewart/Stewart/Stewart.usd@
```

当 `Stewart_full.usd` 被复制到项目 `assets/` 后，这个相对路径解析到了不存在的目录，因此训练时加载失败。

## 修复方式

将依赖资产复制到项目内：

```bash
mkdir -p assets/Stewart/Stewart
cp -a /home/ding/Downloads/Stewart-Platform/Stewart/Stewart/Stewart.usd assets/Stewart/Stewart/
cp -a /home/ding/Downloads/Stewart-Platform/Stewart/Stewart/configuration assets/Stewart/Stewart/
```

然后把 `assets/Stewart_full.usd` 中的 reference 改为项目内相对路径：

```text
./Stewart/Stewart/Stewart.usd
```

可用下面的脚本修改 Sdf reference list：

```bash
PYTHONPATH=/home/ding/Documents/IsaacLab/source/isaaclab:$PYTHONPATH \
/home/ding/anaconda3/envs/env_isaaclab/bin/python - <<'PY'
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Sdf

path = "/home/ding/Documents/stewart_test/stewart_test/assets/Stewart_full.usd"
layer = Sdf.Layer.FindOrOpen(path)

old = "../../Downloads/Stewart-Platform/Stewart/Stewart/Stewart.usd"
new = "./Stewart/Stewart/Stewart.usd"
changed = 0

for prim in layer.rootPrims:
    stack = [prim]
    while stack:
        spec = stack.pop()
        refs = list(spec.referenceList.prependedItems)
        if refs:
            new_refs = []
            for ref in refs:
                if ref.assetPath == old:
                    new_refs.append(Sdf.Reference(new, ref.primPath, ref.layerOffset, ref.customData))
                    changed += 1
                else:
                    new_refs.append(ref)
            spec.referenceList.prependedItems = new_refs
        stack.extend(spec.nameChildren)

print("changed", changed)
layer.Save()
app.close()
PY
```

## 验证修复

再次打开 USD 并打印 reference：

```bash
PYTHONPATH=/home/ding/Documents/IsaacLab/source/isaaclab:$PYTHONPATH \
/home/ding/anaconda3/envs/env_isaaclab/bin/python - <<'PY'
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd

path = "/home/ding/Documents/stewart_test/stewart_test/assets/Stewart_full.usd"
stage = Usd.Stage.Open(path)
print("stage", bool(stage))

if stage:
    for prim in stage.Traverse():
        refs = prim.GetMetadata("references")
        if refs:
            print(prim.GetPath(), refs)

app.close()
PY
```

应能看到：

```text
/World/Stewart SdfReferenceListOp(Prepended Items: [SdfReference(./Stewart/Stewart/Stewart.usd, ...)])
```

最后用小规模训练做 smoke test：

```bash
PYTHONPATH=$PWD/source/stewart_test:$PYTHONPATH \
/home/ding/anaconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Template-Stewart-Wave-System-Direct-v0 \
  --num_envs 2 \
  --headless \
  --max_iterations 1
```

如果环境能创建、Actor/Critic 网络能打印并完成 `Learning iteration 0/1`，说明 USD 资产链路和 Articulation 加载已经正常。

## 常见 Warning

以下 warning 通常不等价于失败：

```text
Unable to resolve info:sourceAsset ... UsdPreviewSurface.mdl
triangle mesh collision ... falling back to convexHull approximation
```

真正需要处理的是：

```text
Could not open asset ...
Failed to find an articulation ...
```

这类错误通常表示 USD reference 断链或目标 USD 没有包含有效的 ArticulationRootAPI。
