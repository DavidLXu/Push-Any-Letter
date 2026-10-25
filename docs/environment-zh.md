> 公开数据与训练入口见 [项目 README](../README.md)。本页同时介绍通用环境（默认摩擦 0.7）与小批量示范生成；公开中心目标数据使用摩擦 1.4，并包含 RGB 图像。

# Push Any Letter

本地运行的二维字母推动仿真。用圆形推杆把橙色字母推到青绿色目标轮廓，覆盖率达到 95% 即成功。提供原生 Pygame 操作窗口、Gymnasium 接口和演示轨迹录制，可在 Mac 上使用 CPU 运行。

## 在 macOS 上启动

双击 **Launch.command**，启动脚本会创建 `.venv` 并安装依赖。也可以从终端运行：

```bash
cd Push-Any-Letter
./Launch.command
```

首次在其他机器上运行，需要 Python 3.9–3.12 和网络安装依赖；安装后可以离线使用。Apple Silicon 的 Python 3.9 环境已在本机实际验证。Intel Mac 使用相同代码，但本次没有 Intel 机器实测。启动脚本会在项目目录创建独立虚拟环境。

## 操作

| 操作 | 效果 |
| --- | --- |
| 在场地内按住鼠标左键并移动 | 设置推杆目标位置；推杆有速度和力限制 |
| 点击右侧字母 | 切换字母并重置场景 |
| Tab / a-z 按钮 | 切换大小写字母列表 |
| N | 下一个字母 |
| R | 新随机场景 |
| Shift+R | 重复当前 seed 的初始场景 |
| Space | 暂停 / 继续 |
| C | 开始录制，或停止并保存 |
| S | 保存；录制中会从当前 seed 重开一局 |
| Esc | 退出，并保存尚未保存的录制 |

松开鼠标后，推杆保持最后一个目标位置。橙色区域是物体，青绿色区域是目标，黑色圆圈是推杆，小十字表示推杆的控制目标。每局最多 600 步，即 30 秒仿真时间；成功后也会暂停，按 R 开始新一局。

## 字符和字体

界面提供 **A–Z 和 a–z**。碰撞网格来自字体的真实外轮廓，保留 A、B、O 等字母内部孔洞；凹形轮廓通过三角剖分组成同一刚体。`i`、`j` 等分离笔画也固定为同一刚体，不会散开。曲线离散化后再以 0.3 像素容差简化。

命令行接受一个可见 Unicode 字符，只要指定字体包含该字符且轮廓能够有效三角剖分：

```bash
./Launch.command --letter a
./Launch.command --letter Ω
./Launch.command --letter 中 --font '/Library/Fonts/Arial Unicode.ttf'
```

默认使用 Mac 的 Arial Bold，其他平台会尝试 DejaVu Sans Bold 或 Pygame 内置字体。字体文件不随项目分发。改变字体也会改变物理几何、质心和目标形状。单个字符的最长边统一为 156 像素；暂不支持词组、emoji 彩色字体和任意 SVG。

## Python / Gymnasium

```python
import numpy as np
from push_any_letter import PushLetterEnv

env = PushLetterEnv(letter="T", render_mode="rgb_array")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(
    np.array([300, 400], dtype=np.float32)
)
image = env.render()  # uint8, (640, 640, 3)
env.close()
```

导入 `push_any_letter` 后，也可使用 `gymnasium.make("PushAnyLetter-v0", letter="B")`。通过 `reset(options={"letter": "C"})` 切换字符。多字符任务中，字符身份见 `info["letter"]`；训练策略需要自行加入字符条件或使用图像观察。

- 动作：推杆的绝对目标坐标 `[x, y]`，范围 `[28, 612]`，超界会裁剪。
- 观察：字典 `agent_pos`（2 维）、`object_pose`（3 维）、`goal_pose`（3 维），均为 float32；姿态是 `[质心 x, 质心 y, 角度]`。
- 坐标：场地为 640×640 像素，左上角为原点，x 向右、y 向下；角度为弧度，屏幕上顺时针为正，观察中规范到 `[0, 2π)`。
- 频率：控制 20 Hz，物理 200 Hz，每个动作执行 10 个物理子步。
- 指标：`coverage = intersection_area / glyph_area`；奖励为 `min(coverage / 0.95, 1)`，达到 95% 时 `terminated=True`。
- 时限：默认 600 控制步；达到时限且未成功时 `truncated=True`。
- 渲染：`None` 用于无窗口训练，`rgb_array` 输出图像，`human` 打开原生窗口。

这是受 Push-T 启发的独立实现。它使用字体字形、动态推杆、边界和不同控制参数，不与原始 Push-T 数据或预训练权重直接兼容。桌面阻力用速度阻尼近似，碰撞使用接触摩擦；不是标定过的真实机器人接触模型。像素、质量和力为仿真内部单位。对称字母按覆盖率判断，不强制唯一角度。

## 录制和回放

点击 Record / 按 C 会重置到当前 seed，然后从第一步开始录制。保存到 `recordings/`，每次生成 `.npz` 数据和 `.json` 元数据。重置、切换字母、关闭窗口时自动保存当前数据。

`.npz` 使用 `allow_pickle=False` 即可读取，包含：

- `observations` / `next_observations`：`N×8` 状态，依次为 agent x/y、object x/y/angle、goal x/y/angle。
- `actions`：`N×2` 目标坐标。
- `rewards`、`terminated`、`truncated`：逐步奖励和环境结束标记。用户提前保存或关闭不会伪造环境终止。
- `metadata`：JSON 字符串，记录字母、seed、字体路径、频率和列名。

没有默认录制图像，以减少文件体积；可以根据动作回放并调用 `render()` 获得图像。相同字体、软件版本和平台下可验证确定性；跨平台浮点结果可能有差异。

```bash
.venv/bin/python examples/replay.py recordings/文件名.npz
.venv/bin/python examples/replay.py recordings/文件名.npz --headless
.venv/bin/python examples/random_rollout.py
```

随机动作示例仅用于说明 API。项目另提供下述自动规划器；未包含学习得到的策略权重。

## 检查

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
.venv/bin/python -m push_any_letter.app --screenshot preview.png
```

测试覆盖 52 个大小写字母的网格、孔洞碰撞、推动产生平移和旋转、强推边界、目标成功、时限、Gymnasium 契约、RGB 渲染及保存后的轨迹回放。依赖的本机精确版本见 `requirements-lock.txt`。

参考：[原始 Diffusion Policy Push-T 环境](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/pusht/pusht_env.py)、[Pymunk](https://www.pymunk.org/)、[FontTools](https://fonttools.readthedocs.io/)。


## 自动规划示范（新增）

原始 Diffusion Policy 的 Push-T 使用[鼠标遥操作采集脚本](https://github.com/real-stanford/diffusion_policy/blob/main/demo_pusht.py)，默认控制频率为 10 Hz。[官方 README](https://github.com/real-stanford/diffusion_policy#replaybuffer) 展示的公开数据包含 206 个 episode。这里新增的是 **基于模型的规划器示范**，与人工遥操作的来源不同；当前环境保持 20 Hz。

规划器使用真实物体状态和已知物理模型，在复制的仿真中搜索接触位置、推动方向和距离，选择改善目标对齐的动作片段，再在原环境执行。绕行、接触和推动都由推杆动作实现；执行中的物体没有瞬移或直接写入目标姿态。每个片段结束后重新规划。对几何对称字母允许等价目标朝向。

生成 A–Z、a–z 各 1 条示范：

```bash
.venv/bin/python -m push_any_letter.dataset --episodes 1 --output datasets/planner_v1
```

生成大写字母各 10 条，使用不同随机初始状态；建议先小规模确认耗时：

```bash
.venv/bin/python -m push_any_letter.dataset --letters ABCDEFGHIJKLMNOPQRSTUVWXYZ --episodes 10 --seed 1000 --output datasets/uppercase_10
```

`--episodes` 是每个字母所需的**通过验证的轨迹数**。默认最多尝试该数量的 3 倍（`--attempts`），不会无限重试。`--seed` 开始按字母分别递增 seed。相同参数和输出路径可继续未完成批次；修改参数时使用新目录。

输出目录包含：

- `episodes/`：仅保存成功且通过回放验证的 `.npz` 与对应 JSON 元数据，文件名使用 Unicode 编码，避免 Mac 大小写不敏感文件系统冲突。
- `attempts/`：未通过验证的尝试，不能作为成功专家示范使用。
- `report.json`：每次尝试、逐字母数量、成功覆盖率、步数、计算耗时和验证结果。`complete` 表示是否满足全部字母配额。
- `overview.png` / `planner_preview.mp4`：运行下面的预览脚本生成。

除基础录制字段外，默认保存 `images`（N×96×96×3、uint8、动作前图像）和 `timestamps`。状态中的目标姿态与图片中的目标轮廓可用于目标条件策略。元数据标记 `source=model_predictive_planner`、`privileged_state=true`，并保存字体 SHA-256、物理库版本、最大步数与规划参数；不要将这些数据标注为人类示范。

每条轨迹先生成，再创建全新的环境，仅重放已保存的动作。验收条件为：最终覆盖率 ≥95%、完整状态最大误差 <0.001、奖励和终止标记一致、有限且合法的动作、物体和推杆留在场地内、接触穿透不超过 3 像素，以及成功后无多余动作。图像与动作前状态在回放时一起导出。

```bash
# 单独重新验证一条轨迹
.venv/bin/python -m push_any_letter.dataset --validate datasets/planner_v1/episodes/U0054_seed42.npz
# 原生窗口回放 T；按该轨迹记录的时间限制重建环境
.venv/bin/python examples/replay.py datasets/planner_v1/episodes/U0054_seed42.npz
# 汇总图和四字母演示视频；视频编码需要 ffmpeg
.venv/bin/python examples/preview_dataset.py datasets/planner_v1 --video
```

规划器默认允许每局 2400 步，即 120 秒仿真时间，区别于基础 Gym 环境默认 600 步。复杂轮廓可能需要绕行、反复接触或重新尝试；这些是验证成功的规划轨迹，不保证最短或具有人工示范的动作分布。初始小样本主要用于验证采集链路，不足以证明训练泛化能力。本次公开数据含 6,240 条高摩擦版本轨迹，全部用于训练；固定未见过的目标角度与新初始状态用于测试。完整协议见 [训练说明](training.md)。
