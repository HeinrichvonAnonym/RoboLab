# H5 转 LeRobot v3 / 浏览器裁剪

在仓库根目录执行命令。工具只读原始 H5，不连接机器人，不上传数据。

## 环境

独立环境已在本机 `.venv-lerobot` 建好。重新安装时使用 Python 3.12 和 CPU 版 PyTorch：

```bash
python3 -m pip install --user uv
uv python install 3.12
uv venv --python 3.12 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python --torch-backend cpu -r scripts/requirements-lerobot.lock
```

`requirements-lerobot.txt` 是直接依赖；`.lock` 固定了已验证的 Linux / Python 3.12 CPU 环境全部依赖。若终端初始化了 ROS 并设置了 `PYTHONPATH`，可在下面的 Python 命令前加 `env -u PYTHONPATH`，避免混用其他 Python 环境。

## 浏览器播放和裁剪

```bash
.venv-lerobot/bin/python scripts/review_h5.py
```

打开 <http://127.0.0.1:8765>。启动时扫描已完成录制，默认选择北京时间 **2026-10-13 21:00:00 起**的 `data/recording_*.h5`，包含后续日期；`.inprogress` 不参与扫描。

1. 左侧选择录制，两路 RGB 和八条 action position 曲线使用同一时间轴。
2. 播放/暂停、左右按钮逐帧、拖动时间条，或者点击 action 曲线定位。空格控制播放，左右方向键逐帧（输入框内不触发）。
3. 暂停后标记 Start、End，或直接填写原始录制内秒数。进度条和全部 action 曲线立即显示绿色 Start、黄色 End 虚线及时间标签；两端齐全后用半透明黄色预览待删除区间。点击“添加删除区间”才确认删除。**首尾帧都删除**。可以添加多段、移除区间、撤销，或剔除整条。
4. “保存方案”只写入 JSON。未保存编辑会提示，且不能启动导出。
5. “导出已保存方案”才生成数据集；导出过程中可继续编辑、保存下一版方案，正在运行的任务始终使用启动时的方案快照。

红色区域表示被剔除的数据；播放仍显示完整原始时间轴，便于检查和恢复剪辑。每个连续保留区间导出为独立 episode。界面显示当前方案是否已同步到已验证的数据集，失败时可查看原因。

播放倍速独立于导出 FPS：

```bash
.venv-lerobot/bin/python scripts/review_h5.py \
  --fps 15 --speed 0.5 --port 8765 \
  --clips outputs/lerobot_review/clips.json \
  --output outputs/lerobot/franka_20261013_210000
```

`--speed` 和页面输入支持 0.25×–4×。页面解码跟不上时跳过显示帧，暂停/逐帧仍可精确选取同步帧；这不会影响导出帧。需要跨机器访问时显式使用 `--host 0.0.0.0`，或通过 SSH 转发本机 8765 端口；默认仅监听 loopback。

## Headless 转换

不使用交互、不剔除区间：

```bash
.venv-lerobot/bin/python scripts/convert_h5_to_lerobot.py
```

复用浏览器保存的方案：

```bash
.venv-lerobot/bin/python scripts/convert_h5_to_lerobot.py \
  --clips outputs/lerobot_review/clips.json
```

先只检查源文件和剪辑，不生成数据集：

```bash
.venv-lerobot/bin/python scripts/convert_h5_to_lerobot.py --scan
```

共同参数包括 `--input-dir`、`--since`、`--output`、`--fps`、`--task`、`--max-gap-ms`、`--repo-id`、`--clips`。FPS 为 1–240 的整数；`--since` 含时区偏移时按对应时刻筛选，不含时区时按上海时间解释。默认 task 为 `franka manipulation`；`--repo-id` 只用于本地元数据。

短片段验证示例：

```bash
.venv-lerobot/bin/python scripts/convert_h5_to_lerobot.py \
  --file recording_20261014_182353_1791973433661981062.h5 \
  --max-frames 45 --output outputs/lerobot_smoke/example
```

`--file` 可重复；`--max-frames` 限制**每个源文件裁剪后**最多导出的帧数，只用于验证。带此限制或只导出部分源文件的结果会在浏览器中标为需要重新全量导出。CLI 未提供 `--clips` 就保留全部有效数据，不会自动读取浏览器方案；显式指定不存在的方案会报错。

## 同步和速度约定

| 字段 | 来源与单位 |
| --- | --- |
| `observation.images.kinec2` | Kinect BGRX 转 RGB，保留原始分辨率（当前 960×540） |
| `observation.images.realsense` | RealSense RGB，保留原始分辨率（当前 640×480） |
| `observation.state` | `franka_state/joints_position`，8 维 |
| `action` | `franka_command/joints_position`，8 维绝对目标 |
| `observation.velocity` | H5 实测前 7 维速度，加位置差分得到的夹爪速度 |
| `action.velocity` | 原始 command 的 8 维位置差分 |

维度名为 `joint_1` … `joint_7`、`gripper_close`。前 7 维单位 rad、rad/s；夹爪为 0–1 闭合比例、比例/s。夹爪目标阶跃的导数可能较大，保留该变化，不做平滑或归零。

- 使用 `timestamps_ns` 的整数纳秒和相对秒数计算，忽略低精度 `sys_time`。
- 在四路共同覆盖区间，从共同起点建立固定 FPS 网格。录制内秒数的零点是四路最早接收时间，因此第一条同步帧可能不在 0 秒。
- action 取当前时刻之前最近的命令；state 和实测 velocity 线性插值；RGB 选择最近帧（等距离选前帧）。
- 相邻源样本间隔超过 `--max-gap-ms`（默认 200 ms）时，缺口内部沿用前一个有效样本，不跨缺口插值，也不因为采样缺口拆分 episode。
- 所有位置差分在原始采样时间轴、按连续有效区间预先计算，使用非等间隔中心差分，端点单边差分；孤立样本沿用上一有效速度，没有上一速度时为零。**裁剪只过滤同步帧，不改变保留帧的速度，也不重新连接删除区间两侧求导。**
- `export_report.json` 保存采样缺口和滞后帧统计；`provenance.parquet` 保存每帧源时间、源 command / RGB 索引、两路图像时间戳，以及各路距前一个源样本的时间（`*_age_ms`，并非最近图像的绝对时间误差）。

方案格式如下，时间单位为原始录制内秒数，区间闭合：

```json
{
  "version": 1,
  "recordings": {
    "recording_20261014_182353_1791973433661981062.h5": {
      "drop": false,
      "exclude": [[1.0, 2.0], [5.0, 6.0]]
    }
  }
}
```

重叠区间自动合并；`drop: true` 删除整条录制。浏览器自动附加文件大小和修改时间作为 `fingerprint`，防止把旧方案用于被替换的 H5；纳秒修改时间以字符串保存，避免 JavaScript 整数精度丢失，人工方案可以省略该指纹。方案引用不在本次时间筛选范围内的文件时导出报错，避免静默漏用剪辑。

## 输出与失败处理

默认输出 `outputs/lerobot/franka_20261013_210000` 是指向当前完整版本的符号链接，可直接作为官方 `LeRobotDataset(..., root=...)` 的 root。每次导出先写入相邻 `.franka_20261013_210000.versions/` 下的独立目录；完成 `finalize()`、官方 loader 首尾帧和视频读取、索引/FPS/统计验证后，才原子替换符号链接。

输出包含官方 `data/`、`videos/`、`meta/`，以及 `clip_plan.json`、`export_report.json`、`provenance.parquet`。视频使用流式 H.264 编码，不先落盘全部 PNG。旧的成功版本保留，便于恢复；重复导出会占用额外空间，可在确认不再需要后自行删除非当前版本。已有普通目录或非本工具管理的符号链接不会被覆盖。

结构损坏、缺少相机或时间戳损坏的源录制整条跳过并记录。当前 96 条候选中 94 条有效，另外两条为：

- `recording_20261013_213801_1791898681631501630.h5`：缺少 RealSense RGB。
- `recording_20261014_181613_1791972973702287166.h5`：尾部时间戳损坏。

编码、写入或验证失败则整次导出失败，保留上一次成功结果。所有录制被剔除时同样报错并保留旧结果。输出旁的 `.export.lock` 防止并发导出；进程强制退出后若锁残留，先检查其中 PID 确认任务已结束，再移除锁并重试。

## 测试与本机验证样本

```bash
.venv-lerobot/bin/python -m unittest discover -s tests -p test_lerobot_conversion.py -v
node tests/test_lerobot_player.js
```

可选的真实 Chromium 自动化检查（依赖仅用于测试）：

```bash
uv pip install --python .venv-lerobot/bin/python playwright==1.51.0
.venv-lerobot/bin/python -m playwright install chromium --only-shell
.venv-lerobot/bin/python -m unittest discover -s tests -p test_lerobot_browser.py -v
```

该检查使用临时合成录制，验证双路图像、播放/暂停/逐帧、倍速、选区、撤销、整条剔除、保存重载和页面导出；截图保存到 `outputs/lerobot_review/browser_smoke.png`。

真实录制短片段验证结果位于 `outputs/lerobot_smoke/franka_real`：使用 `outputs/lerobot_review/smoke_clips.json` 删除中间 10 帧，生成 2 个 episode、45 帧，已通过官方 loader 和双路视频验证。完整数据尚未导出，等待实际裁剪方案。

本次经确认清理的旧录制及缓存清单见 `outputs/cleanup_20261017.json`；96 条选中录制均已核对保留。
