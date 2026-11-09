# Python 推理与实机位置控制

第一版使用 CPU 假策略，无 checkpoint，不需要安装 LingBot 或 PyTorch。保留
RoboLab 的 Zenoh/Protobuf 通信，对齐 LingBot 默认 `use_length=50`、
`chunk_ret=True`：取一次观测，返回 50 步，依次执行后再推理。

## 环境与离线验证

```bash
cd /home/heinrich/roboLab
bash scripts/setup_python_inference.sh
env -u PYTHONPATH .venv-inference/bin/python -m unittest discover -s tests -p test_python_inference.py -v
env -u PYTHONPATH .venv-inference/bin/python scripts/run_python_inference.py \
  --offline-h5 data/recording_20261014_182353_1791973433661981062.h5
```

环境独立于 `.venv-lerobot`，Python >=3.10（本机验证用 3.12）。可以用
`INFERENCE_PYTHON=/absolute/path/to/python3.12` 指定创建环境的解释器。
命令清除 `PYTHONPATH`，避免终端自动加载的 ROS Python 包混入独立环境。
`requirements-inference.txt` 列出直接依赖，setup 使用本机验证后的 `.lock` 固定全部依赖。
C++ 对照测试需要 `g++` 和 `/usr/include/eigen3`。Zenoh 测试禁用 multicast/gossip，
仅连接随机 localhost TCP 端口；离线命令不导入或打开 Zenoh。

离线检查真实 H5 的首/中/尾图像及全部 8D state/action 格式，然后用配置 home
生成一圈和保持段。该轨迹明确使用 **nominal home + 录制首帧夹爪状态**，
不冒充 H5 原始动作或实机反馈。输出到唯一的 `outputs/python_inference/` 目录：
`config.json`、`report.json`、`trajectory.npz`；指定 `--output` 时目标目录必须不存在。

## 实机入口

在两个终端中依次启动（第一终端先等待，第二终端必须保留交互键盘）：

```bash
# 终端 1
env -u PYTHONPATH .venv-inference/bin/python scripts/run_python_inference.py --live

# 终端 2
./start.sh config/bringup_python_inference.yaml
```

确保其他遥操作、回放或推理发布者已停止。专用 bringup 保留 Franka、双相机、
键盘、录制插件，省略会竞争 `franka/command` 的 cartesian_control_plugin。

1. 等相机和机器人状态就绪，F2：`init → gohome → standby`。
2. 再按 F2：进入 `inference`，检查 home（每关节误差 <=0.05 rad）及观测，开始画圈。
3. 基坐标系 XY 平面、半径 3 cm、10 秒一圈，从 +Z 看逆时针；圆心在初始 TCP 的
   X 负方向，初始位置为圆周起点。五次时间插值起停；朝向与夹爪不变。
4. 一圈后保持最终关节目标；按 F2 退出到 standby。再次进入仍须通过 home 检查。
   中途停止后若已偏离 home，现有状态机没有 standby→gohome 的直接转换：需重启
   底层控制会话，再按 F2 回 home，不会自动归位。

状态机 topic 仅在转换时发布，没有历史查询。Python 应先启动；若晚启动，不根据
`RobotObservation.mode` 猜测运行状态，需通过 F2 转换让它收到新的状态。
两次触发之间等日志确认，不快速连续按键。

Python 按 30 Hz（软实时）发布 `RobotCommand.TYPE_JOINT_TARGET`，`mode="position"`。
底层继续使用已有控制器、滤波与限速。推理时暂停发新命令，不并行预取；因此慢推理
会增加 chunk 之间的保持时间，10 秒是轨迹的动作时间，不是硬实时墙钟承诺。
错过调度不会突发补发命令。Python 不是 1 kHz 的底层实时控制器。

退出 inference、输入缺失/过期、非法消息或逆解失败时清空 chunk。故障锁定，
数据恢复不会自动续跑；退出并重新进入后才重置。200 ms 新鲜度针对双相机和
机器人状态，在执行缓存动作期间也检查。**停止发布后底层仍可能跟踪最后目标**，
F2 切换 standby 才是既有停止流程；停止 Python 不是急停。

## 输入输出及替换模型

`config/inference/python_circle.yaml` 保存策略、prompt、topic 和轨迹参数。
运动学直接读取已有 cartesian_control 配置，home 读取 Franka 配置。

| 接口 | 数据 |
| --- | --- |
| `observation['image']['kinec2']` | `(540,960,3)` uint8 RGB HWC，线上的 BGRX/type=1 解码转换 |
| `observation['image']['realsense']` | `(480,640,3)` uint8 RGB HWC，线上的 RGB/type=5 |
| `observation['state']` | `(8,)`：joint_1…joint_7，gripper_close |
| `observation['prompt']` | 配置中的非空文本 |
| `policy.reset()` | 清除轨迹起点、计数与历史 |
| `policy.infer(observation)['action']` | `(50,8)`：绝对角度 rad + `[0,1]` 夹爪闭合比例 |

两路 RGB 保留 H5 分辨率，不拼接、缩放或归一化；无 batch 维。实时模式使用最新帧，
允许复用较慢相机帧，时间戳为本机接收单调时钟，**不是硬件同步**。
`ages_ms` 和 `pose_ee` 是诊断字段。假策略校验所有输入，但不依据图像或 prompt 决策。
假策略额外返回 `target_xyz` 供离线审计，不向机器人发布它。

checkpoint 路径默认 null；选择真实策略会明确报缺 checkpoint 或后端未实现，
不会默默降级到假策略。后续实现新的 Policy，保留执行器与运输层，并复核图像预处理、
state/action 归一化、动作步长和相机键名映射，不直接认为本接口可加载任何 LingBot 权重。

IK 是 NumPy 阻尼最小二乘，沿用已有 DH、TCP 和关节限位，跨 chunk 使用前一目标
关节解保持连续。整段生成/检查成功才发布；默认最大目标关节速度 0.4 rad/s，低于已有
底层 0.5 rad/s 限速。200 ms、1 mm 和 0.5° 为第一版工程阈值，均可配置。
运动学一致性测试只证明与已有 C++ 配置一致，不证明实际工具几何已标定。

## 日志与实机验收

`events.jsonl` 记录状态、故障、每次推理耗时，以及每步动作、机器人关节反馈、
`pose_ee=[x,y,z,qw,qx,qy,qz]`（若存在）、图像/状态年龄、发布时间和调度迟到量。
底层 `pose_ee` 的定义可能不同于配置 TCP；比较前需确认工具变换，不能直接相减。
录制插件在 inference 期间照常写 H5。

实机验收需现场完成一圈及一次 F2 中断，观察运动，检查反馈轨迹、姿态、夹爪和
chunk 边界连续性；只验证目标轨迹不能声称实机画圆成功。本次自动测试不连接机器人。

Python stubs 独立存于 `python/robo_lab/inference/proto/`，源为仓库的两个 proto，
不改旧脚本使用的 stubs。协议改变后用此环境重新生成：

```bash
env -u PYTHONPATH .venv-inference/bin/python -m grpc_tools.protoc -I proto \
  --python_out=python/robo_lab/inference/proto proto/franka.proto proto/kinect.proto
```
