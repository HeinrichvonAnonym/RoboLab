# roboLab（中文说明）

**roboLab** 是一套基于 **Zenoh** 的**去中心化消息层**与**插件运行时**，面向**具身智能 / 机器人**软件。进程通过 **Zenoh**（发布/订阅、发现、可选路由）通信，而不是依赖单一中心化的「ROS 式 master」；机器人与传感器相关逻辑放在**动态加载的插件**里，由 **YAML bringup** 与 **Protobuf** 消息描述驱动。

核心组件：

- **`MessageSystem`** — 基于 [zenoh-c](https://github.com/eclipse-zenoh/zenoh-c) 的 C++ 封装，按 key expression 做发布/订阅。
- **`Framework`** — 读取 **bringup YAML**，`dlopen` 加载插件 `.so`，依次调用 `initialize()` / `run()` / `stop()`，在 **Ctrl+C** 或 **SIGTERM** 时干净退出。
- **插件**（如 Franka、Kinect）— 实现 `Plugin` 接口，用 YAML 读参数，并可通过 Zenoh 交换 **Protobuf** 负载（见 `proto/`）。

---

## 安装与构建（Ubuntu 22.04 x86_64 / ROS 2 Humble）

先安装 ROS 2 Humble，然后在仓库根目录执行：

```bash
bash scripts/install_x86.sh
source /opt/ros/humble/setup.bash
bash build.sh
```

安装脚本通过 sudo 补齐系统开发依赖，只安装 Kinect 所需的 **CUDA 11.8 编译器和运行时开发包**（`cuda-nvcc-11-8`、`cuda-cudart-dev-11-8`），保留现有 NVIDIA 驱动与内核。无需下载完整 Toolkit 中的分析器和数学库。脚本会初始化仓库已锁定的串口库和 CUDA samples 子模块；后者提供 `helper_math.h`，不编译 samples。

Rust **1.93.0** 安装在用户目录，已有默认工具链不变；沿用 **zenoh-c 1.8.0** 及与其配套的锁文件，后续增量构建复用产物。`INSTALL_SYSTEM=0` 只安装 Rust，`INSTALL_RUST=0` 只安装系统依赖，明确使用 CPU Kinect 时可设 `INSTALL_CUDA=0`。`INSTALL_ZENOHD=1` 可选安装同版本路由；一般 peer 通信不需要路由。`build.sh` 不自动安装任何依赖。

| CMake 开关 | 默认值 | 关闭后的效果 |
|---|---|---|
| `ROBOLAB_BUILD_FRANKA` | ON | 不构建 Franka 硬件插件及 libfranka |
| `ROBOLAB_BUILD_KINECT` | ON | 不构建 Kinect 插件及 libfreenect2 |
| `ROBOLAB_BUILD_LIBREALSENSE` | ON | 不构建 RealSense 插件及 SDK |
| `ROBOLAB_LIBFREENECT2_CUDA` | ON | 使用已有 CPU Kinect 路径 |
| `ROBOLAB_BUILD_ROS_BRIDGE` | ON | 不构建 ROS 2 桥接，不查找 ROS 包 |
| `ROBOLAB_BUILD_TENSORRT` | OFF | 不构建推理库及推理示例插件 |

控制、录制回放、键盘及 IMU 插件继续构建。TensorRT 需单独提供兼容 SDK，通过 `bash build.sh -DROBOLAB_BUILD_TENSORRT=ON -DROBOLAB_TENSORRT_ROOT=/path/to/TensorRT` 开启。`ROBOLAB_CUDA_ROOT` 默认 `/usr/local/cuda-11.8`。已开启模块缺少依赖时明确报错，不静默关闭功能。旧 ROS 1 `ros_ws` 保留，本次不构建、不迁移。

`build.sh` 透传 CMake 参数、保留缓存，检测到 ccache 时启用。默认 Release，C/C++ 并行度上限 8、Cargo 上限 4，可分别用 `CMAKE_BUILD_PARALLEL_LEVEL`、`CARGO_BUILD_JOBS` 覆盖。第三方示例、工具和测试默认关闭。单独重编插件可执行 `cmake --build build --target ros_bridge_plugin`。输出位置仍为 `build/apps`、`build/plugins`；自定义构建时请选择匹配的 bringup。

可通过 `CMAKE_PREFIX_PATH`/`zenohc_DIR` 使用兼容的已安装 zenohc 包；`ROBOLAB_ZENOHC_ROOT` 指定其他 zenoh-c **源码目录**。绕过 build.sh 直接编译源码时，先 `source ~/.cargo/env`。默认使用系统 protoc，与系统 Protobuf C++ 库配套，避免混入 conda 版本。

## ROS 2 桥接兼容

保留 YAML 列表及字段名、Zenoh key、Protobuf 定义和插件库名。`ros_type` 同时接受 `sensor_msgs/JointState` 与 `sensor_msgs/msg/JointState` 等两种写法，未知类型延续字节数组回退。

节点名为 `robo_lab_ros_bridge_plugin`。绝对话题保持不变；相对话题沿用原私有命名空间，例如 `franka_4/joint_state` 解析为 `/robo_lab_ros_bridge_plugin/franka_4/joint_state`。默认 QoS 为 keep-last 10、reliable、volatile；每个话题可增加：

```yaml
qos: {depth: 10, reliability: best_effort}
```

接收 best-effort ROS 传感器时应配置相应订阅可靠性。插件使用独立 ROS Context，由 Framework 统一处理 Ctrl+C/SIGTERM 及资源释放。

## 无硬件验证

```bash
source /opt/ros/humble/setup.bash
bash build.sh -DROBOLAB_BUILD_GRIPPER_TESTS=ON \
  -DROBOLAB_BUILD_KINECT_TESTS=ON -DROBOLAB_BUILD_INTEGRATION_TESTS=ON
ctest --test-dir build --output-on-failure
```

覆盖插件加载、夹爪 worker、Kinect 采样、ROS/Zenoh 双向转换、两种消息类型写法、QoS、非法配置/消息、退出及合成 HDF5 录制回放。集成测试使用临时文件和独立测试话题，不初始化机器人或相机。实机运动、相机吞吐及 TensorRT 推理另行验收。回放兼容旧 7 列和现有 8 列关节数据，仍只输出 7 个机械臂关节；桥接修复了旧 JointStateArray 被误读为空 RobotObservation 的问题。

本机验证结果与计时见 [Ubuntu 22.04 验证记录](scripts/README_ubuntu22_validation.md)。

实机验收时连接配置对应的 Franka、相机和 IMU，核对设备及话题；确认 Kinect 日志选择 CUDA，检查图像格式/帧率、ROS 2 话题可见性，以及原操作流程下的录制回放。先验证正常退出，再进行现场监督下的机器人运动。TensorRT 需使用实际 SDK 和模型单独验证。

---

## 配置方式

### 1. Bringup 文件（加载哪些插件）

`start.sh` / `main` 默认使用 `config/bringup_franka_config.yaml`，列出插件，例如：

```yaml
plugins:
  - name: franka_plugin
    config: config/plugins/franka_plugin_config.yaml
    lib: ./build/plugins/libfranka_plugin.so
  - name: kinect_plugin
    config: config/plugins/kinect_plugin_config.yaml
    lib: ./build/plugins/libkinect_plugin.so
```

- **`name`** — 逻辑名称（用于日志）。
- **`config`** — 插件专用 YAML（路径解析规则与 `lib` 相同）。
- **`lib`** — 插件**动态库**路径。

若 bringup 文件位于名为 `config/` 的目录下，相对路径还会相对**仓库根目录**解析，因此在仓库根运行时可正确使用 `./build/plugins/...`。

### 2. 各插件 YAML

每个插件读取自己的配置文件（如 `config/plugins/franka_plugin_config.yaml`）：机器人 IP、Zenoh **topic** 键、控制模式、增益等。

### 3. Protobuf（`proto/`）

`proto/` 下所有 `.proto` 会自动编译进 **`robo_lab_proto`** 库（见顶层 `CMakeLists.txt`）。插件与工具约定线上类型（如 `franka.RobotCommand`）。修改 schema 后执行 `bash scripts/regen_proto_py.sh` 重新生成 Python 桩代码。

---

## 运行 — `start.sh`

**`start.sh`** 是**推荐的启动入口**：在**仓库根目录**启动已编译的二进制 **`build/apps/robo_lab_main`**（保证 YAML 里相对路径正确），并**透传**后续参数。

```bash
./start.sh
# 等价于：
./build/apps/robo_lab_main

# 指定 bringup：
./start.sh /path/to/my_bringup.yaml
```

MuJoCo 查看器默认使用仓库内的 `.venv-teleop`，不再要求名为 `roboLab` 的 Conda 环境。运行 `bash scripts/setup_teleop.sh` 安装 GELLO 和查看器所需的 Python 依赖；没有桌面或依赖时会提示并跳过查看器。可用 `ROBOLAB_SKIP_MUJOCO_VIEWER=1 ./start.sh ...` 禁用查看器，或用 `ROBOLAB_PYTHON=/path/to/python` 指定解释器；显式设置 `ROBOLAB_MUJOCO_CONDA_ENV` 时仍支持已有 Conda 环境。

GELLO → Panda 使用 `./start.sh config/bringup_gello_franka.yaml` 和独立的 `./start_gello.sh --gello-port ...`。这个 bringup 只加载 Franka 与键盘状态切换插件，保留原校准、命令话题和控制算法。环境安装、只读通信检查及现场 F2 操作顺序见 [GELLO / Franka 操作说明](scripts/README_gello_franka.md)。启动 Franka bringup 会进入机器人控制程序，应由现场操作员执行。

本机 GELLO 配置已接入 Robotiq 2F-85 的 USB–RS485 驱动，沿用第 8 个命令值及现有夹爪工作线程处理开合、停止和反馈；无需连接原装 Franka Hand 的 TCP 1338 端口。初始化只读状态，未激活的夹爪需要现场显式激活。查看器会跟随所选 bringup 读取配置。

**内部流程简述**

1. **`robo_lab_main`**（`apps/main.cpp`）加载 bringup 路径（默认 `config/bringup_franka_config.yaml`，或第一个命令行参数）。
2. **`Framework::load_bringup`** 解析 YAML，解析 `lib` / `config`，对每个插件 `dlopen`，解析 `robo_lab_plugin_create` / `destroy`，并对每个插件调用 **`initialize(config_path)`**。
3. **`Framework::run_until_signal`** 在每个插件独立线程上启动 **`run()`**，随后阻塞等待 **SIGINT** 或 **SIGTERM**，再调用 **`stop()`** 并 join 线程。
4. 进程退出时，框架析构会停止并卸载插件。

请始终在**仓库根目录**运行（或使用会自动 `cd` 到根目录的 `start.sh`），这样 YAML 中的 `./build/plugins/...` 与 `config/...` 才能保持有效。

---

## 目录结构（简表）

| 路径 | 作用 |
|------|------|
| `apps/main.cpp` | 可执行入口、bringup 路径、框架运行循环 |
| `include/`、`src/` | `Framework`、`MessageSystem`、`RoboLab` 基类 |
| `src/plugins/` | 插件目标（`.so` 输出到 `build/plugins/`） |
| `config/` | Bringup + 各插件 YAML |
| `proto/` | Protobuf 描述文件 |
| `scripts/` | `install_x86.sh`、演示发布脚本、`regen_proto_py.sh` |
| `third_party/` | libfranka、libfreenect2、zenoh-c 等 |

---

## D405 彩色立体预览与 Google WebRTC

新增可选 D405 工作流：插件经 Zenoh 发布包含时间戳与标定的同步 RGB-D，桌面 OpenGL 工具合成左右彩色视图。
Google libwebrtc 使用锁定 submodule、独立依赖准备和本机 DataChannel/H.264 自检。
安装、构建、参数及验收步骤见 [D405 / WebRTC 操作说明](scripts/README_d405_webrtc.md)。Quest WebXR 视频接入属于下一阶段。

## 许可与第三方

`third_party/` 中随仓库提供的第三方库遵循各自上游许可证，请参阅对应项目说明。

---

*英文版说明见 [README.md](README.md)。*
