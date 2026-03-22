# roboLab（中文说明）

**roboLab** 是一套基于 **Zenoh** 的**去中心化消息层**与**插件运行时**，面向**具身智能 / 机器人**软件。进程通过 **Zenoh**（发布/订阅、发现、可选路由）通信，而不是依赖单一中心化的「ROS 式 master」；机器人与传感器相关逻辑放在**动态加载的插件**里，由 **YAML bringup** 与 **Protobuf** 消息描述驱动。

核心组件：

- **`MessageSystem`** — 基于 [zenoh-c](https://github.com/eclipse-zenoh/zenoh-c) 的 C++ 封装，按 key expression 做发布/订阅。
- **`Framework`** — 读取 **bringup YAML**，`dlopen` 加载插件 `.so`，依次调用 `initialize()` / `run()` / `stop()`，在 **Ctrl+C** 或 **SIGTERM** 时干净退出。
- **插件**（如 Franka、Kinect）— 实现 `Plugin` 接口，用 YAML 读参数，并可通过 Zenoh 交换 **Protobuf** 负载（见 `proto/`）。

---

## 安装（x86_64 Linux）

在 **Debian/Ubuntu x86_64** 上，可先运行辅助脚本（安装编译工具链、CMake、Protobuf、USB/Eigen/Poco 等第三方依赖、用于编译 **zenoh-c** 的 **Rust/Cargo**，以及可选的 **Zenoh 路由** `zenohd`）：

```bash
bash scripts/install_x86.sh
```

然后在**仓库根目录**编译 C++ 工程：

```bash
bash build.sh
```

**为什么需要 Cargo / Zenoh？**

- **Cargo + Rust** — CMake 会编译 **zenoh-c**，其实现为 Rust。需要可用的 `cargo`/`rustc`，除非你已安装带 CMake 配置的 **zenohc** 并通过 `CMAKE_PREFIX_PATH` / `ROBOLAB_ZENOHC_ROOT` 指给顶层 `CMakeLists.txt`。
- **Zenoh** — 对等端常用 **peer** 模式 + 多播发现（往往无需额外安装）。若需要**路由节点**，可执行 `INSTALL_ZENOHD=1 bash scripts/install_x86.sh` 或 `cargo install zenohd --locked`。Python 示例：`pip install -r scripts/requirements.txt`（见 `scripts/publish_demo_command.py`）。

**其他依赖**

- 将 **zenoh-c** 放到 `third_party/zenoh-c`（克隆），或使用系统安装并设置 `CMAKE_PREFIX_PATH` / `ROBOLAB_ZENOHC_ROOT`。
- **libfreenect2** 为可选（Kinect 插件）；默认关闭 CUDA（`ROBOLAB_LIBFREENECT2_CUDA`），避免 `nvcc` 与主机 C++ 编译器配置问题。

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

## 许可与第三方

`third_party/` 中随仓库提供的第三方库遵循各自上游许可证，请参阅对应项目说明。

---

*英文版说明见 [README.md](README.md)。*
