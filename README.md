# roboLab

*中文说明： [README_cn.md](README_cn.md)*

## TODO:

- achieve a log system
- optimize build & re-build system

**roboLab** is a **Zenoh-based, decentralized messaging layer** and **plugin runtime** for **embodied AI / robotics** software. Processes communicate over **Zenoh** (pub/sub, discovery, optional routing) instead of a single central ROS-like master, while robot- and sensor-specific logic lives in **dynamically loaded plugins** driven by **YAML bringup** and **Protobuf** message schemas.

At its core:

- **`MessageSystem`** — C++ helper on top of [zenoh-c](https://github.com/eclipse-zenoh/zenoh-c) for publish/subscribe on key expressions.
- **`Framework`** — reads a **bringup YAML**, `dlopen`s plugin `.so` files, calls `initialize()` / `run()` / `stop()`, and shuts down cleanly on **Ctrl+C** or **SIGTERM**.
- **Plugins** (e.g. Franka, Kinect) — implement `Plugin`, use YAML for parameters, and can exchange **Protobuf** payloads over Zenoh (see `proto/`).

---

## Install (x86_64 Linux)

On **Debian/Ubuntu x86_64**, use the helper script (installs compiler toolchain, CMake, Protobuf, USB/Eigen/Poco deps for bundled third_party libs, **Rust/Cargo** for building **zenoh-c**, and optionally the **Zenoh router** `zenohd`):

```bash
bash scripts/install_x86.sh
```

Then build the C++ tree from the repository root:

```bash
bash build.sh
```

**Why Cargo / Zenoh?**

- **Cargo + Rust** — The CMake build compiles **zenoh-c**, which is implemented in Rust. You need a working `cargo`/`rustc` unless you install a prebuilt **zenohc** CMake package and point CMake at it (see `ROBOLAB_ZENOHC_ROOT` in the top-level `CMakeLists.txt`).
- **Zenoh** — Peers often run in **peer** mode with multicast scouting (no extra install). For a **router**, run `INSTALL_ZENOHD=1 bash scripts/install_x86.sh` or `cargo install zenohd --locked`. Python demos: `pip install -r scripts/requirements.txt` (see `scripts/publish_demo_command.py`).

**Other dependencies**

- Place **zenoh-c** under `third_party/zenoh-c` (clone) or set `CMAKE_PREFIX_PATH` / `ROBOLAB_ZENOHC_ROOT` if you use a system install.
- **libfreenect2** is optional (Kinect plugin); CUDA is **off** by default (`ROBOLAB_LIBFREENECT2_CUDA`) to avoid `nvcc` host-compiler issues.

---

## Configuration model

### 1. Bringup file (which plugins to load)

Default used by `start.sh` / `main` is `config/bringup_franka_config.yaml`. It lists plugins:

```yaml
plugins:
  - name: franka_plugin
    config: config/plugins/franka_plugin_config.yaml
    lib: ./build/plugins/libfranka_plugin.so
  - name: kinect_plugin
    config: config/plugins/kinect_plugin_config.yaml
    lib: ./build/plugins/libkinect_plugin.so
```

- **`name`** — logical name (logging).
- **`config`** — plugin-specific YAML (paths resolved like `lib`).
- **`lib`** — path to the plugin **shared object**.

If the bringup file lives under a directory named `config/`, relative paths are also resolved from the **repository root**, so `./build/plugins/...` works when you run the binary from the repo root.

### 2. Per-plugin YAML

Each plugin reads its own file (e.g. `config/plugins/franka_plugin_config.yaml`): robot IP, Zenoh **topic** key, control mode, gains, etc.

### 3. Protobuf (`proto/`)

`.proto` files under `proto/` are compiled automatically into the **`robo_lab_proto`** library (see top-level `CMakeLists.txt`). Plugins and tools agree on wire types (e.g. `franka.RobotCommand`). Regenerate Python stubs with `bash scripts/regen_proto_py.sh` when schemas change.

---

## Running — `start.sh`

**`start.sh`** is the **recommended launch entry**: it starts the built binary **`build/apps/robo_lab_main`** from the **repository root** (so relative paths in YAML resolve correctly) and forwards any extra arguments.

```bash
./start.sh
# same as:
./build/apps/robo_lab_main

# custom bringup:
./start.sh /path/to/my_bringup.yaml
```

**What happens internally**

1. **`robo_lab_main`** (`apps/main.cpp`) loads the bringup path (default `config/bringup_franka_config.yaml`, or first CLI argument).
2. **`Framework::load_bringup`** parses the YAML, resolves `lib` / `config`, `dlopen`s each plugin, resolves `robo_lab_plugin_create` / `destroy`, calls **`initialize(config_path)`** on each.
3. **`Framework::run_until_signal`** starts each plugin’s **`run()`** on its own thread, then waits for **SIGINT** or **SIGTERM**, calls **`stop()`**, and joins threads.
4. On process exit, the framework destructor stops and unloads plugins.

Always run from the **repo root** (or use `start.sh`, which `cd`s there) so `./build/plugins/...` and `config/...` in YAML stay valid.

---

## Layout (short)

| Path | Role |
|------|------|
| `apps/main.cpp` | Executable entry, bringup path, framework run loop |
| `include/`, `src/` | `Framework`, `MessageSystem`, `RoboLab` base |
| `src/plugins/` | Plugin targets (`.so` → `build/plugins/`) |
| `config/` | Bringup + per-plugin YAML |
| `proto/` | Protobuf schemas |
| `scripts/` | `install_x86.sh`, demo publisher, `regen_proto_py.sh` |
| `third_party/` | libfranka, libfreenect2, zenoh-c, … |

---

## License / third party

Vendored libraries under `third_party/` keep their respective licenses. See each upstream project for details.
