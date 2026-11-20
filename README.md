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

## Install and build (Ubuntu 22.04 x86_64 / ROS 2 Humble)

Install ROS 2 Humble first, then run from the repository root:

```bash
bash scripts/install_x86.sh
source /opt/ros/humble/setup.bash
bash build.sh
```

The installer uses sudo for distro development packages and the **CUDA 11.8 compiler/runtime development packages** (`cuda-nvcc-11-8`, `cuda-cudart-dev-11-8`). It keeps the installed NVIDIA driver and kernel. The full CUDA toolkit, profilers and math libraries are unnecessary for Kinect. It also initializes the already-pinned serial library and CUDA samples submodules; the latter supplies `helper_math.h`, and no samples are built. Rust **1.93.0** is installed in the user's home, without changing an existing default toolchain. Bundled **zenoh-c 1.8.0** builds with its matching lock file and is reused on later builds.

`INSTALL_SYSTEM=0` installs only Rust; `INSTALL_RUST=0` installs only system dependencies; `INSTALL_CUDA=0` skips CUDA installation for a deliberately CPU-only setup. An optional router can be installed with `INSTALL_ZENOHD=1`; normal peer communication does not require one. No dependencies are installed by `build.sh`.

Build defaults:

| CMake option | Default | Effect when disabled |
|---|---|---|
| `ROBOLAB_BUILD_FRANKA` | ON | Skip Franka hardware plugin and libfranka |
| `ROBOLAB_BUILD_KINECT` | ON | Skip Kinect plugin and libfreenect2 |
| `ROBOLAB_BUILD_LIBREALSENSE` | ON | Skip RealSense plugin and SDK |
| `ROBOLAB_LIBFREENECT2_CUDA` | ON | Use the existing CPU Kinect pipeline |
| `ROBOLAB_BUILD_ROS_BRIDGE` | ON | Skip ROS 2 bridge and ROS package discovery |
| `ROBOLAB_BUILD_TENSORRT` | OFF | Skip inference library and demo plugin |

Control, recording/replay, keyboard and IMU plugins remain enabled. TensorRT can be enabled with `bash build.sh -DROBOLAB_BUILD_TENSORRT=ON -DROBOLAB_TENSORRT_ROOT=/path/to/TensorRT`; provide a compatible SDK separately. `ROBOLAB_CUDA_ROOT` defaults to `/usr/local/cuda-11.8`. Required enabled dependencies fail clearly rather than silently removing features. The legacy ROS 1 `ros_ws` is retained separately and is not built or migrated.

`build.sh` forwards CMake arguments, preserves the build cache and uses ccache when available. Default parallelism is capped at 8 C/C++ jobs and 4 Cargo jobs; override with `CMAKE_BUILD_PARALLEL_LEVEL` and `CARGO_BUILD_JOBS`. The default configuration is Release. Third-party examples/tools/tests are off. Build one plugin with `cmake --build build --target ros_bridge_plugin`. Outputs remain under `build/apps` and `build/plugins`; use a bringup matching the enabled plugins.

An installed compatible `zenohc` CMake package can still be supplied via `CMAKE_PREFIX_PATH`/`zenohc_DIR`; `ROBOLAB_ZENOHC_ROOT` selects alternative zenoh-c **sources**. Direct CMake builds of the bundled sources require `source ~/.cargo/env`. Protobuf code generation uses the system protoc by default so it matches Ubuntu's C++ protobuf library.

## ROS 2 bridge compatibility

Existing YAML list/field names, Zenoh keys, Protobuf schemas and plugin library names are unchanged. `ros_type` accepts both `sensor_msgs/JointState` and `sensor_msgs/msg/JointState` (and equivalent spellings for other supported types). Unsupported types retain the byte-array fallback.

The node is `robo_lab_ros_bridge_plugin`. Absolute ROS topics remain absolute; relative topics retain ROS 1's private-node resolution, e.g. `franka_4/joint_state` becomes `/robo_lab_ros_bridge_plugin/franka_4/joint_state`. Default QoS is keep-last 10, reliable, volatile. Each topic may specify:

```yaml
qos: {depth: 10, reliability: best_effort}
```

Use best-effort subscriptions when consuming best-effort ROS sensors. The plugin owns a separate ROS context; the framework handles Ctrl+C/SIGTERM and orderly teardown.

## Hardware-free checks

```bash
source /opt/ros/humble/setup.bash
bash build.sh -DROBOLAB_BUILD_GRIPPER_TESTS=ON \
  -DROBOLAB_BUILD_KINECT_TESTS=ON -DROBOLAB_BUILD_INTEGRATION_TESTS=ON
ctest --test-dir build --output-on-failure
```

Checks cover plugin loading, gripper worker, Kinect sampling, ROS/Zenoh conversions in both directions, legacy/new type names, QoS, invalid configuration/payloads, shutdown and synthetic HDF5 recording/replay. Integration tests use temporary files and isolated topic names; no robot or camera is initialized. Hardware motion, camera throughput and TensorRT inference require separate acceptance. Replay accepts legacy seven-column and current eight-column recordings, still commanding only seven arm joints. The bridge also fixes JointStateArray payloads being misread as empty RobotObservations.

See the [Ubuntu 22.04 validation record](scripts/README_ubuntu22_validation.md) for measured results.

For hardware acceptance, connect the configured Franka/cameras/IMU, verify device identity and configured topics, then check Kinect logs select CUDA, image formats/rates, ROS 2 topic visibility and recording/replay with the existing operating procedure. Verify orderly shutdown before any supervised robot motion. An untested TensorRT backend must be validated with its actual SDK/model.

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

The optional MuJoCo viewer uses the repository's `.venv-teleop` by default; a Conda environment named `roboLab` is no longer required. Run `bash scripts/setup_teleop.sh` to install the GELLO and viewer dependencies. A missing display or Python dependency produces a diagnostic and skips the viewer. Set `ROBOLAB_SKIP_MUJOCO_VIEWER=1` to disable it, or `ROBOLAB_PYTHON=/path/to/python` to select an interpreter. Explicit `ROBOLAB_MUJOCO_CONDA_ENV` overrides remain supported.

For GELLO → Panda, use `./start.sh config/bringup_gello_franka.yaml` and the separate `./start_gello.sh --gello-port ...` publisher. This bringup loads Franka and keyboard state transitions, retaining the existing calibration, topics and control algorithms. See the [GELLO / Franka guide](scripts/README_gello_franka.md) for setup, read-only communication checks and the on-site F2 sequence. The Franka bringup enters the robot control program and must be started by the on-site operator.

The dedicated GELLO configuration selects a Robotiq 2F-85 over USB–RS485. It uses the existing eighth command value and gripper worker for open/close, cancellation and feedback, without connecting to the Franka Hand port (TCP 1338). Initialization only reads status; activation is an explicit on-site operation. The viewer reads the Franka configuration selected by the same bringup.

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

## H5 dataset conversion and review

Use `scripts/convert_h5_to_lerobot.py` for headless LeRobot v3 conversion, or
`scripts/review_h5.py` for dual-camera browser playback and clip exclusion.
See [setup, commands, synchronization and editing rules](scripts/README_lerobot.md).

## D405 stereo preview and Google WebRTC

The optional D405 workflow publishes calibrated, synchronized RGB-D frames over Zenoh and reconstructs
left/right colour views in a desktop OpenGL preview. Google libwebrtc is pinned as a submodule with
an explicit dependency setup step and a separate local DataChannel/H.264 check. See the
[D405/WebRTC setup and acceptance guide](scripts/README_d405_webrtc.md). Quest WebXR streaming is the next milestone.

## Python inference (Franka, Kinect + RealSense)

The CPU mock policy generates 50-step, 8D absolute joint/gripper targets and
executes them over Zenoh at 30 Hz, gated by the existing F2 state machine.
See [setup, offline verification, dedicated bringup and hardware acceptance](scripts/README_python_inference.md).

## License / third party

Vendored libraries under `third_party/` keep their respective licenses. See each upstream project for details.
