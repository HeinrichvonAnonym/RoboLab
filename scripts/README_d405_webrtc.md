# D405 RGB-D stereo preview / Google WebRTC host preparation

This milestone runs **D405 → RealSense plugin → Zenoh → desktop stereo reconstruction**.
Google WebRTC is prepared and checked independently. Quest WebXR, camera-to-WebRTC bridging,
head/controller tracking and robot control are subsequent milestones.

## 依赖与构建 / Dependencies and build

沿用仓库内 librealsense 2.56.5，不更新固件。预览复用已有 GLFW/OpenGL、Eigen、yaml-cpp 和 Protobuf。
先按仓库 README 安装基础依赖。新增 Google 源码库采用锁定 Git submodule；已有依赖不迁移。

```bash
source /opt/ros/humble/setup.bash
bash build.sh -DROBOLAB_BUILD_RGBD_PREVIEW=ON -DROBOLAB_BUILD_RGBD_TESTS=ON
```

All RGB-D/WebRTC additions are opt-in. The normal project remains C++17; WebRTC consumers
use C++20. A camera-only build may disable Franka, Kinect and the ROS bridge using the existing switches.
Bringup paths below assume the usual `build/` output directory.

## 相机与预览 / Camera and preview

在仓库根目录分别启动两个终端：

```bash
# Terminal 1: this bringup loads only the RealSense plugin.
./build/apps/robo_lab_main config/bringup_d405.yaml

# Terminal 2: original RGB | synthetic LEFT | synthetic RIGHT
./build/apps/rgbd_stereo_preview config/rgbd_stereo_preview.yaml
```

相机序列号在 `config/plugins/realsense_d405_plugin_config.yaml` 中指定；更换设备时修改它。
空序列号沿用 SDK 自动选机，但多机使用建议始终指定序列号。支持配置 `width`、`height`、整数 `fps`。
新配置默认 640×480@30，彩色 RGB8 和原始深度 Z16。
旧 `realsense/rgb`、`realsense/depth`、`realsense/ir` 的 `kinect.rgbImage` 格式不变。
原 `enable_ir` 仍指左侧 IR，不代表两路彩色。

The new `realsense/rgbd` topic carries one `camera.RgbdFrame` per accepted SDK frameset:
both raw images, byte strides, per-stream device timestamps and frame numbers, timestamp domains,
host monotonic receipt time, device serial, stream intrinsics/distortion, depth-to-color extrinsics
and the device's actual depth unit in metres. Rotation is column-major; coordinates are x right,
y down, z forward. Z16 bytes are little-endian. Calibration travels with every frame, so a late
subscriber needs no separate metadata handshake. Schema version is currently 1.

`publish_rgbd` 默认关闭；新 D405 配置将其开启。它要求同时启用 RGB 和深度。
仅接收同一 SDK frameset 的完整帧对；时间戳域须一致、每路帧号须递增，时间差不能超过
`max_pair_skew_ms`（默认 5 ms）。不插值、不复用上一帧凑对。深度单位直接从设备读取，不能假设每单位 1 mm。
捕获出错或连续 3 秒无帧后停止相机，记录原因；恢复连接后停止并重新启动 bringup。
预览超过 500 ms 没收到有效数据会清屏并显示 `STALE`，避免把旧画面当实时画面。

## 合成参数 / Reconstruction parameters

修改 `config/rgbd_stereo_preview.yaml` 后重新启动预览：

| Parameter | Default | Meaning |
|---|---:|---|
| `min_depth_m` | 0.07 | Nearest retained measured depth |
| `max_depth_m` | 0.50 | Farthest retained measured depth |
| `eye_separation_m` | 0.018 | Distance between synthetic cameras; 0 produces identical views |
| `convergence_m` | 0.30 | Zero-disparity plane using parallel cameras and off-axis projection |
| `max_depth_jump_m` | 0.02 | Maximum depth change across one mesh triangle |

These distances configure rendering, not the sensor's physical capabilities. The 18 mm baseline
is a conservative desktop experiment, not a calibrated human IPD. Quest viewing scale and comfort
still need headset testing. RGB-D synthesis cannot recover colour hidden behind foreground objects.
Invalid depth, range-clipped pixels, discontinuities and newly exposed regions remain black.
The renderer does not fill holes or generate missing content.

`valid_depth` is the percentage of raw depth pixels within the configured range (before colour-FOV
and mesh-edge rejection). A low value can reflect distance, lighting, texture, reflective surfaces,
occlusion or the chosen range. The preview shows only the newest pending complete message; `replaced`
counts overwritten pending messages, `frame_gaps` counts skipped colour frame numbers, `invalid`
counts rejected messages. `max_processing_age_ms` is host receipt-to-mesh/upload age on this machine,
**not** camera-to-display or Quest end-to-end latency. No cross-machine clock comparison is implied.

```bash
# Geometry/subscription diagnostics without a display (still builds the mesh).
./build/apps/rgbd_stereo_preview --headless --duration 30

# Hidden GL window: actual GPU rendering, 10-minute acceptance, one PPM snapshot.
./build/apps/rgbd_stereo_preview --hidden --duration 600 --snapshot /tmp/d405-preview.ppm
```

## Google WebRTC

Pinned versions are in `scripts/webrtc_versions.sh` and the parent repository gitlinks:
WebRTC M157 `95554e4a0c5c343fdc989387d9d0ce18e2a2294b`, depot_tools
`ec77017fb3e090ff02bffb9c6980ebcdd3006ddc`.
Google's internal Git dependencies and tool packages follow the pinned upstream DEPS mechanism.
Only the two named submodules are initialized; unrelated historical submodule mappings are untouched.

```bash
# Explicit network/bootstrap phase; can download several GB. No sudo/package upgrades.
bash scripts/setup_webrtc.sh

# GN/Ninja only: no network/bootstrap, fixed Clang and system libstdc++.
bash scripts/build_webrtc.sh

source /opt/ros/humble/setup.bash
bash build.sh -DROBOLAB_BUILD_RGBD_PREVIEW=ON -DROBOLAB_BUILD_RGBD_TESTS=ON \
  -DROBOLAB_BUILD_WEBRTC=ON
ctest --test-dir build -R '^(rgbd|rgbd_render|webrtc_smoke)$' --output-on-failure
```

`WEBRTC_SYNC_JOBS` defaults to 4; `WEBRTC_BUILD_JOBS` defaults to 6.
Sources/tools/cache stay under `third_party`; compiled output defaults to `build/third_party/webrtc`.
Use `ROBOLAB_WEBRTC_BUILD_DIR` for the build script and the same CMake cache variable for an alternative.
Resolved dependencies are recorded in `third_party/webrtc/dependencies.resolved.txt`; the build profile
and source revisions are recorded in `robolab-build.json` alongside GN output.
The build/configure steps reject missing or mismatched preparation instead of silently downloading.

The fixed Clang consumer is compiled separately from the GCC-built parent project. Ubuntu's
installed `/usr/bin/g++` selects its matching libstdc++ headers/runtime explicitly (a partial newer
GCC installation must not change that choice). The preparation script applies the tracked
`config/webrtc/libstdcxx11.patch`: use `std::nullptr_t` and construct a string key where libstdc++ 11
lacks heterogeneous pair comparison. The pinned upstream HEAD stays unchanged; this exact patch
is the only allowed tracked source difference, and its hash is recorded and checked at build/import.
The WebRTC submodule therefore intentionally appears modified after preparation.
On a subsequent preparation, the script temporarily reverses only this exact patch for gclient
and restores it afterward (also on a sync failure); any other tracked edits are preserved and rejected.
Chromium's unsafe-buffer lint plugin is disabled because its libc++ iterator assumptions do not
match libstdc++; the profile retains libstdc++ assertions and upstream runtime hardening.
The optional Rust implementation is disabled: this pinned revision's complete static archive
does not bundle its Rust objects (the limitation is documented in upstream `BUILD.gn`).
Public WebRTC headers should be consumed with this Clang profile, not directly by the repository's GCC 11 targets.
Upstream test media and test metadata hooks are disabled because those targets are not built.

`RoboLab::WebRTC` is the CMake import target. This profile enables SCTP DataChannel and H.264,
disables upstream examples/tests/tools, desktop capture and WebRTC debug Protobuf, and does not
enable RTX/NVENC integration. The smoke test creates two real PeerConnections, exchanges SDP/ICE
in memory without external STUN/TURN, verifies binary payloads in both directions, and validates
decoded 640×480 H.264 image content. Three rounds exercise teardown and reconnection.
This validates the local dependency/media stack; it is not a Quest compatibility test.

本机使用固定 Clang 和系统 libstdc++，自检目标单独按 C++20 编译，主项目仍为 C++17。
M157 在 libstdc++ 11 下需要两处小型兼容修正，已记录在 `config/webrtc/libstdcxx11.patch`；
准备脚本负责应用，构建和 CMake 导入检查补丁内容与哈希。源码提交仍锁定，但 WebRTC 子模块会显示
这两处预期修改。该版本的可选 Rust 实现尚不能完整打包进静态库，因此使用其 C++ 实现。
自检使用空音频设备，不访问麦克风；视频明确使用软件 H.264 编解码。

## Acceptance / 验收

- Hardware-free tests: legacy/new configuration, pairing, malformed payloads/calibration, stride,
  device depth units, zero baseline, disparity sign, range clipping and depth discontinuities.
- GL test (requires DISPLAY): a near surface must occlude a later-drawn far surface; zero-eye views
  match; missing triangles remain black. Run in a desktop session or an explicitly supplied X server.
- Hardware: run camera plus preview for 600 seconds. Average received rate should be at least 27 FPS
  for the 30 FPS profile, with no sustained pending-frame accumulation. Inspect `invalid`, `replaced`,
  frame gaps and skew, not just the headline frame rate.
- Physical unplug: with capture running, remove only the D405 USB cable. Preview should become stale
  after 500 ms; capture should report failure within its 3-second no-frame deadline. Reconnect,
  Ctrl+C the bringup, restart it, and confirm live frames return. This requires an operator.
- Stop with Ctrl+C and verify no camera-owning process remains. An independent restart must reopen
  the device. Do not run a second capture application against the same camera concurrently.

```bash
# Optional operator-assisted unplug/replug test. Wait for READY before removing USB.
python3 scripts/check_d405_disconnect.py

# Prove the prepared WebRTC build can run without network access (Linux user namespaces).
unshare -Urn bash scripts/build_webrtc.sh
```

The unplug helper checks USB removal, stale display, capture stop and a fresh bringup after
reconnection. It writes logs/result JSON under `build/qa/d405/disconnect`. Its reported durations
are relative to observing USB removal in sysfs, not an electrical USB disconnect timestamp.

## 本机验收记录 / Local acceptance record — 2026-11-16

环境：Ubuntu 22.04 x86_64，D405 序列号 `262622272972`，640×480@30，NVIDIA OpenGL。
These measurements apply to the local camera/desktop and in-process WebRTC tests, not to Quest.

| Check / 检查 | Result / 结果 |
|---|---|
| D405 → Zenoh → hidden OpenGL preview, 600.003 s | 17,982 received and rendered; average 29.9698 FPS |
| Frame gaps / pending replacements / invalid messages | 0 / 0 / 0 |
| Maximum paired device timestamp difference | 0.309814 ms (configured limit 5 ms) |
| Maximum local receipt-to-mesh/upload age | 57.2709 ms; not end-to-end display latency |
| Physical unplug, stale, stop, reconnect + fresh bringup | Passed with operator; camera stopped after removal and resumed after restart |
| Legacy configuration / missing serial / unsupported stream / invalid FPS | Legacy 640×480@15 starts; the three invalid cases report errors |
| WebRTC binary round trip | Three rounds, 4,096 bytes exactly matched in each direction |
| H.264 software encode/transport/decode | Three rounds, 30 validated 640×480 decoded frames per round |
| Prepared WebRTC build with network namespace isolated | Passed both no-op build and 91-step incremental rebuild after repeat preparation |
| Repeat dependency preparation with compatibility patch installed | Passed; pinned revisions and the exact patch are retained |
| Final CTest suite | 10/10 passed, including geometry, GL occlusion, WebRTC and existing integration tests |

Detailed local artifacts (ignored build output): `build/qa/d405/preview-stability.log`,
`unplug3/result.json`, `config-checks/results.json`, `webrtc-smoke.log`,
`webrtc-offline-build.log`, `webrtc-offline-rebuild.log`, `webrtc-setup-idempotent.log`,
`webrtc-smoke-final.log`, `acceptance-final.log` and `d405-preview.png`.
The QA runs released the camera afterward. No firmware update or robot bringup was performed.

## 下一阶段的接入顺序 / Next integration milestones

1. Add a host video source after `build_mesh` / `RgbdRenderer`, render side-by-side into an offscreen
   framebuffer, then feed a WebRTC video track. Keep the latest-frame policy; measure render,
   readback, encode and receive latency separately. The existing preview remains a diagnostic consumer.
2. Add LAN signaling to exchange SDP/ICE and serve the WebXR page over a headset-trusted HTTPS
   origin. First confirm a plain H.264 video and binary DataChannel on Quest 3 before entering VR.
3. In an immersive WebXR session, map the packed image's left/right halves to the corresponding eyes.
   Validate scale, aspect, orientation, zero disparity and comfortable preview baseline on hardware.
   Camera viewpoints remain fixed until a separate pose-aware rendering design is accepted.
4. Define versioned DataChannel envelopes for ordinary Zenoh messages, including ordering, expiry,
   maximum size and disconnect behavior. Pose input and robot control require a separate later scope.

下一阶段：复用 RGB-D/重建模块生成一帧左右并排视频 → Google WebRTC 编码与媒体轨道 →
Quest 浏览器解码 → WebXR 分眼渲染。一般消息单独走 DataChannel。加入信令、安全访问地址、
网络自适应、端到端延迟测量以及 Quest 实机验收后，才算完整 VR 链路打通。
