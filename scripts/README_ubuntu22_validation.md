# Ubuntu 22.04 / ROS 2 Humble 验证记录

验证日期：2026-11-10（Asia/Jakarta）。本记录针对本次工作区修改，未进行机器人或相机实机联调。

## 环境与依赖

- Ubuntu 22.04.5，5.15.0-1032-realtime 内核；GCC 11.4.0、CMake 3.22.1、Protobuf 3.12.4。
- ROS 2 Humble；Rust 1.93.0；zenoh-c 1.8.0，Cargo 使用 `--locked`，并通过离线锁文件解析检查。
- CUDA 11.8 nvcc/cudart 开发组件；保留 NVIDIA 595.99.02 驱动及原实时内核。必要 CUDA 包约 45.6 MB，避免安装完整 Toolkit。
- RTX 4090 的简单 CUDA kernel 执行成功，返回预期值 42；这不等于 Kinect 实机性能验收。
- 保留 libfranka 0.9.0、librealsense 2.56.5 和其他第三方实现。补齐 CUDA samples、serial 的缺失子模块地址，保持仓库原有 gitlink 提交不变。
- 原 zenoh-c 锁文件仍引用 1.7.2，而 manifest 已指定 1.8.0；本次仅对齐这些 Zenoh 包及其固定提交，其他 crate 版本保持不变。

## 构建结果

| 检查 | 结果 |
|---|---|
| 默认配置，Kinect CUDA 开启，TensorRT 关闭 | 成功；主程序及 10 个插件生成 |
| 全新目录全量 C/C++ 构建 | 成功；禁用 ccache，使用本机已编译的同版本 zenoh-c 安装包 |
| 关闭 Franka、双相机及 ROS 桥接 | 成功；显式禁止查找 CUDA、rclcpp、Poco 后仍可构建 |
| Kinect CPU 路径，关闭 Franka、RealSense、ROS 桥接 | 成功；显式禁止查找 CUDA |
| 已安装 zenohc 的 CMake 接入 | 成功；上述两种精简组合使用同源码产物经上游安装规则生成的本地包 |
| 显式开启 TensorRT，但 SDK 缺失 | 按预期在配置阶段失败，提示 TensorRT 头文件和相关开关 |
| 静态检查 | `bash -n`、Python 编译检查及 `git diff --check` 通过 |

默认插件：Franka、Kinect、RealSense、笛卡尔控制、录制、Franka 回放、键盘、IMU 遥操作、N100 IMU、ROS 桥接。`LIBFREENECT2_WITH_CUDA_SUPPORT` 已在实际生成头文件中确认。N100 依赖缺失时现会明确报错，避免静默少构建一个插件。

## 功能检查

默认构建的 CTest **6/6 通过，用时约 14 秒**：

1. `gripper_worker`：已有无硬件夹爪测试。
2. `kinect_sampling`：已有图像采样测试。
3. `plugin_load`：实际加载全部默认动态库及工厂接口，创建/销毁插件，不初始化硬件。
4. `record_replay`：独立 Zenoh 话题产生合成状态，录制并完成 HDF5 文件，验证插值及回放命令；覆盖旧 7 列数据与非法列数。
5. `ros_bridge`：两个独立 ROS Context，验证支持类型的双向转换、两种类型名称、私有/绝对话题、QoS、异常 payload/配置、初始化失败回收、重复停止及启动前停止。
6. `bridge_shutdown`：在独立进程运行真实入口，验证 SIGINT、SIGTERM 均正常退出。

精简组合与 CPU Kinect 组合各自的插件加载、录制回放测试也均 **2/2 通过**。

测试确认并修复两个原有功能问题：

- JointStateArray 可以被 Protobuf 解析为没有关节的 RobotObservation，导致桥接丢失关节值；现在会正确走 JointStateArray 转换。
- 录制器写入 8 列关节数据，而回放器只接受 7 列；现在兼容两种格式，仍仅控制 7 个机械臂关节，忽略额外夹爪列，不改变线上协议。

## 构建计时

| 场景 | 实测墙钟时间 | 条件 |
|---|---:|---|
| Zenoh 首次构建 | 303.45 秒 | 包含 Rust 依赖下载和编译，Cargo 并行度 4 |
| 全量 C/C++ 构建 | 62.58 秒 | 全新构建目录、禁用 ccache、并行度 8，复用已编译 Zenoh；含配置时间 |
| 无修改重跑 `build.sh` | 2.05 秒 | 没有重新编译或链接 |
| 触碰桥接源文件后重跑 | 1.98 秒 | ccache 命中，仅重编桥接源文件并重链桥接插件 |

首次系统安装下载时间不计入构建计时。这些是本机观测，不保证其他机器或网络上的耗时。Zenoh 与 C/C++ 分开计时，不能据此声称测得一次完整的冷启动总时间。

日志位于本工作区的 `build/validation/`，构建产物及日志不提交仓库。测试命令：

```bash
source /opt/ros/humble/setup.bash
bash build.sh -DROBOLAB_BUILD_GRIPPER_TESTS=ON \
  -DROBOLAB_BUILD_KINECT_TESTS=ON -DROBOLAB_BUILD_INTEGRATION_TESTS=ON
ctest --test-dir build --output-on-failure
```

## 尚未验收

Franka 实机运动与固件兼容、Kinect/RealSense 图像与帧率、N100 串口数据、现场设备权限及 TensorRT 模型推理均未验证。相机 CUDA 处理路径已编译，但没有接入 Kinect 验证实际采样性能。`ros_ws` 与所有 Protobuf 定义保持原样；现场验收步骤见根目录 README。
