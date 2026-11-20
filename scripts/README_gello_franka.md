# GELLO → Franka Panda / Robotiq 2F-85 现场启动

本流程沿用仓库现有 GELLO 校准、Protobuf、Zenoh 话题及 Franka 控制器。独立 bringup `config/bringup_gello_franka.yaml` 仅加载 Franka 和键盘插件；相机、录制和笛卡尔控制器未加载。原默认 bringup 保持不变。

这台 Panda 安装的是 **Robotiq 2F-85**。专用配置 `config/plugins/franka_gello_plugin_config.yaml` 设置 `gripper_type: robotiq_2f85`，机械臂通过 FCI 连接，夹爪通过 USB–RS485 连接。原 `franka_plugin_config.yaml` 仍适用于 Franka Hand；本机请使用上面的专用 bringup。

## 一次性安装

在仓库根目录执行：

```bash
bash scripts/setup_teleop.sh
```

脚本使用系统 Python 3.10 创建 `.venv-teleop`，安装 `scripts/requirements-teleop.txt` 中固定的直接依赖，初始化仓库锁定提交的 DynamixelSDK，并生成 `scripts/gen/franka_pb2.py`。不需要 Conda、PyTorch、TensorRT 或完整仿真训练环境。依赖系统 `python3-venv`、`protobuf-compiler` 和 `lsof`；Ubuntu 开发依赖安装入口仍为 `scripts/install_x86.sh`。已有环境会复用，不会删除。

Robotiq 驱动编入现有 Franka 插件，使用 Linux 串口接口，没有新增运行时软件包。更新代码后构建插件和两个可选诊断工具：

```bash
bash build.sh
cmake --build build --target franka_connection_check robotiq_connection_check --parallel 8
```

当前 Panda GELLO 的串口及标定已存在于 `gello/agents/gello_agent.py`：

```text
/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBYPQ5E-if00-port0
```

使用完整的 by-id 路径；`/dev/ttyUSB0` 会随插拔变化，也不匹配原校准字典。连接别的 GELLO 时需要核实其串号与校准，不能直接套用上述参数。

Robotiq 的串口为 `/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAL8Y3V-if00-port0`，与 GELLO 是两个独立设备。为这两个设备授予当前本地桌面用户权限（本机已完成）：

```bash
sudo install -m 0644 scripts/70-robolab-gello.rules /etc/udev/rules.d/70-robolab-gello.rules
sudo udevadm control --reload-rules
# 重新插拔 GELLO USB；也可针对当前实际设备触发规则：
sudo udevadm trigger --action=change /sys/class/tty/ttyUSB0
sudo udevadm trigger --action=change /sys/class/tty/ttyUSB1
udevadm settle
getfacl /dev/ttyUSB0
```

规则只匹配 `FTBYPQ5E` 和 `DAAL8Y3V`，不改变其他串口权限。设备编号以本机实际 by-id 链接为准。纯 SSH / 无本地活动桌面的会话可使用 `dialout` 组授权，加入组后重新登录。运行插件和 GELLO 发布器不需要 sudo。

## 先检查通信

确认 Franka Desk 已启用 FCI。当前 `robot_ip` 是 `172.16.0.40`；本机网卡地址是 `172.16.0.150`。分别检查机械臂和 Robotiq；下面两条命令只读取状态，不启动控制、激活或开合：

```bash
timeout 15s build/apps/franka_connection_check 172.16.0.40 --arm-only
build/apps/robotiq_connection_check \
  /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAL8Y3V-if00-port0
```

机械臂成功时输出 `ARM OK (TCP 1337)`。原装 Franka Hand 使用另一个端口 1338；旧 libfranka 将两者的连接拒绝都打印成 `FCI refused`，因此不能只凭该字符串判断机械臂 FCI 状态。现在插件日志会标明失败的具体接口。Robotiq 不使用 1338；此处保留 `--arm-only`。修改机器人 IP 或夹爪串口时，编辑专用配置文件。

Robotiq 正常通信会打印 `ROBOTIQ OK ready=... fault=...`；初始化要求已激活（`ready=1`）。[故障码 `9` 表示通信看门狗超时](https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_TM-OMRON_InstructionManual_20190206.pdf)：启动只读检查不会发送恢复或运动命令，代码允许首次现场开合命令尝试恢复通信，实际恢复需现场验证；若没有有效反馈，动作会超时报错并尝试停止。其他故障会阻止插件初始化或开合。若 `ready=0`，先停止所有夹爪控制进程，由现场人员执行下面的显式激活操作；**它会使手指运动进行标定**：

```bash
build/apps/robotiq_connection_check \
  /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAL8Y3V-if00-port0 --activate
```

插件初始化从不自动激活夹爪。激活及寄存器定义参见 [Robotiq 控制手册](https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190227.pdf)。

单独读取 GELLO：

```bash
./start_gello.sh \
  --gello-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBYPQ5E-if00-port0 \
  --dry-run
```

`--dry-run` 会连接真实 GELLO 并关闭其舵机力矩，打印角度，但不发布机器人命令。`q` 是归一化后的 7 个 Panda 关节角，单位 rad；`gripper` 是原有的 0–1 夹爪量；`clipped` 是被限位裁剪的关节编号。Ctrl+C 退出并关闭串口。启动失败不会使用模拟零值，超过 1 秒没有新舵机数据会报错退出；驱动不会杀死串口占用者或自行改权限。

若出现 `comm=-3001`，SDK 的含义是未收到舵机应答，并非已经确认舵机自身报错。先检查舵机外接电源、USB 转接器到首个舵机的总线连接及串口占用，再核实 ID 和波特率。驱动现在要求全部舵机确认关闭力矩后才启动读取线程；失败最多重试 3 次，不再吞掉异常并误报 `Successfully initialized`，也不再在机器人包装层重复发送初始化关闭力矩命令。

## 现场启动和首次运动

以下由现场操作员执行。启动 Franka bringup 即会进入原有实时控制程序；它不是只读检查。确认现场运动空间和停机装置就绪，并且没有其他发布器向 `franka/command` 或 `trigger` 发送数据。

1. 终端 A 启动机器人端：

   ```bash
   ./start.sh config/bringup_gello_franka.yaml
   ```

   等待状态 `init`。查看器会自动打开，并读取同一个 bringup 选中的 Franka 配置：白色模型显示实际状态，蓝色模型显示命令。查看器关闭不等于机器人停止；不需要图形界面时可设置 `ROBOLAB_SKIP_MUJOCO_VIEWER=1`。当前 URDF 的夹爪外观仍为 Franka Hand，仅用于开度示意，不是 Robotiq 的几何或碰撞模型。

2. 在终端 A 按 **F2**，执行 `init → gohome`。这一步会产生首次机械臂运动。等日志 `go_home complete -> standby` 后再继续；回零期间额外的 F2 不会停止回零。

3. 使用上面的 `--dry-run` 在终端 B 检查 GELLO，手动对齐其姿态与 Panda。逐关节核对方向和数值，不应带着明显的姿态差进入跟随。原回零角为 `[-0.32, -0.9, 0.13, -2.75, 0.18, 1.95, 0.49]`，以实际状态为准。角度归一化只消除整圈跳变，不替代机械校准。

4. Ctrl+C 停止 dry-run，然后在终端 B 启动命令发布器：

   ```bash
   ./start_gello.sh \
     --gello-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBYPQ5E-if00-port0
   ```

   默认以 100 Hz 向 `franka/command` 发布 `RobotCommand`，保留第 8 个夹爪值。机器人仍应处于 `standby`；查看器可先对比白色实际姿态与蓝色目标姿态。

5. 在终端 A 按 **F2**，执行 `standby → inference`，开始跟随。先做小幅、单关节动作，再验收夹爪和连续遥操作。GELLO 第 8 个值小于 `0.5` 时请求打开，达到 `0.5` 时请求闭合；沿用原有二态开合逻辑。夹爪命令只在 `inference` 接受，切回 `standby` 时取消正在执行的开合。

6. 结束时先在终端 A 按 **F2** 返回 `standby`，再在终端 B Ctrl+C 停止发布器，最后停止终端 A。仅停止发布器不会使机器人状态机自动回到 `standby`，原控制器可能继续执行最后目标；本次没有改写该控制逻辑。异常运动使用现场停机装置。

F2 必须在运行键盘插件的终端 A 中按下，不是在查看器窗口或 GELLO 终端中。键盘插件需要交互式 TTY。不要同时运行默认 bringup 或另一个控制器，以免多个进程争用机器人或向同一命令话题发数据。

专用配置默认开度 `0.085 m`、速度 `0.03 m/s`、原始力寄存器 `32/255`，力寄存器值不是牛顿。反馈开度由可配置编码器端点 `13/230` 估算，沿用 [上游 2F-85 驱动的换算方式](https://github.com/ros-industrial-attic/robotiq/blob/kinetic-devel/robotiq_2f_gripper_control/src/robotiq_2f_gripper_control/robotiq_2f_gripper_ctrl.py)，实际指尖开度和夹持力需要现场确认。物体接触可能使目标开度不可达，日志会显示未到位；配置关闭自动重试，避免同一闭合请求反复施力。切回 standby 不会自动松开已夹持的物体。

## 本机验证记录

2026-11-10，Ubuntu 22.04 / ROS 2 Humble：

| 项目 | 结果 |
|---|---|
| GELLO Python 环境、SDK、命令行入口 | 已安装，`./start_gello.sh --help` 正常 |
| GELLO 实际串口采样 | 7 个关节 + 夹爪正常，约 100 Hz 发布循环；静止采样没有限位裁剪 |
| GELLO → Python Zenoh 1.5.1 → 主工程 C++ Zenoh 1.8.0 | 独立 `robolab_validation/gello/<随机值>` 话题收到 202 帧有效命令，非法帧 0；未发布到机器人命令话题 |
| Panda FCI 只读状态 | Desk 启用 FCI 后读取 101 帧，`robot_mode=Idle`，正常结束 |
| MuJoCo 查看器 | 修正 URDF 导入丢失网格目录的问题；无 GUI 和桌面 GUI 均可启动、Ctrl+C 正常退出 |
| `start.sh` | 默认 venv、跳过查看器、缺失 Conda、缺失 Python、无桌面共 5 种情况通过；使用只有 ROS 桥接的临时 bringup 测试，并验证 SIGINT/SIGTERM 退出 |
| GELLO Python 回归 | 13 项原消息/归一化测试 + 10 项驱动异常测试通过；覆盖力矩命令持续超时、瞬时超时恢复、失败关闭端口及避免重复初始化命令 |
| 主工程 CTest | 6/6 通过，14.09 秒 |

原 GELLO 校准、Panda 控制算法、默认 Franka 配置和 Protobuf 定义未修改；新增了本机 Robotiq 专用配置。测试采样时 GELLO 第 2 关节约 `1.261 rad`，与 Panda 回零目标 `-0.9 rad` 有明显差异；现场需要先对齐，不能直接切换跟随。

通信日志在 `build/validation/teleop/`。跨版本 Zenoh 接收测试在关闭会话时出现一条 `Unable to publish transport event: session closed` 日志，两端均正常结束且 202 帧解析成功；不将其作为实时控制性能结论。

本次没有启动 Franka 控制模式、回零、机械臂跟随或夹爪动作；实时性能、运动方向、机械校准和运动中的故障处置仍需现场验收。

同日故障复查：曾观察到 8 个舵机均不回应，随后在未修改波特率、ID 或校准的情况下全部于 57600 baud 正常回应。物理超时原因未确定。修复初始化错误处理后，9 秒 dry-run 正常结束；Panda 只读检查再次收到 101 帧 `Idle` 状态。日志分别为 `gello_dry_run_retry.log`、`gello_retry_tests.log` 和 `franka_read_retry.log`，位于上述验证目录。

可重复的无硬件回归命令：

```bash
env -u PYTHONPATH .venv-teleop/bin/python -m unittest discover -s tests -p 'test_gello_*.py' -v
ctest --test-dir build --output-on-failure
```

### 2026-11-16 Robotiq 接入验证

- 实际 USB–RS485 状态读取成功：115200 baud、slave 9，`ready=1`、`fault=9`、原始位置 3，估算开度 0.085 m。没有发送激活或开合命令。
- Desk 开启 FCI 后，机械臂 TCP 1337 的 libfranka 握手及状态读取成功。
- 通过 `Framework::load_bringup` 加载实际 `config/bringup_gello_franka.yaml`，Panda、Robotiq 和键盘插件初始化均成功；测试程序没有调用 `run()` 或 `run_until_signal()`。
- 新增 11 项伪终端 Modbus 测试：只读初始化、分片/错误/异常报文、串口独占与释放、开合寄存器、显式激活及失败清理、停止与超时、故障/接触处理、命令应答校验、现有工作线程的开合与 standby 取消。
- 默认构建及 CTest **7/7 通过，14.98 秒**。新增驱动没有引入 ROS 控制器、Python 夹爪进程或新的系统依赖。
- 查看器选择 Robotiq 配置（0.085 m）及 `start.sh` 带空格路径传参、SIGINT 退出检查通过。
- 当日再次进行 5 秒实际 GELLO dry-run，关节及夹爪采样正常、无裁剪，正常退出并释放串口；查看器使用专用 bringup 的无 GUI 启动检查也通过。

日志保存在 `build/validation/teleop/`，主要文件为 `panda_robotiq_initialize_20261116.log`、`robotiq_ctest_20261116.log` 和 `robotiq_full_build_20261116.log`。初始化退出仍可能出现已知 Zenoh 管理事件的 `session closed` 日志，进程已正常退出。

**尚未验收：** 实际 Robotiq 激活、开合方向、夹持力、带物体接触、运动中取消/断线，以及 Panda 实际运动和实时性能。协议与异常路径通过模拟测试不代表这些现场项目已经通过。
