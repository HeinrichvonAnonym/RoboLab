#pragma once

#include "message_system.h"
#include "plugin_interface.h"
#include "plugins/franka_replayer.h"

#include <atomic>
#include <memory>
#include <string>

namespace robo_lab {

// Replays a recorded franka_state trajectory by publishing
// franka.RobotCommand (TYPE_JOINT_TARGET) messages onto a Zenoh topic.
//
// Publishes at a fixed user-configurable rate using linear interpolation,
// so the consumer (e.g. franka_plugin) sees a smooth signal even though the
// recording was sampled at ~30 Hz.
class FrankaReplayerPlugin : public Plugin {
 public:
  FrankaReplayerPlugin() = default;
  ~FrankaReplayerPlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  bool load_config(const std::string& config_path);
  bool publish_command(const std::array<double, 7>& q, double sys_time);

  std::string config_path_;
  std::string trajectory_file_;
  std::string cmd_topic_{"franka/command"};
  std::string control_mode_{"position"};

  // Timing of the publication loop.
  double publish_hz_{100.0};
  // 1.0 = realtime; 2.0 = twice as fast; 0.5 = half speed.
  double rate_factor_{1.0};
  // Optional warm-up delay before the first command is published. Useful so
  // franka_plugin has time to finish go_home before commands start streaming.
  double startup_delay_s_{1.0};

  std::atomic<bool> stop_{false};
  std::unique_ptr<MessageSystem> message_system_;
  std::unique_ptr<FrankaReplayer> replayer_;
  uint32_t sequence_{0};
};

}  // namespace robo_lab
