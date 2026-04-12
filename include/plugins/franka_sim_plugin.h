#pragma once

#include "plugin_interface.h"
#include "message_system.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <franka/robot.h>
#include <franka/model.h>
#include <franka/gripper.h>
#include <Eigen/Dense>
#include <franka.pb.h>


namespace robo_lab {

/// Franka robot plugin: receives joint commands, executes PD control, publishes state.
class FrankaPlugin : public Plugin {
 public:
  FrankaPlugin() = default;
  ~FrankaPlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  std::string config_path_;
  std::string robot_ip_;
  std::string cmd_topic_;
  std::string state_topic_;
  std::string control_mode_;
  std::vector<double> kp_gains_;
  std::vector<double> kd_gains_;
  std::atomic<bool> stop_{false};

  std::unique_ptr<MessageSystem> message_system_;

  void cmd_subscriber_callback(const std::string& key, const std::string& payload);

  // std::unique_ptr<franka::Robot> robot_;
  // std::unique_ptr<franka::Model> model_;
  // std::unique_ptr<franka::Gripper> gripper_; 
  bool publish_state();
  uint32_t state_sequence_{0};

  // Thread-safe target joint positions (from cartesian controller commands)
  mutable std::mutex target_mutex_;
  std::array<double, 7> q_target_{};
  std::atomic<bool> has_target_{false};
  
  // Safety limits for joint position change per step
  static constexpr double kMaxJointStep = 0.01;  // rad per control cycle (~1ms)
};
}  // namespace robo_lab
