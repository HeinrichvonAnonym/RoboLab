#pragma once

#include "plugin_interface.h"
#include "message_system.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <franka/robot.h>
#include <franka/model.h>
#include <franka/gripper.h>
#include <Eigen/Dense>


namespace robo_lab {

/// Franka robot plugin: receives joint commands, executes PD control, publishes state.
class FrankaPlugin : public Plugin {
 public:
  FrankaPlugin() = default;
  ~FrankaPlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

  /// Velocity-based go-home: moves joints toward arm_home_ at go_home_vel_q_ rad/s.
  /// Returns true when all joints reach home position (within threshold).
  bool go_home_cmd(const franka::RobotState& robot_state, std::array<double, 7>& q_cmd);

 private:
  enum class ControlState {
    kInit,
    kGoHome,
    kStandby,
    kInference,
  };

  std::string config_path_;
  std::string robot_ip_;
  std::string cmd_topic_;
  std::string state_topic_;
  std::string trigger_topic_{"trigger"};
  std::string control_mode_;
  std::vector<double> kp_gains_;
  std::vector<double> kd_gains_;
  std::atomic<bool> stop_{false};
  std::atomic<ControlState> control_state_{ControlState::kInit};
  std::array<double, 7> arm_home_;

  std::unique_ptr<MessageSystem> message_system_;

  void cmd_subscriber_callback(const std::string& key, const std::string& payload);
  void trigger_subscriber_callback(const std::string& key, const std::string& payload);

  std::unique_ptr<franka::Robot> robot_;
  std::unique_ptr<franka::Model> model_;
  std::unique_ptr<franka::Gripper> gripper_;

  bool publish_state(const franka::RobotState& robot_state);
  void reset_control_session(const franka::RobotState& robot_state);
  void enter_inference_from_control(const franka::RobotState& robot_state);
  static const char* control_state_name(ControlState state);
  void gripper_action_loop();
  void gripper_interrupt_loop();
  void stop_gripper_worker();
  uint32_t state_sequence_{0};

  // Thread-safe target joint positions (from cartesian controller commands)
  mutable std::mutex target_mutex_;
  std::array<double, 7> q_target_{};
  std::atomic<bool> has_target_{false};

  // First-order IIR low-pass on the incoming command stream. Updated each
  // control cycle as: q_target_filtered = alpha * q_target + (1-alpha) *
  // q_target_filtered. alpha=1.0 disables the filter (raw command is used).
  double cmd_filter_alpha_{1.0};
  std::array<double, 7> q_target_filtered_{};
  bool cmd_filter_primed_{false};

  bool enable_gripper_{true};
  double gripper_max_width_{0.08};
  double gripper_closed_width_{0.0};
  double gripper_speed_{0.1};
  double gripper_close_threshold_{0.5};
  std::thread gripper_action_thread_;
  std::thread gripper_interrupt_thread_;
  std::mutex gripper_mutex_;
  std::condition_variable gripper_action_cv_;
  std::condition_variable gripper_interrupt_cv_;
  bool gripper_stop_{false};
  bool has_gripper_target_{false};
  bool gripper_target_closed_{false};
  bool gripper_move_in_progress_{false};
  bool gripper_active_closed_{false};
  double gripper_target_close_norm_{0.0};
  uint64_t gripper_target_seq_{0};
  uint64_t gripper_active_seq_{0};
  uint64_t gripper_stop_requested_seq_{0};

  // Safety limits for joint position change per step
  static constexpr double kMaxJointStep = 0.0005;  // rad per control cycle (~1ms) = 0.5 rad/s
};
}  // namespace robo_lab
