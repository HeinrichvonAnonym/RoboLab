#pragma once

#include "message_system.h"
#include "plugin_interface.h"

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>

namespace robo_lab {

// Reads single keystrokes from the controlling terminal (stdin in cbreak
// mode) and publishes a franka.CartesianDPoseCmd delta on every press.
//
// Each press emits one delta on a single axis; all six fields of
// CartesianDPoseCmd are zero except the matching one, which is set to
// +linear_step / -linear_step (translation axes) or +angular_step /
// -angular_step (rotation axes).
//
// This is the minimum publisher needed to drive the upcoming cartesian
// controller; no integration / target-pose accumulation happens here.
class KeyboardPlugin : public Plugin {
 public:
  KeyboardPlugin() = default;
  ~KeyboardPlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  // Direction tag selected by a single key. Each tag picks one axis of
  // CartesianDPoseCmd and the sign applied to its step.
  enum class Direction {
    kXPos, kXNeg,
    kYPos, kYNeg,
    kZPos, kZNeg,
    kRollPos, kRollNeg,
    kPitchPos, kPitchNeg,
    kYawPos, kYawNeg,
  };

  bool load_config(const std::string& config_path);
  bool publish_direction(Direction d);
  bool publish_next_record();
  bool enter_raw_mode();
  void restore_mode();
  static bool parse_direction_tag_(const std::string& s, Direction* out);
  static const char* direction_name(Direction d);

  std::string config_path_;
  std::string cmd_topic_{"franka/cartesian_dpose"};
  std::string next_record_topic_{"next_record"};
  // Keys are stored as the literal byte read from stdin so case is preserved
  // (the YAML allows mapping 'W' and 'w' to different actions).
  std::unordered_map<char, Direction> key_map_;
  std::string pending_input_;

  // Step magnitudes per single press.
  double linear_step_{0.005};   // m
  double angular_step_{0.02};   // rad

  std::atomic<bool> stop_{false};
  std::unique_ptr<MessageSystem> message_system_;

  // Saved termios state to restore on stop().
  struct TermiosState;
  std::unique_ptr<TermiosState> saved_termios_;
  bool raw_mode_active_{false};
};

}  // namespace robo_lab
