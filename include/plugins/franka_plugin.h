#pragma once

#include "plugin_interface.h"
#include "message_system.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <franka/robot.h>
#include <franka/model.h>
#include <franka/gripper.h>

namespace robo_lab {

/// Minimal sample plugin: reads a few keys from its YAML and runs until stop().
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
  std::string topic_;
  std::string control_mode_;
  std::vector<double> kp_gains_;
  std::vector<double> kd_gains_;
  std::atomic<bool> stop_{false};

  std::unique_ptr<MessageSystem> message_system_;

  void cmd_subscriber_callback(const std::string& key, const std::string& payload);

  std::unique_ptr<franka::Robot> robot_;
  std::unique_ptr<franka::Model> model_;
  std::unique_ptr<franka::Gripper> gripper_;
};
}  // namespace robo_lab
