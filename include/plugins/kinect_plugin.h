#pragma once

#include "plugin_interface.h"
#include "message_system.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <kinect.pb.h>


namespace robo_lab {

/// Minimal sample plugin: reads a few keys from its YAML and runs until stop().
class KinectPlugin : public Plugin {
 public:
  KinectPlugin() = default;
  ~KinectPlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  std::string config_path_;
  std::string robot_ip_;
  std::string topic_;
  std::string control_mode_;
  std::atomic<bool> stop_{false};

  std::unique_ptr<MessageSystem> message_system_;
};
}  // namespace robo_lab
