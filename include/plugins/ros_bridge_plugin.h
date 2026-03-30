#pragma once

#include "message_system.h"
#include "plugin_interface.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <ros/ros.h>

namespace robo_lab {

class RosBridgePlugin : public Plugin {
 public:
  RosBridgePlugin() = default;
  ~RosBridgePlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  struct BridgeRule {
    std::string zenoh_msg;
    std::string zenoh_type;
    std::string ros_msg;
    std::string ros_type;
  };

  bool setup_robo_to_ros(const BridgeRule& rule);
  bool setup_ros_to_robo(const BridgeRule& rule);

  std::string config_path_;
  std::atomic<bool> stop_{false};
  std::unique_ptr<MessageSystem> message_system_;
  std::unique_ptr<ros::NodeHandle> nh_;
  std::vector<ros::Publisher> ros_publishers_;
  std::vector<ros::Subscriber> ros_subscribers_;
};

}  // namespace robo_lab
