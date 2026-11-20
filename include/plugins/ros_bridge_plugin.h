#pragma once

#include "message_system.h"
#include "plugin_interface.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace robo_lab {

class RosBridgePlugin : public Plugin {
 public:
  RosBridgePlugin();
  ~RosBridgePlugin() override;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  struct BridgeTopic {
    std::string zenoh_msg;
    std::string zenoh_type;
    std::string ros_msg;
    std::string ros_type;
    std::string frame;  // optional ROS header.frame_id for stamped/pose messages
    std::vector<std::string> joint_names;
    size_t qos_depth = 10;
    bool best_effort = false;
  };

  void cleanup();
  bool load_config(const std::string& config_path);
  static bool parse_topic_list(const YAML::Node& node, std::vector<BridgeTopic>* out);

  std::string config_path_;
  std::atomic<bool> stop_{false};
  std::unique_ptr<MessageSystem> message_system_;

  std::vector<BridgeTopic> robo_to_ros_topics_;
  std::vector<BridgeTopic> ros_to_robo_topics_;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace robo_lab
