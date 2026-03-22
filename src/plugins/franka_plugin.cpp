#include "plugins/franka_plugin.h"

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "franka.pb.h"

namespace robo_lab {

namespace {

bool read_double_sequence(const YAML::Node& node, const char* key, std::vector<double>* out) {
  if (!node || !node[key] || !node[key].IsSequence()) {
    return false;
  }
  out->clear();
  for (const auto& item : node[key]) {
    if (!item.IsScalar()) {
      return false;
    }
    out->push_back(item.as<double>());
  }
  return true;
}

}  // namespace

bool FrankaPlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;

  YAML::Node root;
  try {
    
    root = YAML::LoadFile(config_path);

    message_system_ = std::make_unique<MessageSystem>();
    message_system_->initialize();

    if (root["dynamic"]) {
        const YAML::Node dyn = root["dynamic"];
        const bool has_kp = dyn["kp"] && dyn["kp"].IsSequence();
        const bool has_kd = dyn["kd"] && dyn["kd"].IsSequence();
        if (has_kp != has_kd) {
          std::cerr << "franka_plugin: provide both dynamic.kp and dynamic.kd or omit dynamic in "
                    << config_path << '\n';
          return false;
        }
        if (has_kp) {
          if (!read_double_sequence(dyn, "kp", &kp_gains_) ||
              !read_double_sequence(dyn, "kd", &kd_gains_)) {
            std::cerr << "franka_plugin: dynamic.kp / dynamic.kd must be numeric sequences in " << config_path
                      << '\n';
            return false;
          }
          if (kp_gains_.size() != kd_gains_.size()) {
            std::cerr << "franka_plugin: dynamic.kp and dynamic.kd length mismatch in " << config_path << '\n';
            return false;
          }
        }
    }

    if (!root["robot_ip"]) {
      std::cerr << "franka_plugin: robot_ip missing in " << config_path << '\n';
      return false;
    }
    robot_ip_ = root["robot_ip"].as<std::string>();

    if (root["topic"]) {
      topic_ = root["topic"].as<std::string>();
    }
    if (topic_.empty()) {
      topic_ = "robot/command";
    }
    message_system_->subscribe(
        topic_, std::bind(&FrankaPlugin::cmd_subscriber_callback, this, std::placeholders::_1, std::placeholders::_2));

    if (root["control_mode"]) {
      control_mode_ = root["control_mode"].as<std::string>();
    }

    
  } catch (const YAML::Exception& e) {
    std::cerr << "franka_plugin: YAML error in " << config_path << ": " << e.what() << '\n';
    return false;
  }

  std::cout << "franka_plugin: initialized (robot_ip=" << robot_ip_ << ", topic=" << topic_
            << ", control_mode=" << control_mode_;
  if (!kp_gains_.empty()) {
    std::cout << ", gains n=" << kp_gains_.size();
  }
  std::cout << ")\n";
  return true;
}

void FrankaPlugin::cmd_subscriber_callback(const std::string& key, const std::string& payload) {
  franka::RobotCommand cmd;
  if (!cmd.ParseFromString(payload)) {
    std::cerr << "franka_plugin: RobotCommand protobuf parse failed (key=" << key << ", bytes=" << payload.size()
              << ")\n";
    return;
  }
  std::cout << "franka_plugin: RobotCommand key=" << key << " type=" << static_cast<int>(cmd.type())
            << " seq=" << cmd.sequence() << " mode=" << cmd.mode() << " note=" << cmd.note()
            << " joints=" << cmd.joints_size();
  for (int i = 0; i < cmd.joints_size(); ++i) {
    const auto& j = cmd.joints(i);
    std::cout << " [" << i << "]: pos=" << j.position() << " vel=" << j.velocity() << " effort=" << j.effort();
  }
  std::cout << '\n';
}

void FrankaPlugin::run() {
  stop_ = false;
  std::cout << "franka_plugin: run loop started\n";
  int tick = 0;
  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ++tick;
    if (tick % 10 == 0) {
      // 
    }
  }
  std::cout << "franka_plugin: run loop exited\n";
}

void FrankaPlugin::stop() {
  stop_ = true;
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::FrankaPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}
