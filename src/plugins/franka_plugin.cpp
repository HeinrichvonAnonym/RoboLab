#include "plugins/franka_plugin.h"

#include <chrono>
#include <array>
#include <cmath>
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
    robot_ = std::make_unique<franka::Robot>(robot_ip_);

    if (root["cmd_topic"]) {
      cmd_topic_ = root["cmd_topic"].as<std::string>();
    }
    if (cmd_topic_.empty()) {
      cmd_topic_ = "robot/command";
    }
    if(root["state_topic"]) {
      state_topic_ = root["state_topic"].as<std::string>();
    }
    if(state_topic_.empty()) {
      state_topic_ = "robot/state";
    }

    message_system_->subscribe(
        cmd_topic_, std::bind(&FrankaPlugin::cmd_subscriber_callback, this, std::placeholders::_1, std::placeholders::_2));

    if (root["control_mode"]) {
      control_mode_ = root["control_mode"].as<std::string>();
    }

    
  } catch (const YAML::Exception& e) {
    std::cerr << "franka_plugin: YAML error in " << config_path << ": " << e.what() << '\n';
    return false;
  }

  std::cout << "franka_plugin: initialized (robot_ip=" << robot_ip_ << ", cmd_topic=" << cmd_topic_ << ", state_topic=" << state_topic_
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
  if (!robot_) {
    std::cerr << "franka_plugin: robot not initialized\n";
    return;
  }

  if (kp_gains_.size() != 7 || kd_gains_.size() != 7) {
    std::cerr << "franka_plugin: control requires dynamic.kp and dynamic.kd with 7 values each\n";
    return;
  }

  const franka::RobotState initial_state = robot_->readOnce();
  std::array<double, 7> q_des = initial_state.q;
  
  constexpr double kTauLimit = 60.0;  // Conservative software saturation.

  // Joint-space PD torque loop:
  // tau = Kp * (q_des - q) + Kd * (dq_des - dq), with dq_des = 0.
  robot_->control(
      [&](const franka::RobotState& robot_state, franka::Duration /*duration*/) -> franka::Torques {
        std::array<double, 7> tau_d{};
        for (size_t i = 0; i < 7; ++i) {
          const double pos_err = q_des[i] - robot_state.q[i];  
          const double vel_err = -robot_state.dq[i];
          double tau = kp_gains_[i] * pos_err + kd_gains_[i] * vel_err;
          if (tau > kTauLimit) {
            tau = kTauLimit;
          } else if (tau < -kTauLimit) {
            tau = -kTauLimit;
          }
          tau_d[i] = tau;
        }

        franka::Torques cmd(tau_d);
        if (stop_) {
          return franka::MotionFinished(cmd);
        }
        return cmd;
      },
      true,
      1000.0);

  std::cout << "franka_plugin: run loop exited\n";
}

void FrankaPlugin::stop() {
  message_system_->close();
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
