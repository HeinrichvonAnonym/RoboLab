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

    if (root["arm_home"]) {
      for (int i = 0; i < 7; ++i) {
        arm_home_[i] = root["arm_home"][i].as<double>();
      }
    } else {
      arm_home_ = {-0.32, -0.9, 0.13, -2.75, 0.18, 1.95, 0.49};
    }

    if (root["skip_go_home"]) {
      skip_go_home_ = root["skip_go_home"].as<bool>();
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
  
  if (cmd.joints_size() != 7) {
    std::cerr << "franka_plugin: expected 7 joints, got " << cmd.joints_size() << "\n";
    return;
  }
  
  // Update target positions (thread-safe)
  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    for (int i = 0; i < 7; ++i) {
      q_target_[i] = cmd.joints(i).position();
    }
    has_target_ = true;
  }
  
  // Debug output
  static int cmd_counter = 0;
  if (cmd_counter++ % 50 == 0) {
    std::cout << "[FrankaPlugin] CMD: [";
    for (int i = 0; i < 7; ++i) {
      std::cout << cmd.joints(i).position();
      if (i < 6) std::cout << ", ";
    }
    std::cout << "]\n";
  }
}

bool FrankaPlugin::go_home_cmd(const franka::RobotState& robot_state, std::array<double, 7>& q_cmd) {
  // Command home position directly; rate limiter (kMaxJointStep) controls velocity.
  // Returns true when all joints are within threshold of home (homing complete).
  constexpr double kHomeThreshold = 0.05;  // rad (~1.1 deg)

  bool all_at_home = true;
  for (int i = 0; i < 7; ++i) {
    const double error = arm_home_[i] - robot_state.q[i];
    const double abs_error = std::abs(error);
    
    if (abs_error > kHomeThreshold) {
      all_at_home = false;
    }
    
    // Always command home position; rate limiter in control loop handles velocity
    q_cmd[i] = arm_home_[i];
  }

  static int go_home_debug = 0;
  if (go_home_debug++ % 500 == 0) {
    std::cout << "[FrankaPlugin] go_home: ";
    for (int i = 0; i < 7; ++i) {
      std::cout << "j" << i << "=" << std::abs(arm_home_[i] - robot_state.q[i]) << " ";
    }
    std::cout << (all_at_home ? "COMPLETE" : "...") << "\n";
  }

  return all_at_home;
}

void FrankaPlugin::run() {
  stop_ = false;
  std::cout << "franka_plugin: run loop started\n";
  if (!robot_) {
    std::cerr << "franka_plugin: robot not initialized\n";
    if (message_system_) {
      message_system_->close();
    }
    return;
  }

  if (kp_gains_.size() != 7 || kd_gains_.size() != 7) {
    std::cerr << "franka_plugin: control requires dynamic.kp and dynamic.kd with 7 values each\n";
    if (message_system_) {
      message_system_->close();
    }
    return;
  }
  constexpr double kTauLimit = 60.0;  // Conservative software saturation.

  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();


  while(!stop_) {
  try{
    const franka::RobotState initial_state = robot_->readOnce();
    
    // Initialize q_des to current position, q_target to home if no external command yet
    std::array<double, 7> q_des = initial_state.q;
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      if (!has_target_) {
        q_target_ = arm_home_;
      }
    }

    // Skip go_home if configured
    if (skip_go_home_ && !control_is_activated_) {
      control_is_activated_ = true;
      std::cout << "[FrankaPlugin] skip_go_home=true, control activated immediately\n";
    }


    robot_->control(
        [&](const franka::RobotState& robot_state, franka::Duration /*duration*/) -> franka::Torques {
          // Get target command
          std::array<double, 7> q_cmd;
          if (!control_is_activated_) {
            // Go-home phase: move to arm_home_ using velocity-based control
            if (go_home_cmd(robot_state, q_cmd)) {
              // Homing complete, switch to normal command mode
              control_is_activated_ = true;
              std::cout << "[FrankaPlugin] go_home complete, control activated\n";
            }
          } else {
            // Normal mode: follow external commands
            std::lock_guard<std::mutex> lock(target_mutex_);
            q_cmd = q_target_;
          }
          
          // Rate-limit q_des towards q_cmd for safety
          for (size_t i = 0; i < 7; ++i) {
            double delta = q_cmd[i] - q_des[i];
            if (delta > kMaxJointStep) {
              delta = kMaxJointStep;
            } else if (delta < -kMaxJointStep) {
              delta = -kMaxJointStep;
            }
            q_des[i] += delta;
          }
          
          // PD control: tau = Kp * (q_des - q) + Kd * (0 - dq)
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
          
          // Debug: show position error direction periodically
          static int ctrl_debug = 0;
          if (ctrl_debug++ % 1000 == 0) {
            std::cout << "[FrankaPlugin] q_des[0]=" << q_des[0] 
                      << " q=" << robot_state.q[0]
                      << " err=" << (q_des[0] - robot_state.q[0])
                      << " tau=" << tau_d[0] << "\n";
          }

          franka::Torques cmd(tau_d);
          if (stop_) {
            return franka::MotionFinished(cmd);
          }
          
          // Publish state at ~50Hz
          std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
          if(end_time - start_time > std::chrono::milliseconds(20)){
            start_time = end_time;
            if (!publish_state(robot_state)){
              std::cout << "franka_plugin: publish_state failed\n"<<std::endl;
              return franka::MotionFinished(cmd);
            }
          }
          return cmd;
        },
        true,
        1000.0);
      } catch (...) {
        const franka::RobotState robot_state = robot_->readOnce();
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        if(end_time - start_time > std::chrono::milliseconds(20)){
          start_time = end_time;
          if (!publish_state(robot_state)){
            std::cout << "franka_plugin: publish_state failed\n"<<std::endl;
            continue;
          }
        }
      }
  }

  std::cout << "franka_plugin: run loop exited\n";
  robot_->stop();
  // Close Zenoh on the run thread so no concurrent publish_state() + close() race.
  if (message_system_) {
    message_system_->close();
  }
}

bool FrankaPlugin::publish_state(const franka::RobotState& robot_state) {
  try
  {
    // std::cout << "franka_plugin: publish_state started\n"<<std::endl;
    franka::RobotObservation obs;
    obs.set_type(franka::RobotObservation::TYPE_JOINT_TARGET);
    obs.set_mode(control_mode_);
    for(int i = 0; i < 7; ++i) {
      auto* joint = obs.add_joints();
      joint->set_position(robot_state.q[i]);
      joint->set_velocity(robot_state.dq[i]);
      joint->set_effort(robot_state.tau_J[i]);
    }
    obs.set_sys_time(robot_state.time.toSec());
    // obs.set_sequence(state_sequence_++);

    std::string payload;
    if (!obs.SerializeToString(&payload)) {
      std::cout << "franka_plugin: publish_state SerializeToString failed\n"<<std::endl;
      return false;
    }
    // std::cout << "franka_plugin: publish_state payload size: " << payload.size() << std::endl;
    if (!message_system_->publish(state_topic_, payload)){
      std::cout << "franka_plugin: publish_state publish failed\n"<<std::endl;
      return false;
    }
    return true;
  }
 catch (...) {
  std::cout << "franka_plugin: publish_state exception\n"<<std::endl;
  return false;
}
}

void FrankaPlugin::stop() {
  if (stop_.exchange(true)) {
    return;
  }
  std::cout << "franka_plugin: stopping\n";
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
