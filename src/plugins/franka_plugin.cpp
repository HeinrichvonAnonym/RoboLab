#include "plugins/franka_plugin.h"

#include <array>
#include <chrono>
#include <cmath>
#include <exception>
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

double clamp_double(double value, double low, double high) {
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
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
    if (root["trigger_topic"]) {
      trigger_topic_ = root["trigger_topic"].as<std::string>();
    }
    if (trigger_topic_.empty()) {
      trigger_topic_ = "trigger";
    }
    if (root["state_machine_topic"]) {
      state_machine_topic_ = root["state_machine_topic"].as<std::string>();
    }
    if (state_machine_topic_.empty()) {
      state_machine_topic_ = "franka/state_machine";
    }

    message_system_->subscribe(
        cmd_topic_, std::bind(&FrankaPlugin::cmd_subscriber_callback, this, std::placeholders::_1, std::placeholders::_2));
    message_system_->subscribe(
        trigger_topic_,
        std::bind(&FrankaPlugin::trigger_subscriber_callback, this,
                  std::placeholders::_1, std::placeholders::_2));

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

    if (root["cmd_filter_alpha"]) {
      cmd_filter_alpha_ = root["cmd_filter_alpha"].as<double>();
      if (cmd_filter_alpha_ <= 0.0 || cmd_filter_alpha_ > 1.0) {
        std::cerr << "franka_plugin: cmd_filter_alpha must be in (0, 1] (got "
                  << cmd_filter_alpha_ << ")\n";
        return false;
      }
    }

    if (root["enable_gripper"]) {
      enable_gripper_ = root["enable_gripper"].as<bool>();
    }
    if (root["gripper_max_width"]) {
      gripper_max_width_ = root["gripper_max_width"].as<double>();
      if (gripper_max_width_ <= 0.0) {
        std::cerr << "franka_plugin: gripper_max_width must be positive (got "
                  << gripper_max_width_ << ")\n";
        return false;
      }
    }
    if (root["gripper_closed_width"]) {
      gripper_closed_width_ = root["gripper_closed_width"].as<double>();
      if (gripper_closed_width_ < 0.0 || gripper_closed_width_ > gripper_max_width_) {
        std::cerr << "franka_plugin: gripper_closed_width must be in [0, gripper_max_width] (got "
                  << gripper_closed_width_ << ")\n";
        return false;
      }
    }
    if (root["gripper_speed"]) {
      gripper_speed_ = root["gripper_speed"].as<double>();
      if (gripper_speed_ <= 0.0) {
        std::cerr << "franka_plugin: gripper_speed must be positive (got "
                  << gripper_speed_ << ")\n";
        return false;
      }
    }
    if (root["gripper_close_threshold"]) {
      gripper_close_threshold_ = root["gripper_close_threshold"].as<double>();
      if (gripper_close_threshold_ < 0.0 || gripper_close_threshold_ > 1.0) {
        std::cerr << "franka_plugin: gripper_close_threshold must be in [0, 1] (got "
                  << gripper_close_threshold_ << ")\n";
        return false;
      }
    }
    if (enable_gripper_) {
      gripper_ = std::make_unique<franka::Gripper>(robot_ip_);
    }

  } catch (const YAML::Exception& e) {
    std::cerr << "franka_plugin: YAML error in " << config_path << ": " << e.what() << '\n';
    return false;
  } catch (const std::exception& e) {
    std::cerr << "franka_plugin: failed to initialize Franka interface: " << e.what() << '\n';
    return false;
  }

  std::cout << "franka_plugin: initialized (robot_ip=" << robot_ip_ << ", cmd_topic=" << cmd_topic_ << ", state_topic=" << state_topic_
            << ", trigger_topic=" << trigger_topic_
            << ", state_machine_topic=" << state_machine_topic_
            << ", control_mode=" << control_mode_
            << ", cmd_filter_alpha=" << cmd_filter_alpha_
            << ", enable_gripper=" << (enable_gripper_ ? "true" : "false");
  if (enable_gripper_) {
    std::cout << ", gripper_max_width=" << gripper_max_width_
              << ", gripper_closed_width=" << gripper_closed_width_
              << ", gripper_speed=" << gripper_speed_
              << ", gripper_close_threshold=" << gripper_close_threshold_;
  }
  if (!kp_gains_.empty()) {
    std::cout << ", gains n=" << kp_gains_.size();
  }
  std::cout << ")\n";
  return true;
}

const char* FrankaPlugin::control_state_name(ControlState state) {
  switch (state) {
    case ControlState::kInit:
      return "init";
    case ControlState::kGoHome:
      return "gohome";
    case ControlState::kStandby:
      return "standby";
    case ControlState::kInference:
      return "inference";
  }
  return "?";
}

bool FrankaPlugin::publish_control_state(ControlState state) {
  if (!message_system_ || state_machine_topic_.empty()) {
    return false;
  }
  const char* state_name = control_state_name(state);
  std::lock_guard<std::mutex> lock(publish_mutex_);
  if (!message_system_->publish(state_machine_topic_, state_name)) {
    std::cout << "franka_plugin: publish_control_state failed (" << state_name << ")\n";
    return false;
  }
  return true;
}

void FrankaPlugin::reset_control_session(const franka::RobotState& robot_state) {
  control_state_.store(ControlState::kInit);
  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    q_target_ = robot_state.q;
    has_target_ = false;
  }
  q_target_filtered_ = robot_state.q;
  cmd_filter_primed_ = true;
  publish_control_state(ControlState::kInit);
  std::cout << "[FrankaPlugin] control session reset -> init\n";
}

void FrankaPlugin::enter_inference_from_control(const franka::RobotState& robot_state) {
  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    q_target_ = robot_state.q;
    has_target_ = true;
  }
  q_target_filtered_ = robot_state.q;
  std::cout << "[FrankaPlugin] standby trigger accepted -> inference\n";
}

void FrankaPlugin::trigger_subscriber_callback(const std::string& key, const std::string& payload) {
  if (payload != "trigger") {
    std::cout << "[FrankaPlugin] ignoring trigger payload on '" << key
              << "' (bytes=" << payload.size() << ")\n";
    return;
  }

  ControlState expected = ControlState::kInit;
  if (control_state_.compare_exchange_strong(expected, ControlState::kGoHome)) {
    publish_control_state(ControlState::kGoHome);
    std::cout << "[FrankaPlugin] trigger: init -> gohome\n";
    return;
  }

  expected = ControlState::kStandby;
  if (control_state_.compare_exchange_strong(expected, ControlState::kInference)) {
    publish_control_state(ControlState::kInference);
    return;
  }

  expected = ControlState::kInference;
  if (control_state_.compare_exchange_strong(expected, ControlState::kStandby)) {
    publish_control_state(ControlState::kStandby);
    std::cout << "[FrankaPlugin] trigger: inference -> standby\n";
    return;
  }

  const ControlState state = control_state_.load();
  std::cout << "[FrankaPlugin] trigger ignored in " << control_state_name(state) << "\n";
}

void FrankaPlugin::cmd_subscriber_callback(const std::string& key, const std::string& payload) {
  const ControlState state = control_state_.load();
  if (state != ControlState::kInference) {
    static int dropped_cmd_counter = 0;
    if (dropped_cmd_counter++ % 50 == 0) {
      std::cout << "[FrankaPlugin] dropping command on '" << key
                << "' while state=" << control_state_name(state) << "\n";
    }
    return;
  }

  franka::RobotCommand cmd;
  if (!cmd.ParseFromString(payload)) {
    std::cerr << "franka_plugin: RobotCommand protobuf parse failed (key=" << key << ", bytes=" << payload.size()
              << ")\n";
    return;
  }
  
  if (cmd.joints_size() != 7 && cmd.joints_size() != 8) {
    std::cerr << "franka_plugin: expected 7 or 8 command joints, got " << cmd.joints_size() << "\n";
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

  if (cmd.joints_size() == 8) {
    if (!enable_gripper_) {
      static bool warned_gripper_disabled = false;
      if (!warned_gripper_disabled) {
        std::cerr << "franka_plugin: received 8D command but gripper is disabled\n";
        warned_gripper_disabled = true;
      }
    } else {
      const double close_norm = clamp_double(cmd.joints(7).position(), 0.0, 1.0);
      const bool target_closed = close_norm >= gripper_close_threshold_;
      bool target_changed = false;
      {
        std::lock_guard<std::mutex> lock(gripper_mutex_);
        target_changed = !has_gripper_target_ || gripper_target_closed_ != target_closed;
        gripper_target_close_norm_ = close_norm;
        gripper_target_closed_ = target_closed;
        has_gripper_target_ = true;
        if (target_changed) {
          ++gripper_target_seq_;
        }
      }
      if (target_changed) {
        gripper_action_cv_.notify_one();
        gripper_interrupt_cv_.notify_one();
      }
    }
  }
  
  // Debug output
  static int cmd_counter = 0;
  if (cmd_counter++ % 50 == 0) {
    std::cout << "[FrankaPlugin] CMD: [";
    for (int i = 0; i < 7; ++i) {
      std::cout << cmd.joints(i).position();
      if (i < 6) std::cout << ", ";
    }
    std::cout << "]";
    if (cmd.joints_size() == 8) {
      std::cout << " gripper=" << cmd.joints(7).position();
    }
    std::cout << "\n";
  }
}

void FrankaPlugin::gripper_action_loop() {
  uint64_t handled_seq = 0;
  while (true) {
    bool target_closed = false;
    double close_norm = 0.0;
    uint64_t target_seq = 0;
    {
      std::unique_lock<std::mutex> lock(gripper_mutex_);
      gripper_action_cv_.wait(lock, [&] {
        return gripper_stop_ || (has_gripper_target_ && gripper_target_seq_ != handled_seq);
      });
      if (gripper_stop_) {
        break;
      }
      target_closed = gripper_target_closed_;
      close_norm = gripper_target_close_norm_;
      target_seq = gripper_target_seq_;
      gripper_move_in_progress_ = true;
      gripper_active_closed_ = target_closed;
      gripper_active_seq_ = target_seq;
      gripper_stop_requested_seq_ = 0;
    }
    gripper_interrupt_cv_.notify_one();

    try {
      if (!gripper_) {
        handled_seq = target_seq;
      } else {
        const double width = target_closed ? gripper_closed_width_ : gripper_max_width_;
        std::cout << "[FrankaPlugin] gripper "
                  << (target_closed ? "close" : "open")
                  << " (gello=" << close_norm << ", width=" << width
                  << ", seq=" << target_seq << ")\n";
        gripper_->move(width, gripper_speed_);
        handled_seq = target_seq;
      }
    } catch (const std::exception& e) {
      bool interrupted = false;
      {
        std::lock_guard<std::mutex> lock(gripper_mutex_);
        interrupted = gripper_stop_ || gripper_target_seq_ != target_seq;
      }
      if (interrupted) {
        std::cout << "[FrankaPlugin] gripper move interrupted: " << e.what() << '\n';
      } else {
        std::cerr << "franka_plugin: gripper move failed: " << e.what() << '\n';
      }
      handled_seq = target_seq;
    }

    {
      std::lock_guard<std::mutex> lock(gripper_mutex_);
      if (gripper_active_seq_ == target_seq) {
        gripper_move_in_progress_ = false;
      }
    }
    gripper_interrupt_cv_.notify_one();
    gripper_action_cv_.notify_one();
  }
}

void FrankaPlugin::gripper_interrupt_loop() {
  while (true) {
    uint64_t active_seq = 0;
    bool should_stop = false;
    {
      std::unique_lock<std::mutex> lock(gripper_mutex_);
      gripper_interrupt_cv_.wait(lock, [&] {
        return gripper_stop_ ||
               (gripper_move_in_progress_ && has_gripper_target_ &&
                gripper_target_seq_ != gripper_active_seq_ &&
                gripper_target_closed_ != gripper_active_closed_ &&
                gripper_stop_requested_seq_ != gripper_active_seq_);
      });
      if (gripper_stop_) {
        break;
      }
      active_seq = gripper_active_seq_;
      gripper_stop_requested_seq_ = active_seq;
      should_stop = true;
    }

    if (!should_stop || !gripper_) {
      continue;
    }
    try {
      std::cout << "[FrankaPlugin] gripper stop active seq=" << active_seq
                << " for target seq change\n";
      gripper_->stop();
    } catch (const std::exception& e) {
      std::cerr << "franka_plugin: gripper stop failed: " << e.what() << '\n';
    }
  }
}

void FrankaPlugin::stop_gripper_worker() {
  {
    std::lock_guard<std::mutex> lock(gripper_mutex_);
    gripper_stop_ = true;
  }
  gripper_action_cv_.notify_one();
  gripper_interrupt_cv_.notify_one();
  if (gripper_) {
    try {
      gripper_->stop();
    } catch (const std::exception& e) {
      std::cerr << "franka_plugin: gripper stop during shutdown failed: " << e.what() << '\n';
    }
  }
  if (gripper_interrupt_thread_.joinable()) {
    gripper_interrupt_thread_.join();
  }
  if (gripper_action_thread_.joinable()) {
    gripper_action_thread_.join();
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
  if (enable_gripper_ && gripper_ && !gripper_action_thread_.joinable() &&
      !gripper_interrupt_thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(gripper_mutex_);
      gripper_stop_ = false;
    }
    gripper_action_thread_ = std::thread(&FrankaPlugin::gripper_action_loop, this);
    gripper_interrupt_thread_ = std::thread(&FrankaPlugin::gripper_interrupt_loop, this);
  }
  constexpr double kTauLimit = 60.0;  // Conservative software saturation.

  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();


  while(!stop_) {
  try{
    const franka::RobotState initial_state = robot_->readOnce();
    
    // Every new libfranka control session starts inert and waits for triggers.
    std::array<double, 7> q_des = initial_state.q;
    reset_control_session(initial_state);
    ControlState previous_state = ControlState::kInit;


    robot_->control(
        [&](const franka::RobotState& robot_state, franka::Duration /*duration*/) -> franka::Torques {
          // Get target command
          std::array<double, 7> q_cmd = q_des;
          const ControlState state = control_state_.load();
          if (previous_state != ControlState::kInference &&
              state == ControlState::kInference) {
            enter_inference_from_control(robot_state);
          }

          if (state == ControlState::kInit) {
            q_target_filtered_ = robot_state.q;
          } else if (state == ControlState::kGoHome) {
            if (go_home_cmd(robot_state, q_cmd)) {
              control_state_.store(ControlState::kStandby);
              publish_control_state(ControlState::kStandby);
              std::cout << "[FrankaPlugin] go_home complete -> standby\n";
            }
            q_target_filtered_ = robot_state.q;
          } else if (state == ControlState::kStandby) {
            q_target_filtered_ = robot_state.q;
          } else if (state == ControlState::kInference) {
            std::array<double, 7> q_target_snapshot;
            bool has_target_snapshot = false;
            {
              std::lock_guard<std::mutex> lock(target_mutex_);
              q_target_snapshot = q_target_;
              has_target_snapshot = has_target_.load();
            }
            if (!has_target_snapshot) {
              q_target_snapshot = robot_state.q;
            }
            // First-order IIR low-pass to absorb step jitter on q_target_ that
            // arrives at coarse rates (recorded data, replayer, IK at 30 Hz).
            // alpha=1.0 -> passthrough; smaller alpha smooths more.
            const double a = cmd_filter_alpha_;
            for (size_t i = 0; i < 7; ++i) {
              q_target_filtered_[i] =
                  a * q_target_snapshot[i] + (1.0 - a) * q_target_filtered_[i];
            }
            q_cmd = q_target_filtered_;
          }
          previous_state = control_state_.load();

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
  stop_gripper_worker();
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
    {
      std::lock_guard<std::mutex> lock(publish_mutex_);
      if (!message_system_->publish(state_topic_, payload)){
        std::cout << "franka_plugin: publish_state publish failed\n"<<std::endl;
        return false;
      }
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
  stop_gripper_worker();
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
