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

class FrankaGripperDevice : public GripperDevice {
 public:
  explicit FrankaGripperDevice(franka::Gripper& gripper) : gripper_(gripper) {}
  bool move(double width, double speed) override { return gripper_.move(width, speed); }
  bool stop() override { return gripper_.stop(); }
 private:
  franka::Gripper& gripper_;
};

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

void set_pose_from_transform(const std::array<double, 16>& transform, franka::Pose* pose) {
  if (!pose) {
    return;
  }

  pose->mutable_pos()->set_x(transform[12]);
  pose->mutable_pos()->set_y(transform[13]);
  pose->mutable_pos()->set_z(transform[14]);

  const double r00 = transform[0];
  const double r01 = transform[4];
  const double r02 = transform[8];
  const double r10 = transform[1];
  const double r11 = transform[5];
  const double r12 = transform[9];
  const double r20 = transform[2];
  const double r21 = transform[6];
  const double r22 = transform[10];

  double qw = 1.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  const double trace = r00 + r11 + r22;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    qw = 0.25 * s;
    qx = (r21 - r12) / s;
    qy = (r02 - r20) / s;
    qz = (r10 - r01) / s;
  } else if (r00 > r11 && r00 > r22) {
    const double s = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
    qw = (r21 - r12) / s;
    qx = 0.25 * s;
    qy = (r01 + r10) / s;
    qz = (r02 + r20) / s;
  } else if (r11 > r22) {
    const double s = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
    qw = (r02 - r20) / s;
    qx = (r01 + r10) / s;
    qy = 0.25 * s;
    qz = (r12 + r21) / s;
  } else {
    const double s = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
    qw = (r10 - r01) / s;
    qx = (r02 + r20) / s;
    qy = (r12 + r21) / s;
    qz = 0.25 * s;
  }

  pose->mutable_rot()->set_w(qw);
  pose->mutable_rot()->set_x(qx);
  pose->mutable_rot()->set_y(qy);
  pose->mutable_rot()->set_z(qz);
}

}  // namespace

FrankaPlugin::~FrankaPlugin() { stop_gripper_worker(); }

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
      gripper_options_.open_width = gripper_max_width_;
      gripper_options_.closed_width = gripper_closed_width_;
      gripper_options_.speed = gripper_speed_;
      gripper_options_.close_threshold = gripper_close_threshold_;
      if (root["gripper_retry_count"]) {
        gripper_options_.retry_count = root["gripper_retry_count"].as<int>();
      }
      if (root["gripper_retry_interval_ms"]) {
        gripper_options_.retry_interval = std::chrono::milliseconds(root["gripper_retry_interval_ms"].as<int>());
      }
      if (root["gripper_width_tolerance"]) {
        gripper_options_.width_tolerance = root["gripper_width_tolerance"].as<double>();
      }
      if (root["gripper_state_max_age_ms"]) {
        gripper_options_.state_max_age = std::chrono::milliseconds(root["gripper_state_max_age_ms"].as<int>());
      }
      if (root["gripper_confirmation_wait_ms"]) {
        gripper_options_.confirmation_wait = std::chrono::milliseconds(root["gripper_confirmation_wait_ms"].as<int>());
      }
      gripper_ = std::make_unique<franka::Gripper>(robot_ip_);
      gripper_worker_ = std::make_unique<GripperWorker>(
          std::make_shared<FrankaGripperDevice>(*gripper_), gripper_options_,
          [](const std::string& message) { std::cout << "[FrankaPlugin] gripper " << message << '\n'; });
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
              << ", gripper_close_threshold=" << gripper_close_threshold_
              << ", gripper_retry_count=" << gripper_options_.retry_count
              << ", gripper_retry_interval_ms=" << gripper_options_.retry_interval.count()
              << ", gripper_width_tolerance=" << gripper_options_.width_tolerance
              << ", gripper_state_max_age_ms=" << gripper_options_.state_max_age.count()
              << ", gripper_confirmation_wait_ms=" << gripper_options_.confirmation_wait.count();
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
  std::lock_guard<std::mutex> transition(control_transition_mutex_);
  control_state_.store(ControlState::kInit);
  if (gripper_worker_) gripper_worker_->set_enabled(false);
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
  std::lock_guard<std::mutex> transition(control_transition_mutex_);
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
    if (gripper_worker_) gripper_worker_->set_enabled(true);
    publish_control_state(ControlState::kInference);
    return;
  }

  expected = ControlState::kInference;
  if (control_state_.compare_exchange_strong(expected, ControlState::kStandby)) {
    if (gripper_worker_) gripper_worker_->set_enabled(false);
    publish_control_state(ControlState::kStandby);
    std::cout << "[FrankaPlugin] trigger: inference -> standby\n";
    return;
  }

  const ControlState state = control_state_.load();
  std::cout << "[FrankaPlugin] trigger ignored in " << control_state_name(state) << "\n";
}

void FrankaPlugin::cmd_subscriber_callback(const std::string& key, const std::string& payload) {
  uint64_t gripper_session = 0;
  {
    std::lock_guard<std::mutex> transition(control_transition_mutex_);
    if (control_state_.load() != ControlState::kInference) {
      return;
    }
    // Capture the session atomically with the state check, before parsing.
    gripper_session = gripper_worker_ ? gripper_worker_->session() : 0;
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
    } else if (gripper_worker_) {
      gripper_worker_->submit(cmd.joints(7).position(), gripper_session);
    }
  }
  
  // Debug output
  static int cmd_counter = 0;
  if (cmd_counter++ % 50 == 0) {
  //   std::cout << "[FrankaPlugin] CMD: [";
  //   for (int i = 0; i < 7; ++i) {
  //     std::cout << cmd.joints(i).position();
  //     if (i < 6) std::cout << ", ";
  //   }
  //   std::cout << "]";
  //   if (cmd.joints_size() == 8) {
  //     std::cout << " gripper=" << cmd.joints(7).position();
  //   }
  //   std::cout << "\n";
  }
}

void FrankaPlugin::gripper_state_loop() {
  while (!gripper_reader_stop_) {
    try {
      const franka::GripperState state = gripper_->readOnce();
      gripper_worker_->update_state(state.width);
      std::unique_lock<std::mutex> lock(gripper_reader_mutex_);
      gripper_reader_cv_.wait_for(lock, std::chrono::milliseconds(20), [&] {
        return gripper_reader_stop_.load();
      });
    } catch (const std::exception& e) {
      if (gripper_reader_stop_) break;
      const auto feedback = gripper_worker_->feedback();
      std::cerr << "franka_plugin: gripper state read failed: " << e.what();
      if (feedback.valid) {
        std::cerr << " cached_width=" << feedback.width << " feedback_age_ms="
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         GripperWorker::Clock::now() - feedback.collected).count();
      }
      std::cerr << '\n';
      std::unique_lock<std::mutex> lock(gripper_reader_mutex_);
      gripper_reader_cv_.wait_for(lock, std::chrono::milliseconds(500), [&] {
        return gripper_reader_stop_.load();
      });
    }
  }
}

void FrankaPlugin::stop_gripper_worker() {
  std::lock_guard<std::mutex> lifecycle(gripper_lifecycle_mutex_);
  gripper_reader_stop_ = true;
  gripper_reader_cv_.notify_all();
  // Only the worker's interrupt thread calls the SDK stop operation. Its stop()
  // drains the current move and all claimed stops before joining either thread.
  if (gripper_worker_) gripper_worker_->stop();
  if (gripper_state_thread_.joinable()) gripper_state_thread_.join();
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
  if (enable_gripper_ && gripper_worker_) {
    std::lock_guard<std::mutex> lifecycle(gripper_lifecycle_mutex_);
    if (!stop_ && !gripper_state_thread_.joinable()) {
      gripper_worker_->start();
      gripper_reader_stop_ = false;
      gripper_state_thread_ = std::thread(&FrankaPlugin::gripper_state_loop, this);
    }
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
            // std::cout << "[FrankaPlugin] q_des[0]=" << q_des[0] 
            //           << " q=" << robot_state.q[0]
            //           << " err=" << (q_des[0] - robot_state.q[0])
            //           << " tau=" << tau_d[0] << "\n";
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
        if (gripper_worker_) gripper_worker_->set_enabled(false);
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
    auto* force = obs.mutable_force();
    force->set_fx(robot_state.O_F_ext_hat_K[0]);
    force->set_fy(robot_state.O_F_ext_hat_K[1]);
    force->set_fz(robot_state.O_F_ext_hat_K[2]);
    force->set_tx(robot_state.O_F_ext_hat_K[3]);
    force->set_ty(robot_state.O_F_ext_hat_K[4]);
    force->set_tz(robot_state.O_F_ext_hat_K[5]);
    set_pose_from_transform(robot_state.O_T_EE, obs.mutable_pose_ee());
    set_pose_from_transform(robot_state.O_T_EE_d, obs.mutable_pose_ee_d());
    if (enable_gripper_ && gripper_worker_) {
      const auto feedback = gripper_worker_->feedback();
      if (feedback.valid) {
        const double gripper_width = feedback.width;
        const double width_range = gripper_max_width_ - gripper_closed_width_;
        const double gripper_close_norm =
            width_range > 1e-9
                ? clamp_double((gripper_max_width_ - gripper_width) / width_range, 0.0, 1.0)
                : 0.0;
        auto* joint = obs.add_joints();
        joint->set_position(gripper_close_norm);
        joint->set_velocity(0.0);
        joint->set_effort(0.0);
      }
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
