#include "plugins/franka_replayer_plugin.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "franka.pb.h"

namespace robo_lab {

bool FrankaReplayerPlugin::load_config(const std::string& config_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    std::cerr << "franka_replayer_plugin: YAML error in " << config_path
              << ": " << e.what() << '\n';
    return false;
  }

  if (!root["trajectory_file"]) {
    std::cerr << "franka_replayer_plugin: trajectory_file missing in "
              << config_path << '\n';
    return false;
  }
  trajectory_file_ = root["trajectory_file"].as<std::string>();

  if (root["cmd_topic"]) {
    cmd_topic_ = root["cmd_topic"].as<std::string>();
  }
  if (root["control_mode"]) {
    control_mode_ = root["control_mode"].as<std::string>();
  }
  if (root["publish_hz"]) {
    publish_hz_ = root["publish_hz"].as<double>();
  }
  if (root["rate_factor"]) {
    rate_factor_ = root["rate_factor"].as<double>();
  }
  if (root["loop"]) {
    // Looping is intentionally not supported: the gap between the last and
    // first recorded poses can be large and would command a sudden jump on
    // the real robot. The plugin exits cleanly after the last frame instead.
    std::cerr << "franka_replayer_plugin: 'loop' option is not supported "
                 "(unsafe pose gap on wrap-around); ignoring.\n";
  }
  if (root["startup_delay_s"]) {
    startup_delay_s_ = root["startup_delay_s"].as<double>();
  }

  if (publish_hz_ <= 0.0) {
    std::cerr << "franka_replayer_plugin: publish_hz must be > 0\n";
    return false;
  }
  if (rate_factor_ <= 0.0) {
    std::cerr << "franka_replayer_plugin: rate_factor must be > 0\n";
    return false;
  }
  return true;
}

bool FrankaReplayerPlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;
  if (!load_config(config_path)) {
    return false;
  }

  replayer_ = std::make_unique<FrankaReplayer>();
  if (!replayer_->load(trajectory_file_)) {
    return false;
  }

  message_system_ = std::make_unique<MessageSystem>();
  message_system_->initialize();
  if (!message_system_->is_open()) {
    std::cerr << "franka_replayer_plugin: failed to open Zenoh session\n";
    return false;
  }

  std::cout << "franka_replayer_plugin: initialized"
            << " (file=" << trajectory_file_
            << ", cmd_topic=" << cmd_topic_
            << ", publish_hz=" << publish_hz_
            << ", rate_factor=" << rate_factor_
            << ", startup_delay_s=" << startup_delay_s_ << ")\n";
  return true;
}

bool FrankaReplayerPlugin::publish_command(const std::array<double, 7>& q,
                                           double sys_time) {
  franka::RobotCommand cmd;
  cmd.set_type(franka::RobotCommand::TYPE_JOINT_TARGET);
  cmd.set_mode(control_mode_);
  cmd.set_sequence(sequence_++);
  cmd.set_sys_time(static_cast<float>(sys_time));
  for (int i = 0; i < 7; ++i) {
    auto* j = cmd.add_joints();
    j->set_position(q[i]);
    j->set_velocity(0.0);
    j->set_effort(0.0);
  }
  std::string payload;
  if (!cmd.SerializeToString(&payload)) {
    return false;
  }
  return message_system_->publish(cmd_topic_, payload);
}

void FrankaReplayerPlugin::run() {
  stop_ = false;
  std::cout << "franka_replayer_plugin: run loop started\n";

  if (!replayer_ || replayer_->empty()) {
    std::cerr << "franka_replayer_plugin: no trajectory loaded\n";
    if (message_system_) {
      message_system_->close();
    }
    return;
  }

  // Warm-up: give downstream consumers (e.g. franka_plugin go_home) a moment
  // to settle before we start streaming targets.
  if (startup_delay_s_ > 0.0 && !stop_) {
    std::cout << "franka_replayer_plugin: startup delay " << startup_delay_s_ << "s\n";
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(static_cast<int64_t>(startup_delay_s_ * 1000.0));
    while (!stop_ && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  const auto period = std::chrono::nanoseconds(
      static_cast<int64_t>(1.0e9 / publish_hz_));
  const int64_t total_ns =
      static_cast<int64_t>(replayer_->duration_s() * 1.0e9);

  const auto start = std::chrono::steady_clock::now();
  auto next_tick = start;

  while (!stop_) {
    const auto now = std::chrono::steady_clock::now();
    const int64_t wall_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
    const int64_t play_ns =
        static_cast<int64_t>(static_cast<double>(wall_ns) * rate_factor_);

    // Past the end of the recording -> publish the final frame once and stop.
    // Looping back to frame 0 is intentionally not supported because the
    // first/last poses may differ by a lot; that would jump the real robot.
    if (play_ns >= total_ns) {
      std::array<double, 7> q_last;
      if (replayer_->sample_at(total_ns, q_last)) {
        publish_command(q_last, static_cast<double>(total_ns) * 1e-9);
      }
      std::cout << "franka_replayer_plugin: trajectory finished, holding last "
                   "frame and exiting\n";
      break;
    }

    std::array<double, 7> q;
    if (!replayer_->sample_at(play_ns, q)) {
      break;
    }
    publish_command(q, static_cast<double>(play_ns) * 1e-9);

    next_tick += period;
    std::this_thread::sleep_until(next_tick);
  }

  std::cout << "franka_replayer_plugin: run loop exited\n";
  if (message_system_) {
    message_system_->close();
  }
}

void FrankaReplayerPlugin::stop() {
  if (stop_.exchange(true)) {
    return;
  }
  std::cout << "franka_replayer_plugin: stopping\n";
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::FrankaReplayerPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}
