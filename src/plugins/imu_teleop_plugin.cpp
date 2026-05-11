#include "plugins/imu_teleop_plugin.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "franka.pb.h"
#include "imu.pb.h"

namespace robo_lab {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

bool ImuTeleopPlugin::load_config(const std::string& config_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    std::cerr << "imu_teleop_plugin: YAML error in " << config_path
              << ": " << e.what() << '\n';
    return false;
  }

  if (root["imu_topic"])           imu_topic_           = root["imu_topic"].as<std::string>();
  if (root["cmd_topic"])           cmd_topic_           = root["cmd_topic"].as<std::string>();
  if (root["highpass_alpha"])      highpass_alpha_      = root["highpass_alpha"].as<float>();
  if (root["lowpass_beta"])        lowpass_beta_        = root["lowpass_beta"].as<float>();
  if (root["velocity_decay"])      velocity_decay_      = root["velocity_decay"].as<float>();
  if (root["velocity_threshold"])  velocity_threshold_  = root["velocity_threshold"].as<float>();
  if (root["linear_scale"])        linear_scale_        = root["linear_scale"].as<float>();
  if (root["angular_scale"])       angular_scale_       = root["angular_scale"].as<float>();
  if (root["max_linear_vel"])      max_linear_vel_      = root["max_linear_vel"].as<float>();
  if (root["max_angular_vel"])     max_angular_vel_     = root["max_angular_vel"].as<float>();
  return true;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float ImuTeleopPlugin::clamp(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// Bandpass filter + dead-zone for linear acceleration → velocity → dpose
// ---------------------------------------------------------------------------

ImuTeleopPlugin::Vec3 ImuTeleopPlugin::filter_accel(const Vec3& accel, float dt) {
  const float a[3] = {accel.x, accel.y, accel.z};
  float out[3]{};

  for (int i = 0; i < 3; ++i) {
    auto& f = axis_filters_[static_cast<size_t>(i)];

    // High-pass: remove gravity / DC bias.
    f.hp_out = highpass_alpha_ * (f.hp_out + a[i] - f.prev_accel);
    f.prev_accel = a[i];

    // Low-pass: smooth out high-frequency noise.
    f.lp_out = lowpass_beta_ * f.hp_out + (1.0f - lowpass_beta_) * f.lp_out;

    // Integrate to velocity.
    f.velocity += f.lp_out * dt;

    // Decay to fight drift.
    f.velocity *= velocity_decay_;

    // Dead-zone.
    if (std::fabs(f.velocity) < velocity_threshold_) {
      f.velocity = 0.0f;
    }

    // Clamp for safety.
    f.velocity = clamp(f.velocity, -max_linear_vel_, max_linear_vel_);

    // Output: velocity * scale * dt → position increment.
    out[i] = f.velocity * linear_scale_;
  }

  return {out[0], out[1], out[2]};
}

// ---------------------------------------------------------------------------
// Orientation delta: diff between current and previous quaternion → euler deltas
// ---------------------------------------------------------------------------

ImuTeleopPlugin::Vec3 ImuTeleopPlugin::compute_orientation_delta(
    float qw, float qx, float qy, float qz) {
  if (!prev_quat_valid_) {
    prev_quat_ = {qw, qx, qy, qz};
    prev_quat_valid_ = true;
    return {0, 0, 0};
  }

  // dq = q_curr * conj(q_prev)
  const auto& p = prev_quat_;
  const Quaternion pc = p.conjugate();
  Quaternion dq;
  dq.w = qw * pc.w - qx * pc.x - qy * pc.y - qz * pc.z;
  dq.x = qw * pc.x + qx * pc.w + qy * pc.z - qz * pc.y;
  dq.y = qw * pc.y - qx * pc.z + qy * pc.w + qz * pc.x;
  dq.z = qw * pc.z + qx * pc.y - qy * pc.x + qz * pc.w;

  prev_quat_ = {qw, qx, qy, qz};

  // Small-angle approximation: for unit quaternion close to identity,
  // euler angles ≈ 2 * (x, y, z) component of dq.
  float droll  = 2.0f * dq.x * angular_scale_;
  float dpitch = 2.0f * dq.y * angular_scale_;
  float dyaw   = 2.0f * dq.z * angular_scale_;

  droll  = clamp(droll,  -max_angular_vel_, max_angular_vel_);
  dpitch = clamp(dpitch, -max_angular_vel_, max_angular_vel_);
  dyaw   = clamp(dyaw,   -max_angular_vel_, max_angular_vel_);

  return {droll, dpitch, dyaw};
}

// ---------------------------------------------------------------------------
// Zenoh callback
// ---------------------------------------------------------------------------

void ImuTeleopPlugin::on_imu_message(const std::string& /*key*/,
                                     const std::string& payload) {
  imu::ImuStamped imu_msg;
  if (!imu_msg.ParseFromString(payload)) return;
  // std::cout << "imu_msg: " << imu_msg.DebugString() << std::endl;

  // Use a fixed dt based on N100 typical output rate (~200 Hz).
  constexpr float dt = 0.005f;

  Vec3 dpos{};
  if (imu_msg.has_imu()) {
    const auto& acc = imu_msg.imu().acceleration();
    dpos = filter_accel({acc.x(), acc.y(), acc.z()}, dt);
  }

  Vec3 drot{};
  if (imu_msg.has_ahrs()) {
    const auto& q = imu_msg.ahrs().quaternion();
    drot = compute_orientation_delta(q.w(), q.x(), q.y(), q.z());
  }

  // Only publish if there's meaningful motion.
  if (dpos.x == 0 && dpos.y == 0 && dpos.z == 0 &&
      drot.x == 0 && drot.y == 0 && drot.z == 0) {
    return;
  }

  std::cout << "dpos: " << dpos.x << ", " << dpos.y << ", " << dpos.z << std::endl;
  std::cout << "drot: " << drot.x << ", " << drot.y << ", " << drot.z << std::endl;
  franka::CartesianDPoseCmd cmd;
  cmd.set_dx(dpos.x);
  cmd.set_dy(dpos.y);
  cmd.set_dz(dpos.z);
  cmd.set_droll(- drot.x);
  cmd.set_dpitch(drot.y);
  cmd.set_dyaw(- drot.z);

  std::string out;
  if (cmd.SerializeToString(&out)) {
    message_system_->publish(cmd_topic_, out);
  }
}

// ---------------------------------------------------------------------------
// Plugin lifecycle
// ---------------------------------------------------------------------------

bool ImuTeleopPlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;
  if (!load_config(config_path)) return false;

  message_system_ = std::make_unique<MessageSystem>();
  message_system_->initialize();
  if (!message_system_->is_open()) {
    std::cerr << "imu_teleop_plugin: failed to open Zenoh session\n";
    return false;
  }

  message_system_->subscribe(
      imu_topic_,
      [this](const std::string& key, const std::string& payload) {
        on_imu_message(key, payload);
      });

  std::cout << "imu_teleop_plugin: initialized (imu=" << imu_topic_
            << ", cmd=" << cmd_topic_ << ")\n";
  return true;
}

void ImuTeleopPlugin::run() {
  stop_ = false;
  std::cout << "imu_teleop_plugin: run loop started\n";
  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (message_system_) message_system_->close();
  std::cout << "imu_teleop_plugin: run loop exited\n";
}

void ImuTeleopPlugin::stop() {
  if (stop_.exchange(true)) return;
  std::cout << "imu_teleop_plugin: stopping\n";
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::ImuTeleopPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}
