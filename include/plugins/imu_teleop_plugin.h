#pragma once

#include "message_system.h"
#include "plugin_interface.h"

#include <array>
#include <atomic>
#include <memory>
#include <string>

namespace robo_lab {

// Subscribes to imu/data (imu::ImuStamped), applies bandpass filtering and
// dead-zone thresholding, then publishes franka::CartesianDPoseCmd increments
// for cartesian_control_plugin to consume.
class ImuTeleopPlugin : public Plugin {
 public:
  ImuTeleopPlugin() = default;
  ~ImuTeleopPlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  struct Vec3 { float x{}, y{}, z{}; };

  struct AxisFilter {
    float prev_accel{0};
    float hp_out{0};
    float lp_out{0};
    float velocity{0};
  };

  struct Quaternion {
    float w{1}, x{0}, y{0}, z{0};
    Quaternion conjugate() const { return {w, -x, -y, -z}; }
  };

  bool load_config(const std::string& config_path);
  void on_imu_message(const std::string& key, const std::string& payload);
  Vec3 filter_accel(const Vec3& accel, float dt);
  Vec3 compute_orientation_delta(float qw, float qx, float qy, float qz);
  static float clamp(float v, float lo, float hi);

  std::string config_path_;
  std::string imu_topic_{"imu/data"};
  std::string cmd_topic_{"franka/cartesian_dpose"};

  // Filter parameters.
  float highpass_alpha_{0.98f};
  float lowpass_beta_{0.2f};
  float velocity_decay_{0.95f};
  float velocity_threshold_{0.02f};
  float linear_scale_{0.001f};
  float angular_scale_{0.01f};
  float max_linear_vel_{0.1f};
  float max_angular_vel_{0.3f};

  // Per-axis filter state (x, y, z).
  std::array<AxisFilter, 3> axis_filters_{};

  // Previous quaternion for orientation differencing.
  Quaternion prev_quat_{1, 0, 0, 0};
  bool prev_quat_valid_{false};

  std::atomic<bool> stop_{false};
  std::unique_ptr<MessageSystem> message_system_;
};

}  // namespace robo_lab
