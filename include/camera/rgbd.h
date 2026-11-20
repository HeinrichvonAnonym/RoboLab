#pragma once

#include <camera.pb.h>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>
#include <cstdint>
#include <string>
#include <vector>

namespace robo_lab::camera_support {

struct CaptureConfig {
  std::string serial;
  int width{640}, height{480}, fps{15};
  bool rgb{true}, depth{true}, ir{false}, publish_rgbd{false}, use_opengl{false};
  double max_pair_skew_ms{5.0};
  std::string rgb_topic{"realsense/rgb"}, depth_topic{"realsense/depth"};
  std::string ir_topic{"realsense/ir"}, rgbd_topic{"realsense/rgbd"};
  static CaptureConfig load(const YAML::Node& node);
};

struct PairGate {
  uint64_t last_color{0}, last_depth{0};
  bool have_previous{false};
  bool accept(const camera::Image& color, const camera::Image& depth, double max_skew_ms);
};

// Throws std::invalid_argument before a consumer reads image bytes or calibration.
void validate_frame(const camera::RgbdFrame& frame);

struct PreviewConfig {
  std::string topic{"realsense/rgbd"};
  float min_depth_m{0.07f}, max_depth_m{0.50f};
  float eye_separation_m{0.018f}, convergence_m{0.30f};
  float max_depth_jump_m{0.02f};
  static PreviewConfig load(const YAML::Node& node);
  void validate() const;
};

struct Vertex {
  Eigen::Vector3f position; // Colour camera coordinates: +x right, +y down, +z forward.
  Eigen::Vector2f uv;
};
struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  size_t valid_depth_pixels{0};
  size_t total_depth_pixels{0};
};

Mesh build_mesh(const camera::RgbdFrame& frame, const PreviewConfig& config);
// Parallel virtual cameras with off-axis frusta. eye_x is measured from the RGB camera.
Eigen::Matrix4f eye_matrix(const camera::Intrinsics& intrinsics,
                           const PreviewConfig& config, float eye_x);

}  // namespace robo_lab::camera_support
