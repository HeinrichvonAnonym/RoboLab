#pragma once
#include "camera/rgbd.h"
#include <memory>

namespace robo_lab::camera_support {
// Construct/destroy on the thread owning a current OpenGL 3.3 context.
class RgbdRenderer {
 public:
  RgbdRenderer();
  ~RgbdRenderer();
  RgbdRenderer(const RgbdRenderer&) = delete;
  RgbdRenderer& operator=(const RgbdRenderer&) = delete;
  void upload(const camera::RgbdFrame& frame, const Mesh& mesh);
  void draw_original(int x, int y, int width, int height);
  void draw_eye(const Eigen::Matrix4f& matrix, int x, int y, int width, int height);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace robo_lab::camera_support
