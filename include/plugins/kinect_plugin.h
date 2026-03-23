#pragma once

#include "plugin_interface.h"
#include "message_system.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <kinect.pb.h>

#include "../../third_party/libfreenect2/examples/viewer.h"


namespace robo_lab {

/// Minimal sample plugin: reads a few keys from its YAML and runs until stop().
class KinectPlugin : public Plugin {
 public:
  KinectPlugin() = default;
  ~KinectPlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  bool publish_frame(libfreenect2::Frame* frame,
                     const std::string& topic,
                     int32_t proto_type);
  void update_viewer(libfreenect2::Frame* rgb,
                     libfreenect2::Frame* ir,
                     libfreenect2::Frame* depth);

  float fps_;
  bool enable_rgb_{true};
  bool enable_depth_{true};
  bool enable_viewer_{true};
  std::string config_path_;
  std::string robot_ip_;
  std::string depth_topic_;
  std::string rgb_topic_;
  std::string ir_topic_;

  std::string serial_;

  std::atomic<bool> stop_{false};
  std::atomic<bool> running_{false};

  std::unique_ptr<MessageSystem> message_system_;

  std::unique_ptr<Viewer> viewer_;

  // Active libfreenect2 device pointer used for safe shutdown from `stop()`.
  libfreenect2::Freenect2Device* dev_{nullptr};
  std::mutex dev_mutex_;
};
}  // namespace robo_lab
