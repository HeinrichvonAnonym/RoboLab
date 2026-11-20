#pragma once
#include "plugin_interface.h"
#include "camera/rgbd.h"
#include "message_system.h"

#include <librealsense2/rs.hpp>
#include <kinect.pb.h> // use same message type as kinect plugin

#include <atomic>
#include <memory>
#include <string>

namespace robo_lab {

class RealsensePlugin : public Plugin {
 public:
  RealsensePlugin() = default;
  ~RealsensePlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  bool publish_frame(const rs2::video_frame& frame, const std::string& topic, int32_t proto_type);

  camera_support::CaptureConfig config_;
  std::atomic<bool> stop_{false};

  std::unique_ptr<MessageSystem> message_system_;

  std::unique_ptr<rs2::pipeline> pipeline_;

};

}  // namespace robo_lab