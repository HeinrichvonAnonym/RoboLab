#include "plugin_interface.h"
#include "message_system.h"

#include <librealsense2/rs.hpp>
#include <kinect.pb.h> // use same message type as kinect plugin

#include <atomic>
#include <memory>
#include <mutex>
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

  std::string config_path_;
  std::string depth_topic_{"realsense/depth"};
  std::string rgb_topic_{"realsense/rgb"};
  std::string ir_topic_{"realsense/ir"};

  float fps_{15.0f};
  bool enable_rgb_{true};
  bool enable_depth_{true};
  bool enable_ir_{false};
  bool use_opengl_{false};

  std::atomic<bool> stop_{false};
  std::atomic<bool> running_{false};

  std::unique_ptr<MessageSystem> message_system_;

  std::unique_ptr<rs2::pipeline> pipeline_;
  std::mutex pipeline_mutex_;

};

}  // namespace robo_lab