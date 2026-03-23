#include "plugins/realsense_plugin.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include <yaml-cpp/yaml.h>

namespace robo_lab {

bool RealsensePlugin::publish_frame(const rs2::video_frame& frame,
                                    const std::string& topic,
                                    int32_t proto_type) {
  if (!frame || topic.empty()) {
    return false;
  }
  if (!message_system_ || !message_system_->is_open()) {
    return false;
  }

  kinect::rgbImage msg;
  msg.set_width(frame.get_width());
  msg.set_height(frame.get_height());
  msg.set_channels(frame.get_bytes_per_pixel());
  msg.set_step(frame.get_stride_in_bytes());
  msg.set_type(proto_type);
  msg.set_image(reinterpret_cast<const char*>(frame.get_data()), frame.get_data_size());

  std::string payload;
  msg.SerializeToString(&payload);
  return message_system_->publish(topic, payload);
}

bool RealsensePlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;
  message_system_ = std::make_unique<MessageSystem>();
  message_system_->initialize();

  const YAML::Node root = YAML::LoadFile(config_path_);
  if (!root) {
    std::cerr << "realsense_plugin: failed to load yaml: " << config_path_ << "\n";
    return false;
  }

  if (root["topic"]) {
    if (root["topic"]["rgb"]) {
      rgb_topic_ = root["topic"]["rgb"].as<std::string>();
    }
    if (root["topic"]["depth"]) {
      depth_topic_ = root["topic"]["depth"].as<std::string>();
    }
    if (root["topic"]["ir"]) {
      ir_topic_ = root["topic"]["ir"].as<std::string>();
    }
  }

  if (root["fps"]) {
    fps_ = root["fps"].as<float>();
  }
  if (root["enable_rgb"]) {
    enable_rgb_ = root["enable_rgb"].as<bool>();
  }
  if (root["enable_depth"]) {
    enable_depth_ = root["enable_depth"].as<bool>();
  }
  if (root["enable_ir"]) {
    enable_ir_ = root["enable_ir"].as<bool>();
  }
  if (root["use_opengl"]) {
    use_opengl_ = root["use_opengl"].as<bool>();
  }

  if (!enable_rgb_ && !enable_depth_ && !enable_ir_) {
    std::cerr << "realsense_plugin: all streams are disabled in config\n";
    return false;
  }

  if (fps_ <= 0.0f) {
    fps_ = 15.0f;
  }

  pipeline_ = std::make_unique<rs2::pipeline>();
  return true;
}

void RealsensePlugin::run() {
  stop_ = false;
  running_ = true;

  if (message_system_ && !message_system_->is_open()) {
    std::cerr << "realsense_plugin: Zenoh not open yet; publish is skipped until available\n";
  }

  rs2::config cfg;
  const int stream_fps = std::max(1, static_cast<int>(fps_));

  if (enable_rgb_) {
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, stream_fps);
  }
  if (enable_depth_) {
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, stream_fps);
  }
  if (enable_ir_) {
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, stream_fps);
  }

  try {
    std::lock_guard<std::mutex> lk(pipeline_mutex_);
    if (use_opengl_) {
      std::cout << "realsense_plugin: OpenGL acceleration requested; using default librealsense backend\n";
    }
    pipeline_->start(cfg);
  } catch (const rs2::error& e) {
    std::cerr << "realsense_plugin: failed to start pipeline: " << e.what() << "\n";
    running_ = false;
    return;
  }

  const auto publish_period =
      std::chrono::milliseconds(std::max(1, static_cast<int>(1000.0f / fps_)));
  auto last_pub = std::chrono::steady_clock::now() - publish_period;

  while (!stop_) {
    rs2::frameset frames;
    try {
      std::lock_guard<std::mutex> lk(pipeline_mutex_);
      if (!pipeline_) {
        break;
      }
      if (!pipeline_->poll_for_frames(&frames)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
    } catch (const rs2::error& e) {
      std::cerr << "realsense_plugin: frame polling error: " << e.what() << "\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - last_pub < publish_period) {
      continue;
    }
    last_pub = now;

    if (enable_rgb_) {
        // std::cout << "realsense_plugin: publishing rgb frame\n";
      rs2::video_frame color = frames.get_color_frame();
      (void)publish_frame(color, rgb_topic_, static_cast<int32_t>(RS2_FORMAT_RGB8));
    }
    if (enable_depth_) {
      rs2::depth_frame depth = frames.get_depth_frame();
      (void)publish_frame(depth, depth_topic_, static_cast<int32_t>(RS2_FORMAT_Z16));
    }
    if (enable_ir_) {
      rs2::video_frame ir = frames.get_infrared_frame(1);
      (void)publish_frame(ir, ir_topic_, static_cast<int32_t>(RS2_FORMAT_Y8));
    }
  }

  {
    std::lock_guard<std::mutex> lk(pipeline_mutex_);
    if (pipeline_) {
      try {
        pipeline_->stop();
      } catch (...) {
      }
    }
  }

  running_ = false;
}

void RealsensePlugin::stop() {
  std::cout << "realsense_plugin: stopping\n";
  stop_ = true;
  {
    std::lock_guard<std::mutex> lk(pipeline_mutex_);
    if (pipeline_) {
      try {
        pipeline_->stop();
      } catch (...) {
      }
    }
  }
  if (message_system_) {
    message_system_->close();
  }
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::RealsensePlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}