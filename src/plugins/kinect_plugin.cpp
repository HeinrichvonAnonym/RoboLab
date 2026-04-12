#include "plugins/kinect_plugin.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <thread>

#include <yaml-cpp/yaml.h>

#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/color_settings.h>
#include <libfreenect2/packet_pipeline.h>



namespace robo_lab {

bool KinectPlugin::publish_frame(libfreenect2::Frame* frame,
                                 const std::string& topic,
                                 int32_t proto_type) {
  if (frame == nullptr || topic.empty()) {
    return false;
  }
  if (!message_system_ || !message_system_->is_open()) {
    return false;
  }

  const size_t img_size = frame->width * frame->height * frame->bytes_per_pixel;
  kinect::rgbImage msg;
  msg.set_width(static_cast<int32_t>(frame->width));
  msg.set_height(static_cast<int32_t>(frame->height));
  msg.set_channels(static_cast<int32_t>(frame->bytes_per_pixel));
  msg.set_step(static_cast<int32_t>(frame->width * frame->bytes_per_pixel));
  msg.set_type(proto_type);
  msg.set_image(frame->data, img_size);

  std::string payload;
  msg.SerializeToString(&payload);
  return message_system_->publish(topic, payload);
}

void KinectPlugin::update_viewer(libfreenect2::Frame* rgb,
                                 libfreenect2::Frame* ir,
                                 libfreenect2::Frame* depth) {
  if (!enable_viewer_ || !viewer_) {
    return;
  }
  if (depth) {
    viewer_->addFrame("depth", depth);
  }
  if (ir) {
    viewer_->addFrame("ir", ir);
  }
  if (rgb) {
    viewer_->addFrame("rgb", rgb);
  }
  (void)viewer_->render();
}

bool KinectPlugin::initialize(const std::string& config_path) {
    config_path_ = config_path;
    message_system_ = std::make_unique<MessageSystem>();
    message_system_->initialize();

    const YAML::Node root = YAML::LoadFile(config_path);
    if (!root) {
      std::cerr << "kinect_plugin: failed to load yaml: " << config_path << '\n';
      return false;
    }

    // Reasonable defaults; can be overridden in YAML.
    depth_topic_ = "kinect/depth";
    rgb_topic_ = "kinect/rgb";
    ir_topic_ = "kinect/ir";

    if (root["topic"]) {
      if (root["topic"]["depth"]) {
        depth_topic_ = root["topic"]["depth"].as<std::string>();
      } else{
        depth_topic_ = "kinect/depth";
      }
      if (root["topic"]["rgb"]) {
        rgb_topic_ = root["topic"]["rgb"].as<std::string>();
      } else {
        rgb_topic_ = "kinect/rgb";
      }
      if (root["topic"]["ir"]) {
        ir_topic_ = root["topic"]["ir"].as<std::string>();
      } else {
        ir_topic_ = "kinect/ir";
      }
    }

    if (root["fps"]) {
      fps_ = root["fps"].as<float>();
    } else {
      fps_ = 30;
    }
    if (root["enable_rgb"]) {
      enable_rgb_ = root["enable_rgb"].as<bool>();
    }
    if (root["enable_depth"]) {
      enable_depth_ = root["enable_depth"].as<bool>();
    }
    if (root["enable_viewer"]) {
      enable_viewer_ = root["enable_viewer"].as<bool>();
    }
    if (!enable_rgb_ && !enable_depth_) {
      std::cerr << "kinect_plugin: both enable_rgb and enable_depth are false\n";
      return false;
    }

    if (!root["serial"]) {
      std::cerr << "kinect_plugin: missing `serial` in " << config_path << '\n';
      return false;
    }

    serial_ = root["serial"].as<std::string>();
    if (serial_.empty()) {
      std::cerr << "kinect_plugin: serial is empty in " << config_path << '\n';
      return false;
    }
    if (serial_.size() != 12) {
      std::cerr << "kinect_plugin: serial must be 12 characters in " << config_path << '\n';
      return false;
    }

    if (enable_viewer_) {
      viewer_ = std::make_unique<Viewer>();
    }


    return true;
}

void KinectPlugin::run() {
  stop_ = false;
  running_ = true;
  libfreenect2::setGlobalLogger(nullptr);  // disable libfreenect2 logs

  // This loop is intentionally structured like libfreenect2's Protonect.cpp:
  // wait for frames -> optional viewer update -> copy payload for publish -> release -> publish.
  struct FramePayloadCopy {
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 0;  // bytes_per_pixel
    int32_t step = 0;
    int32_t proto_type = 0;
    std::string data;
  };

  if (message_system_ && !message_system_->is_open()) {
    std::cerr << "kinect_plugin: Zenoh not open yet; will skip publishes until it is.\n";
  }

  libfreenect2::Freenect2 freenect2;
  const int num_devices = freenect2.enumerateDevices();
  if (num_devices <= 0) {
    std::cerr << "kinect_plugin: no Kinect v2 devices found\n";
    running_ = false;
    return;
  }

  bool serial_found = false;
  for (int i = 0; i < num_devices; ++i) {
    if (freenect2.getDeviceSerialNumber(i) == serial_) {
      serial_found = true;
      break;
    }
  }
  if (!serial_found) {
    std::cerr << "kinect_plugin: Kinect device serial " << serial_ << " not found\n";
    running_ = false;
    return;
  }

  const float effective_fps = (fps_ > 0.1f) ? fps_ : 1.0f;
  const float publish_fps = (effective_fps > 60.0f) ? 60.0f : effective_fps;
  const int kPublishEveryMs = std::max(1, static_cast<int>(1000.0f / publish_fps));

  int types = 0;
  if (enable_rgb_) {
    types |= libfreenect2::Frame::Color;
  }
  if (enable_depth_) {
    types |= libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
  }
  if (types == 0) {
    std::cerr << "kinect_plugin: nothing to stream (enable_rgb/enable_depth both false)\n";
    running_ = false;
    return;
  }

  bool viewer_initialized = false;
  int reconnect_attempt = 0;
  while (!stop_) {
    // Open device by serial number. Pipeline is owned/freed by libfreenect2.
    libfreenect2::PacketPipeline* pipeline = nullptr;
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
    pipeline = new libfreenect2::CudaPacketPipeline();
    std::cout << "kinect_plugin: using CUDA packet pipeline\n";
#else
    pipeline = new libfreenect2::CpuPacketPipeline();
    std::cout << "kinect_plugin: CUDA unavailable, using CPU packet pipeline\n";
#endif
    libfreenect2::Freenect2Device* dev = freenect2.openDevice(serial_, pipeline);
    if (dev == nullptr) {
      std::cerr << "kinect_plugin: failed to open Kinect device by serial: " << serial_ << '\n';
      reconnect_attempt++;
      const int backoff_ms = std::min(3000, 300 * reconnect_attempt);
      std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      continue;
    }
    dev_ = dev;

    libfreenect2::SyncMultiFrameListener listener(types);
    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);

    bool started = false;
    if (enable_rgb_ && enable_depth_) {
      started = dev->start();
    } else {
      started = dev->startStreams(enable_rgb_, enable_depth_);
    }

    if (!started) {
      std::cerr << "kinect_plugin: failed to start Kinect streams\n";
      dev->close();
      delete dev;
      dev_ = nullptr;
      reconnect_attempt++;
      const int backoff_ms = std::min(3000, 300 * reconnect_attempt);
      std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      continue;
    }
    reconnect_attempt = 0;

    std::cout << "kinect_plugin: connected to Kinect v2 serial=" << dev->getSerialNumber() << '\n';

    if (!viewer_initialized && enable_viewer_ && viewer_) {
      viewer_->initialize();
      viewer_initialized = true;
    }

    auto last_pub = std::chrono::steady_clock::now() - std::chrono::milliseconds(kPublishEveryMs);
    int timeout_count = 0;
    int received_count = 0;
    int consecutive_timeouts = 0;
    bool reconnect_requested = false;

    int frame_count = 0;

    while (!stop_) {
      libfreenect2::FrameMap frames;

      // Shorter timeout improves recovery latency when USB stalls.
      if (!listener.waitForNewFrame(frames, 2 * 1000)) {
        timeout_count++;
        consecutive_timeouts++;
        std::cerr << "kinect_plugin: waitForNewFrame timed out x" << timeout_count
                  << " (consecutive=" << consecutive_timeouts
                  << ", last received=" << received_count
                  << "). Check USB bandwidth/power and cable.\n";

        // Recover from likely USB stall/reset by reopening the device.
        if (consecutive_timeouts >= 5) {
          std::cerr << "kinect_plugin: repeated timeouts; reopening device\n";
          reconnect_requested = true;
          break;
        }
        continue;
      }

      consecutive_timeouts = 0;

      received_count++;

      // Extract pointers (valid until listener.release(frames)).
      libfreenect2::Frame* rgb = nullptr;
      libfreenect2::Frame* ir = nullptr;
      libfreenect2::Frame* depth = nullptr;
      auto it_rgb = frames.find(libfreenect2::Frame::Color);
      if (it_rgb != frames.end()) {
        rgb = it_rgb->second;
      }
      auto it_ir = frames.find(libfreenect2::Frame::Ir);
      if (it_ir != frames.end()) {
        ir = it_ir->second;
      }
      auto it_depth = frames.find(libfreenect2::Frame::Depth);
      if (it_depth != frames.end()) {
        depth = it_depth->second;
      }

      if (enable_viewer_) {
      update_viewer(rgb, ir, depth);
      }

      const auto now = std::chrono::steady_clock::now();
      const bool do_publish = (now - last_pub) >= std::chrono::milliseconds(kPublishEveryMs);

      // Copy frame data for protobuf publishing, then release listener frames.
      // This prevents heavy protobuf serialization from stalling capture.
      FramePayloadCopy rgb_copy;
      FramePayloadCopy depth_copy;
      FramePayloadCopy ir_copy;
      bool have_rgb_copy = false;
      bool have_depth_copy = false;
      bool have_ir_copy = false;

      auto copy_one = [&](libfreenect2::Frame* f,
                           int32_t proto_type,
                           FramePayloadCopy* out,
                           bool* have_out) {
        if (!f) {
          return;
        }
        out->width = static_cast<int32_t>(f->width);
        out->height = static_cast<int32_t>(f->height);
        out->channels = static_cast<int32_t>(f->bytes_per_pixel);
        out->step = static_cast<int32_t>(f->width * f->bytes_per_pixel);
        out->proto_type = proto_type;

        const size_t img_size = f->width * f->height * f->bytes_per_pixel;
        out->data.assign(reinterpret_cast<const char*>(f->data), img_size);
        *have_out = true;
      };

      if (do_publish && message_system_ && message_system_->is_open()) {
        if (enable_rgb_) {
          copy_one(rgb, static_cast<int32_t>(libfreenect2::Frame::Color), &rgb_copy, &have_rgb_copy);
        }
        if (enable_depth_) {
          copy_one(depth, static_cast<int32_t>(libfreenect2::Frame::Depth), &depth_copy, &have_depth_copy);
          copy_one(ir, static_cast<int32_t>(libfreenect2::Frame::Ir), &ir_copy, &have_ir_copy);
        }
        last_pub = now;
      }

      listener.release(frames);

      // Publish after release (Protonect doesn't publish, so this is our best-effort async-ish path).
      if (do_publish && message_system_ && message_system_->is_open()) {

        if (frame_count++ % 100 == 0 && frame_count > 1) {
          frame_count = 0;
          const auto wall_now = std::chrono::system_clock::now();
          const std::time_t t = std::chrono::system_clock::to_time_t(wall_now);
          const std::tm* lt = std::localtime(&t);
          if (lt) {
            std::cout << "[kinect_plugin]: publishing frames at "
                    << std::setw(2) << std::setfill('0') << lt->tm_hour
                    << ":" << std::setw(2) << std::setfill('0') << lt->tm_min
                    << ":" << std::setw(2) << std::setfill('0') << lt->tm_sec
                      << std::endl;
          }
        }
        if (enable_rgb_ && have_rgb_copy) {
          kinect::rgbImage msg;
          msg.set_width(rgb_copy.width);
          msg.set_height(rgb_copy.height);
          msg.set_channels(rgb_copy.channels);
          msg.set_step(rgb_copy.step);
          msg.set_type(rgb_copy.proto_type);
          msg.set_image(rgb_copy.data.data(), rgb_copy.data.size());
          std::string payload;
          msg.SerializeToString(&payload);
          (void)message_system_->publish(rgb_topic_, payload);
        }

        if (enable_depth_) {
          if (have_depth_copy) {
            kinect::rgbImage msg;
            msg.set_width(depth_copy.width);
            msg.set_height(depth_copy.height);
            msg.set_channels(depth_copy.channels);
            msg.set_step(depth_copy.step);
            msg.set_type(depth_copy.proto_type);
            msg.set_image(depth_copy.data.data(), depth_copy.data.size());
            std::string payload;
            msg.SerializeToString(&payload);
            (void)message_system_->publish(depth_topic_, payload);
          }
          if (have_ir_copy) {
            kinect::rgbImage msg;
            msg.set_width(ir_copy.width);
            msg.set_height(ir_copy.height);
            msg.set_channels(ir_copy.channels);
            msg.set_step(ir_copy.step);
            msg.set_type(ir_copy.proto_type);
            msg.set_image(ir_copy.data.data(), ir_copy.data.size());
            std::string payload;
            msg.SerializeToString(&payload);
            (void)message_system_->publish(ir_topic_, payload);
          }
        }
      }
    }

    if (dev_) {
      dev_->stop();
      dev_->close();
      delete dev_;
      dev_ = nullptr;
    }

    if (!stop_ && reconnect_requested) {
      reconnect_attempt++;
      const int backoff_ms = std::min(3000, 300 * reconnect_attempt);
      std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
    }
  }

  running_ = false;
  if (message_system_) {
    message_system_->close();
  }
}

void KinectPlugin::stop() {
  if (stop_.exchange(true)) {
    return;
  }
  std::cout << "kinect_plugin: stopping\n";
}

}
extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::KinectPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}