#include "plugins/kinect_plugin.h"
#include "plugins/kinect_sampling.h"

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
    viewer_->addFrame("RGB", rgb);  // Viewer selects its color shader with this key.
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
    rgb_downsample_ = depth_downsample_ = ir_downsample_ = 1;
    try {
      const YAML::Node downsample = root["downsample"];
      if (downsample.IsDefined()) {
        if (!downsample.IsMap()) {
          std::cerr << "kinect_plugin: downsample must be a map of positive integers\n";
          return false;
        }
        auto read_factor = [&](const char* stream, int& factor) {
          const YAML::Node value = downsample[stream];
          if (value.IsDefined()) {
            factor = value.as<int>();
          }
          if (factor <= 0) {
            std::cerr << "kinect_plugin: downsample." << stream
                      << " must be a positive integer\n";
            return false;
          }
          return true;
        };
        if (!read_factor("rgb", rgb_downsample_) ||
            !read_factor("depth", depth_downsample_) ||
            !read_factor("ir", ir_downsample_)) {
          return false;
        }
      }
    } catch (const YAML::Exception& error) {
      std::cerr << "kinect_plugin: invalid downsample configuration (expected positive integers): "
                << error.what() << '\n';
      return false;
    }
    std::cout << "kinect_plugin: downsample rgb=" << rgb_downsample_
              << " depth=" << depth_downsample_ << " ir=" << ir_downsample_ << '\n';
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
  // Keep device/USB errors visible; LIBFREENECT2_LOGGER_LEVEL controls verbosity.
  libfreenect2::setGlobalLogger(libfreenect2::createConsoleLoggerWithDefaultLevel());

  // Keep the existing capture and reconnect sequence. Only copy sampled pixels
  // after wait succeeds, then release native frames before rendering/publishing.
  struct FrameRelease {
    libfreenect2::SyncMultiFrameListener& listener;
    libfreenect2::FrameMap& frames;
    bool active = true;
    ~FrameRelease() { if (active) listener.release(frames); }
    void release() { listener.release(frames); active = false; }
  };
  kinect_detail::SampledImage rgb_copy, depth_copy, ir_copy;
  std::string payload;

  // Frame's external-data constructor leaves rawdata null; these views never
  // free the sampled strings. Their addresses remain valid throughout run().
  unsigned char unused = 0;
  libfreenect2::Frame rgb_view(0, 0, 0, &unused);
  libfreenect2::Frame depth_view(0, 0, 0, &unused);
  libfreenect2::Frame ir_view(0, 0, 0, &unused);

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
    // A USB reset/replug can change the device address; refresh cached handles before reopening.
    freenect2.enumerateDevices();
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

    auto listener = std::make_unique<libfreenect2::SyncMultiFrameListener>(types);
    dev->setColorFrameListener(listener.get());
    dev->setIrAndDepthFrameListener(listener.get());

    bool started = false;
    if (enable_rgb_ && enable_depth_) {
      started = dev->start();
    } else {
      started = dev->startStreams(enable_rgb_, enable_depth_);
    }

    if (!started) {
      std::cerr << "kinect_plugin: failed to start Kinect streams\n";
      dev->close();
      listener.reset();
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
    bool output_sizes_logged = false;

    while (!stop_) {
      libfreenect2::FrameMap frames;

      // Shorter timeout improves recovery latency when USB stalls.
      if (!listener->waitForNewFrame(frames, 2 * 1000)) {
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
      FrameRelease frame_release{*listener, frames};

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

      const auto now = std::chrono::steady_clock::now();
      const bool do_publish = (now - last_pub) >= std::chrono::milliseconds(kPublishEveryMs);

      // Copy frame data for protobuf publishing, then release listener frames.
      // This prevents heavy protobuf serialization from stalling capture.
      bool have_rgb_copy = false;
      bool have_depth_copy = false;
      bool have_ir_copy = false;

      auto copy_one = [&](libfreenect2::Frame* f,
                           int32_t proto_type,
                           int factor, const char* name,
                           kinect_detail::SampledImage* out,
                           libfreenect2::Frame* view,
                           bool* have_out) {
        if (!f) {
          return;
        }
        kinect_detail::sample_image(f->data, f->width, f->height,
                                   f->bytes_per_pixel, factor, proto_type, *out);
        view->width = out->width;
        view->height = out->height;
        view->bytes_per_pixel = out->channels;
        view->data = reinterpret_cast<unsigned char*>(&out->data[0]);
        view->format = f->format;
        *have_out = true;
        if (!output_sizes_logged) {
          std::cout << "kinect_plugin: " << name << " downsample=" << factor
                    << " " << f->width << "x" << f->height << " -> "
                    << out->width << "x" << out->height
                    << " step=" << out->step << " bytes=" << out->data.size() << '\n';
        }
      };

      const bool can_publish = message_system_ && message_system_->is_open();
      if (do_publish && (can_publish || enable_viewer_)) {
        try {
          if (enable_rgb_) {
            copy_one(rgb, static_cast<int32_t>(libfreenect2::Frame::Color),
                     rgb_downsample_, "rgb", &rgb_copy, &rgb_view, &have_rgb_copy);
          }
          if (enable_depth_) {
            copy_one(depth, static_cast<int32_t>(libfreenect2::Frame::Depth),
                     depth_downsample_, "depth", &depth_copy, &depth_view, &have_depth_copy);
            copy_one(ir, static_cast<int32_t>(libfreenect2::Frame::Ir),
                     ir_downsample_, "ir", &ir_copy, &ir_view, &have_ir_copy);
          }
        } catch (const std::exception& error) {
          std::cerr << "kinect_plugin: failed to copy sampled frame: " << error.what() << '\n';
          continue;  // FrameRelease still releases the acquired native frames.
        }
        output_sizes_logged = true;
        last_pub = now;
      }

      frame_release.release();

      if (do_publish && enable_viewer_) {
        update_viewer(have_rgb_copy ? &rgb_view : nullptr,
                      have_ir_copy ? &ir_view : nullptr,
                      have_depth_copy ? &depth_view : nullptr);
      }

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
        auto publish_copy = [&](const kinect_detail::SampledImage& image,
                                const std::string& topic) {
          kinect_detail::serialize_image(image, payload);
          (void)message_system_->publish(topic, payload);
        };
        if (enable_rgb_ && have_rgb_copy) {
          publish_copy(rgb_copy, rgb_topic_);
        }
        if (enable_depth_) {
          if (have_depth_copy) {
            publish_copy(depth_copy, depth_topic_);
          }
          if (have_ir_copy) {
            publish_copy(ir_copy, ir_topic_);
          }
        }
      }
    }

    if (dev_) {
      dev_->stop();
      dev_->close();
      // Release queued CUDA frames while the device's pipeline allocators still exist.
      listener.reset();
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
