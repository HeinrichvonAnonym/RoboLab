#include "plugins/realsense_plugin.h"

#include <chrono>
#include <iostream>
#include <thread>

namespace robo_lab {
namespace {
void fill_intrinsics(const rs2::stream_profile& profile, camera::Intrinsics* out) {
  const auto k=profile.as<rs2::video_stream_profile>().get_intrinsics();
  out->set_width(k.width); out->set_height(k.height); out->set_fx(k.fx); out->set_fy(k.fy);
  out->set_ppx(k.ppx); out->set_ppy(k.ppy); out->set_distortion_model(k.model);
  for (float coefficient : k.coeffs) out->add_coefficients(coefficient);
}
void fill_image(const rs2::video_frame& frame, camera::Image::Encoding encoding, camera::Image* out) {
  out->set_width(frame.get_width()); out->set_height(frame.get_height());
  out->set_stride(frame.get_stride_in_bytes()); out->set_encoding(encoding);
  out->set_data(frame.get_data(),frame.get_data_size());
  out->set_frame_number(frame.get_frame_number()); out->set_device_timestamp_ms(frame.get_timestamp());
  out->set_timestamp_domain(frame.get_frame_timestamp_domain());
}
}  // namespace

bool RealsensePlugin::publish_frame(const rs2::video_frame& frame,const std::string& topic,int32_t type) {
  if (!frame || topic.empty()) return true;
  kinect::rgbImage msg;
  msg.set_width(frame.get_width()); msg.set_height(frame.get_height());
  msg.set_channels(frame.get_bytes_per_pixel()); msg.set_step(frame.get_stride_in_bytes());
  msg.set_type(type); msg.set_image(frame.get_data(),frame.get_data_size());
  std::string payload;
  return msg.SerializeToString(&payload) && message_system_->publish(topic,payload);
}

bool RealsensePlugin::initialize(const std::string& path) {
  try {
    config_=camera_support::CaptureConfig::load(YAML::LoadFile(path));
    stop_=false;
    message_system_=std::make_unique<MessageSystem>();
    message_system_->initialize();
    if (!message_system_->is_open()) throw std::runtime_error("Zenoh session could not be opened");
    pipeline_=std::make_unique<rs2::pipeline>();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "realsense_plugin: initialization failed: " << e.what() << '\n';
    if (message_system_) message_system_->close();
    return false;
  }
}

void RealsensePlugin::run() {
  bool started=false;
  uint64_t published=0, rejected=0;
  try {
    if (stop_) { message_system_->close(); return; }
    rs2::config cfg;
    if (!config_.serial.empty()) cfg.enable_device(config_.serial);
    if (config_.rgb) cfg.enable_stream(RS2_STREAM_COLOR,config_.width,config_.height,RS2_FORMAT_RGB8,config_.fps);
    if (config_.depth) cfg.enable_stream(RS2_STREAM_DEPTH,config_.width,config_.height,RS2_FORMAT_Z16,config_.fps);
    if (config_.ir) cfg.enable_stream(RS2_STREAM_INFRARED,1,config_.width,config_.height,RS2_FORMAT_Y8,config_.fps);
    if (config_.use_opengl) std::cout << "realsense_plugin: using the default SDK capture backend (legacy use_opengl setting)\n";
    const auto profile=pipeline_->start(cfg);
    started=true;
    const std::string serial=profile.get_device().get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    std::cout << "realsense_plugin: serial=" << serial << " " << config_.width << 'x' << config_.height << '@' << config_.fps << " RGB-D=" << config_.publish_rgbd << std::endl;
    camera::RgbdFrame packet;
    packet.set_schema_version(1); packet.set_serial(serial);
    if (config_.publish_rgbd) {
      const auto color=profile.get_stream(RS2_STREAM_COLOR), depth=profile.get_stream(RS2_STREAM_DEPTH);
      fill_intrinsics(color,packet.mutable_color_intrinsics());
      fill_intrinsics(depth,packet.mutable_depth_intrinsics());
      const auto extrinsics=depth.get_extrinsics_to(color);
      for (float x : extrinsics.rotation) packet.mutable_depth_to_color()->add_rotation(x);
      for (float x : extrinsics.translation) packet.mutable_depth_to_color()->add_translation_m(x);
    }
    camera_support::PairGate pairs;
    auto last_frame=std::chrono::steady_clock::now();
    auto last_report=last_frame;
    while (!stop_) {
      rs2::frameset frames;
      if (!pipeline_->poll_for_frames(&frames)) {
        if (std::chrono::steady_clock::now()-last_frame > std::chrono::seconds(3))
          throw std::runtime_error("no frames for 3 seconds; camera disconnected or stalled; reconnect and restart bringup");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      last_frame=std::chrono::steady_clock::now();
      auto color=frames.get_color_frame(); auto depth=frames.get_depth_frame();
      bool ok=true;
      if(config_.rgb) ok=publish_frame(color,config_.rgb_topic,RS2_FORMAT_RGB8) && ok;
      if(config_.depth) ok=publish_frame(depth,config_.depth_topic,RS2_FORMAT_Z16) && ok;
      if(config_.ir) ok=publish_frame(frames.get_infrared_frame(1),config_.ir_topic,RS2_FORMAT_Y8) && ok;
      if(config_.publish_rgbd) {
        if(color && depth) {
          fill_image(color,camera::Image::RGB8,packet.mutable_color());
          fill_image(depth,camera::Image::Z16,packet.mutable_depth());
          packet.set_depth_units_m(depth.get_units());
          packet.set_host_received_monotonic_ns(std::chrono::duration_cast<std::chrono::nanoseconds>(last_frame.time_since_epoch()).count());
          if(pairs.accept(packet.color(),packet.depth(),config_.max_pair_skew_ms)) {
            std::string payload;
            ok=packet.SerializeToString(&payload) && message_system_->publish(config_.rgbd_topic,payload) && ok;
            ++published;
          } else ++rejected;
        } else ++rejected;
      }
      if (!ok) throw std::runtime_error("Zenoh publish failed; check configured topic keys and session");
      if (last_frame-last_report >= std::chrono::seconds(5)) {
        std::cout << "realsense_plugin: rgbd_published=" << published << " pairs_rejected=" << rejected << std::endl;
        last_report=last_frame;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "realsense_plugin: capture stopped: " << e.what()
              << " (requested serial=" << (config_.serial.empty()?"auto":config_.serial)
              << " " << config_.width << 'x' << config_.height << '@' << config_.fps
              << " RGB=" << config_.rgb << " depth=" << config_.depth << " IR=" << config_.ir << ')' << std::endl;
  }
  // Only this thread publishes, stops the pipeline, and closes Zenoh.
  if (started) { try { pipeline_->stop(); } catch (const std::exception& e) { std::cerr << "realsense_plugin: stop: " << e.what() << '\n'; } }
  if (message_system_) message_system_->close();
  std::cout << "realsense_plugin: stopped; rgbd_published=" << published << " pairs_rejected=" << rejected << std::endl;
}

void RealsensePlugin::stop() { stop_=true; }
}  // namespace robo_lab

extern "C" {
ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() { return new robo_lab::RealsensePlugin(); }
ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) { delete plugin; }
}
