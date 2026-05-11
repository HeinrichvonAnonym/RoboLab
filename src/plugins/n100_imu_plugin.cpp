#include "plugins/n100_imu_plugin.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#include <serial/serial.h>
#include <yaml-cpp/yaml.h>

#include "imu.pb.h"

namespace robo_lab {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

bool N100ImuPlugin::load_config(const std::string& config_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    std::cerr << "n100_imu_plugin: YAML error in " << config_path
              << ": " << e.what() << '\n';
    return false;
  }

  if (root["port"])      port_      = root["port"].as<std::string>();
  if (root["baudrate"])  baudrate_  = root["baudrate"].as<uint32_t>();
  if (root["imu_topic"]) imu_topic_ = root["imu_topic"].as<std::string>();
  return true;
}

// ---------------------------------------------------------------------------
// Serial helpers
// ---------------------------------------------------------------------------

bool N100ImuPlugin::open_serial() {
  try {
    serial_ = std::make_unique<serial::Serial>(
        port_, baudrate_, serial::Timeout::simpleTimeout(100));
  } catch (const serial::IOException& e) {
    std::cerr << "n100_imu_plugin: failed to open " << port_
              << " @ " << baudrate_ << ": " << e.what() << '\n';
    return false;
  }
  if (!serial_->isOpen()) {
    std::cerr << "n100_imu_plugin: port " << port_ << " not open\n";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Byte-level helpers
// ---------------------------------------------------------------------------

float N100ImuPlugin::decode_float(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, sizeof(float));
  return v;
}

// ---------------------------------------------------------------------------
// Frame parser
// ---------------------------------------------------------------------------
//
// N100 FDI-link frame layout:
//   [0]          0xFC       frame header
//   [1]          type_id    0x40=IMU, 0x41=AHRS, 0x42=INSGPS, ...
//   [2]          data_len   payload byte count (0x38 for IMU, 0x30 for AHRS)
//   [3]          SN         sequence number (increments per frame)
//   [4]          CRC8       polynomial CRC over header
//   [5]          CRC16_H    \  polynomial CRC over data payload
//   [6]          CRC16_L    /
//   [7 .. 7+N-1] data[N]
//   [7+N]        0xFD       frame tail
//
// Total frame size = 7 + data_len + 1.

bool N100ImuPlugin::try_parse_frame(const std::vector<uint8_t>& buf,
                                    size_t& consumed) {
  bool got_frame = false;

  while (consumed + kHeaderLen + kTailLen <= buf.size()) {
    if (buf[consumed] != kFrameHeader) {
      ++consumed;
      continue;
    }

    const size_t hdr = consumed;
    const uint8_t type_id  = buf[hdr + 1];
    const uint8_t data_len = buf[hdr + 2];

    if (type_id != kTypeImu && type_id != kTypeAhrs) {
      ++consumed;
      continue;
    }

    if (type_id == kTypeImu  && data_len != kImuDataLen)  { ++consumed; continue; }
    if (type_id == kTypeAhrs && data_len != kAhrsDataLen) { ++consumed; continue; }

    const size_t frame_len = kHeaderLen + data_len + kTailLen;
    if (hdr + frame_len > buf.size()) break;

    if (buf[hdr + frame_len - 1] != kFrameEnd) {
      ++consumed;
      continue;
    }

    const uint8_t* payload = &buf[hdr + kHeaderLen];
    switch (type_id) {
      case kTypeImu:  handle_imu_frame(payload, data_len);  break;
      case kTypeAhrs: handle_ahrs_frame(payload, data_len); break;
      default: break;
    }

    consumed = hdr + frame_len;
    got_frame = true;
  }

  return got_frame;
}

// ---------------------------------------------------------------------------
// IMU frame (0x40, 56 bytes):
//   gyro_x/y/z  [0..11]   (3 x float, rad/s)
//   accel_x/y/z [12..23]  (3 x float, m/s^2)
//   mag_x/y/z   [24..35]  (3 x float)
//   pressure     [36..39]  (float, Pa)
// ---------------------------------------------------------------------------

void N100ImuPlugin::handle_imu_frame(const uint8_t* data, size_t len) {
  if (len < 40) return;
  latest_imu_.gx = decode_float(data + 0);
  latest_imu_.gy = decode_float(data + 4);
  latest_imu_.gz = decode_float(data + 8);
  latest_imu_.ax = decode_float(data + 12);
  latest_imu_.ay = decode_float(data + 16);
  latest_imu_.az = decode_float(data + 20);
  latest_imu_.mx = decode_float(data + 24);
  latest_imu_.my = decode_float(data + 28);
  latest_imu_.mz = decode_float(data + 32);
  latest_imu_.pressure = decode_float(data + 36);
  latest_imu_.valid = true;
}

// ---------------------------------------------------------------------------
// AHRS frame (0x41, 48 bytes):
//   roll_speed / pitch_speed / heading_speed [0..11]  (3 x float, rad/s)
//   roll / pitch / heading                   [12..23] (3 x float, rad)
//   Q1 / Q2 / Q3 / Q4                       [24..39] (4 x float)
// ---------------------------------------------------------------------------

void N100ImuPlugin::handle_ahrs_frame(const uint8_t* data, size_t len) {
  if (len < 40) return;
  latest_ahrs_.roll_speed    = decode_float(data + 0);
  latest_ahrs_.pitch_speed   = decode_float(data + 4);
  latest_ahrs_.heading_speed = decode_float(data + 8);
  latest_ahrs_.roll    = decode_float(data + 12);
  latest_ahrs_.pitch   = decode_float(data + 16);
  latest_ahrs_.heading = decode_float(data + 20);
  latest_ahrs_.q1 = decode_float(data + 24);
  latest_ahrs_.q2 = decode_float(data + 28);
  latest_ahrs_.q3 = decode_float(data + 32);
  latest_ahrs_.q4 = decode_float(data + 36);
  latest_ahrs_.valid = true;
}

// ---------------------------------------------------------------------------
// Publish
// ---------------------------------------------------------------------------

void N100ImuPlugin::publish_if_ready() {
  if (!latest_imu_.valid && !latest_ahrs_.valid) return;

  imu::ImuStamped msg;

  if (latest_imu_.valid) {
    auto* imu_d = msg.mutable_imu();
    auto* acc = imu_d->mutable_acceleration();
    acc->set_x(latest_imu_.ax);
    acc->set_y(latest_imu_.ay);
    acc->set_z(latest_imu_.az);
    auto* gyr = imu_d->mutable_gyroscope();
    gyr->set_x(latest_imu_.gx);
    gyr->set_y(latest_imu_.gy);
    gyr->set_z(latest_imu_.gz);
    auto* mag = imu_d->mutable_magnetometer();
    mag->set_x(latest_imu_.mx);
    mag->set_y(latest_imu_.my);
    mag->set_z(latest_imu_.mz);
    imu_d->set_pressure(latest_imu_.pressure);
  }

  if (latest_ahrs_.valid) {
    auto* ahrs_d = msg.mutable_ahrs();
    auto* euler = ahrs_d->mutable_euler();
    euler->set_roll(latest_ahrs_.roll);
    euler->set_pitch(latest_ahrs_.pitch);
    euler->set_yaw(latest_ahrs_.heading);
    auto* quat = ahrs_d->mutable_quaternion();
    quat->set_w(latest_ahrs_.q1);
    quat->set_x(latest_ahrs_.q2);
    quat->set_y(latest_ahrs_.q3);
    quat->set_z(latest_ahrs_.q4);
  }

  using clock = std::chrono::steady_clock;
  const auto now = clock::now();
  const float sys_sec =
      std::chrono::duration<float>(now.time_since_epoch()).count();
  msg.set_sys_time(sys_sec);

  std::string payload;
  if (msg.SerializeToString(&payload)) {
    message_system_->publish(imu_topic_, payload);
  }

  latest_imu_.valid  = false;
  latest_ahrs_.valid = false;
}

// ---------------------------------------------------------------------------
// Plugin lifecycle
// ---------------------------------------------------------------------------

bool N100ImuPlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;
  if (!load_config(config_path)) return false;

  if (!open_serial()) return false;

  message_system_ = std::make_unique<MessageSystem>();
  message_system_->initialize();
  if (!message_system_->is_open()) {
    std::cerr << "n100_imu_plugin: failed to open Zenoh session\n";
    return false;
  }

  std::cout << "n100_imu_plugin: initialized (port=" << port_
            << ", baud=" << baudrate_
            << ", topic=" << imu_topic_ << ")\n";
  return true;
}

void N100ImuPlugin::read_loop() {
  std::vector<uint8_t> ring;
  ring.reserve(4096);
  uint8_t tmp[512];

  while (!stop_) {
    size_t avail = 0;
    try {
      avail = serial_->read(tmp, sizeof(tmp));
    } catch (const serial::IOException& e) {
      std::cerr << "n100_imu_plugin: serial read error: " << e.what() << '\n';
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    if (avail == 0) continue;

    ring.insert(ring.end(), tmp, tmp + avail);

    size_t consumed = 0;
    try_parse_frame(ring, consumed);
    publish_if_ready();

    if (consumed > 0) {
      ring.erase(ring.begin(), ring.begin() + static_cast<ptrdiff_t>(consumed));
    }

    if (ring.size() > 8192) {
      ring.erase(ring.begin(), ring.end() - 2048);
    }
  }
}

void N100ImuPlugin::run() {
  stop_ = false;
  std::cout << "n100_imu_plugin: run loop started\n";
  read_loop();
  if (serial_ && serial_->isOpen()) serial_->close();
  if (message_system_) message_system_->close();
  std::cout << "n100_imu_plugin: run loop exited\n";
}

void N100ImuPlugin::stop() {
  if (stop_.exchange(true)) return;
  std::cout << "n100_imu_plugin: stopping\n";
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::N100ImuPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}
