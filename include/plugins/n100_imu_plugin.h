#pragma once

#include "message_system.h"
#include "plugin_interface.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace serial {
class Serial;
}

namespace robo_lab {

// Reads IMU/AHRS frames from a Wheeltech N100 over UART and publishes
// imu.ImuStamped on a configurable Zenoh topic each cycle.
//
// N100 frame format (FDI-link protocol):
//   [0xFC] [type] [len] [SN] [CRC8] [CRC16_H] [CRC16_L] [data[len]]
//   Header is 7 bytes, followed by `len` bytes of payload.
//
// Supported type IDs:
//   0x40  IMU   (gyro / accel / mag)       len = 0x38 (56 bytes)
//   0x41  AHRS  (angular vel / euler / quat) len = 0x30 (48 bytes)
class N100ImuPlugin : public Plugin {
 public:
  N100ImuPlugin() = default;
  ~N100ImuPlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  static constexpr uint8_t kFrameHeader = 0xFC;
  static constexpr uint8_t kFrameEnd    = 0xFD;
  static constexpr uint8_t kTypeImu     = 0x40;
  static constexpr uint8_t kTypeAhrs    = 0x41;
  static constexpr size_t  kHeaderLen   = 7;   // head(1)+type(1)+len(1)+sn(1)+crc8(1)+crc16(2)
  static constexpr size_t  kTailLen     = 1;   // 0xFD
  static constexpr uint8_t kImuDataLen  = 0x38;  // 56
  static constexpr uint8_t kAhrsDataLen = 0x30;  // 48

  struct ImuRaw {
    float gx{}, gy{}, gz{};           // gyroscope   (rad/s)
    float ax{}, ay{}, az{};           // accel       (m/s^2)
    float mx{}, my{}, mz{};           // magnetometer
    float pressure{};
    bool valid{false};
  };

  struct AhrsRaw {
    float roll_speed{}, pitch_speed{}, heading_speed{};  // angular vel (rad/s)
    float roll{}, pitch{}, heading{};                    // euler       (rad)
    float q1{}, q2{}, q3{}, q4{};                       // quaternion
    bool valid{false};
  };

  bool load_config(const std::string& config_path);
  bool open_serial();
  void read_loop();
  bool try_parse_frame(const std::vector<uint8_t>& buf, size_t& consumed);
  void handle_imu_frame(const uint8_t* data, size_t len);
  void handle_ahrs_frame(const uint8_t* data, size_t len);
  void publish_if_ready();

  static float decode_float(const uint8_t* p);

  std::string config_path_;
  std::string port_{"/dev/ttyUSB0"};
  uint32_t baudrate_{921600};
  std::string imu_topic_{"imu/data"};

  std::atomic<bool> stop_{false};
  std::unique_ptr<serial::Serial> serial_;
  std::unique_ptr<MessageSystem> message_system_;

  ImuRaw  latest_imu_;
  AhrsRaw latest_ahrs_;
};

}  // namespace robo_lab
