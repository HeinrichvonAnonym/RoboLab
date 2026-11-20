#pragma once

#include "plugins/gripper_worker.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace robo_lab {

// Robotiq 2F-85 Modbus RTU adapter. Construction and read_state() never write
// control registers. Activation is an explicit on-site operation.
class RobotiqGripper : public GripperDevice {
 public:
  struct Options {
    std::string port;
    int baudrate{115200};
    int slave_id{9};
    int timeout_ms{200};
    int motion_timeout_ms{5000};
    int force{32};  // Raw Robotiq register, not newtons.
    int open_position{13};
    int closed_position{230};
    double max_width{0.085};
  };
  struct State {
    bool activated{false};
    int activation_status{0};
    int object_status{0};
    int fault{0};
    int requested_position{0};
    int position{0};
    double width{0.0};  // Estimated from configured encoder endpoints.
    bool ready() const { return activated && activation_status == 3; }
  };
  explicit RobotiqGripper(Options options);
  ~RobotiqGripper() override;
  State read_state();
  void activate();  // May move fingers; never called by plugin initialization.
  bool move(double width, double speed) override;
  bool stop() override;

 private:
  std::vector<uint8_t> transact(std::vector<uint8_t> request, std::size_t response_size);
  void command(uint8_t action, uint8_t position, uint8_t speed);
  Options options_;
  int fd_{-1};
  std::mutex io_mutex_;
  std::mutex command_mutex_;
  std::atomic<uint64_t> cancellation_{0};
};

}  // namespace robo_lab
