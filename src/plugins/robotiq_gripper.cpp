#include "plugins/robotiq_gripper.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <utility>

namespace robo_lab {
namespace {
using Clock = std::chrono::steady_clock;
uint16_t crc16(const uint8_t* bytes, std::size_t size) {
  uint16_t crc = 0xffff;
  for (std::size_t i = 0; i < size; ++i) {
    crc ^= bytes[i];
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc & 1) ? (crc >> 1) ^ 0xa001 : crc >> 1;
  }
  return crc;
}
speed_t baud_constant(int baud) {
  switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    default: throw std::invalid_argument("Unsupported Robotiq baudrate");
  }
}
}  // namespace

RobotiqGripper::RobotiqGripper(Options options) : options_(std::move(options)) {
  if (options_.port.empty() || options_.slave_id < 1 || options_.slave_id > 247 ||
      options_.timeout_ms <= 0 || options_.motion_timeout_ms <= 0 ||
      options_.force < 0 || options_.force > 255 || options_.open_position < 0 ||
      options_.closed_position > 255 || options_.open_position >= options_.closed_position ||
      !std::isfinite(options_.max_width) || options_.max_width <= 0)
    throw std::invalid_argument("Invalid Robotiq serial/position configuration");
  const auto baud = baud_constant(options_.baudrate);
  fd_ = ::open(options_.port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
  if (fd_ < 0) throw std::runtime_error("Robotiq open " + options_.port + ": " + std::strerror(errno));
  try {
    if (ioctl(fd_, TIOCEXCL) < 0) throw std::runtime_error("Cannot reserve Robotiq serial port");
    termios tty{};
    if (tcgetattr(fd_, &tty) < 0) throw std::runtime_error("Cannot read Robotiq serial attributes");
    cfmakeraw(&tty);
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);
    if (tcsetattr(fd_, TCSANOW, &tty) < 0) throw std::runtime_error("Cannot configure Robotiq serial port");
  } catch (...) {
    ioctl(fd_, TIOCNXCL);
    ::close(fd_);
    fd_ = -1;
    throw;
  }
}

RobotiqGripper::~RobotiqGripper() {
  if (fd_ >= 0) { ioctl(fd_, TIOCNXCL); ::close(fd_); }
}

std::vector<uint8_t> RobotiqGripper::transact(std::vector<uint8_t> request, std::size_t response_size) {
  std::lock_guard<std::mutex> lock(io_mutex_);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));  // Modbus RTU inter-frame silence.
  const uint16_t crc = crc16(request.data(), request.size());
  request.push_back(crc & 0xff);
  request.push_back(crc >> 8);
  // Discard any late reply to an expired transaction before starting a new one.
  if (tcflush(fd_, TCIFLUSH) < 0) throw std::runtime_error("Robotiq serial disconnected");
  const auto deadline = Clock::now() + std::chrono::milliseconds(options_.timeout_ms);
  auto wait_io = [&](short event) {
    while (true) {
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
      if (remaining <= 0) throw std::runtime_error("Robotiq Modbus timeout on " + options_.port);
      pollfd pfd{fd_, event, 0};
      const int result = poll(&pfd, 1, static_cast<int>(remaining));
      if (result < 0 && errno == EINTR) continue;
      if (result < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)))
        throw std::runtime_error("Robotiq serial I/O error on " + options_.port);
      if (result > 0 && (pfd.revents & event)) return;
    }
  };
  std::size_t sent = 0;
  while (sent < request.size()) {
    wait_io(POLLOUT);
    const auto n = ::write(fd_, request.data() + sent, request.size() - sent);
    if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (n <= 0) throw std::runtime_error("Robotiq serial write failed");
    sent += static_cast<std::size_t>(n);
  }
  std::vector<uint8_t> response;
  while (response.size() < response_size) {
    wait_io(POLLIN);
    uint8_t buffer[32];
    const auto n = ::read(fd_, buffer, std::min(sizeof(buffer), response_size - response.size()));
    if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (n <= 0) throw std::runtime_error("Robotiq serial read failed");
    response.insert(response.end(), buffer, buffer + n);
    if (response.size() >= 2 && response[1] == (request[1] | 0x80)) response_size = 5;
  }
  const auto size = response.size();
  if (size < 5 || response[0] != options_.slave_id ||
      crc16(response.data(), size - 2) != (response[size - 2] | (response[size - 1] << 8)))
    throw std::runtime_error("Invalid Robotiq Modbus address or CRC");
  if (response[1] == (request[1] | 0x80))
    throw std::runtime_error("Robotiq Modbus exception " + std::to_string(response[2]));
  if (response[1] != request[1]) throw std::runtime_error("Unexpected Robotiq Modbus function");
  return response;
}

RobotiqGripper::State RobotiqGripper::read_state() {
  const auto reply = transact({static_cast<uint8_t>(options_.slave_id), 3, 7, 0xd0, 0, 3}, 11);
  if (reply[2] != 6) throw std::runtime_error("Invalid Robotiq status length");
  State state;
  state.activated = reply[3] & 1;
  state.activation_status = (reply[3] >> 4) & 3;
  state.object_status = (reply[3] >> 6) & 3;
  state.fault = reply[5] & 0x0f;
  state.requested_position = reply[6];
  state.position = reply[7];
  state.width = options_.max_width * std::clamp(
      double(options_.closed_position - state.position) /
          (options_.closed_position - options_.open_position), 0.0, 1.0);
  return state;
}

void RobotiqGripper::command(uint8_t action, uint8_t position, uint8_t speed) {
  const auto reply = transact({static_cast<uint8_t>(options_.slave_id), 0x10, 3, 0xe8, 0, 3, 6,
                              action, 0, 0, position, speed, static_cast<uint8_t>(options_.force)}, 8);
  if (reply[2] != 3 || reply[3] != 0xe8 || reply[4] != 0 || reply[5] != 3)
    throw std::runtime_error("Invalid Robotiq command acknowledgement");
}

void RobotiqGripper::activate() {
  // Operator-only routine: activation can move the fingers through calibration.
  std::lock_guard<std::mutex> lock(command_mutex_);
  try {
    command(0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    command(1, 0, 0);
    const auto deadline = Clock::now() + std::chrono::milliseconds(options_.motion_timeout_ms);
    while (Clock::now() < deadline) {
      const auto state = read_state();
      if (state.ready() && state.fault == 0) return;
      if (state.fault >= 0x0a) throw std::runtime_error("Robotiq activation fault " + std::to_string(state.fault));
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("Robotiq activation timed out");
  } catch (...) {
    try { command(0, 0, 0); } catch (...) {}
    throw;
  }
}

bool RobotiqGripper::move(double width, double speed) {
  if (!std::isfinite(width) || !std::isfinite(speed) || width < 0 ||
      width > options_.max_width || speed <= 0)
    throw std::invalid_argument("Invalid Robotiq width/speed");
  const auto generation = cancellation_.load();
  const uint8_t position = static_cast<uint8_t>(std::lround(options_.closed_position -
      width / options_.max_width * (options_.closed_position - options_.open_position)));
  // Approximate speed conversion used by the upstream 2F-85 ROS driver.
  const uint8_t speed_byte = static_cast<uint8_t>(std::clamp(255.0 * (speed - 0.013) / 0.087, 0.0, 255.0));
  bool command_attempted = false;
  try {
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      const auto state = read_state();
      if (!state.ready()) throw std::runtime_error("Robotiq is not activated; run the on-site --activate check first");
      // 0x09 is the communication watchdog fault, cleared by resuming commands.
      if (state.fault != 0 && state.fault != 9)
        throw std::runtime_error("Robotiq fault " + std::to_string(state.fault));
      if (generation != cancellation_.load()) return false;
      command_attempted = true;
      command(9, position, speed_byte);
    }
    const auto deadline = Clock::now() + std::chrono::milliseconds(options_.motion_timeout_ms);
    while (Clock::now() < deadline) {
      if (generation != cancellation_.load()) return false;
      const auto state = read_state();
      if (state.fault != 0 && state.fault != 9)
        throw std::runtime_error("Robotiq motion fault " + std::to_string(state.fault));
      if (state.fault == 0 && state.requested_position == position && state.object_status != 0)
        return state.object_status == 3;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("Robotiq motion timed out");
  } catch (...) {
    if (command_attempted) { try { stop(); } catch (...) {} }
    throw;
  }
}

bool RobotiqGripper::stop() {
  std::lock_guard<std::mutex> lock(command_mutex_);
  ++cancellation_;
  const auto state = read_state();
  // Do not activate an inactive gripper as a side effect of stopping it.
  command(state.ready() ? 1 : 0, 0, 0);
  return true;
}
}  // namespace robo_lab
