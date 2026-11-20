#include "plugins/robotiq_gripper.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <future>
#include <poll.h>
#include <pty.h>
#include <thread>
#include <unistd.h>

using robo_lab::GripperWorker;
using robo_lab::RobotiqGripper;
using namespace std::chrono_literals;

namespace {
uint16_t crc(const std::vector<uint8_t>& data) {
  uint16_t value = 0xffff;
  for (auto byte : data) {
    value ^= byte;
    for (int i = 0; i < 8; ++i) value = (value & 1) ? (value >> 1) ^ 0xa001 : value >> 1;
  }
  return value;
}
bool wait_for(const std::function<bool()>& predicate) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!predicate() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(2ms);
  return predicate();
}

// A Modbus server on a pseudo-terminal exercises the actual serial transport.
// It never opens a hardware device. All commands and CRCs are checked here.
class SimulatedGripper {
 public:
  enum Error { None, BadCrc, WrongSlave, WrongFunction, Exception, Silence, BadAck };
  struct Command { int action, position, speed, force; };
  std::atomic<Error> error{None};
  std::atomic<bool> hold_motion{false}, contact{false}, fragmented{false};
  std::atomic<int> requests{0}, invalid_requests{0};
  std::atomic<uint8_t> activation_fault{0};
  SimulatedGripper() {
    char name[128];
    if (openpty(&master_, &slave_, name, nullptr, nullptr)) throw std::runtime_error("openpty failed");
    path_ = name;
    thread_ = std::thread([this] { serve(); });
  }
  ~SimulatedGripper() {
    done_ = true;
    thread_.join();
    close(master_);
    close(slave_);
  }
  RobotiqGripper::Options options() const {
    RobotiqGripper::Options result;
    result.port = path_;
    result.timeout_ms = 60;
    result.motion_timeout_ms = 200;
    return result;
  }
  void status(uint8_t status, uint8_t fault, uint8_t position = 13) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status; fault_ = fault; position_ = position;
  }
  std::vector<Command> commands() {
    std::lock_guard<std::mutex> lock(mutex_);
    return commands_;
  }
 private:
  void serve() {
    std::vector<uint8_t> incoming;
    while (!done_) {
      pollfd pfd{master_, POLLIN, 0};
      if (poll(&pfd, 1, 10) <= 0 || !(pfd.revents & POLLIN)) continue;
      uint8_t buffer[64];
      const auto n = read(master_, buffer, sizeof(buffer));
      if (n <= 0) continue;
      incoming.insert(incoming.end(), buffer, buffer + n);
      while (incoming.size() >= 2) {
        const std::size_t size = incoming[1] == 0x10 ? 15 : 8;
        if (incoming.size() < size) break;
        std::vector<uint8_t> request(incoming.begin(), incoming.begin() + size);
        incoming.erase(incoming.begin(), incoming.begin() + size);
        respond(request);
      }
    }
  }
  void respond(const std::vector<uint8_t>& request) {
    ++requests;
    const auto size = request.size();
    const std::vector<uint8_t> bytes(request.begin(), request.end() - 2);
    if (request[0] != 9 || crc(bytes) != (request[size - 2] | (request[size - 1] << 8))) {
      ++invalid_requests; return;
    }
    const auto mode = error.load();
    std::vector<uint8_t> reply;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (request[1] == 3 && request[2] == 7 && request[3] == 0xd0 && request[4] == 0 && request[5] == 3) {
        reply = {9, 3, 6, status_, 0, fault_, requested_, position_, 0};
      } else if (request[1] == 0x10 && request[2] == 3 && request[3] == 0xe8 && request[4] == 0 &&
                 request[5] == 3 && request[6] == 6 && request[8] == 0 && request[9] == 0) {
        commands_.push_back({request[7], request[10], request[11], request[12]});
        fault_ = 0;
        if (request[7] == 9) {
          requested_ = request[10];
          status_ = hold_motion ? 0x39 : contact ? 0xb9 : 0xf9;
          if (!hold_motion) position_ = contact ? 100 : requested_;
        } else {
          status_ = request[7] == 1 ? 0x31 : 0;
          if (request[7] == 1) fault_ = activation_fault.load();
        }
        reply = {9, 0x10, 3, 0xe8, 0, 3};
        if (mode == BadAck) reply[3] = 0xe9;
      } else { ++invalid_requests; return; }
    }
    if (mode == Silence) return;
    if (mode == Exception) reply = {9, static_cast<uint8_t>(request[1] | 0x80), 2};
    if (mode == WrongSlave) reply[0] = 10;
    if (mode == WrongFunction) reply[1] = 4;
    const auto checksum = crc(reply);
    reply.push_back(checksum & 0xff); reply.push_back(checksum >> 8);
    if (mode == BadCrc) reply.back() ^= 1;
    if (fragmented) {
      for (auto byte : reply) {
        if (write(master_, &byte, 1) != 1) return;
        std::this_thread::sleep_for(1ms);
      }
    } else { if (write(master_, reply.data(), reply.size()) != static_cast<ssize_t>(reply.size())) return; }
  }
  int master_, slave_;
  std::string path_;
  std::thread thread_;
  std::atomic<bool> done_{false};
  std::mutex mutex_;
  uint8_t status_{0xf9}, fault_{0}, requested_{13}, position_{13};
  std::vector<Command> commands_;
};

TEST(Robotiq, ConnectionAndFeedbackAreReadOnly) {
  SimulatedGripper server;
  {
    RobotiqGripper device(server.options());
    EXPECT_EQ(server.requests, 0);
    const auto state = device.read_state();
    EXPECT_TRUE(state.ready());
    EXPECT_DOUBLE_EQ(state.width, 0.085);
    server.status(0xf9, 0, 230);
    EXPECT_DOUBLE_EQ(device.read_state().width, 0.0);
    EXPECT_TRUE(server.commands().empty());
  }
  EXPECT_TRUE(server.commands().empty());
  EXPECT_EQ(server.invalid_requests, 0);
}

TEST(Robotiq, ParsesFragmentedReplyAndRejectsCorruptOrExceptionalReplies) {
  SimulatedGripper server;
  RobotiqGripper device(server.options());
  server.fragmented = true;
  EXPECT_TRUE(device.read_state().ready());
  server.fragmented = false;
  for (auto error : {SimulatedGripper::BadCrc, SimulatedGripper::WrongSlave,
                     SimulatedGripper::WrongFunction, SimulatedGripper::Exception,
                     SimulatedGripper::Silence}) {
    server.error = error;
    EXPECT_THROW(device.read_state(), std::runtime_error);
  }
  server.error = SimulatedGripper::None;
  EXPECT_TRUE(device.read_state().ready());
}

TEST(Robotiq, SerialOwnershipIsReleasedAfterDestruction) {
  SimulatedGripper server;
  {
    RobotiqGripper device(server.options());
    EXPECT_THROW(RobotiqGripper another(server.options()), std::runtime_error);
  }
  RobotiqGripper reopened(server.options());
  EXPECT_TRUE(reopened.read_state().ready());
}

TEST(Robotiq, OpenAndCloseUseBoundedSpeedForceAndPositionRegisters) {
  SimulatedGripper server;
  RobotiqGripper device(server.options());
  ASSERT_TRUE(device.move(0, 0.03));
  ASSERT_TRUE(device.move(0.085, 0.03));
  const auto commands = server.commands();
  ASSERT_EQ(commands.size(), 2);
  EXPECT_EQ(commands[0].action, 9);
  EXPECT_EQ(commands[0].position, 230);
  EXPECT_EQ(commands[1].position, 13);
  EXPECT_EQ(commands[0].speed, 49);
  EXPECT_EQ(commands[0].force, 32);
  EXPECT_EQ(server.invalid_requests, 0);
}

TEST(Robotiq, InvalidInputAndInactiveDeviceNeverStartMotion) {
  SimulatedGripper server;
  RobotiqGripper device(server.options());
  EXPECT_THROW(device.move(-1, 0.03), std::invalid_argument);
  EXPECT_THROW(device.move(0, 0), std::invalid_argument);
  server.status(0, 0);
  EXPECT_THROW(device.move(0, 0.03), std::runtime_error);
  EXPECT_TRUE(server.commands().empty());
}

TEST(Robotiq, ActivationIsExplicitAndUsesResetThenActivate) {
  SimulatedGripper server;
  server.status(0, 0);
  RobotiqGripper device(server.options());
  EXPECT_FALSE(device.read_state().ready());
  EXPECT_TRUE(server.commands().empty());
  device.activate();
  const auto commands = server.commands();
  ASSERT_EQ(commands.size(), 2);
  EXPECT_EQ(commands[0].action, 0);
  EXPECT_EQ(commands[1].action, 1);
  EXPECT_TRUE(device.read_state().ready());
}

TEST(Robotiq, FailedActivationResetsAndStopCannotActivateAnUnreadyGripper) {
  SimulatedGripper server;
  RobotiqGripper device(server.options());
  server.activation_fault = 0x0a;
  EXPECT_THROW(device.activate(), std::runtime_error);
  ASSERT_EQ(server.commands().size(), 3);
  EXPECT_EQ(server.commands().back().action, 0);
  server.status(0x11, 0);
  EXPECT_TRUE(device.stop());
  EXPECT_EQ(server.commands().back().action, 0);
}

TEST(Robotiq, StopInterruptsMoveAndTimeoutAttemptsStop) {
  SimulatedGripper server;
  server.hold_motion = true;
  RobotiqGripper device(server.options());
  auto move = std::async(std::launch::async, [&] { return device.move(0, 0.03); });
  ASSERT_TRUE(wait_for([&] { return !server.commands().empty(); }));
  EXPECT_TRUE(device.stop());
  EXPECT_FALSE(move.get());
  EXPECT_EQ(server.commands().back().action, 1);
  EXPECT_THROW(device.move(0, 0.03), std::runtime_error);
  EXPECT_EQ(server.commands().back().action, 1);
}

TEST(Robotiq, WatchdogFaultCanResumeButOtherFaultsAndContactCannotReportReached) {
  SimulatedGripper server;
  RobotiqGripper device(server.options());
  server.status(0x31, 9);
  EXPECT_TRUE(device.move(0, 0.03));
  server.status(0xf9, 0x0a);
  EXPECT_THROW(device.move(0.085, 0.03), std::runtime_error);
  server.status(0xf9, 0);
  server.contact = true;
  EXPECT_FALSE(device.move(0, 0.03));
}

TEST(Robotiq, BadCommandAcknowledgementAttemptsStop) {
  SimulatedGripper server;
  RobotiqGripper device(server.options());
  server.error = SimulatedGripper::BadAck;
  EXPECT_THROW(device.move(0, 0.03), std::runtime_error);
  ASSERT_EQ(server.commands().size(), 2);
  EXPECT_EQ(server.commands().back().action, 1);
}

TEST(Robotiq, ExistingWorkerControlsOpenCloseAndCancelsOnStandby) {
  SimulatedGripper server;
  auto device = std::make_shared<RobotiqGripper>(server.options());
  GripperWorker::Options options;
  options.open_width = 0.085; options.speed = 0.03; options.retry_count = 0;
  std::atomic<int> reached{0}, cancelled{0};
  GripperWorker worker(device, options, [&](const std::string& line) {
    if (line.find("outcome=reached") != std::string::npos) ++reached;
    if (line.find("outcome=cancelled") != std::string::npos) ++cancelled;
  });
  worker.start();
  std::atomic<bool> finish{false};
  std::thread feedback([&] {
    while (!finish) { worker.update_state(device->read_state().width); std::this_thread::sleep_for(5ms); }
  });
  EXPECT_FALSE(worker.submit(1));
  worker.set_enabled(true);
  EXPECT_TRUE(worker.submit(1));
  EXPECT_TRUE(wait_for([&] { return reached == 1; }));
  EXPECT_TRUE(worker.submit(0));
  EXPECT_TRUE(wait_for([&] { return reached == 2; }));
  server.hold_motion = true;
  EXPECT_TRUE(worker.submit(1));
  EXPECT_TRUE(wait_for([&] { return server.commands().size() >= 3; }));
  worker.set_enabled(false);
  EXPECT_TRUE(wait_for([&] { return cancelled == 1; }));
  worker.stop();
  finish = true; feedback.join();
  EXPECT_EQ(server.commands().back().action, 1);
  EXPECT_EQ(server.invalid_requests, 0);
}
}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
