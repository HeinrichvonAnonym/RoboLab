#include "plugins/gripper_worker.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

using robo_lab::GripperDevice;
using robo_lab::GripperWorker;
using namespace std::chrono_literals;

namespace {

bool wait_until(const std::function<bool()>& predicate, std::chrono::milliseconds timeout = 2000ms) {
  const auto deadline = GripperWorker::Clock::now() + timeout;
  while (!predicate() && GripperWorker::Clock::now() < deadline) std::this_thread::sleep_for(1ms);
  return predicate();
}

class Gate {
 public:
  bool wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, 3s, [&] { return open_; });
  }
  void open() {
    std::lock_guard<std::mutex> lock(mutex_);
    open_ = true;
    cv_.notify_all();
  }
 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool open_{false};
};

class FakeDevice : public GripperDevice {
 public:
  std::atomic<int> moves{0}, stops{0}, active_moves{0};
  std::function<bool(double, double)> on_move;
  std::function<bool()> on_stop;

  bool move(double width, double speed) override {
    ++moves;
    ++active_moves;
    struct Guard {
      std::atomic<int>& count;
      ~Guard() { --count; }
    } guard{active_moves};
    return on_move ? on_move(width, speed) : true;
  }
  bool stop() override {
    ++stops;
    return on_stop ? on_stop() : true;
  }
};

struct Harness {
  std::shared_ptr<FakeDevice> device = std::make_shared<FakeDevice>();
  std::mutex log_mutex;
  std::vector<std::string> logs;
  std::unique_ptr<GripperWorker> worker;

  void start(GripperWorker::Options options = {}) {
    worker = std::make_unique<GripperWorker>(device, options, [this](const std::string& message) {
      std::lock_guard<std::mutex> lock(log_mutex);
      logs.push_back(message);
    });
    worker->start();
    worker->set_enabled(true);
  }
  int count(const std::string& text) {
    std::lock_guard<std::mutex> lock(log_mutex);
    int result = 0;
    for (const auto& line : logs) result += line.find(text) != std::string::npos;
    return result;
  }
  ~Harness() { if (worker) worker->stop(); }
};

GripperWorker::Options fast_options() {
  GripperWorker::Options options;
  options.retry_interval = 20ms;
  options.confirmation_wait = 20ms;
  return options;
}

TEST(GripperWorker, RequiresEnabledSessionAndFreshFeedback) {
  Harness h;
  h.device->on_move = [&](double width, double) { h.worker->update_state(width); return true; };
  h.start();
  h.worker->set_enabled(false);
  EXPECT_FALSE(h.worker->submit(1));
  h.worker->set_enabled(true);
  std::this_thread::sleep_for(30ms);
  EXPECT_EQ(h.device->moves, 0);
  EXPECT_TRUE(h.worker->submit(1));
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
  for (int i = 0; i < 100; ++i) EXPECT_TRUE(h.worker->submit(1));
  std::this_thread::sleep_for(30ms);
  EXPECT_EQ(h.device->moves, 1);
  EXPECT_EQ(h.count("feedback_age_ms=unknown"), 0);
  EXPECT_TRUE(h.worker->submit(0));
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 2; }));
  EXPECT_DOUBLE_EQ(h.worker->feedback().width, 0.08);
}

TEST(GripperWorker, LateStopCannotCancelNextMove) {
  Gate finish_first, finish_stop;
  std::atomic<bool> first_moving{false}, stop_entered{false}, stop_returned{false};
  Harness h;
  h.device->on_move = [&](double width, double) {
    if (h.device->moves == 1) {
      first_moving = true;
      EXPECT_TRUE(finish_first.wait());
    } else {
      EXPECT_TRUE(stop_returned.load());
      EXPECT_EQ(h.device->active_moves, 1);
    }
    h.worker->update_state(width);
    return true;
  };
  h.device->on_stop = [&] {
    stop_entered = true;
    EXPECT_TRUE(finish_stop.wait());
    stop_returned = true;
    return true;
  };
  h.start();
  h.worker->submit(1);
  EXPECT_TRUE(wait_until([&] { return first_moving.load(); }));
  h.worker->submit(0);
  EXPECT_TRUE(wait_until([&] { return stop_entered.load(); }));
  // Let move return naturally while its claimed stop is still delayed.
  finish_first.open();
  EXPECT_TRUE(wait_until([&] { return h.device->active_moves == 0; }));
  std::this_thread::sleep_for(40ms);
  EXPECT_EQ(h.device->moves, 1);
  finish_stop.open();
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
  EXPECT_EQ(h.device->moves, 2);
  EXPECT_EQ(h.count("outcome=cancelled"), 1);
  EXPECT_DOUBLE_EQ(h.worker->feedback().width, 0.08);
}

TEST(GripperWorker, StopBeforeSdkLaunchDoesNotConsumeCancellation) {
  Gate enter_sdk;
  std::atomic<bool> announced{false}, launched{false}, interrupted{false};
  auto device = std::make_shared<FakeDevice>();
  std::unique_ptr<GripperWorker> worker;
  std::atomic<int> reached{0};
  device->on_move = [&](double width, double) {
    if (device->moves == 1) {
      launched = true;
      EXPECT_TRUE(wait_until([&] { return interrupted.load(); }));
      return false;
    }
    worker->update_state(width);
    return true;
  };
  device->on_stop = [&] {
    if (launched) interrupted = true;
    return true;
  };
  worker = std::make_unique<GripperWorker>(device, fast_options(), [&](const std::string& line) {
    if (line.find("event=move") != std::string::npos && !announced.exchange(true)) {
      EXPECT_TRUE(enter_sdk.wait());
    }
    if (line.find("outcome=reached") != std::string::npos) ++reached;
  });
  worker->start();
  worker->set_enabled(true);
  worker->submit(1);
  EXPECT_TRUE(wait_until([&] { return announced.load(); }));
  worker->submit(0);
  EXPECT_TRUE(wait_until([&] { return device->stops >= 1; }));
  EXPECT_FALSE(launched);
  enter_sdk.open();
  EXPECT_TRUE(wait_until([&] { return reached == 1; }));
  EXPECT_GE(device->stops, 2);
  EXPECT_EQ(device->moves, 2);
  worker->stop();
}

TEST(GripperWorker, RepeatedStopTimerCannotCarryOverToNextAction) {
  Gate finish_first, finish_second;
  std::atomic<bool> first_entered{false}, second_entered{false};
  std::atomic<int> stops_on_new_action{0};
  Harness h;
  h.device->on_move = [&](double width, double) {
    if (h.device->moves == 1) {
      first_entered = true;
      EXPECT_TRUE(finish_first.wait());
    } else {
      second_entered = true;
      EXPECT_TRUE(finish_second.wait());
    }
    h.worker->update_state(width);
    return true;
  };
  h.device->on_stop = [&] {
    if (h.device->moves >= 2) ++stops_on_new_action;
    // Keep the old move pending, so cancellation enters its 20 ms repeat timer.
    return false;
  };
  h.start();
  h.worker->submit(1);
  EXPECT_TRUE(wait_until([&] { return first_entered.load(); }));
  h.worker->submit(0);
  EXPECT_TRUE(wait_until([&] { return h.device->stops >= 1; }));
  std::this_thread::sleep_for(5ms);
  finish_first.open();
  EXPECT_TRUE(wait_until([&] { return second_entered.load(); }));
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(stops_on_new_action, 0);
  finish_second.open();
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
}

TEST(GripperWorker, FalseResultsAndExceptionsRetryWithoutResettingBudget) {
  for (bool exception : {false, true}) {
    Harness h;
    auto options = fast_options();
    h.device->on_move = [&](double width, double) {
      // Even an at-target sample cannot turn an unsuccessful SDK result into success.
      h.worker->update_state(width);
      if (exception) throw std::runtime_error("simulated failure");
      return false;
    };
    h.start(options);
    h.worker->submit(1);
    EXPECT_TRUE(wait_until([&] { return h.count("outcome=exhausted") == 1; }));
    for (int i = 0; i < 100; ++i) h.worker->submit(0.7);
    std::this_thread::sleep_for(80ms);
    EXPECT_EQ(h.device->moves, 3);
    EXPECT_EQ(h.count("outcome=reached"), 0);
    EXPECT_EQ(h.count("outcome=failed"), 2);
    EXPECT_EQ(h.count("sdk_success=0"), 3);
    h.worker->submit(0);
    EXPECT_TRUE(wait_until([&] { return h.count("outcome=exhausted") == 2; }));
    EXPECT_EQ(h.device->moves, 6);
  }
}

TEST(GripperWorker, TransientFailureRecoversOnRetry) {
  Harness h;
  h.device->on_move = [&](double width, double) {
    if (h.device->moves == 1) return false;
    h.worker->update_state(width);
    return true;
  };
  h.start(fast_options());
  h.worker->submit(1);
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
  EXPECT_EQ(h.device->moves, 2);
  EXPECT_EQ(h.count("outcome=failed"), 1);
}

TEST(GripperWorker, TargetChangeInterruptsRetryWait) {
  Harness h;
  auto options = fast_options();
  options.retry_interval = 1000ms;
  h.device->on_move = [&](double width, double) {
    if (width == 0.0) return false;
    h.worker->update_state(width);
    return true;
  };
  h.start(options);
  h.worker->submit(1);
  EXPECT_TRUE(wait_until([&] { return h.count("retry_in_ms=1000") == 1; }));
  h.worker->submit(0);
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }, 300ms));
  EXPECT_EQ(h.device->moves, 2);
}

TEST(GripperWorker, MissingExpiredOrWrongFeedbackCannotConfirmSuccess) {
  for (int mode : {0, 1, 2}) {
    Harness h;
    auto options = fast_options();
    options.state_max_age = 10ms;
    h.device->on_move = [&](double width, double) {
      if (mode == 1) h.worker->update_state(width, GripperWorker::Clock::now() - 1s);
      if (mode == 2) h.worker->update_state(0.05);
      return true;
    };
    h.start(options);
    h.worker->submit(1);
    EXPECT_TRUE(wait_until([&] { return h.count("outcome=exhausted") == 1; }));
    EXPECT_EQ(h.device->moves, 3);
    EXPECT_EQ(h.count("outcome=reached"), 0);
  }
}

TEST(GripperWorker, DelayedFeedbackWithinConfirmationWindowCanComplete) {
  Harness h;
  h.start();
  h.worker->submit(1);
  EXPECT_TRUE(wait_until([&] { return h.device->moves == 1; }));
  std::this_thread::sleep_for(20ms);
  h.worker->update_state(0.0005);
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
  EXPECT_EQ(h.device->moves, 1);
}

TEST(GripperWorker, SessionExitCancelsAndReentryRequiresNewCommand) {
  std::atomic<bool> cancelled{false};
  Harness h;
  h.device->on_move = [&](double width, double) {
    if (h.device->moves == 1) {
      EXPECT_TRUE(wait_until([&] { return cancelled.load(); }));
      return false;
    }
    h.worker->update_state(width);
    return true;
  };
  h.device->on_stop = [&] { cancelled = true; return true; };
  h.start();
  h.worker->submit(1);
  EXPECT_TRUE(wait_until([&] { return h.device->active_moves == 1; }));
  h.worker->set_enabled(false);
  EXPECT_FALSE(h.worker->submit(0));
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=cancelled") == 1; }));
  h.worker->set_enabled(true);
  std::this_thread::sleep_for(30ms);
  EXPECT_EQ(h.device->moves, 1);
  h.worker->submit(0);
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
}

TEST(GripperWorker, ConcurrentStopIsIdempotentAndDrainsAllDeviceCalls) {
  std::atomic<bool> cancelled{false};
  Harness h;
  h.device->on_move = [&](double, double) {
    EXPECT_TRUE(wait_until([&] { return cancelled.load(); }));
    return false;
  };
  h.device->on_stop = [&] { cancelled = true; return true; };
  h.start();
  h.worker->submit(1);
  EXPECT_TRUE(wait_until([&] { return h.device->active_moves == 1; }));
  std::vector<std::thread> stoppers;
  for (int i = 0; i < 4; ++i) stoppers.emplace_back([&] { h.worker->stop(); });
  for (auto& thread : stoppers) thread.join();
  h.worker->stop();
  EXPECT_EQ(h.device->active_moves, 0);
  const int stops = h.device->stops;
  std::this_thread::sleep_for(30ms);
  EXPECT_EQ(h.device->stops, stops);
  EXPECT_FALSE(h.worker->submit(0));
  h.device->on_move = [&](double width, double) { h.worker->update_state(width); return true; };
  h.worker->start();
  h.worker->set_enabled(true);
  h.worker->submit(0);
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
}

TEST(GripperWorker, OldSessionCommandIsRejectedAfterReentry) {
  Harness h;
  h.device->on_move = [&](double width, double) { h.worker->update_state(width); return true; };
  h.start();
  const uint64_t old_session = h.worker->session();
  h.worker->set_enabled(false);
  h.worker->set_enabled(true);
  EXPECT_NE(h.worker->session(), old_session);
  EXPECT_FALSE(h.worker->submit(1, old_session));
  std::this_thread::sleep_for(30ms);
  EXPECT_EQ(h.device->moves, 0);
  EXPECT_TRUE(h.worker->submit(0, h.worker->session()));
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
}

TEST(GripperWorker, RapidCommandsKeepOnlyLatestTarget) {
  Gate finish_first;
  std::atomic<bool> entered{false};
  Harness h;
  h.device->on_move = [&](double width, double) {
    if (h.device->moves == 1) {
      entered = true;
      EXPECT_TRUE(finish_first.wait());
      return false;
    }
    h.worker->update_state(width);
    return true;
  };
  h.start();
  h.worker->submit(1);
  EXPECT_TRUE(wait_until([&] { return entered.load(); }));
  for (int i = 0; i < 1000; ++i) h.worker->submit(i % 2);
  h.worker->submit(0);
  finish_first.open();
  EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
  EXPECT_EQ(h.device->moves, 2);
  EXPECT_DOUBLE_EQ(h.worker->feedback().width, 0.08);
}

TEST(GripperWorker, FailedOrThrowingStopsKeepCancellationPending) {
  for (bool exception : {false, true}) {
    std::atomic<bool> cancelled{false};
    Harness h;
    h.device->on_move = [&](double width, double) {
      if (h.device->moves == 1) {
        EXPECT_TRUE(wait_until([&] { return cancelled.load(); }));
        return false;
      }
      h.worker->update_state(width);
      return true;
    };
    h.device->on_stop = [&] {
      if (h.device->stops == 1) {
        if (exception) throw std::runtime_error("stop failed");
        return false;
      }
      cancelled = true;
      return true;
    };
    h.start();
    h.worker->submit(1);
    EXPECT_TRUE(wait_until([&] { return h.device->active_moves == 1; }));
    h.worker->submit(0);
    EXPECT_TRUE(wait_until([&] { return h.count("outcome=reached") == 1; }));
    EXPECT_GE(h.device->stops, 2);
    EXPECT_EQ(h.device->moves, 2);
    EXPECT_EQ(h.count("sdk_success=0"), 2); // failed stop and cancelled move
  }
}

TEST(GripperWorker, RejectsInvalidCommandsFeedbackAndConfiguration) {
  Harness h;
  h.start();
  EXPECT_FALSE(h.worker->submit(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(h.worker->submit(std::numeric_limits<double>::infinity()));
  h.worker->update_state(std::numeric_limits<double>::quiet_NaN());
  h.worker->update_state(-1);
  h.worker->update_state(0.08, GripperWorker::Clock::now() + 1s);
  EXPECT_FALSE(h.worker->feedback().valid);
  h.worker->update_state(0.04);
  h.worker->update_state(0.08, GripperWorker::Clock::now() - 1s);
  EXPECT_DOUBLE_EQ(h.worker->feedback().width, 0.04);
  for (int mode = 0; mode < 7; ++mode) {
    auto options = fast_options();
    if (mode == 0) options.retry_count = -1;
    if (mode == 1) options.retry_interval = 0ms;
    if (mode == 2) options.width_tolerance = std::numeric_limits<double>::quiet_NaN();
    if (mode == 3) options.state_max_age = 0ms;
    if (mode == 4) options.confirmation_wait = 0ms;
    if (mode == 5) options.speed = 0;
    if (mode == 6) options.closed_width = 0.09;
    EXPECT_THROW(GripperWorker(h.device, options, {}), std::invalid_argument);
  }
}

// Real-time replay uses a simulated actuator and a 100 Hz state stream. It
// never instantiates libfranka, connects Zenoh, or addresses a physical robot.
class SimulatedActuator : public GripperDevice {
 public:
  std::atomic<int> moves{0}, stops{0};
  double width() {
    std::lock_guard<std::mutex> lock(mutex_);
    return width_;
  }
  bool move(double target, double speed) override {
    ++moves;
    std::unique_lock<std::mutex> lock(mutex_);
    interrupted_ = false;
    running_ = true;
    auto previous = GripperWorker::Clock::now();
    while (std::abs(width_ - target) > 1e-6 && !interrupted_) {
      cv_.wait_for(lock, 5ms, [&] { return interrupted_; });
      const auto now = GripperWorker::Clock::now();
      const double step = speed * std::chrono::duration<double>(now - previous).count();
      previous = now;
      width_ += std::copysign(std::min(step, std::abs(target - width_)), target - width_);
    }
    running_ = false;
    return !interrupted_;
  }
  bool stop() override {
    ++stops;
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) interrupted_ = true;
    cv_.notify_all();
    return true;
  }
 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  double width_{0.08};
  bool running_{false}, interrupted_{false};
};

int replay(const char* filename) {
  std::ifstream input(filename);
  std::vector<std::pair<double, double>> samples;
  double seconds, norm;
  while (input >> seconds >> norm) samples.emplace_back(seconds, norm);
  if (samples.empty()) throw std::runtime_error("No replay samples");
  auto device = std::make_shared<SimulatedActuator>();
  std::atomic<int> reached{0}, failed{0};
  GripperWorker worker(device, {}, [&](const std::string& line) {
    if (line.find("outcome=reached") != std::string::npos) ++reached;
    if (line.find("outcome=exhausted") != std::string::npos) ++failed;
  });
  worker.start();
  worker.set_enabled(true);
  std::atomic<bool> sampling{true};
  std::thread state_thread([&] {
    while (sampling) {
      worker.update_state(device->width());
      std::this_thread::sleep_for(10ms);
    }
  });
  const auto started = GripperWorker::Clock::now();
  double next_report = 10;
  for (const auto& sample : samples) {
    std::this_thread::sleep_until(started + std::chrono::duration_cast<GripperWorker::Clock::duration>(
        std::chrono::duration<double>(sample.first - samples.front().first)));
    worker.submit(sample.second);
    if (sample.first - samples.front().first >= next_report) {
      std::cout << "replay t=" << next_report << "s width=" << device->width() << '\n' << std::flush;
      next_report += 10;
    }
  }
  const double target = samples.back().second >= 0.5 ? 0.0 : 0.08;
  const bool followed = wait_until([&] { return std::abs(device->width() - target) <= 0.001; }, 3000ms);
  // Also wait for the SDK operation and feedback confirmation to finish.
  std::this_thread::sleep_for(50ms);
  worker.stop();
  sampling = false;
  state_thread.join();
  std::cout << "replay samples=" << samples.size() << " moves=" << device->moves
            << " stops=" << device->stops << " reached=" << reached << " exhausted=" << failed
            << " final_width=" << device->width() << " target_width=" << target << '\n';
  return followed && failed == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--replay") return replay(argv[2]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
