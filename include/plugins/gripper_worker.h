#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace robo_lab {

// Blocking device operations may run concurrently so stop() can interrupt move().
class GripperDevice {
 public:
  virtual ~GripperDevice() = default;
  virtual bool move(double width, double speed) = 0;
  virtual bool stop() = 0;
};

class GripperWorker {
 public:
  using Clock = std::chrono::steady_clock;
  struct Options {
    double open_width{0.08};
    double closed_width{0.0};
    double speed{0.1};
    double close_threshold{0.5};
    int retry_count{2};
    std::chrono::milliseconds retry_interval{200};
    double width_tolerance{0.001};
    std::chrono::milliseconds state_max_age{200};
    std::chrono::milliseconds confirmation_wait{200};
  };
  struct Feedback {
    bool valid{false};
    double width{0.0};
    Clock::time_point collected{};
  };
  using Logger = std::function<void(const std::string&)>;

  GripperWorker(std::shared_ptr<GripperDevice> device, Options options, Logger logger);
  ~GripperWorker();
  GripperWorker(const GripperWorker&) = delete;
  GripperWorker& operator=(const GripperWorker&) = delete;

  void start();
  void stop();
  // Session changes invalidate all pending targets and latch cancellation.
  void set_enabled(bool enabled);
  uint64_t session() const;
  bool submit(double close_norm);
  bool submit(double close_norm, uint64_t session);
  void update_state(double width, Clock::time_point collected = Clock::now());
  Feedback feedback() const;

 private:
  void action_loop();
  void interrupt_loop();
  bool cancelled_locked(uint64_t target_seq) const;
  bool reached_locked(double width, Clock::time_point started) const;
  std::string feedback_text_locked() const;
  void log(const std::string& message);

  const std::shared_ptr<GripperDevice> device_;
  const Options options_;
  const Logger logger_;
  std::mutex lifecycle_mutex_;
  std::mutex log_mutex_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::thread action_thread_;
  std::thread interrupt_thread_;
  bool stopping_{true};
  bool enabled_{false};
  bool has_target_{false};
  bool target_closed_{false};
  double target_norm_{0.0};
  uint64_t target_seq_{0};
  uint64_t session_seq_{0};
  uint64_t finished_seq_{0};
  uint64_t action_seq_{0};
  bool move_running_{false};
  bool cancel_active_{false};
  bool stop_in_flight_{false};
  Feedback feedback_;
};

}  // namespace robo_lab
