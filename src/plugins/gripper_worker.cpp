#include "plugins/gripper_worker.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace robo_lab {

GripperWorker::GripperWorker(std::shared_ptr<GripperDevice> device, Options options, Logger logger)
    : device_(std::move(device)), options_(options), logger_(std::move(logger)) {
  if (!device_ || !std::isfinite(options_.open_width) || options_.open_width <= 0.0 ||
      !std::isfinite(options_.closed_width) || options_.closed_width < 0.0 ||
      options_.closed_width > options_.open_width || !std::isfinite(options_.speed) ||
      options_.speed <= 0.0 || !std::isfinite(options_.close_threshold) ||
      options_.close_threshold < 0.0 || options_.close_threshold > 1.0 ||
      options_.retry_count < 0 || options_.retry_interval.count() <= 0 ||
      !std::isfinite(options_.width_tolerance) || options_.width_tolerance <= 0.0 ||
      options_.state_max_age.count() <= 0 || options_.confirmation_wait.count() <= 0) {
    throw std::invalid_argument("Invalid gripper worker configuration");
  }
}

GripperWorker::~GripperWorker() { stop(); }

void GripperWorker::start() {
  std::lock_guard<std::mutex> lifecycle(lifecycle_mutex_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stopping_) return;
    stopping_ = false;
    ++session_seq_;
    enabled_ = false;
    has_target_ = false;
    feedback_ = Feedback{};
  }
  try {
    action_thread_ = std::thread(&GripperWorker::action_loop, this);
    interrupt_thread_ = std::thread(&GripperWorker::interrupt_loop, this);
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      cancel_active_ = true;
    }
    cv_.notify_all();
    if (action_thread_.joinable()) action_thread_.join();
    throw;
  }
}

void GripperWorker::stop() {
  // Both the plugin's run thread and framework stop thread may call this.
  std::lock_guard<std::mutex> lifecycle(lifecycle_mutex_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    enabled_ = false;
    has_target_ = false;
    cancel_active_ = true;
    ++target_seq_;
    ++session_seq_;
  }
  cv_.notify_all();
  if (action_thread_.joinable()) action_thread_.join();
  if (interrupt_thread_.joinable()) interrupt_thread_.join();
}

void GripperWorker::set_enabled(bool enabled) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || enabled_ == enabled) return;
    enabled_ = enabled;
    has_target_ = false;
    cancel_active_ = true;
    ++target_seq_;
    ++session_seq_;
  }
  cv_.notify_all();
}

bool GripperWorker::submit(double close_norm) {
  return submit(close_norm, session());
}

uint64_t GripperWorker::session() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return session_seq_;
}

bool GripperWorker::submit(double close_norm, uint64_t session) {
  if (!std::isfinite(close_norm)) return false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || !enabled_ || session != session_seq_) return false;
    close_norm = std::clamp(close_norm, 0.0, 1.0);
    const bool closed = close_norm >= options_.close_threshold;
    if (!has_target_ || target_closed_ != closed) {
      ++target_seq_;
      if (move_running_) cancel_active_ = true;
    }
    has_target_ = true;
    target_closed_ = closed;
    target_norm_ = close_norm;
  }
  cv_.notify_all();
  return true;
}

void GripperWorker::update_state(double width, Clock::time_point collected) {
  if (!std::isfinite(width) || width < 0.0 || collected > Clock::now()) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (feedback_.valid && collected < feedback_.collected) return;
    feedback_ = Feedback{true, width, collected};
  }
  cv_.notify_all();
}

GripperWorker::Feedback GripperWorker::feedback() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return feedback_;
}

bool GripperWorker::cancelled_locked(uint64_t seq) const {
  return stopping_ || !enabled_ || !has_target_ || target_seq_ != seq || cancel_active_;
}

bool GripperWorker::reached_locked(double width, Clock::time_point started) const {
  return feedback_.valid && feedback_.collected >= started &&
         Clock::now() - feedback_.collected <= options_.state_max_age &&
         std::abs(feedback_.width - width) <= options_.width_tolerance;
}

std::string GripperWorker::feedback_text_locked() const {
  if (!feedback_.valid) return " actual_width=unknown feedback_age_ms=unknown";
  std::ostringstream text;
  text << " actual_width=" << feedback_.width << " feedback_age_ms="
       << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - feedback_.collected).count();
  return text.str();
}

void GripperWorker::log(const std::string& message) {
  if (logger_) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    logger_(message);
  }
}

void GripperWorker::action_loop() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    cv_.wait(lock, [&] {
      return stopping_ || (enabled_ && has_target_ && target_seq_ != finished_seq_);
    });
    if (stopping_) return;
    const uint64_t target_seq = target_seq_;
    const double width = target_closed_ ? options_.closed_width : options_.open_width;

    for (int retry = 0; retry <= options_.retry_count; ++retry) {
      if (stopping_ || !enabled_ || !has_target_ || target_seq_ != target_seq) break;
      // An old stop must finish before this action is announced or launched.
      cv_.wait(lock, [&] { return !stop_in_flight_; });
      if (stopping_ || !enabled_ || !has_target_ || target_seq_ != target_seq) break;
      move_running_ = true;
      cancel_active_ = false;
      const uint64_t action_seq = ++action_seq_;
      const auto started = Clock::now();
      std::ostringstream start_message;
      start_message << "event=move target_seq=" << target_seq << " action_seq=" << action_seq
                    << " retry=" << retry << " gello=" << target_norm_ << " width=" << width;
      lock.unlock();
      log(start_message.str());
      cv_.notify_all();

      bool sdk_success = false;
      std::string error;
      try {
        sdk_success = device_->move(width, options_.speed);
      } catch (const std::exception& e) {
        error = e.what();
      } catch (...) {
        error = "unknown exception";
      }

      lock.lock();
      move_running_ = false;
      cv_.notify_all();
      // The interrupt thread may already have claimed a stop for this action.
      // Even a naturally completed move must drain that stop before proceeding.
      cv_.wait(lock, [&] { return !stop_in_flight_; });
      if (!cancelled_locked(target_seq) && sdk_success) {
        cv_.wait_until(lock, Clock::now() + options_.confirmation_wait, [&] {
          return cancelled_locked(target_seq) || reached_locked(width, started);
        });
      }
      const bool cancelled = cancelled_locked(target_seq);
      const bool reached = !cancelled && sdk_success && reached_locked(width, started);
      const bool exhausted = !cancelled && !reached && retry == options_.retry_count;
      if (reached || exhausted) finished_seq_ = target_seq;
      std::ostringstream result_message;
      result_message << "event=result target_seq=" << target_seq << " action_seq=" << action_seq
                     << " retry=" << retry << " sdk_success=" << sdk_success << " outcome="
                     << (cancelled ? "cancelled" : reached ? "reached" : exhausted ? "exhausted" : "failed")
                     << feedback_text_locked();
      if (!error.empty()) result_message << " error=" << error;
      if (!cancelled && !reached && !exhausted) {
        result_message << " retry_in_ms=" << options_.retry_interval.count();
      }
      lock.unlock();
      log(result_message.str());
      lock.lock();
      if (cancelled || reached || exhausted) break;
      cv_.wait_until(lock, Clock::now() + options_.retry_interval, [&] {
        return cancelled_locked(target_seq);
      });
      if (cancelled_locked(target_seq)) break;
    }
  }
}

void GripperWorker::interrupt_loop() {
  std::unique_lock<std::mutex> lock(mutex_);
  auto next_stop = Clock::time_point::min();
  auto last_log = Clock::time_point::min();
  uint64_t last_action = 0;
  bool last_success = false;
  std::string last_error;
  while (true) {
    cv_.wait(lock, [&] {
      return stopping_ || (move_running_ && cancel_active_);
    });
    if (stopping_ && !move_running_) return;
    if (!move_running_ || !cancel_active_) continue;
    const uint64_t action_seq = action_seq_;
    if (action_seq != last_action) next_stop = Clock::time_point::min();
    if (Clock::now() < next_stop) {
      cv_.wait_until(lock, next_stop, [&] {
        return !move_running_ || action_seq_ != action_seq || !cancel_active_;
      });
    }
    // A different action may have started while the repeated-stop timer was
    // waiting. Revalidate its identity and cancellation before claiming stop.
    if (!move_running_ || action_seq_ != action_seq || !cancel_active_) continue;
    // Claim under the same lock used when the action thread launches a move.
    stop_in_flight_ = true;
    const uint64_t target_seq = target_seq_;
    lock.unlock();
    bool sdk_success = false;
    std::string error;
    try {
      sdk_success = device_->stop();
    } catch (const std::exception& e) {
      error = e.what();
    } catch (...) {
      error = "unknown exception";
    }
    lock.lock();
    stop_in_flight_ = false;
    next_stop = Clock::now() + std::chrono::milliseconds(20);
    // A stop sent before move() entered the SDK cannot consume cancellation.
    // Keep checking every 20 ms until that move has actually returned.
    cv_.notify_all();
    if (action_seq != last_action || sdk_success != last_success || error != last_error ||
        Clock::now() - last_log >= std::chrono::seconds(1)) {
      std::ostringstream message;
      message << "event=stop action_seq=" << action_seq << " target_seq=" << target_seq
              << " sdk_success=" << sdk_success << feedback_text_locked();
      if (!error.empty()) message << " error=" << error;
      last_log = Clock::now();
      lock.unlock();
      log(message.str());
      lock.lock();
    }
    last_action = action_seq;
    last_success = sdk_success;
    last_error = error;
  }
}

}  // namespace robo_lab
