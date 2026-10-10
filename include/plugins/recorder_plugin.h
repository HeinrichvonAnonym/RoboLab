#pragma once

#include "plugin_interface.h"
#include "message_system.h"

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace robo_lab {

class RecorderPlugin : public Plugin {
 public:
  RecorderPlugin() = default;
  ~RecorderPlugin() override = default;

  bool initialize(const std::string& config_path) override;
  void run() override;
  void stop() override;

 private:
  struct RecordedMessage {
    std::string topic;
    std::string payload;
    int64_t timestamp_ns;
  };

  void subscribe_callback(const std::string& key, const std::string& payload);
  void next_record_callback(const std::string& key, const std::string& payload);
  void control_state_callback(const std::string& key, const std::string& payload);

  /// Generate a timestamped .h5 path inside data_dir_.
  std::string make_h5_path() const;

  /// Append messages to the current HDF5 recording file, creating it if needed.
  void append_hdf5(const std::deque<RecordedMessage>& messages);
  void finish_recording();
  void flush_anchor_if_requested();
  void flush_all();

  std::string config_path_;
  std::vector<std::string> topics_;
  std::unordered_map<std::string, std::string> topic_proto_;
  float record_frequency_{30.0f};
  std::string anchor_topic_{"franka/state"};
  size_t flush_anchor_count_{30};
  size_t flush_buffer_bytes_{256 * 1024 * 1024};
  std::string next_record_topic_{"next_record"};
  std::string state_machine_topic_{"franka/state_machine"};
  std::string recording_state_{"inference"};
  std::string data_dir_{"data"};

  std::atomic<bool> stop_{false};
  std::atomic<bool> running_{false};
  bool accepting_{false};
  bool recording_active_{false};
  size_t anchor_count_since_flush_{0};
  bool anchor_flush_requested_{false};
  int64_t anchor_flush_boundary_ns_{0};
  std::string current_h5_path_;

  std::unique_ptr<MessageSystem> message_system_;

  std::mutex buffer_mutex_;
  std::mutex file_mutex_;
  std::mutex flush_mutex_;
  std::mutex control_mutex_;
  std::deque<RecordedMessage> buffer_;
  size_t buffer_bytes_{0};
  std::unordered_map<std::string, int64_t> last_record_ns_by_topic_;
};

}  // namespace robo_lab
