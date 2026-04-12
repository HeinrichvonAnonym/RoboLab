#pragma once

#include "plugin_interface.h"
#include "message_system.h"

#include <atomic>
#include <chrono>
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

  /// Generate a timestamped .h5 path inside data_dir_.
  std::string make_h5_path() const;

  /// Writes to a temporary file, then renames to `path` so readers never see a half-written file.
  void write_hdf5(const std::string& path, const std::deque<RecordedMessage>& messages);

  std::string config_path_;
  std::vector<std::string> topics_;
  std::unordered_map<std::string, std::string> topic_proto_;
  float record_frequency_{30.0f};
  float save_interval_s_{0.0f};
  std::string data_dir_{"data"};

  std::atomic<bool> stop_{false};
  std::atomic<bool> running_{false};
  bool accepting_{true};

  std::unique_ptr<MessageSystem> message_system_;

  std::mutex buffer_mutex_;
  std::deque<RecordedMessage> buffer_;
  std::unordered_map<std::string, int64_t> last_record_ns_by_topic_;
};

}  // namespace robo_lab
