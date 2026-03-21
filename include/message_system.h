#pragma once

#include "robo_lab.h"

#include <functional>
#include <memory>
#include <string>

namespace robo_lab {

/// Zenoh pub/sub helper built on [zenoh-c](https://github.com/eclipse-zenoh/zenoh-c).
/// Subscribers are invoked from Zenoh threads; keep callbacks short and thread-safe.
class MessageSystem : public RoboLab {
 public:
  MessageSystem();
  ~MessageSystem() override;

  void initialize() override;
  void close() override;

  /// Publish a payload on a key expression (one publisher per key is cached for reuse).
  bool publish(const std::string& keyexpr, const std::string& payload);

  /// Subscribe to a key expression; the callback may run on a Zenoh worker thread.
  bool subscribe(const std::string& keyexpr,
                 std::function<void(const std::string& key, const std::string& payload)> callback);

  [[nodiscard]] bool is_open() const;

  MessageSystem(const MessageSystem&) = delete;
  MessageSystem& operator=(const MessageSystem&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace robo_lab
