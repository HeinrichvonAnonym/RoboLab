#pragma once

#include "plugin_interface.h"

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace robo_lab {

/// Loads bringup YAML, dlopens plugin libraries, drives initialize / run / stop.
///
/// Bringup format (subset of YAML): a top-level `plugins:` list; each entry has `name`, `config`, and
/// `lib` (paths may be absolute or relative). Relative paths are tried against the current working
/// directory, the bringup file directory, and — if the bringup file lives in a directory named
/// `config/` — the parent of that directory (repository root), so entries like
/// `./build/plugins/libfranka_plugin.so` work when you run the binary from the repo root.
class Framework {
 public:
  Framework();
  ~Framework();

  Framework(const Framework&) = delete;
  Framework& operator=(const Framework&) = delete;

  /// Parse bringup file and load plugin shared objects; calls initialize() on each plugin.
  bool load_bringup(const std::string& bringup_yaml_path);

  /// Block SIGINT/SIGTERM for worker threads, start each plugin::run() on its own thread, block until
  /// Ctrl-C or SIGTERM, then call stop() and join workers.
  void run_until_signal();

  void request_stop();

 private:
  struct Loaded;
  std::vector<std::unique_ptr<Loaded>> loaded_;
  std::atomic<bool> stop_requested_{false};
  std::filesystem::path bringup_path_;
  std::filesystem::path repo_root_;
};

}  // namespace robo_lab
