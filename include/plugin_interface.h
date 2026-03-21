#pragma once

#include <string>

namespace robo_lab {

/// Shared by the host and every plugin .so (same vtable layout required).
class Plugin {
 public:
  virtual ~Plugin() = default;
  virtual bool initialize(const std::string& config_path) = 0;
  virtual void run() = 0;
  virtual void stop() = 0;
};

}  // namespace robo_lab

#if defined(_WIN32)
#define ROBO_LAB_PLUGIN_EXPORT __declspec(dllexport)
#else
#define ROBO_LAB_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create();
ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin);
}
