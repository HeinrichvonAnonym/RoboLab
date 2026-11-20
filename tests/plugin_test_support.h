#pragma once
#include "plugin_interface.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

// Load the actual module, with Framework's visibility flags; never initialize hardware.
struct TestPlugin {
  void* handle = nullptr;
  robo_lab::Plugin* plugin = nullptr;
  void (*destroy)(robo_lab::Plugin*) = nullptr;
  std::thread worker;
  explicit TestPlugin(const char* path) {
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(dlerror());
    auto create = reinterpret_cast<robo_lab::Plugin* (*)()>(dlsym(handle, "robo_lab_plugin_create"));
    destroy = reinterpret_cast<void (*)(robo_lab::Plugin*)>(dlsym(handle, "robo_lab_plugin_destroy"));
    if (!create || !destroy) {
      dlclose(handle);
      throw std::runtime_error("missing plugin factory");
    }
    plugin = create();
  }
  void run() { worker = std::thread([this] { plugin->run(); }); }
  void stop() {
    plugin->stop();
    if (worker.joinable()) worker.join();
  }
  ~TestPlugin() {
    stop();
    destroy(plugin);
    dlclose(handle);
  }
};

struct TestDirectory {
  std::filesystem::path path;
  TestDirectory() {
    char name[] = "/tmp/robolab-test-XXXXXX";
    const char* result = mkdtemp(name);
    if (!result) throw std::runtime_error("mkdtemp failed");
    path = result;
  }
  std::string write(const std::string& name, const std::string& contents) const {
    const auto file = path / name;
    std::ofstream(file) << contents;
    return file.string();
  }
  ~TestDirectory() { std::filesystem::remove_all(path); }
};
