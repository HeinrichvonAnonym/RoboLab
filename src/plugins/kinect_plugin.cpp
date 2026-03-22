#include "plugins/kinect_plugin.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <yaml-cpp/yaml.h>

namespace robo_lab {

bool KinectPlugin::initialize(const std::string& config_path) {
    config_path_ = config_path;
    message_system_ = std::make_unique<MessageSystem>();
    message_system_->initialize();
    return true;
}

void KinectPlugin::run() {
  stop_ = false;
  std::cout << "kinect_plugin: run loop started\n";
  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  std::cout << "kinect_plugin: run loop exited\n";
}

void KinectPlugin::stop() {
  stop_ = true;
}

}
extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::KinectPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}