#include "plugins/franka_plugin.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

namespace robo_lab {

namespace {

std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  if (s.size() >= 2 &&
      ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
    s = s.substr(1, s.size() - 2);
  }
  return s;
}

std::string strip_comment(std::string line) {
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '#') {
      line.resize(i);
      break;
    }
  }
  return trim(line);
}

}  // namespace

bool FrankaPlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;
  std::ifstream f(config_path);
  if (!f) {
    std::cerr << "franka_plugin: cannot open config: " << config_path << '\n';
    return false;
  }

  std::string line;
  while (std::getline(f, line)) {
    line = strip_comment(line);
    if (line.empty()) {
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = trim(line.substr(0, colon));
    std::string val = trim(line.substr(colon + 1));
    if (key == "robot_ip") {
      robot_ip_ = val;
    } else if (key == "topic") {
      topic_ = val;
    } else if (key == "control_mode") {
      control_mode_ = val;
    }
  }

  if (robot_ip_.empty()) {
    std::cerr << "franka_plugin: robot_ip missing in " << config_path << '\n';
    return false;
  }

  if (topic_.empty()) {
    topic_ = "robot/command";
  }

  std::cout << "franka_plugin: initialized (robot_ip=" << robot_ip_ << ", topic=" << topic_
            << ", control_mode=" << control_mode_ << ")\n";
  return true;
}

void FrankaPlugin::run() {
  stop_ = false;
  std::cout << "franka_plugin: run loop started\n";
  int tick = 0;
  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ++tick;
    if (tick % 10 == 0) {
      std::cout << "franka_plugin: heartbeat tick=" << tick << " topic=" << topic_
                << " robot_ip=" << robot_ip_ << " mode=" << control_mode_ << '\n';
    }
  }
  std::cout << "franka_plugin: run loop exited\n";
}

void FrankaPlugin::stop() {
  stop_ = true;
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::FrankaPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}
