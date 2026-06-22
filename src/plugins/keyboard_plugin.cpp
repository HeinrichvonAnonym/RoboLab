#include "plugins/keyboard_plugin.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <poll.h>
#include <termios.h>

#include <yaml-cpp/yaml.h>

#include "franka.pb.h"

namespace robo_lab {

struct KeyboardPlugin::TermiosState {
  struct termios tio {};
};

namespace {

// Extracts a single byte from a YAML mapping key. Accepts a one-character
// scalar; returns 0 if the key is empty.
char first_char_of_key(const std::string& key) {
  return key.empty() ? '\0' : key[0];
}

bool starts_with(const std::string& value, const char* prefix) {
  const std::string p(prefix);
  return value.size() >= p.size() && value.compare(0, p.size(), p) == 0;
}

}  // namespace

// Human-readable label for log messages.
const char* KeyboardPlugin::direction_name(Direction d) {
  switch (d) {
    case Direction::kXPos:     return "x_positive";
    case Direction::kXNeg:     return "x_negative";
    case Direction::kYPos:     return "y_positive";
    case Direction::kYNeg:     return "y_negative";
    case Direction::kZPos:     return "z_positive";
    case Direction::kZNeg:     return "z_negative";
    case Direction::kRollPos:  return "roll_positive";
    case Direction::kRollNeg:  return "roll_negative";
    case Direction::kPitchPos: return "pitch_positive";
    case Direction::kPitchNeg: return "pitch_negative";
    case Direction::kYawPos:   return "yaw_positive";
    case Direction::kYawNeg:   return "yaw_negative";
  }
  return "?";
}

bool KeyboardPlugin::parse_direction_tag_(const std::string& s, Direction* out) {
  using D = Direction;
  static const std::unordered_map<std::string, D> kTable = {
      {"x_positive",     D::kXPos},     {"x_negative",     D::kXNeg},
      {"y_positive",     D::kYPos},     {"y_negative",     D::kYNeg},
      {"z_positive",     D::kZPos},     {"z_negative",     D::kZNeg},
      {"roll_positive",  D::kRollPos},  {"roll_negative",  D::kRollNeg},
      {"pitch_positive", D::kPitchPos}, {"pitch_negative", D::kPitchNeg},
      {"yaw_positive",   D::kYawPos},   {"yaw_negative",   D::kYawNeg},
  };
  auto it = kTable.find(s);
  if (it == kTable.end()) {
    return false;
  }
  *out = it->second;
  return true;
}

bool KeyboardPlugin::load_config(const std::string& config_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    std::cerr << "keyboard_plugin: YAML error in " << config_path
              << ": " << e.what() << '\n';
    return false;
  }

  if (root["cmd_topic"]) {
    cmd_topic_ = root["cmd_topic"].as<std::string>();
  }
  if (root["next_record_topic"]) {
    next_record_topic_ = root["next_record_topic"].as<std::string>();
  }
  if (root["linear_step"]) {
    linear_step_ = root["linear_step"].as<double>();
  }
  if (root["angular_step"]) {
    angular_step_ = root["angular_step"].as<double>();
  }

  if (!root["keys_mapping"] || !root["keys_mapping"].IsMap()) {
    std::cerr << "keyboard_plugin: keys_mapping (map) missing in "
              << config_path << '\n';
    return false;
  }
  key_map_.clear();
  for (const auto& kv : root["keys_mapping"]) {
    const std::string key_str = kv.first.as<std::string>();
    const std::string dir_str = kv.second.as<std::string>();
    const char ch = first_char_of_key(key_str);
    if (ch == '\0') {
      std::cerr << "keyboard_plugin: empty key in keys_mapping; skipping\n";
      continue;
    }
    Direction d;
    if (!parse_direction_tag_(dir_str, &d)) {
      std::cerr << "keyboard_plugin: unknown direction '" << dir_str
                << "' for key '" << key_str << "'; skipping\n";
      continue;
    }
    key_map_[ch] = d;
  }
  if (key_map_.empty()) {
    std::cerr << "keyboard_plugin: no valid key mappings\n";
    return false;
  }
  return true;
}

bool KeyboardPlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;
  if (!load_config(config_path)) {
    return false;
  }

  message_system_ = std::make_unique<MessageSystem>();
  message_system_->initialize();
  if (!message_system_->is_open()) {
    std::cerr << "keyboard_plugin: failed to open Zenoh session\n";
    return false;
  }

  // Pre-publish a zero-delta sample so the Zenoh publisher is declared before
  // run() starts. This also lets late subscribers (recorder etc.) see traffic
  // on the topic immediately rather than only after the first key press.
  franka::CartesianDPoseCmd warmup;
  warmup.set_dx(0.0f);
  warmup.set_dy(0.0f);
  warmup.set_dz(0.0f);
  warmup.set_droll(0.0f);
  warmup.set_dpitch(0.0f);
  warmup.set_dyaw(0.0f);
  std::string warmup_payload;
  if (warmup.SerializeToString(&warmup_payload)) {
    if (!message_system_->publish(cmd_topic_, warmup_payload)) {
      std::cerr << "keyboard_plugin: warmup publish on '" << cmd_topic_
                << "' failed (subscribers may not see the topic until first key)\n";
    }
  }

  std::cout << "keyboard_plugin: initialized (cmd_topic=" << cmd_topic_
            << ", next_record_topic=" << next_record_topic_
            << ", linear_step=" << linear_step_
            << ", angular_step=" << angular_step_
            << ", keys=" << key_map_.size() << ")\n";
  return true;
}

bool KeyboardPlugin::enter_raw_mode() {
  if (!isatty(STDIN_FILENO)) {
    std::cerr << "keyboard_plugin: stdin is not a TTY; keyboard input "
                 "disabled (run from an interactive terminal)\n";
    return false;
  }
  saved_termios_ = std::make_unique<TermiosState>();
  if (tcgetattr(STDIN_FILENO, &saved_termios_->tio) != 0) {
    std::cerr << "keyboard_plugin: tcgetattr failed: " << std::strerror(errno) << '\n';
    saved_termios_.reset();
    return false;
  }
  struct termios raw = saved_termios_->tio;
  // Disable line-buffering and local echo, but keep ISIG so Ctrl-C still
  // delivers SIGINT to the framework's signal handler.
  raw.c_lflag &= ~(ICANON | ECHO);
  raw.c_cc[VMIN] = 0;   // poll handles blocking
  raw.c_cc[VTIME] = 0;
  if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
    std::cerr << "keyboard_plugin: tcsetattr failed: " << std::strerror(errno) << '\n';
    saved_termios_.reset();
    return false;
  }
  raw_mode_active_ = true;
  return true;
}

void KeyboardPlugin::restore_mode() {
  if (raw_mode_active_ && saved_termios_) {
    tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios_->tio);
  }
  raw_mode_active_ = false;
  saved_termios_.reset();
}

bool KeyboardPlugin::publish_direction(Direction d) {
  franka::CartesianDPoseCmd msg;
  msg.set_dx(0.0f);
  msg.set_dy(0.0f);
  msg.set_dz(0.0f);
  msg.set_droll(0.0f);
  msg.set_dpitch(0.0f);
  msg.set_dyaw(0.0f);

  const float ls = static_cast<float>(linear_step_);
  const float as = static_cast<float>(angular_step_);
  switch (d) {
    case Direction::kXPos:     msg.set_dx(+ls); break;
    case Direction::kXNeg:     msg.set_dx(-ls); break;
    case Direction::kYPos:     msg.set_dy(+ls); break;
    case Direction::kYNeg:     msg.set_dy(-ls); break;
    case Direction::kZPos:     msg.set_dz(+ls); break;
    case Direction::kZNeg:     msg.set_dz(-ls); break;
    case Direction::kRollPos:  msg.set_droll(+as); break;
    case Direction::kRollNeg:  msg.set_droll(-as); break;
    case Direction::kPitchPos: msg.set_dpitch(+as); break;
    case Direction::kPitchNeg: msg.set_dpitch(-as); break;
    case Direction::kYawPos:   msg.set_dyaw(+as); break;
    case Direction::kYawNeg:   msg.set_dyaw(-as); break;
  }

  std::string payload;
  if (!msg.SerializeToString(&payload)) {
    return false;
  }
  return message_system_->publish(cmd_topic_, payload);
}

bool KeyboardPlugin::publish_next_record() {
  return message_system_ && message_system_->publish(next_record_topic_, "next_record");
}

void KeyboardPlugin::run() {
  stop_ = false;
  std::cout << "keyboard_plugin: run loop started\n";

  if (!enter_raw_mode()) {
    // Without a TTY we still need to honour stop(); just idle.
    while (!stop_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (message_system_) {
      message_system_->close();
    }
    return;
  }

  std::cout << "keyboard_plugin: ready, listening for keys (Ctrl-C to quit)\n";

  while (!stop_) {
    struct pollfd pfd{};
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;
    const int rc = poll(&pfd, 1, 100);  // 100 ms tick lets stop_ break the loop
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      std::cerr << "keyboard_plugin: poll failed: " << std::strerror(errno) << '\n';
      break;
    }
    if (rc == 0 || !(pfd.revents & POLLIN)) {
      continue;
    }
    char buf[16];
    const ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
    if (n <= 0) {
      continue;
    }
    pending_input_.append(buf, static_cast<size_t>(n));
    // Drain complete key sequences; one publish per recognised key or F1 press.
    while (!pending_input_.empty()) {
      if (starts_with(pending_input_, "\x1bOP")) {
        const bool ok = publish_next_record();
        std::cout << "keyboard_plugin: F1 -> next_record"
                  << (ok ? " (published)" : " (publish FAILED)")
                  << " on '" << next_record_topic_ << "'\n";
        pending_input_.erase(0, 3);
        continue;
      }
      if (starts_with(pending_input_, "\x1b[11~")) {
        const bool ok = publish_next_record();
        std::cout << "keyboard_plugin: F1 -> next_record"
                  << (ok ? " (published)" : " (publish FAILED)")
                  << " on '" << next_record_topic_ << "'\n";
        pending_input_.erase(0, 5);
        continue;
      }
      if (pending_input_[0] == '\x1b' &&
          (pending_input_.size() < 3 || (pending_input_[1] == '[' && pending_input_.size() < 5))) {
        break;
      }

      const char ch = pending_input_[0];
      pending_input_.erase(0, 1);
      auto it = key_map_.find(ch);
      if (it == key_map_.end()) {
        // Unknown key -- log so users can see why nothing was published.
        // Printable chars only, fall back to numeric byte for control chars.
        std::cout << "keyboard_plugin: unmapped key '"
                  << (ch >= 0x20 && ch < 0x7f ? std::string(1, ch)
                                              : std::string("\\x"))
                  << "' (byte=0x" << std::hex << (int)(unsigned char)ch
                  << std::dec << ")\n";
        continue;
      }
      const bool ok = publish_direction(it->second);
      std::cout << "keyboard_plugin: key '" << ch << "' -> "
                << direction_name(it->second)
                << (ok ? " (published)" : " (publish FAILED)")
                << " on '" << cmd_topic_ << "'\n";
    }
  }

  restore_mode();
  if (message_system_) {
    message_system_->close();
  }
  std::cout << "keyboard_plugin: run loop exited\n";
}

void KeyboardPlugin::stop() {
  if (stop_.exchange(true)) {
    return;
  }
  std::cout << "keyboard_plugin: stopping\n";
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::KeyboardPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}
