#include "framework.h"

#include <algorithm>
#include <cctype>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <sstream>
#include <string>
#include <string_view>

namespace robo_lab {

namespace {

struct PluginSpec {
  std::string name;
  std::string config;
  std::string lib;
};

bool is_complete(const PluginSpec& s) { return !s.name.empty() && !s.config.empty() && !s.lib.empty(); }

std::string trim_copy(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  if (s.size() >= 2 &&
      ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
    s = s.substr(1, s.size() - 2);
  }
  return s;
}

std::string strip_inline_comment(std::string line) {
  std::size_t i = 0;
  while (i < line.size()) {
    if (line[i] == '#') {
      line.resize(i);
      break;
    }
    ++i;
  }
  return trim_copy(line);
}

bool parse_key_value(const std::string& line, std::string* key, std::string* val) {
  const auto colon = line.find(':');
  if (colon == std::string::npos) {
    return false;
  }
  *key = trim_copy(line.substr(0, colon));
  *val = trim_copy(line.substr(colon + 1));
  return !key->empty();
}

std::vector<PluginSpec> parse_bringup_yaml(std::string_view text) {
  std::vector<PluginSpec> specs;
  // Brace-init avoids the "most vexing parse" (otherwise `in` looks like a function declaration).
  std::istringstream in{std::string(text)};
  std::string line;

  bool in_plugins = false;
  bool in_entry = false;
  PluginSpec cur;

  auto flush = [&]() {
    if (is_complete(cur)) {
      specs.push_back(std::move(cur));
    } else if (in_entry && (!cur.name.empty() || !cur.config.empty() || !cur.lib.empty())) {
      std::cerr << "framework: incomplete plugin entry (need name, config, lib)\n";
    }
    cur = {};
    in_entry = false;
  };

  while (std::getline(in, line)) {
    line = strip_inline_comment(line);
    if (line.empty()) {
      continue;
    }

    if (!in_plugins) {
      if (trim_copy(line) == "plugins:") {
        in_plugins = true;
      }
      continue;
    }

    if (!line.empty() && line[0] == '-') {
      flush();
      in_entry = true;
      line = trim_copy(line.substr(1));
      if (line.empty()) {
        continue;
      }
    } else if (!in_entry) {
      continue;
    }

    std::string key;
    std::string val;
    if (!parse_key_value(line, &key, &val)) {
      continue;
    }
    if (key == "name") {
      cur.name = val;
    } else if (key == "config") {
      cur.config = val;
    } else if (key == "lib") {
      cur.lib = val;
    }
  }
  flush();
  return specs;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f) {
    return {};
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::filesystem::path strip_dot_relative(const std::filesystem::path& p) {
  const std::string s = p.generic_string();
  if (s.size() >= 2 && s[0] == '.' && (s[1] == '/' || s[1] == '\\')) {
    return std::filesystem::path(s.substr(2));
  }
  return p;
}

std::filesystem::path guess_repo_root(const std::filesystem::path& bringup_canonical) {
  namespace fs = std::filesystem;
  const fs::path parent = bringup_canonical.parent_path();
  if (parent.filename() == "config") {
    return parent.parent_path();
  }
  return parent;
}

std::filesystem::path resolve_existing_file(const std::filesystem::path& bringup_file,
                                            const std::string& rel_or_abs,
                                            const std::filesystem::path& repo_root) {
  namespace fs = std::filesystem;
  fs::path raw(rel_or_abs);
  if (raw.is_absolute() && fs::exists(raw)) {
    return fs::weakly_canonical(raw);
  }

  const fs::path rel = strip_dot_relative(raw);
  const fs::path cwd = fs::current_path();
  const fs::path bringup_dir = bringup_file.parent_path();

  const fs::path candidates[] = {
      cwd / raw,
      cwd / rel,
      bringup_dir / raw,
      bringup_dir / rel,
      repo_root / raw,
      repo_root / rel,
  };

  for (const fs::path& c : candidates) {
    std::error_code ec;
    const fs::path n = fs::weakly_canonical(c, ec);
    if (fs::exists(n)) {
      return fs::weakly_canonical(n);
    }
    if (fs::exists(c)) {
      return fs::weakly_canonical(c);
    }
  }
  return cwd / rel;
}

}  // namespace

struct Framework::Loaded {
  void* handle = nullptr;
  std::unique_ptr<Plugin, void (*)(Plugin*)> plugin{nullptr, nullptr};
  std::thread worker;
};

Framework::Framework() = default;

Framework::~Framework() {
  for (auto& L : loaded_) {
    if (L->plugin) {
      L->plugin->stop();
    }
  }
  for (auto& L : loaded_) {
    if (L->worker.joinable()) {
      L->worker.join();
    }
  }
  for (auto& L : loaded_) {
    L->plugin.reset();
    if (L->handle != nullptr) {
      dlclose(L->handle);
      L->handle = nullptr;
    }
  }
  loaded_.clear();
}

void Framework::request_stop() { stop_requested_ = true; }

bool Framework::load_bringup(const std::string& bringup_yaml_path) {
  namespace fs = std::filesystem;
  bringup_path_ = fs::weakly_canonical(fs::path(bringup_yaml_path));
  if (!fs::exists(bringup_path_)) {
    std::cerr << "framework: bringup file not found: " << bringup_path_ << '\n';
    return false;
  }

  repo_root_ = guess_repo_root(bringup_path_);

  const std::string yaml = read_file(bringup_path_);
  if (yaml.empty()) {
    std::cerr << "framework: empty or unreadable bringup: " << bringup_path_ << '\n';
    return false;
  }

  const std::vector<PluginSpec> specs = parse_bringup_yaml(yaml);
  if (specs.empty()) {
    std::cerr << "framework: no plugins listed in " << bringup_path_ << '\n';
    return false;
  }

  for (const PluginSpec& spec : specs) {
    const fs::path lib_path = resolve_existing_file(bringup_path_, spec.lib, repo_root_);
    const fs::path cfg_path = resolve_existing_file(bringup_path_, spec.config, repo_root_);

    if (!fs::exists(lib_path)) {
      std::cerr << "framework: plugin library not found: " << lib_path << " (from entry '" << spec.name
                << "')\n";
      return false;
    }
    if (!fs::exists(cfg_path)) {
      std::cerr << "framework: plugin config not found: " << cfg_path << " (from entry '" << spec.name
                << "')\n";
      return false;
    }

    void* handle = dlopen(lib_path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      std::cerr << "framework: dlopen failed for " << lib_path << ": " << dlerror() << '\n';
      return false;
    }

    dlerror();
    auto* create_fn = reinterpret_cast<Plugin* (*)()>(dlsym(handle, "robo_lab_plugin_create"));
    auto* destroy_fn = reinterpret_cast<void (*)(Plugin*)>(dlsym(handle, "robo_lab_plugin_destroy"));
    const char* sym_err = dlerror();
    if (sym_err != nullptr || create_fn == nullptr || destroy_fn == nullptr) {
      std::cerr << "framework: missing robo_lab_plugin_create/destroy in " << lib_path << '\n';
      dlclose(handle);
      return false;
    }

    Plugin* raw = create_fn();
    if (raw == nullptr) {
      std::cerr << "framework: robo_lab_plugin_create returned null for " << spec.name << '\n';
      dlclose(handle);
      return false;
    }

    auto loaded = std::make_unique<Loaded>();
    loaded->handle = handle;
    loaded->plugin = std::unique_ptr<Plugin, void (*)(Plugin*)>(raw, destroy_fn);

    if (!loaded->plugin->initialize(cfg_path.string())) {
      std::cerr << "framework: initialize() failed for plugin '" << spec.name << "'\n";
      loaded->plugin.reset();
      dlclose(handle);
      return false;
    }

    std::cout << "framework: loaded plugin '" << spec.name << "' from " << lib_path << '\n';
    loaded_.push_back(std::move(loaded));
  }

  return true;
}

void Framework::run_until_signal() {
  if (loaded_.empty()) {
    return;
  }

  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGINT);
  sigaddset(&set, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &set, nullptr);

  stop_requested_ = false;

  for (auto& L : loaded_) {
    Plugin* p = L->plugin.get();
    L->worker = std::thread([p]() { p->run(); });
  }

  int sig = 0;
  sigwait(&set, &sig);
  (void)sig;
  stop_requested_ = true;
  std::cout << "framework: shutdown signal received, stopping plugins...\n";

  // sleep for 1 s
  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (auto& L : loaded_) {
    if (L->plugin) {
      L->plugin->stop();
    }
  }
  for (auto& L : loaded_) {
    if (L->worker.joinable()) {
      L->worker.join();
    }
  }
}

}  // namespace robo_lab
