#include <iostream>
#include <string>

#include "framework.h"

int main(int argc, char** argv) {
  std::string bringup = "config/bringup_franka_config.yaml";
  if (argc > 1) {
    bringup = argv[1];
  }

  robo_lab::Framework framework;
  std::cout << "main: loading bringup: " << bringup << '\n';
  if (!framework.load_bringup(bringup)) {
    std::cerr << "main: load_bringup failed (check paths in YAML; run from repo root if using relative "
                 "paths)\n";
    return 1;
  }

  std::cout << "main: plugins initialized; press Ctrl-C (or send SIGTERM) to stop.\n";
  framework.run_until_signal();
  std::cout << "main: shutdown complete\n";
  return 0;
}
