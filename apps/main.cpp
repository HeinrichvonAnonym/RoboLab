#include <iostream>
#include <string>
#include <csignal>
#include <pthread.h>

#include "framework.h"

int main(int argc, char** argv) {
  // Zenoh/ROS/driver threads created during initialize() must inherit this mask.
  // Framework::run_until_signal() is the single owner of shutdown signals.
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  if (pthread_sigmask(SIG_BLOCK, &signals, nullptr) != 0) {
    std::cerr << "main: failed to block shutdown signals\n";
    return 1;
  }
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
