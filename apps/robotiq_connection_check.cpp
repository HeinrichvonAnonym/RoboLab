#include "plugins/robotiq_gripper.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2 && !(argc == 3 && std::string(argv[2]) == "--activate")) {
    std::cerr << "Usage: " << argv[0] << " <serial-port> [--activate]\n"
              << "Default is read-only. --activate may move the fingers for calibration.\n";
    return 2;
  }
  try {
    robo_lab::RobotiqGripper::Options options;
    options.port = argv[1];
    robo_lab::RobotiqGripper gripper(options);
    if (argc == 3) {
      std::cout << "Activating Robotiq: fingers may move for calibration." << std::endl;
      gripper.activate();
    }
    const auto state = gripper.read_state();
    std::cout << "ROBOTIQ OK ready=" << state.ready() << " fault=" << state.fault
              << " position=" << state.position << " estimated_width=" << state.width << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ROBOTIQ FAILED: " << error.what() << '\n';
    return 1;
  }
}
