#include <franka/gripper.h>
#include <franka/robot.h>

#include <exception>
#include <iostream>
#include <string>

// Only connection handshakes and state reads. Never call control, homing,
// move, grasp, recovery or stop; the on-site operator owns robot motion.
int main(int argc, char** argv) {
  if (argc != 2 && !(argc == 3 && std::string(argv[2]) == "--arm-only")) {
    std::cerr << "Usage: " << argv[0] << " <robot-ip> [--arm-only]\n";
    return 2;
  }
  const bool check_gripper = argc == 2;
  bool success = true;
  std::cout << "Read-only connection check for " << argv[1] << std::endl;
  try {
    franka::Robot robot(argv[1]);
    const auto state = robot.readOnce();
    std::cout << "ARM OK (TCP 1337), server version=" << robot.serverVersion()
              << ", q=[";
    for (std::size_t i = 0; i < state.q.size(); ++i) {
      std::cout << (i ? ", " : "") << state.q[i];
    }
    std::cout << "]" << std::endl;
  } catch (const std::exception& e) {
    success = false;
    std::cerr << "ARM FAILED (TCP 1337): " << e.what() << std::endl;
  }
  if (check_gripper) {
    try {
      franka::Gripper gripper(argv[1]);
      const auto state = gripper.readOnce();
      std::cout << "GRIPPER OK (TCP 1338), width=" << state.width
                << ", max_width=" << state.max_width
                << ", server version=" << gripper.serverVersion() << std::endl;
    } catch (const std::exception& e) {
      success = false;
      std::cerr << "GRIPPER FAILED (TCP 1338): " << e.what() << std::endl;
    }
  } else {
    std::cout << "GRIPPER NOT CHECKED (--arm-only)" << std::endl;
  }
  return success ? 0 : 1;
}
