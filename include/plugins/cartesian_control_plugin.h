#pragma once

#include "plugin_interface.h"
#include "message_system.h"
#include "cartesian_control/cartesian_controller.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace robo_lab {

// Stage 1 cartesian-control plugin: subscribes to the robot-state topic,
// runs forward kinematics, and publishes per-link world poses as a
// franka::RobotLinkTransforms payload. No commands are produced yet.
class CartesianControlPlugin : public Plugin {
public:
    CartesianControlPlugin() = default;
    ~CartesianControlPlugin() override = default;

    bool initialize(const std::string& config_path) override;
    void run() override;
    void stop() override;

private:
    std::string config_path_;

    // Zenoh topics
    std::string state_topic_;   // Subscribe: current robot state
    std::string pose_topic_;    // Publish:   per-link world poses

    std::atomic<bool> stop_{false};

    std::unique_ptr<MessageSystem> message_system_;
    std::unique_ptr<CartesianController> controller_;

    void state_callback(const std::string& key, const std::string& payload);
    bool publish_link_transforms(const std::vector<CartesianPose>& poses);

    bool load_config(const std::string& config_path);
};

}  // namespace robo_lab
