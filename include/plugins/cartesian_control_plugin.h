#pragma once

#include "plugin_interface.h"
#include "message_system.h"
#include "cartesian_control/cartesian_controller.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace robo_lab {

// Cartesian control plugin.
//
// Subscribes:
//   - state_topic_  (franka.RobotObservation): current joint state. First
//     observation seeds the controller's internal q_target. Every
//     observation triggers FK and a per-link pose publication for viz.
//   - dpose_topic_  (franka.CartesianDPoseCmd): incremental Cartesian
//     command in base frame. Each message is converted to a joint delta
//     via the resolved-rate (DLS) controller, integrated into q_target,
//     and republished as a franka.RobotCommand on cmd_topic_.
//
// Publishes:
//   - pose_topic_   (franka.RobotLinkTransforms): link FK for viz.
//   - cmd_topic_    (franka.RobotCommand): joint position targets to
//     consume by franka_plugin.
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
    std::string state_topic_;     // SUB:  franka.RobotObservation
    std::string pose_topic_;      // PUB:  franka.RobotLinkTransforms (viz)
    std::string dpose_topic_;     // SUB:  franka.CartesianDPoseCmd
    std::string cmd_topic_;       // PUB:  franka.RobotCommand

    std::atomic<bool> stop_{false};
    std::atomic<uint32_t> cmd_sequence_{0};

    std::unique_ptr<MessageSystem> message_system_;
    std::unique_ptr<CartesianController> controller_;

    // Guards controller_ state and the latched q_current_ (touched from both
    // the state and dpose Zenoh callback threads).
    std::mutex controller_mutex_;
    std::vector<double> q_current_;  // Latest measured joint positions.
    bool have_current_state_{false};

    void state_callback(const std::string& key, const std::string& payload);
    void dpose_callback(const std::string& key, const std::string& payload);

    bool publish_link_transforms(const std::vector<CartesianPose>& poses);
    bool publish_joint_command(const std::vector<double>& q_target);

    bool load_config(const std::string& config_path);
};

}  // namespace robo_lab
