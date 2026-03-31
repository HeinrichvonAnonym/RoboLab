#pragma once

#include "plugin_interface.h"
#include "message_system.h"
#include "cartesian_control/cartesian_controller.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace robo_lab {

class CartesianControlPlugin : public Plugin {
public:
    CartesianControlPlugin() = default;
    ~CartesianControlPlugin() override = default;

    bool initialize(const std::string& config_path) override;
    void run() override;
    void stop() override;

private:
    std::string config_path_;
    std::string control_mode_;
    
    // Zenoh topics
    std::string state_topic_;        // Subscribe: current robot state
    std::string target_pose_topic_;  // Subscribe: target pose
    std::string cmd_topic_;          // Publish: joint commands
    std::string pose_topic_;         // Publish: current EEF pose
    
    double control_rate_{50.0};      // Hz
    
    std::atomic<bool> stop_{false};
    
    std::unique_ptr<MessageSystem> message_system_;
    std::unique_ptr<CartesianController> controller_;
    
    void state_callback(const std::string& key, const std::string& payload);
    void target_pose_callback(const std::string& key, const std::string& payload);
    
    bool publish_joint_command(const std::vector<double>& q);
    bool publish_current_pose(const CartesianPose& pose);
    
    bool load_config(const std::string& config_path);
};

}  // namespace robo_lab
