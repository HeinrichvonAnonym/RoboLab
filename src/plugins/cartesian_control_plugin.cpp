#include "plugins/cartesian_control_plugin.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <yaml-cpp/yaml.h>
#include "franka.pb.h"

namespace robo_lab {

bool CartesianControlPlugin::load_config(const std::string& config_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        std::cerr << "cartesian_control_plugin: YAML error in " << config_path
                  << ": " << e.what() << '\n';
        return false;
    }

    std::string control_config_path;
    if (root["control_config"]) {
        control_config_path = root["control_config"].as<std::string>();
    }
    if (control_config_path.empty()) {
        std::cerr << "cartesian_control_plugin: control_config not specified in "
                  << config_path << '\n';
        return false;
    }

    YAML::Node ctrl_root;
    try {
        ctrl_root = YAML::LoadFile(control_config_path);
    } catch (const YAML::Exception& e) {
        std::cerr << "cartesian_control_plugin: YAML error in " << control_config_path
                  << ": " << e.what() << '\n';
        return false;
    }

    if (ctrl_root["state_topic"]) {
        state_topic_ = ctrl_root["state_topic"].as<std::string>();
    }
    if (ctrl_root["pose_topic"]) {
        pose_topic_ = ctrl_root["pose_topic"].as<std::string>();
    }

    if (ctrl_root["dh_params"] && ctrl_root["dh_params"].IsSequence()) {
        std::vector<std::array<double, 4>> dh_params;
        for (const auto& row : ctrl_root["dh_params"]) {
            if (row.IsSequence() && row.size() == 4) {
                std::array<double, 4> dh;
                dh[0] = row[0].as<double>();
                dh[1] = row[1].as<double>();
                dh[2] = row[2].as<double>();
                dh[3] = row[3].as<double>();
                dh_params.push_back(dh);
            }
        }
        controller_->set_dh_params(dh_params);
        std::cout << "cartesian_control_plugin: loaded " << dh_params.size()
                  << " DH params from config\n";
    } else {
        controller_->set_franka_default_dh();
        std::cout << "cartesian_control_plugin: using default Franka DH params\n";
    }

    return true;
}

bool CartesianControlPlugin::initialize(const std::string& config_path) {
    config_path_ = config_path;
    controller_ = std::make_unique<CartesianController>();

    if (!load_config(config_path_)) {
        return false;
    }

    message_system_ = std::make_unique<MessageSystem>();
    message_system_->initialize();
    if (!message_system_->is_open()) {
        std::cerr << "cartesian_control_plugin: failed to open Zenoh session\n";
        return false;
    }

    if (state_topic_.empty() || pose_topic_.empty()) {
        std::cerr << "cartesian_control_plugin: state_topic / pose_topic must be set\n";
        return false;
    }

    message_system_->subscribe(
        state_topic_,
        std::bind(&CartesianControlPlugin::state_callback, this,
                  std::placeholders::_1, std::placeholders::_2));

    std::cout << "cartesian_control_plugin: initialized"
              << " (state=" << state_topic_
              << ", pose=" << pose_topic_
              << ", links=" << controller_->num_joints() << ")\n";
    return true;
}

void CartesianControlPlugin::state_callback(const std::string& /*key*/,
                                            const std::string& payload) {
    franka::RobotObservation obs;
    if (!obs.ParseFromString(payload)) {
        std::cerr << "cartesian_control_plugin: failed to parse RobotObservation\n";
        return;
    }

    JointState state;
    state.position.resize(obs.joints_size());
    state.velocity.resize(obs.joints_size());
    state.effort.resize(obs.joints_size());
    for (int i = 0; i < obs.joints_size(); ++i) {
        state.position[i] = obs.joints(i).position();
        state.velocity[i] = obs.joints(i).velocity();
        state.effort[i] = obs.joints(i).effort();
    }
    state.timestamp = obs.sys_time();

    auto poses = controller_->compute_link_poses(state);
    if (poses.empty()) {
        return;
    }
    publish_link_transforms(poses);
}

bool CartesianControlPlugin::publish_link_transforms(
    const std::vector<CartesianPose>& poses) {
    franka::RobotLinkTransforms msg;
    for (const auto& p : poses) {
        franka::StampedPose* sp = msg.add_transforms();
        sp->mutable_pose()->mutable_pos()->set_x(p.position.x());
        sp->mutable_pose()->mutable_pos()->set_y(p.position.y());
        sp->mutable_pose()->mutable_pos()->set_z(p.position.z());
        sp->mutable_pose()->mutable_rot()->set_w(p.orientation.w());
        sp->mutable_pose()->mutable_rot()->set_x(p.orientation.x());
        sp->mutable_pose()->mutable_rot()->set_y(p.orientation.y());
        sp->mutable_pose()->mutable_rot()->set_z(p.orientation.z());
        sp->set_sys_time(static_cast<float>(p.timestamp));
    }

    std::string payload;
    if (!msg.SerializeToString(&payload)) {
        return false;
    }
    return message_system_->publish(pose_topic_, payload);
}

void CartesianControlPlugin::run() {
    stop_ = false;
    std::cout << "cartesian_control_plugin: run loop started\n";
    while (!stop_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "cartesian_control_plugin: run loop exited\n";
}

void CartesianControlPlugin::stop() {
    if (stop_.exchange(true)) {
        return;
    }
    if (message_system_) {
        message_system_->close();
    }
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
    return new robo_lab::CartesianControlPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
    delete plugin;
}

}
