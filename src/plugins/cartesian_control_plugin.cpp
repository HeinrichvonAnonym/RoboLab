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
        std::cerr << "cartesian_control_plugin: YAML error in " << config_path << ": " << e.what() << '\n';
        return false;
    }

    if (root["control_mode"]) {
        control_mode_ = root["control_mode"].as<std::string>();
    }

    // Load control config
    std::string control_config_path;
    if (root["control_config"]) {
        control_config_path = root["control_config"].as<std::string>();
    }

    if (control_config_path.empty()) {
        std::cerr << "cartesian_control_plugin: control_config not specified in " << config_path << '\n';
        return false;
    }

    YAML::Node ctrl_root;
    try {
        ctrl_root = YAML::LoadFile(control_config_path);
    } catch (const YAML::Exception& e) {
        std::cerr << "cartesian_control_plugin: YAML error in " << control_config_path << ": " << e.what() << '\n';
        return false;
    }

    // Topics
    if (ctrl_root["state_topic"]) {
        state_topic_ = ctrl_root["state_topic"].as<std::string>();
    }
    if (ctrl_root["target_pose_topic"]) {
        target_pose_topic_ = ctrl_root["target_pose_topic"].as<std::string>();
    }
    if (ctrl_root["cmd_topic"]) {
        cmd_topic_ = ctrl_root["cmd_topic"].as<std::string>();
    }
    if (ctrl_root["pose_topic"]) {
        pose_topic_ = ctrl_root["pose_topic"].as<std::string>();
    }

    // DH parameters
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
        std::cout << "cartesian_control_plugin: loaded " << dh_params.size() << " DH params from config\n";
    } else {
        controller_->set_franka_default_dh();
        std::cout << "cartesian_control_plugin: using default Franka DH params\n";
    }

    // Limits
    CartesianLimits limits;
    if (ctrl_root["max_d_pos"]) {
        limits.max_pos_error = ctrl_root["max_d_pos"].as<double>();
    }
    if (ctrl_root["max_d_vel"]) {
        limits.max_vel = ctrl_root["max_d_vel"].as<double>();
    }
    if (ctrl_root["max_d_acc"]) {
        limits.max_acc = ctrl_root["max_d_acc"].as<double>();
    }
    controller_->set_limits(limits);

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

    // Subscribe to robot state
    if (!state_topic_.empty()) {
        message_system_->subscribe(
            state_topic_,
            std::bind(&CartesianControlPlugin::state_callback, this,
                      std::placeholders::_1, std::placeholders::_2));
    }

    // Subscribe to target pose
    if (!target_pose_topic_.empty()) {
        message_system_->subscribe(
            target_pose_topic_,
            std::bind(&CartesianControlPlugin::target_pose_callback, this,
                      std::placeholders::_1, std::placeholders::_2));
    }

    std::cout << "cartesian_control_plugin: initialized"
              << " (state=" << state_topic_
              << ", target=" << target_pose_topic_
              << ", cmd=" << cmd_topic_
              << ", pose=" << pose_topic_ << ")\n";
    return true;
}

void CartesianControlPlugin::state_callback(const std::string& key, const std::string& payload) {
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

    controller_->set_current_joint_state(state);

    // Publish current pose
    CartesianPose current_pose = controller_->get_current_pose();
    
    static int debug_counter = 0;
    if (debug_counter++ % 100 == 0) {
        std::cout << "[CartesianPlugin] FK pose: pos=("
                  << current_pose.position.x() << ", "
                  << current_pose.position.y() << ", "
                  << current_pose.position.z() << ") quat=("
                  << current_pose.orientation.w() << ", "
                  << current_pose.orientation.x() << ", "
                  << current_pose.orientation.y() << ", "
                  << current_pose.orientation.z() << ")\n";
    }
    
    publish_current_pose(current_pose);
}

void CartesianControlPlugin::target_pose_callback(const std::string& key, const std::string& payload) {
    franka::StampedPose msg;
    if (!msg.ParseFromString(payload)) {
        std::cerr << "cartesian_control_plugin: failed to parse StampedPose\n";
        return;
    }

    CartesianPose pose;
    pose.position = Eigen::Vector3d(
        msg.pose().pos().x(),
        msg.pose().pos().y(),
        msg.pose().pos().z());
    pose.orientation = Eigen::Quaterniond(
        msg.pose().rot().w(),
        msg.pose().rot().x(),
        msg.pose().rot().y(),
        msg.pose().rot().z());
    pose.timestamp = msg.sys_time();

    controller_->set_target_pose(pose);
    
    CartesianPose current = controller_->get_current_pose();
    
    std::cout << "cartesian_control_plugin: target=("
              << pose.position.x() << ", "
              << pose.position.y() << ", "
              << pose.position.z() << ") current=("
              << current.position.x() << ", "
              << current.position.y() << ", "
              << current.position.z() << ") delta=("
              << (pose.position.x() - current.position.x()) << ", "
              << (pose.position.y() - current.position.y()) << ", "
              << (pose.position.z() - current.position.z()) << ")\n";
}

bool CartesianControlPlugin::publish_joint_command(const std::vector<double>& q) {
    franka::RobotCommand cmd;
    cmd.set_type(franka::RobotCommand::TYPE_JOINT_TARGET);
    cmd.set_mode(control_mode_);

    for (size_t i = 0; i < q.size(); ++i) {
        auto* joint = cmd.add_joints();
        joint->set_position(q[i]);
        joint->set_velocity(0.0);
        joint->set_effort(0.0);
    }

    std::string payload;
    if (!cmd.SerializeToString(&payload)) {
        return false;
    }
    return message_system_->publish(cmd_topic_, payload);
}

bool CartesianControlPlugin::publish_current_pose(const CartesianPose& pose) {
    franka::StampedPose msg;
    msg.mutable_pose()->mutable_pos()->set_x(pose.position.x());
    msg.mutable_pose()->mutable_pos()->set_y(pose.position.y());
    msg.mutable_pose()->mutable_pos()->set_z(pose.position.z());
    msg.mutable_pose()->mutable_rot()->set_w(pose.orientation.w());
    msg.mutable_pose()->mutable_rot()->set_x(pose.orientation.x());
    msg.mutable_pose()->mutable_rot()->set_y(pose.orientation.y());
    msg.mutable_pose()->mutable_rot()->set_z(pose.orientation.z());
    msg.set_sys_time(pose.timestamp);

    std::string payload;
    if (!msg.SerializeToString(&payload)) {
        return false;
    }
    return message_system_->publish(pose_topic_, payload);
}

void CartesianControlPlugin::run() {
    stop_ = false;
    std::cout << "cartesian_control_plugin: control loop started\n";

    const double dt = 1.0 / control_rate_;
    const auto period = std::chrono::duration<double>(dt);

    int loop_counter = 0;
    while (!stop_) {
        auto t_start = std::chrono::steady_clock::now();

        // Compute and publish joint command
        std::vector<double> q_cmd = controller_->compute_joint_command(dt);
        if (!q_cmd.empty()) {
            publish_joint_command(q_cmd);
            
            // Debug every 100 iterations
            if (loop_counter++ % 100 == 0) {
                JointState state = controller_->get_current_joint_state();
                if (!state.position.empty()) {
                    std::cout << "[CartesianPlugin] q_cur[0]=" << state.position[0]
                              << " q_cmd[0]=" << q_cmd[0]
                              << " delta=" << (q_cmd[0] - state.position[0]) << "\n";
                }
            }
        }

        // Sleep for remaining time
        auto t_elapsed = std::chrono::steady_clock::now() - t_start;
        if (t_elapsed < period) {
            std::this_thread::sleep_for(period - t_elapsed);
        }
    }

    std::cout << "cartesian_control_plugin: control loop exited\n";
}

void CartesianControlPlugin::stop() {
    stop_ = true;
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
