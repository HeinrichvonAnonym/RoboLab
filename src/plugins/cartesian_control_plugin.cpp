#include "plugins/cartesian_control_plugin.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include "franka.pb.h"

namespace robo_lab {

namespace {

// Build a 4x4 transform from a 7-element list [x, y, z, qx, qy, qz, qw].
Eigen::Matrix4d tool_offset_from_yaml(const YAML::Node& node) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    if (!node || !node.IsSequence() || node.size() != 7) {
        return T;
    }
    const double x  = node[0].as<double>();
    const double y  = node[1].as<double>();
    const double z  = node[2].as<double>();
    const double qx = node[3].as<double>();
    const double qy = node[4].as<double>();
    const double qz = node[5].as<double>();
    const double qw = node[6].as<double>();
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T(0, 3) = x;
    T(1, 3) = y;
    T(2, 3) = z;
    return T;
}

}  // namespace

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

    if (ctrl_root["state_topic"])  state_topic_  = ctrl_root["state_topic"].as<std::string>();
    if (ctrl_root["pose_topic"])   pose_topic_   = ctrl_root["pose_topic"].as<std::string>();
    if (ctrl_root["dpose_topic"])  dpose_topic_  = ctrl_root["dpose_topic"].as<std::string>();
    if (ctrl_root["cmd_topic"])    cmd_topic_    = ctrl_root["cmd_topic"].as<std::string>();

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

    if (ctrl_root["damping"]) {
        controller_->set_damping(ctrl_root["damping"].as<double>());
    }

    if (ctrl_root["max_dq_per_step"] && ctrl_root["max_dq_per_step"].IsSequence()) {
        std::vector<double> caps;
        for (const auto& v : ctrl_root["max_dq_per_step"]) {
            caps.push_back(v.as<double>());
        }
        controller_->set_max_dq_per_step(caps);
    }

    if (ctrl_root["joint_lower"] && ctrl_root["joint_upper"] &&
        ctrl_root["joint_lower"].IsSequence() && ctrl_root["joint_upper"].IsSequence()) {
        std::vector<double> lower, upper;
        for (const auto& v : ctrl_root["joint_lower"]) lower.push_back(v.as<double>());
        for (const auto& v : ctrl_root["joint_upper"]) upper.push_back(v.as<double>());
        controller_->set_joint_limits(lower, upper);
    }

    if (ctrl_root["tool_offset"]) {
        controller_->set_tool_offset(tool_offset_from_yaml(ctrl_root["tool_offset"]));
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

    if (state_topic_.empty() || pose_topic_.empty() ||
        dpose_topic_.empty() || cmd_topic_.empty()) {
        std::cerr << "cartesian_control_plugin: state/pose/dpose/cmd topics must all be set\n";
        return false;
    }

    message_system_->subscribe(
        state_topic_,
        std::bind(&CartesianControlPlugin::state_callback, this,
                  std::placeholders::_1, std::placeholders::_2));

    message_system_->subscribe(
        dpose_topic_,
        std::bind(&CartesianControlPlugin::dpose_callback, this,
                  std::placeholders::_1, std::placeholders::_2));

    std::cout << "cartesian_control_plugin: initialized"
              << " (state=" << state_topic_
              << ", pose=" << pose_topic_
              << ", dpose=" << dpose_topic_
              << ", cmd=" << cmd_topic_
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

    {
        std::lock_guard<std::mutex> lk(controller_mutex_);
        q_current_ = state.position;
        have_current_state_ = !q_current_.empty();
        if (have_current_state_ && !controller_->is_seeded()) {
            controller_->seed_target(q_current_);
            std::cout << "cartesian_control_plugin: seeded q_target from first observation\n";
        }
    }

    auto poses = controller_->compute_link_poses(state);
    if (!poses.empty()) {
        publish_link_transforms(poses);
    }
}

void CartesianControlPlugin::dpose_callback(const std::string& /*key*/,
                                            const std::string& payload) {
    franka::CartesianDPoseCmd dpose;
    if (!dpose.ParseFromString(payload)) {
        std::cerr << "cartesian_control_plugin: failed to parse CartesianDPoseCmd\n";
        return;
    }

    const std::array<double, 6> dx = {
        static_cast<double>(dpose.dx()),
        static_cast<double>(dpose.dy()),
        static_cast<double>(dpose.dz()),
        static_cast<double>(dpose.droll()),
        static_cast<double>(dpose.dpitch()),
        static_cast<double>(dpose.dyaw()),
    };

    // Skip zero-deltas (warmup pings from the keyboard plugin etc.).
    bool all_zero = true;
    for (double v : dx) {
        if (v != 0.0) { all_zero = false; break; }
    }
    if (all_zero) {
        return;
    }

    std::vector<double> q_target_new;
    {
        std::lock_guard<std::mutex> lk(controller_mutex_);
        if (!controller_->is_seeded()) {
            std::cerr << "cartesian_control_plugin: dropping dpose -- "
                         "q_target not yet seeded (waiting for first state)\n";
            return;
        }
        if (!controller_->apply_dpose(dx, &q_target_new)) {
            std::cerr << "cartesian_control_plugin: apply_dpose failed\n";
            return;
        }
    }

    publish_joint_command(q_target_new);
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

bool CartesianControlPlugin::publish_joint_command(
    const std::vector<double>& q_target) {
    franka::RobotCommand cmd;
    cmd.set_type(franka::RobotCommand::TYPE_JOINT_TARGET);
    cmd.set_sequence(cmd_sequence_.fetch_add(1));
    cmd.set_mode("position");
    cmd.set_note("cartesian_control");
    const float t_s =
        static_cast<float>(std::chrono::duration<double>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count());
    cmd.set_sys_time(t_s);
    for (double q : q_target) {
        franka::JointCommand* jc = cmd.add_joints();
        jc->set_position(q);
        jc->set_velocity(0.0);
        jc->set_effort(0.0);
    }

    std::string payload;
    if (!cmd.SerializeToString(&payload)) {
        return false;
    }
    return message_system_->publish(cmd_topic_, payload);
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
