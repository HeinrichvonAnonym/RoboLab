#include "plugins/ros_bridge_plugin.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "franka.pb.h"
#include "kinect.pb.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/ByteMultiArray.h>
#include <yaml-cpp/yaml.h>

namespace robo_lab {
namespace {

bool publish_zenoh_to_ros(const std::string& ros_type,
                          const std::vector<std::string>& joint_names,
                          const std::string& payload,
                          const ros::Publisher& pub) {
  if (ros_type == "sensor_msgs/JointState") {
    sensor_msgs::JointState out;
    out.header.stamp = ros::Time::now();

    franka::RobotObservation obs;
    if (obs.ParseFromString(payload)) {
      const int n = obs.joints_size();
      out.name.resize(static_cast<size_t>(n));
      out.position.resize(static_cast<size_t>(n));
      out.velocity.resize(static_cast<size_t>(n));
      out.effort.resize(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) {
        if (i < static_cast<int>(joint_names.size())) {
          out.name[static_cast<size_t>(i)] = joint_names[static_cast<size_t>(i)];
        } else {
          out.name[static_cast<size_t>(i)] = "panda_joint" + std::to_string(i + 1);
        }
        out.position[static_cast<size_t>(i)] = obs.joints(i).position();
        out.velocity[static_cast<size_t>(i)] = obs.joints(i).velocity();
        out.effort[static_cast<size_t>(i)] = obs.joints(i).effort();
      }
      // If configured with extra joints (e.g. gripper), publish zeros for missing values.
      for (size_t i = static_cast<size_t>(n); i < joint_names.size(); ++i) {
        out.name.push_back(joint_names[i]);
        out.position.push_back(0.0);
        out.velocity.push_back(0.0);
        out.effort.push_back(0.0);
      }
      pub.publish(out);
      //std::cout << "ros_bridge_plugin: converted franka/state RobotObservation -> sensor_msgs/JointState (n=" << n
              //  << ")\n";
      return true;
    }

    franka::JointStateArray arr;
    if (arr.ParseFromString(payload)) {
      const int n = arr.states_size();
      out.name.resize(static_cast<size_t>(n));
      out.position.resize(static_cast<size_t>(n));
      out.velocity.resize(static_cast<size_t>(n));
      out.effort.resize(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) {
        if (i < static_cast<int>(joint_names.size())) {
          out.name[static_cast<size_t>(i)] = joint_names[static_cast<size_t>(i)];
        } else {
          out.name[static_cast<size_t>(i)] = "panda_joint" + std::to_string(i + 1);
        }
        out.position[static_cast<size_t>(i)] = arr.states(i).position();
        out.velocity[static_cast<size_t>(i)] = arr.states(i).velocity();
        out.effort[static_cast<size_t>(i)] = arr.states(i).effort();
      }
      for (size_t i = static_cast<size_t>(n); i < joint_names.size(); ++i) {
        out.name.push_back(joint_names[i]);
        out.position.push_back(0.0);
        out.velocity.push_back(0.0);
        out.effort.push_back(0.0);
      }
      pub.publish(out);
      std::cout << "ros_bridge_plugin: converted franka/state JointStateArray -> sensor_msgs/JointState (n=" << n
                << ")\n";
      return true;
    }
    return false;
  }

  if (ros_type == "sensor_msgs/Image") {
    kinect::rgbImage img;
    if (!img.ParseFromString(payload)) {
      return false;
    }
    sensor_msgs::Image out;
    out.header.stamp = ros::Time::now();
    out.width = static_cast<uint32_t>(img.width());
    out.height = static_cast<uint32_t>(img.height());
    out.step = static_cast<uint32_t>(img.step());
    out.is_bigendian = false;
    if (img.channels() == 3) {
      out.encoding = "rgb8";
    } else if (img.channels() == 1) {
      out.encoding = "mono8";
    } else {
      out.encoding = "8UC" + std::to_string(img.channels());
    }
    out.data.assign(img.image().begin(), img.image().end());
    pub.publish(out);
    return true;
  }

  if (ros_type == "geometry_msgs/PoseStamped") {
    franka::StampedPose pose_msg;
    if (!pose_msg.ParseFromString(payload)) {
      return false;
    }
    geometry_msgs::PoseStamped out;
    out.header.stamp = ros::Time::now();
    out.header.frame_id = "panda_link0";
    out.pose.position.x = pose_msg.pose().pos().x();
    out.pose.position.y = pose_msg.pose().pos().y();
    out.pose.position.z = pose_msg.pose().pos().z();
    out.pose.orientation.w = pose_msg.pose().rot().w();
    out.pose.orientation.x = pose_msg.pose().rot().x();
    out.pose.orientation.y = pose_msg.pose().rot().y();
    out.pose.orientation.z = pose_msg.pose().rot().z();
    pub.publish(out);
    return true;
  }

  std_msgs::ByteMultiArray raw;
  raw.data.assign(payload.begin(), payload.end());
  pub.publish(raw);
  return true;
}

}  // namespace

struct RosBridgePlugin::Impl {
  std::unique_ptr<ros::NodeHandle> nh;
  std::unique_ptr<ros::AsyncSpinner> spinner;
  std::vector<ros::Publisher> robo_to_ros_publishers;
  std::vector<ros::Subscriber> ros_to_robo_subscribers;
};

bool RosBridgePlugin::parse_topic_list(const YAML::Node& node, std::vector<BridgeTopic>* out) {
  if (!node || !node.IsSequence()) {
    return false;
  }
  out->clear();
  for (const auto& item : node) {
    if (!item.IsMap() || !item["zenoh_msg"] || !item["ros_msg"]) {
      return false;
    }
    BridgeTopic t;
    t.zenoh_msg = item["zenoh_msg"].as<std::string>();
    t.ros_msg = item["ros_msg"].as<std::string>();
    if (item["zenoh_type"]) {
      t.zenoh_type = item["zenoh_type"].as<std::string>();
    }
    if (item["ros_type"]) {
      t.ros_type = item["ros_type"].as<std::string>();
    }
    if (item["joint_name"] && item["joint_name"].IsSequence()) {
      for (const auto& name_node : item["joint_name"]) {
        if (!name_node.IsScalar()) {
          return false;
        }
        t.joint_names.push_back(name_node.as<std::string>());
      }
    }
    out->push_back(std::move(t));
  }
  return true;
}

bool RosBridgePlugin::load_config(const std::string& config_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    std::cerr << "ros_bridge_plugin: YAML error in " << config_path << ": " << e.what() << '\n';
    return false;
  }

  // Accept both naming styles for compatibility.
  const YAML::Node robo_to_ros =
      root["roboLab_2_ros_topics"] ? root["roboLab_2_ros_topics"] : root["robo_lab_2_ros_topics"];
  const YAML::Node ros_to_robo =
      root["ros_2_roboLab_topics"] ? root["ros_2_roboLab_topics"] : root["ros_2_robo_lab_topics"];

  if (robo_to_ros && !parse_topic_list(robo_to_ros, &robo_to_ros_topics_)) {
    std::cerr << "ros_bridge_plugin: invalid roboLab_2_ros_topics in " << config_path << '\n';
    return false;
  }
  if (ros_to_robo && !parse_topic_list(ros_to_robo, &ros_to_robo_topics_)) {
    std::cerr << "ros_bridge_plugin: invalid ros_2_roboLab_topics in " << config_path << '\n';
    return false;
  }
  return true;
}

bool RosBridgePlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;
  if (!load_config(config_path_)) {
    return false;
  }

  message_system_ = std::make_unique<MessageSystem>();
  message_system_->initialize();
  if (!message_system_->is_open()) {
    std::cerr << "ros_bridge_plugin: failed to open Zenoh session\n";
    return false;
  }

  int argc = 0;
  char** argv = nullptr;
  if (!ros::isInitialized()) {
    ros::init(argc, argv, "robo_lab_ros_bridge_plugin", ros::init_options::NoSigintHandler);
  }

  impl_ = std::make_unique<Impl>();
  impl_->nh = std::make_unique<ros::NodeHandle>("~");
  impl_->spinner = std::make_unique<ros::AsyncSpinner>(1);

  impl_->robo_to_ros_publishers.reserve(robo_to_ros_topics_.size());
  for (const auto& bridge : robo_to_ros_topics_) {
    ros::Publisher pub;
    if (bridge.ros_type == "sensor_msgs/JointState") {
      pub = impl_->nh->advertise<sensor_msgs::JointState>(bridge.ros_msg, 10);
    } else if (bridge.ros_type == "sensor_msgs/Image") {
      pub = impl_->nh->advertise<sensor_msgs::Image>(bridge.ros_msg, 10);
    } else if (bridge.ros_type == "geometry_msgs/PoseStamped") {
      pub = impl_->nh->advertise<geometry_msgs::PoseStamped>(bridge.ros_msg, 10);
    } else {
      pub = impl_->nh->advertise<std_msgs::ByteMultiArray>(bridge.ros_msg, 10);
      std::cout << "ros_bridge_plugin: unsupported ros_type '" << bridge.ros_type
                << "' on " << bridge.ros_msg << ", using std_msgs/ByteMultiArray fallback\n";
    }
    impl_->robo_to_ros_publishers.push_back(pub);

    const size_t idx = impl_->robo_to_ros_publishers.size() - 1;
    const BridgeTopic bridge_cfg = bridge;
    message_system_->subscribe(
        bridge.zenoh_msg, [this, idx, bridge_cfg](const std::string& key, const std::string& payload) {
          if (!publish_zenoh_to_ros(
                  bridge_cfg.ros_type, bridge_cfg.joint_names, payload, impl_->robo_to_ros_publishers[idx])) {
            std::cerr << "ros_bridge_plugin: failed to decode Zenoh payload for key=" << key
                      << " as ros_type=" << bridge_cfg.ros_type << '\n';
          }
        });
  }

  impl_->ros_to_robo_subscribers.reserve(ros_to_robo_topics_.size());
  for (const auto& bridge : ros_to_robo_topics_) {
    ros::Subscriber sub;
    if (bridge.ros_type == "sensor_msgs/JointState") {
      // std::cout << "ros_bridge_plugin: subscribing to " << bridge.ros_msg << std::endl;
      sub = impl_->nh->subscribe<sensor_msgs::JointState>(
          bridge.ros_msg, 10,
          [this, bridge](const sensor_msgs::JointState::ConstPtr& msg) {
            franka::JointStateArray out;
            const size_t n = msg->position.size();
            for (size_t i = 0; i < n; ++i) {
              auto* s = out.add_states();
              s->set_position(msg->position[i]);
              if (i < msg->velocity.size()) {
                s->set_velocity(msg->velocity[i]);
              }
              if (i < msg->effort.size()) {
                s->set_effort(msg->effort[i]);
              }
            }
            std::string payload;
            if (!out.SerializeToString(&payload)) {
              return;
            }
            message_system_->publish(bridge.zenoh_msg, payload);
          });
    } else if (bridge.ros_type == "sensor_msgs/Image") {
      sub = impl_->nh->subscribe<sensor_msgs::Image>(
          bridge.ros_msg, 10,
          [this, bridge](const sensor_msgs::Image::ConstPtr& msg) {
            kinect::rgbImage out;
            out.set_image(msg->data.data(), msg->data.size());
            out.set_width(static_cast<int32_t>(msg->width));
            out.set_height(static_cast<int32_t>(msg->height));
            out.set_step(static_cast<int32_t>(msg->step));
            if (msg->encoding == "rgb8") {
              out.set_channels(3);
            } else if (msg->encoding == "mono8") {
              out.set_channels(1);
            } else {
              out.set_channels(0);
            }
            out.set_type(0);
            std::string payload;
            if (!out.SerializeToString(&payload)) {
              return;
            }
            message_system_->publish(bridge.zenoh_msg, payload);
          });
    } else if (bridge.ros_type == "geometry_msgs/PoseStamped") {
      sub = impl_->nh->subscribe<geometry_msgs::PoseStamped>(
          bridge.ros_msg, 10,
          [this, bridge](const geometry_msgs::PoseStamped::ConstPtr& msg) {
            franka::StampedPose out;
            out.mutable_pose()->mutable_pos()->set_x(msg->pose.position.x);
            out.mutable_pose()->mutable_pos()->set_y(msg->pose.position.y);
            out.mutable_pose()->mutable_pos()->set_z(msg->pose.position.z);
            out.mutable_pose()->mutable_rot()->set_w(msg->pose.orientation.w);
            out.mutable_pose()->mutable_rot()->set_x(msg->pose.orientation.x);
            out.mutable_pose()->mutable_rot()->set_y(msg->pose.orientation.y);
            out.mutable_pose()->mutable_rot()->set_z(msg->pose.orientation.z);
            out.set_sys_time(msg->header.stamp.toSec());
            std::string payload;
            if (!out.SerializeToString(&payload)) {
              return;
            }
            message_system_->publish(bridge.zenoh_msg, payload);
          });
    } else {
      sub = impl_->nh->subscribe<std_msgs::ByteMultiArray>(
          bridge.ros_msg, 10,
          [this, bridge](const std_msgs::ByteMultiArray::ConstPtr& msg) {
            std::string payload(msg->data.begin(), msg->data.end());
            message_system_->publish(bridge.zenoh_msg, payload);
          });
      std::cout << "ros_bridge_plugin: unsupported ros_type '" << bridge.ros_type
                << "' on " << bridge.ros_msg << ", using std_msgs/ByteMultiArray fallback\n";
    }
    impl_->ros_to_robo_subscribers.push_back(sub);
  }

  impl_->spinner->start();
  std::cout << "ros_bridge_plugin: initialized (robo->ros=" << robo_to_ros_topics_.size()
            << ", ros->robo=" << ros_to_robo_topics_.size() << ")\n";
  return true;
}

void RosBridgePlugin::run() {
  stop_ = false;
  std::cout << "ros_bridge_plugin: run loop started\n";
  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  std::cout << "ros_bridge_plugin: run loop exited\n";
}

void RosBridgePlugin::stop() {
  stop_ = true;
  if (impl_ && impl_->spinner) {
    impl_->spinner->stop();
  }
  if (message_system_) {
    message_system_->close();
  }
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::RosBridgePlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}  // extern "C"
