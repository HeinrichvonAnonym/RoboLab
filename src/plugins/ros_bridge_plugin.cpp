#include "plugins/ros_bridge_plugin.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "franka.pb.h"
#include "imu.pb.h"
#include "kinect.pb.h"
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/byte_multi_array.hpp>
#include <yaml-cpp/yaml.h>

namespace robo_lab {
namespace {

// Preserve ROS 1 NodeHandle("~") resolution for relative topic names.
std::string private_topic(const std::string& name) {
  return (!name.empty() && name.front() != '/' && name.front() != '~') ? "~/" + name : name;
}

struct RosPublisher {
  rclcpp::PublisherBase::SharedPtr publisher;
  rclcpp::Clock::SharedPtr clock;

  template<class Message>
  void publish(const Message& message) const {
    std::static_pointer_cast<rclcpp::Publisher<Message>>(publisher)->publish(message);
  }
};

bool publish_zenoh_to_ros(const std::string& ros_type,
                          const std::vector<std::string>& joint_names,
                          const std::string& frame,
                          const std::string& payload,
                          const RosPublisher& pub) {
  const std::string frame_id = frame.empty() ? std::string("panda_link0") : frame;
  if (ros_type == "sensor_msgs/JointState") {
    sensor_msgs::msg::JointState out;
    out.header.stamp = pub.clock->now();

    franka::RobotObservation obs;
    if (obs.ParseFromString(payload) && obs.joints_size() > 0) {
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
    sensor_msgs::msg::Image out;
    out.header.stamp = pub.clock->now();
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
    geometry_msgs::msg::PoseStamped out;
    out.header.stamp = pub.clock->now();
    out.header.frame_id = frame_id;
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

  if (ros_type == "geometry_msgs/PoseArray") {
    franka::RobotLinkTransforms link_msg;
    if (!link_msg.ParseFromString(payload)) {
      return false;
    }
    geometry_msgs::msg::PoseArray out;
    out.header.stamp = pub.clock->now();
    out.header.frame_id = frame_id;
    out.poses.reserve(static_cast<size_t>(link_msg.transforms_size()));
    for (const auto& sp : link_msg.transforms()) {
      geometry_msgs::msg::Pose p;
      p.position.x = sp.pose().pos().x();
      p.position.y = sp.pose().pos().y();
      p.position.z = sp.pose().pos().z();
      p.orientation.w = sp.pose().rot().w();
      p.orientation.x = sp.pose().rot().x();
      p.orientation.y = sp.pose().rot().y();
      p.orientation.z = sp.pose().rot().z();
      out.poses.push_back(p);
    }
    pub.publish(out);
    return true;
  }

  if (ros_type == "sensor_msgs/Imu") {
    imu::ImuStamped imu_msg;
    if (!imu_msg.ParseFromString(payload)) {
      return false;
    }
    sensor_msgs::msg::Imu out;
    out.header.stamp = pub.clock->now();
    out.header.frame_id = frame.empty() ? std::string("imu_link") : frame;
    if (imu_msg.has_ahrs()) {
      out.orientation.w = imu_msg.ahrs().quaternion().w();
      out.orientation.x = imu_msg.ahrs().quaternion().x();
      out.orientation.y = imu_msg.ahrs().quaternion().y();
      out.orientation.z = imu_msg.ahrs().quaternion().z();
    }
    if (imu_msg.has_imu()) {
      out.angular_velocity.x = imu_msg.imu().gyroscope().x();
      out.angular_velocity.y = imu_msg.imu().gyroscope().y();
      out.angular_velocity.z = imu_msg.imu().gyroscope().z();
      out.linear_acceleration.x = imu_msg.imu().acceleration().x();
      out.linear_acceleration.y = imu_msg.imu().acceleration().y();
      out.linear_acceleration.z = imu_msg.imu().acceleration().z();
    }
    pub.publish(out);
    return true;
  }

  if (ros_type == "nav_msgs/Odometry") {
    imu::ImuStamped imu_msg;
    if (!imu_msg.ParseFromString(payload)) {
      return false;
    }
    nav_msgs::msg::Odometry out;
    out.header.stamp = pub.clock->now();
    out.header.frame_id = frame.empty() ? std::string("odom") : frame;
    out.child_frame_id = "imu_link";
    if (imu_msg.has_ahrs()) {
      out.pose.pose.orientation.w = imu_msg.ahrs().quaternion().w();
      out.pose.pose.orientation.x = imu_msg.ahrs().quaternion().x();
      out.pose.pose.orientation.y = imu_msg.ahrs().quaternion().y();
      out.pose.pose.orientation.z = imu_msg.ahrs().quaternion().z();
    }
    if (imu_msg.has_imu()) {
      out.twist.twist.angular.x = imu_msg.imu().gyroscope().x();
      out.twist.twist.angular.y = imu_msg.imu().gyroscope().y();
      out.twist.twist.angular.z = imu_msg.imu().gyroscope().z();
    }
    pub.publish(out);
    return true;
  }

  std_msgs::msg::ByteMultiArray raw;
  raw.data.assign(payload.begin(), payload.end());
  pub.publish(raw);
  return true;
}

}  // namespace

struct RosBridgePlugin::Impl {
  rclcpp::Context::SharedPtr context;
  rclcpp::Node::SharedPtr node;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  std::vector<RosPublisher> robo_to_ros_publishers;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> ros_to_robo_subscribers;
};

RosBridgePlugin::RosBridgePlugin() = default;
RosBridgePlugin::~RosBridgePlugin() { cleanup(); }

void RosBridgePlugin::cleanup() {
  // Called after run's executor has stopped, or before it was ever started.
  // Drain Zenoh callbacks while ROS publishers and their context are still alive.
  if (message_system_) message_system_->close();
  message_system_.reset();
  if (impl_) {
    impl_->ros_to_robo_subscribers.clear();
    impl_->robo_to_ros_publishers.clear();
    impl_->executor.reset();
    impl_->node.reset();
    if (impl_->context && impl_->context->is_valid()) impl_->context->shutdown("bridge stopped");
    impl_.reset();
  }
}

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
    // Store a canonical old-style name so the existing conversion dispatch stays intact.
    const auto msg_segment = t.ros_type.find("/msg/");
    if (msg_segment != std::string::npos) t.ros_type.replace(msg_segment, 5, "/");
    if (t.zenoh_msg.empty() || t.ros_msg.empty()) return false;
    if (const auto qos = item["qos"]) {
      if (!qos.IsMap()) return false;
      if (qos["depth"]) {
        const int depth = qos["depth"].as<int>();
        if (depth <= 0) return false;
        t.qos_depth = static_cast<size_t>(depth);
      }
      if (qos["reliability"]) {
        const auto reliability = qos["reliability"].as<std::string>();
        if (reliability != "reliable" && reliability != "best_effort") return false;
        t.best_effort = reliability == "best_effort";
      }
    }
    if (item["frame"]) {
      t.frame = item["frame"].as<std::string>();
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
  cleanup();
  stop_ = false;
  robo_to_ros_topics_.clear();
  ros_to_robo_topics_.clear();
  try {
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

    impl_ = std::make_unique<Impl>();
    impl_->context = std::make_shared<rclcpp::Context>();
    impl_->context->init(0, nullptr); // Does not install global signal handlers.
    rclcpp::NodeOptions node_options;
    node_options.context(impl_->context);
    impl_->node = std::make_shared<rclcpp::Node>("robo_lab_ros_bridge_plugin", node_options);
    rclcpp::ExecutorOptions executor_options;
    executor_options.context = impl_->context;
    impl_->executor = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(executor_options);
    impl_->executor->add_node(impl_->node);

    impl_->robo_to_ros_publishers.reserve(robo_to_ros_topics_.size());
    for (const auto& bridge : robo_to_ros_topics_) {
      RosPublisher pub;
      pub.clock = impl_->node->get_clock();
      auto qos = rclcpp::QoS(rclcpp::KeepLast(bridge.qos_depth)).durability_volatile();
      if (bridge.best_effort) qos.best_effort();
      const auto topic = private_topic(bridge.ros_msg);
      if (bridge.ros_type == "sensor_msgs/JointState") {
        pub.publisher = impl_->node->create_publisher<sensor_msgs::msg::JointState>(topic, qos);
      } else if (bridge.ros_type == "sensor_msgs/Image") {
        pub.publisher = impl_->node->create_publisher<sensor_msgs::msg::Image>(topic, qos);
      } else if (bridge.ros_type == "geometry_msgs/PoseStamped") {
        pub.publisher = impl_->node->create_publisher<geometry_msgs::msg::PoseStamped>(topic, qos);
      } else if (bridge.ros_type == "geometry_msgs/PoseArray") {
        pub.publisher = impl_->node->create_publisher<geometry_msgs::msg::PoseArray>(topic, qos);
      } else if (bridge.ros_type == "sensor_msgs/Imu") {
        pub.publisher = impl_->node->create_publisher<sensor_msgs::msg::Imu>(topic, qos);
      } else if (bridge.ros_type == "nav_msgs/Odometry") {
        pub.publisher = impl_->node->create_publisher<nav_msgs::msg::Odometry>(topic, qos);
      } else {
        pub.publisher = impl_->node->create_publisher<std_msgs::msg::ByteMultiArray>(topic, qos);
        std::cout << "ros_bridge_plugin: unsupported ros_type '" << bridge.ros_type
                  << "' on " << bridge.ros_msg << ", using std_msgs/ByteMultiArray fallback\n";
      }
      impl_->robo_to_ros_publishers.push_back(pub);

      if (!message_system_->subscribe(
          bridge.zenoh_msg, [pub, bridge](const std::string& key, const std::string& payload) {
            try {
              if (!publish_zenoh_to_ros(bridge.ros_type, bridge.joint_names, bridge.frame, payload, pub)) {
                std::cerr << "ros_bridge_plugin: failed to decode payload for " << key << '\n';
              }
            } catch (const std::exception& e) {
              std::cerr << "ros_bridge_plugin: publish failed for " << key << ": " << e.what() << '\n';
            }
          })) {
        throw std::runtime_error("failed to subscribe to Zenoh key: " + bridge.zenoh_msg);
      }
    }

    impl_->ros_to_robo_subscribers.reserve(ros_to_robo_topics_.size());
    for (const auto& bridge : ros_to_robo_topics_) {
      rclcpp::SubscriptionBase::SharedPtr sub;
      auto qos = rclcpp::QoS(rclcpp::KeepLast(bridge.qos_depth)).durability_volatile();
      if (bridge.best_effort) qos.best_effort();
      const auto topic = private_topic(bridge.ros_msg);
      if (bridge.ros_type == "sensor_msgs/JointState") {
        // std::cout << "ros_bridge_plugin: subscribing to " << bridge.ros_msg << std::endl;
        sub = impl_->node->create_subscription<sensor_msgs::msg::JointState>(
            topic, qos,
            [this, bridge](const sensor_msgs::msg::JointState::ConstSharedPtr msg) {
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
        sub = impl_->node->create_subscription<sensor_msgs::msg::Image>(
            topic, qos,
            [this, bridge](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
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
        sub = impl_->node->create_subscription<geometry_msgs::msg::PoseStamped>(
            topic, qos,
            [this, bridge](const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
              franka::StampedPose out;
              out.mutable_pose()->mutable_pos()->set_x(msg->pose.position.x);
              out.mutable_pose()->mutable_pos()->set_y(msg->pose.position.y);
              out.mutable_pose()->mutable_pos()->set_z(msg->pose.position.z);
              out.mutable_pose()->mutable_rot()->set_w(msg->pose.orientation.w);
              out.mutable_pose()->mutable_rot()->set_x(msg->pose.orientation.x);
              out.mutable_pose()->mutable_rot()->set_y(msg->pose.orientation.y);
              out.mutable_pose()->mutable_rot()->set_z(msg->pose.orientation.z);
              out.set_sys_time(rclcpp::Time(msg->header.stamp).seconds());
              std::string payload;
              if (!out.SerializeToString(&payload)) {
                return;
              }
              message_system_->publish(bridge.zenoh_msg, payload);
            });
      } else {
        sub = impl_->node->create_subscription<std_msgs::msg::ByteMultiArray>(
            topic, qos,
            [this, bridge](const std_msgs::msg::ByteMultiArray::ConstSharedPtr msg) {
              std::string payload(msg->data.begin(), msg->data.end());
              message_system_->publish(bridge.zenoh_msg, payload);
            });
        std::cout << "ros_bridge_plugin: unsupported ros_type '" << bridge.ros_type
                  << "' on " << bridge.ros_msg << ", using std_msgs/ByteMultiArray fallback\n";
      }
      impl_->ros_to_robo_subscribers.push_back(sub);
    }

    std::cout << "ros_bridge_plugin: initialized (robo->ros=" << robo_to_ros_topics_.size()
              << ", ros->robo=" << ros_to_robo_topics_.size() << ")\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "ros_bridge_plugin: initialization failed: " << e.what() << '\n';
    cleanup();
    return false;
  }
}

void RosBridgePlugin::run() {
  if (!impl_) return;
  try {
    while (!stop_ && impl_->context->is_valid()) {
      impl_->executor->spin_some(std::chrono::milliseconds(10));
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  } catch (const std::exception& e) {
    std::cerr << "ros_bridge_plugin: executor stopped: " << e.what() << '\n';
  }
  cleanup();
}

void RosBridgePlugin::stop() {
  // No reset in run(): stop-before-run must also be respected.
  stop_ = true;
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
