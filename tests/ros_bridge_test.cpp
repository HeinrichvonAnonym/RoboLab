#include "plugin_test_support.h"
#include "message_system.h"
#include "franka.pb.h"
#include "kinect.pb.h"
#include "imu.pb.h"
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/byte_multi_array.hpp>
#include <chrono>
#include <mutex>
#include <sstream>
using namespace std::chrono_literals;

class BridgeTest : public ::testing::Test {
 protected:
  TestDirectory files;
  TestPlugin bridge{BRIDGE_LIBRARY};
  robo_lab::MessageSystem bus;
  rclcpp::Context::SharedPtr context;
  rclcpp::Node::SharedPtr node;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  std::string prefix = "robolab_test_" + std::to_string(getpid());

  void SetUp() override {
    context = std::make_shared<rclcpp::Context>();
    context->init(0, nullptr);
    node = std::make_shared<rclcpp::Node>("bridge_test", rclcpp::NodeOptions().context(context));
    rclcpp::ExecutorOptions options;
    options.context = context;
    executor = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(options);
    executor->add_node(node);
    bus.initialize();
    ASSERT_TRUE(bus.is_open());
  }
  void TearDown() override {
    bridge.stop();
    bus.close();
    executor.reset();
    node.reset();
    context->shutdown("test complete");
  }
  template<class Message>
  typename Message::SharedPtr receive_ros(const std::string& suffix, const std::string& payload,
                                         bool private_name = false) {
    typename Message::SharedPtr result;
    const auto topic = (private_name ? "/robo_lab_ros_bridge_plugin/" : "/") + prefix + "/" + suffix;
    auto sub = node->create_subscription<Message>(topic, 10,
        [&](typename Message::SharedPtr message) { result = message; });
    const auto deadline = std::chrono::steady_clock::now() + 8s;
    while (!result && std::chrono::steady_clock::now() < deadline) {
      bus.publish(prefix + "/" + suffix, payload);
      executor->spin_some();
      std::this_thread::sleep_for(20ms);
    }
    EXPECT_NE(result, nullptr) << suffix;
    return result;
  }
  template<class Message>
  std::string receive_zenoh(const std::string& suffix, const Message& message) {
    struct Received { std::mutex mutex; bool ready = false; std::string payload; };
    auto result = std::make_shared<Received>();
    EXPECT_TRUE(bus.subscribe(prefix + "/" + suffix, [result](const auto&, const auto& payload) {
      std::lock_guard<std::mutex> lock(result->mutex);
      result->payload = payload;
      result->ready = true;
    }));
    auto pub = node->create_publisher<Message>("/" + prefix + "/" + suffix,
        rclcpp::QoS(10).best_effort());
    const auto deadline = std::chrono::steady_clock::now() + 8s;
    while (std::chrono::steady_clock::now() < deadline) {
      pub->publish(message);
      executor->spin_some();
      std::this_thread::sleep_for(20ms);
      std::lock_guard<std::mutex> lock(result->mutex);
      if (result->ready) return result->payload;
    }
    ADD_FAILURE() << "Zenoh delivery timed out: " << suffix;
    return {};
  }
};

TEST_F(BridgeTest, TypesDirectionsAliasesNamespacesAndQos) {
  std::ostringstream config;
  config << "roboLab_2_ros_topics:\n";
  const std::pair<const char*, const char*> outgoing[] = {
    {"joints", "sensor_msgs/JointState"}, {"array", "sensor_msgs/msg/JointState"},
    {"image", "sensor_msgs/msg/Image"}, {"pose", "geometry_msgs/PoseStamped"},
    {"poses", "geometry_msgs/msg/PoseArray"}, {"imu", "sensor_msgs/Imu"},
    {"odom", "nav_msgs/msg/Odometry"}, {"raw", "custom/raw"}};
  for (const auto& entry : outgoing) {
    const std::string suffix = entry.first;
    config << "  - zenoh_msg: " << prefix << '/' << suffix << "\n    ros_msg: "
           << (suffix == "joints" ? "" : "/") << prefix << '/' << suffix
           << "\n    ros_type: " << entry.second << "\n    frame: test_frame\n"
           << "    joint_name: [joint_a, joint_b]\n";
  }
  config << "ros_2_roboLab_topics:\n";
  for (const auto& entry : {outgoing[0], outgoing[2], outgoing[3], outgoing[7]}) {
    config << "  - zenoh_msg: " << prefix << "/in_" << entry.first
           << "\n    ros_msg: /" << prefix << "/in_" << entry.first
           << "\n    ros_type: " << entry.second
           << "\n    qos: {depth: 3, reliability: best_effort}\n";
  }
  ASSERT_TRUE(bridge.plugin->initialize(files.write("bridge.yaml", config.str())));
  bridge.run();
  std::this_thread::sleep_for(1500ms);

  franka::RobotObservation observation;
  auto* joint = observation.add_joints();
  joint->set_position(1.25); joint->set_velocity(2.5); joint->set_effort(3.75);
  auto joints = receive_ros<sensor_msgs::msg::JointState>("joints", observation.SerializeAsString(), true);
  ASSERT_TRUE(joints); ASSERT_EQ(joints->position.size(), 2u);
  EXPECT_EQ(joints->name[0], "joint_a"); EXPECT_DOUBLE_EQ(joints->position[0], 1.25);
  EXPECT_DOUBLE_EQ(joints->velocity[0], 2.5); EXPECT_DOUBLE_EQ(joints->effort[0], 3.75);
  EXPECT_DOUBLE_EQ(joints->position[1], 0);
  EXPECT_GT(rclcpp::Time(joints->header.stamp).nanoseconds(), 0);
  franka::JointStateArray array;
  array.add_states()->set_position(4.5);
  auto states = receive_ros<sensor_msgs::msg::JointState>("array", array.SerializeAsString());
  ASSERT_TRUE(states); ASSERT_EQ(states->position.size(), 2u); EXPECT_DOUBLE_EQ(states->position[0], 4.5);

  // A truncated protobuf must not be published, and must not poison the session.
  int invalid_count = 0;
  auto invalid_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>("/" + prefix + "/pose", 10,
      [&](const geometry_msgs::msg::PoseStamped&) { ++invalid_count; });
  for (int i = 0; i < 10; ++i) {
    bus.publish(prefix + "/pose", std::string(1, '\x80'));
    executor->spin_some(); std::this_thread::sleep_for(20ms);
  }
  EXPECT_EQ(invalid_count, 0);
  invalid_sub.reset();

  kinect::rgbImage pixels;
  pixels.set_width(2); pixels.set_height(1); pixels.set_channels(3); pixels.set_step(6);
  pixels.set_image(std::string("\0\1\2\3\4\5", 6));
  auto image = receive_ros<sensor_msgs::msg::Image>("image", pixels.SerializeAsString());
  ASSERT_TRUE(image); EXPECT_EQ(image->encoding, "rgb8"); EXPECT_EQ(image->width, 2u);
  EXPECT_EQ(image->step, 6u); ASSERT_EQ(image->data.size(), 6u); EXPECT_EQ(image->data[5], 5u);

  franka::StampedPose pose;
  pose.mutable_pose()->mutable_pos()->set_x(0.25);
  pose.mutable_pose()->mutable_rot()->set_w(1.0);
  auto ros_pose = receive_ros<geometry_msgs::msg::PoseStamped>("pose", pose.SerializeAsString());
  ASSERT_TRUE(ros_pose); EXPECT_EQ(ros_pose->header.frame_id, "test_frame");
  EXPECT_DOUBLE_EQ(ros_pose->pose.position.x, 0.25); EXPECT_DOUBLE_EQ(ros_pose->pose.orientation.w, 1);
  franka::RobotLinkTransforms transforms;
  *transforms.add_transforms() = pose;
  auto poses = receive_ros<geometry_msgs::msg::PoseArray>("poses", transforms.SerializeAsString());
  ASSERT_TRUE(poses); ASSERT_EQ(poses->poses.size(), 1u); EXPECT_DOUBLE_EQ(poses->poses[0].position.x, 0.25);

  imu::ImuStamped inertial;
  inertial.mutable_ahrs()->mutable_quaternion()->set_w(1);
  inertial.mutable_imu()->mutable_gyroscope()->set_z(2);
  inertial.mutable_imu()->mutable_acceleration()->set_x(3);
  auto imu_msg = receive_ros<sensor_msgs::msg::Imu>("imu", inertial.SerializeAsString());
  ASSERT_TRUE(imu_msg); EXPECT_DOUBLE_EQ(imu_msg->orientation.w, 1);
  EXPECT_DOUBLE_EQ(imu_msg->angular_velocity.z, 2); EXPECT_DOUBLE_EQ(imu_msg->linear_acceleration.x, 3);
  auto odom = receive_ros<nav_msgs::msg::Odometry>("odom", inertial.SerializeAsString());
  ASSERT_TRUE(odom); EXPECT_EQ(odom->child_frame_id, "imu_link");
  EXPECT_EQ(odom->header.frame_id, "test_frame"); EXPECT_DOUBLE_EQ(odom->twist.twist.angular.z, 2);
  const std::string bytes("\0\xff\x7f", 3);
  auto raw = receive_ros<std_msgs::msg::ByteMultiArray>("raw", bytes);
  ASSERT_TRUE(raw); EXPECT_EQ(std::string(raw->data.begin(), raw->data.end()), bytes);

  auto pub_info = node->get_publishers_info_by_topic("/" + prefix + "/pose");
  ASSERT_FALSE(pub_info.empty());
  EXPECT_EQ(pub_info.front().qos_profile().reliability(), rclcpp::ReliabilityPolicy::Reliable);
  EXPECT_EQ(pub_info.front().qos_profile().durability(), rclcpp::DurabilityPolicy::Volatile);
  auto sub_info = node->get_subscriptions_info_by_topic("/" + prefix + "/in_image");
  ASSERT_FALSE(sub_info.empty());
  EXPECT_EQ(sub_info.front().qos_profile().reliability(), rclcpp::ReliabilityPolicy::BestEffort);

  sensor_msgs::msg::JointState input_joints;
  input_joints.position = {0.2, 0.4}; input_joints.velocity = {0.6};
  franka::JointStateArray decoded_joints;
  ASSERT_TRUE(decoded_joints.ParseFromString(receive_zenoh("in_joints", input_joints)));
  ASSERT_EQ(decoded_joints.states_size(), 2);
  EXPECT_DOUBLE_EQ(decoded_joints.states(0).position(), 0.2);
  EXPECT_DOUBLE_EQ(decoded_joints.states(0).velocity(), 0.6);
  EXPECT_DOUBLE_EQ(decoded_joints.states(1).velocity(), 0);
  kinect::rgbImage decoded_image;
  ASSERT_TRUE(decoded_image.ParseFromString(receive_zenoh("in_image", *image)));
  EXPECT_EQ(decoded_image.image(), pixels.image()); EXPECT_EQ(decoded_image.channels(), 3);
  ros_pose->header.stamp.sec = 12; ros_pose->header.stamp.nanosec = 500000000;
  franka::StampedPose decoded_pose;
  ASSERT_TRUE(decoded_pose.ParseFromString(receive_zenoh("in_pose", *ros_pose)));
  EXPECT_FLOAT_EQ(decoded_pose.sys_time(), 12.5f); EXPECT_DOUBLE_EQ(decoded_pose.pose().pos().x(), 0.25);
  EXPECT_EQ(receive_zenoh("in_raw", *raw), bytes);
  bridge.stop();
  bridge.stop();
  // Stopping this plugin must not shut down another ROS context in the process.
  EXPECT_TRUE(context->is_valid());
}

TEST_F(BridgeTest, InvalidConfigurationAndStopBeforeRun) {
  for (const auto& qos : {"{depth: 0}", "{reliability: typo}", "{depth: nope}", "[]"}) {
    const auto config = files.write("invalid.yaml", "roboLab_2_ros_topics:\n  - zenoh_msg: test/key\n"
        "    ros_msg: /test/key\n    qos: " + std::string(qos) + "\n");
    EXPECT_FALSE(bridge.plugin->initialize(config));
  }
  EXPECT_FALSE(bridge.plugin->initialize(files.write("invalid_name.yaml",
      "roboLab_2_ros_topics:\n  - {zenoh_msg: test/key, ros_msg: 'invalid topic'}\n")));
  // This fails after ROS publishers were created, exercising partial initialization cleanup.
  EXPECT_FALSE(bridge.plugin->initialize(files.write("invalid_zenoh.yaml",
      "roboLab_2_ros_topics:\n  - {zenoh_msg: 'invalid//key', ros_msg: /valid_topic}\n")));
  EXPECT_TRUE(context->is_valid());
  ASSERT_TRUE(bridge.plugin->initialize(files.write("empty.yaml",
      "robo_lab_2_ros_topics: []\nros_2_robo_lab_topics: []\n")));
  bridge.plugin->stop();
  bridge.plugin->stop();
  bridge.run();
  bridge.worker.join();
  EXPECT_TRUE(context->is_valid());
}
