#include "plugin_test_support.h"
#include "message_system.h"
#include "plugins/franka_replayer.h"
#include "franka.pb.h"
#include <gtest/gtest.h>
#include <hdf5.h>
#include <chrono>
#include <mutex>
#include <sstream>
using namespace std::chrono_literals;

TEST(Recording, SyntheticEpisodeAndReplay) {
  TestDirectory files;
  const std::string prefix = "robolab_test_" + std::to_string(getpid());
  // The replayer's existing HDF5 contract uses /franka_state. A leading slash
  // becomes '_' in recorder groups, so use an isolated key then rename the group.
  const auto state_key = prefix + "/state";
  std::ostringstream config;
  config << "topic: [" << state_key << "]\n"
         << "topic_proto: {" << state_key << ": franka.RobotObservation}\n"
         << "state_machine_topic: " << prefix << "/mode\n"
         << "next_record_topic: " << prefix << "/checkpoint\n"
         << "anchor_topic: " << state_key << "\nrecord_frequency: 100\n"
         << "recording_state: inference\nflush_anchor_count: 2\ndata_dir: " << files.path.string() << '\n';
  TestPlugin recorder(RECORDER_LIBRARY);
  ASSERT_TRUE(recorder.plugin->initialize(files.write("recorder.yaml", config.str())));
  recorder.run();
  robo_lab::MessageSystem bus;
  bus.initialize();
  ASSERT_TRUE(bus.is_open());
  std::this_thread::sleep_for(1500ms);
  for (int i = 0; i < 5; ++i) { bus.publish(prefix + "/mode", "inference"); std::this_thread::sleep_for(100ms); }
  for (int i = 0; i < 10; ++i) {
    franka::RobotObservation state;
    for (int j = 0; j < 7; ++j) state.add_joints()->set_position(j + i * 0.01);
    ASSERT_TRUE(bus.publish(state_key, state.SerializeAsString()));
    std::this_thread::sleep_for(50ms);
  }
  bus.publish(prefix + "/mode", "standby");
  std::this_thread::sleep_for(200ms);
  recorder.stop();
  std::filesystem::path recording;
  for (const auto& entry : std::filesystem::directory_iterator(files.path)) {
    ASSERT_NE(entry.path().extension(), ".inprogress");
    if (entry.path().extension() == ".h5") recording = entry.path();
  }
  ASSERT_FALSE(recording.empty());
  // Keep live test keys isolated while exercising the existing fixed replay schema.
  hid_t h5 = H5Fopen(recording.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  ASSERT_GE(h5, 0);
  const auto group = prefix + "_state";
  const auto rename = H5Lmove(h5, group.c_str(), h5, "franka_state", H5P_DEFAULT, H5P_DEFAULT);
  H5Fclose(h5);
  ASSERT_GE(rename, 0);
  robo_lab::FrankaReplayer trajectory;
  ASSERT_TRUE(trajectory.load(recording.string()));
  ASSERT_GE(trajectory.num_frames(), 5u);
  std::array<double, 7> first, last, middle;
  ASSERT_TRUE(trajectory.sample_normalized(0, first));
  ASSERT_TRUE(trajectory.sample_normalized(1, last));
  ASSERT_TRUE(trajectory.sample_normalized(0.5, middle));
  EXPECT_NEAR(first[0], 0, 0.011);
  EXPECT_NEAR(last[6], 6.09, 1e-9);
  EXPECT_GT(middle[0], first[0]);
  EXPECT_LT(middle[0], last[0]);

  std::mutex mutex;
  std::vector<franka::RobotCommand> commands;
  ASSERT_TRUE(bus.subscribe(prefix + "/replay", [&](const auto&, const auto& payload) {
    franka::RobotCommand command;
    if (command.ParseFromString(payload)) { std::lock_guard<std::mutex> lock(mutex); commands.push_back(command); }
  }));
  TestPlugin replay(REPLAYER_LIBRARY);
  std::ostringstream replay_config;
  replay_config << "trajectory_file: " << recording.string() << "\ncmd_topic: " << prefix
                << "/replay\npublish_hz: 50\nstartup_delay_s: 1.5\nrate_factor: 1\n";
  ASSERT_TRUE(replay.plugin->initialize(files.write("replay.yaml", replay_config.str())));
  replay.run();
  replay.worker.join();
  std::this_thread::sleep_for(100ms);
  { std::lock_guard<std::mutex> lock(mutex);
    ASSERT_GE(commands.size(), 5u);
    EXPECT_EQ(commands.back().mode(), "position");
    ASSERT_EQ(commands.back().joints_size(), 7);
    EXPECT_DOUBLE_EQ(commands.back().joints(6).position(), last[6]);
  }
  bus.close();
}

TEST(Recording, LegacySevenColumnsAndInvalidWidth) {
  TestDirectory files;
  for (hsize_t width : {7, 6}) {
    const auto file = (files.path / "legacy.h5").string();
    hid_t h5 = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    ASSERT_GE(h5, 0);
    hid_t group = H5Gcreate2(h5, "franka_state", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t shape[2] = {2, width};
    hid_t space = H5Screate_simple(2, shape, nullptr);
    hid_t dataset = H5Dcreate2(group, "joints_position", H5T_NATIVE_DOUBLE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> positions(2 * width);
    for (size_t i = 0; i < positions.size(); ++i) positions[i] = i;
    EXPECT_GE(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions.data()), 0);
    H5Dclose(dataset); H5Sclose(space);
    const hsize_t count = 2;
    space = H5Screate_simple(1, &count, nullptr);
    dataset = H5Dcreate2(group, "timestamps_ns", H5T_NATIVE_INT64, space,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const int64_t timestamps[2] = {1000000000, 2000000000};
    EXPECT_GE(H5Dwrite(dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, timestamps), 0);
    H5Dclose(dataset); H5Sclose(space); H5Gclose(group); H5Fclose(h5);
    robo_lab::FrankaReplayer trajectory;
    if (width == 7) {
      ASSERT_TRUE(trajectory.load(file));
      std::array<double, 7> middle;
      ASSERT_TRUE(trajectory.sample_normalized(0.5, middle));
      EXPECT_DOUBLE_EQ(middle[0], 3.5);
      EXPECT_DOUBLE_EQ(middle[6], 9.5);
    } else {
      EXPECT_FALSE(trajectory.load(file));
    }
  }
}
