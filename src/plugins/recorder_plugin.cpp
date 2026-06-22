#include "plugins/recorder_plugin.h"

#include "plugin_interface.h"

#include "demo_inference.pb.h"
#include "franka.pb.h"
#include "kinect.pb.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include <hdf5.h>
#include <yaml-cpp/yaml.h>

namespace robo_lab {

namespace {

int64_t now_ns_wall() {
  const auto now = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

hid_t OpenOrCreateGroup(hid_t file_id, const std::string& group_name) {
  if (H5Lexists(file_id, group_name.c_str(), H5P_DEFAULT) > 0) {
    return H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);
  }
  return H5Gcreate2(file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

hid_t OpenOrCreateDataset1D(hid_t group_id, const char* name, hid_t type, hsize_t chunk_hint) {
  if (H5Lexists(group_id, name, H5P_DEFAULT) > 0) {
    return H5Dopen2(group_id, name, H5P_DEFAULT);
  }

  hsize_t dims[1] = {0};
  hsize_t max_dims[1] = {H5S_UNLIMITED};
  hid_t space = H5Screate_simple(1, dims, max_dims);
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t chunk[1] = {std::max<hsize_t>(1, std::min<hsize_t>(chunk_hint, 1024))};
  H5Pset_chunk(dcpl, 1, chunk);
  hid_t ds = H5Dcreate2(group_id, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
  H5Pclose(dcpl);
  H5Sclose(space);
  return ds;
}

hid_t OpenOrCreateDataset2D(hid_t group_id, const char* name, hid_t type, hsize_t ncols,
                            hsize_t chunk_rows_hint) {
  if (H5Lexists(group_id, name, H5P_DEFAULT) > 0) {
    return H5Dopen2(group_id, name, H5P_DEFAULT);
  }

  hsize_t dims[2] = {0, ncols};
  hsize_t max_dims[2] = {H5S_UNLIMITED, ncols};
  hid_t space = H5Screate_simple(2, dims, max_dims);
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t chunk[2] = {std::max<hsize_t>(1, std::min<hsize_t>(chunk_rows_hint, 256)), ncols};
  H5Pset_chunk(dcpl, 2, chunk);
  hid_t ds = H5Dcreate2(group_id, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
  H5Pclose(dcpl);
  H5Sclose(space);
  return ds;
}

template <typename T>
bool AppendDataset1D(hid_t group_id, const char* name, hid_t type, const std::vector<T>& data) {
  hid_t ds = OpenOrCreateDataset1D(group_id, name, type, data.size());
  if (ds < 0) {
    return false;
  }

  hid_t old_space = H5Dget_space(ds);
  hsize_t old_dims[1] = {0};
  H5Sget_simple_extent_dims(old_space, old_dims, nullptr);
  H5Sclose(old_space);

  if (data.empty()) {
    H5Dclose(ds);
    return true;
  }

  hsize_t new_dims[1] = {old_dims[0] + data.size()};
  H5Dset_extent(ds, new_dims);
  hid_t file_space = H5Dget_space(ds);
  hsize_t start[1] = {old_dims[0]};
  hsize_t count[1] = {data.size()};
  H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr);
  hid_t mem_space = H5Screate_simple(1, count, nullptr);
  H5Dwrite(ds, type, mem_space, file_space, H5P_DEFAULT, data.data());
  H5Sclose(mem_space);
  H5Sclose(file_space);
  H5Dclose(ds);
  return true;
}

template <typename T>
bool AppendDataset2D(hid_t group_id, const char* name, hid_t type, const std::vector<T>& data,
                     hsize_t nrows, hsize_t ncols) {
  if (nrows == 0 || ncols == 0) return true;
  hid_t ds = OpenOrCreateDataset2D(group_id, name, type, ncols, nrows);
  if (ds < 0) {
    return false;
  }

  hid_t old_space = H5Dget_space(ds);
  hsize_t old_dims[2] = {0, ncols};
  H5Sget_simple_extent_dims(old_space, old_dims, nullptr);
  H5Sclose(old_space);

  hsize_t new_dims[2] = {old_dims[0] + nrows, ncols};
  H5Dset_extent(ds, new_dims);
  hid_t file_space = H5Dget_space(ds);
  hsize_t start[2] = {old_dims[0], 0};
  hsize_t count[2] = {nrows, ncols};
  H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr);
  hid_t mem_space = H5Screate_simple(2, count, nullptr);
  H5Dwrite(ds, type, mem_space, file_space, H5P_DEFAULT, data.data());
  H5Sclose(mem_space);
  H5Sclose(file_space);
  H5Dclose(ds);
  return true;
}

hsize_t Dataset1DSize(hid_t group_id, const char* name) {
  if (H5Lexists(group_id, name, H5P_DEFAULT) <= 0) {
    return 0;
  }
  hid_t ds = H5Dopen2(group_id, name, H5P_DEFAULT);
  if (ds < 0) {
    return 0;
  }
  hid_t space = H5Dget_space(ds);
  hsize_t dims[1] = {0};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  H5Sclose(space);
  H5Dclose(ds);
  return dims[0];
}

}  // namespace

void RecorderPlugin::subscribe_callback(const std::string& key, const std::string& payload) {
  const int64_t t_ns = now_ns_wall();

  {
    std::lock_guard<std::mutex> lk(buffer_mutex_);
    if (!accepting_) {
      return;
    }

    if (!anchor_seen_ && key != anchor_topic_) {
      return;
    }

    const float hz = std::max(record_frequency_, 0.1f);
    const int64_t min_interval_ns = static_cast<int64_t>(1e9f / hz);
    int64_t& last = last_record_ns_by_topic_[key];
    if (last != 0 && (t_ns - last) < min_interval_ns) {
      return;
    }
    last = t_ns;

    if (key == anchor_topic_) {
      anchor_seen_ = true;
      ++anchor_count_since_flush_;
    }

    RecordedMessage msg;
    msg.topic = key;
    msg.payload = payload;
    msg.timestamp_ns = t_ns;
    buffer_.push_back(std::move(msg));

    if (key == anchor_topic_ && anchor_count_since_flush_ >= flush_anchor_count_) {
      anchor_flush_requested_ = true;
      anchor_flush_boundary_ns_ = t_ns;
      anchor_count_since_flush_ = 0;
    }
  }
}

void RecorderPlugin::next_record_callback(const std::string& key, const std::string& payload) {
  (void)key;
  (void)payload;
  std::cout << "recorder_plugin: next_record requested\n";
  std::lock_guard<std::mutex> flush_lk(flush_mutex_);
  flush_all();
  rotate_record_file();
}

bool RecorderPlugin::initialize(const std::string& config_path) {
  config_path_ = config_path;
  message_system_ = std::make_unique<MessageSystem>();
  message_system_->initialize();

  const YAML::Node root = YAML::LoadFile(config_path_);
  if (!root) {
    std::cerr << "recorder_plugin: failed to load yaml: " << config_path_ << "\n";
    return false;
  }

  if (root["topic"]) {
    if (root["topic"].IsSequence()) {
      for (const auto& topic_node : root["topic"]) {
        topics_.push_back(topic_node.as<std::string>());
      }
    }
  }

  if (topics_.empty()) {
    std::cerr << "recorder_plugin: no topics specified in config\n";
    return false;
  }

  if (root["record_frequency"]) {
    record_frequency_ = root["record_frequency"].as<float>();
  }

  if (root["anchor_topic"]) {
    anchor_topic_ = root["anchor_topic"].as<std::string>();
  }

  if (root["flush_anchor_count"]) {
    const int count = root["flush_anchor_count"].as<int>();
    if (count > 0) {
      flush_anchor_count_ = static_cast<size_t>(count);
    }
  }

  if (root["next_record_topic"]) {
    next_record_topic_ = root["next_record_topic"].as<std::string>();
  }

  if (root["data_dir"]) {
    data_dir_ = root["data_dir"].as<std::string>();
  }
#ifdef ROBOLAB_RECORDER_DEFAULT_DATA_DIR
  else {
    data_dir_ = ROBOLAB_RECORDER_DEFAULT_DATA_DIR;
  }
#endif

  if (root["topic_proto"] && root["topic_proto"].IsMap()) {
    for (const auto& it : root["topic_proto"]) {
      topic_proto_[it.first.as<std::string>()] = it.second.as<std::string>();
    }
  }

  std::cout << "recorder_plugin: initialized, topics=" << topics_.size()
            << " topic_proto_entries=" << topic_proto_.size()
            << " record_frequency_hz=" << record_frequency_
            << " anchor_topic=" << anchor_topic_
            << " flush_anchor_count=" << flush_anchor_count_
            << " next_record_topic=" << next_record_topic_
            << " data_dir=" << data_dir_ << "\n";
  return true;
}

void RecorderPlugin::run() {
  stop_ = false;
  accepting_ = true;
  anchor_seen_ = false;
  anchor_count_since_flush_ = 0;
  anchor_flush_requested_ = false;
  anchor_flush_boundary_ns_ = 0;
  current_h5_path_.clear();
  last_record_ns_by_topic_.clear();
  running_ = true;

  if (!message_system_ || !message_system_->is_open()) {
    std::cerr << "recorder_plugin: Zenoh not open; subscriptions may fail\n";
  }

  for (const auto& topic : topics_) {
    message_system_->subscribe(
        topic, [this](const std::string& key, const std::string& payload) {
          subscribe_callback(key, payload);
        });
    std::cout << "recorder_plugin: subscribed to '" << topic << "'\n";
  }

  message_system_->subscribe(
      next_record_topic_, [this](const std::string& key, const std::string& payload) {
        next_record_callback(key, payload);
      });
  std::cout << "recorder_plugin: subscribed to control topic '" << next_record_topic_ << "'\n";
  std::cout << "recorder_plugin: waiting for anchor topic '" << anchor_topic_ << "'\n";

  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    flush_anchor_if_requested();
  }

  {
    std::lock_guard<std::mutex> lk(buffer_mutex_);
    accepting_ = false;
  }
  {
    std::lock_guard<std::mutex> flush_lk(flush_mutex_);
    flush_all();
  }

  if (message_system_) {
    message_system_->close();
  }
  running_ = false;
}

void RecorderPlugin::stop() {
  if (stop_.exchange(true)) {
    return;
  }
  std::cout << "recorder_plugin: stopping\n";
}

std::string RecorderPlugin::make_h5_path() const {
  std::error_code ec;
  std::filesystem::create_directories(data_dir_, ec);

  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  const std::tm* lt = std::localtime(&t);

  std::ostringstream filename;
  filename << data_dir_;
  if (!data_dir_.empty() && data_dir_.back() != '/') {
    filename << '/';
  }
  filename << "recording_";
  if (lt) {
    filename << (1900 + lt->tm_year) << std::setw(2) << std::setfill('0') << (1 + lt->tm_mon)
             << std::setw(2) << std::setfill('0') << lt->tm_mday << "_" << std::setw(2)
             << std::setfill('0') << lt->tm_hour << std::setw(2) << std::setfill('0') << lt->tm_min
             << std::setw(2) << std::setfill('0') << lt->tm_sec;
  } else {
    filename << "unknown";
  }
  // Unique suffix so two saves in the same wall-clock second cannot clobber each other.
  const int64_t epoch_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  filename << "_" << epoch_ms << ".h5";
  return filename.str();
}

void RecorderPlugin::rotate_record_file() {
  std::lock_guard<std::mutex> file_lk(file_mutex_);
  current_h5_path_ = make_h5_path();

  hid_t file_id = H5Fcreate(current_h5_path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0) {
    std::cerr << "recorder_plugin: failed to create new HDF5 file: " << current_h5_path_ << "\n";
    current_h5_path_.clear();
    return;
  }
  H5Fclose(file_id);
  std::cout << "recorder_plugin: next recording file: " << current_h5_path_ << "\n";
}

void RecorderPlugin::flush_anchor_if_requested() {
  std::lock_guard<std::mutex> flush_lk(flush_mutex_);

  std::deque<RecordedMessage> snapshot;
  {
    std::lock_guard<std::mutex> lk(buffer_mutex_);
    if (!anchor_flush_requested_) {
      return;
    }

    std::deque<RecordedMessage> keep;
    while (!buffer_.empty()) {
      if (buffer_.front().timestamp_ns <= anchor_flush_boundary_ns_) {
        snapshot.push_back(std::move(buffer_.front()));
      } else {
        keep.push_back(std::move(buffer_.front()));
      }
      buffer_.pop_front();
    }
    buffer_.swap(keep);
    anchor_flush_requested_ = false;
    anchor_flush_boundary_ns_ = 0;
  }

  if (!snapshot.empty()) {
    append_hdf5(snapshot);
  }
}

void RecorderPlugin::flush_all() {
  std::deque<RecordedMessage> snapshot;
  {
    std::lock_guard<std::mutex> lk(buffer_mutex_);
    snapshot.swap(buffer_);
    anchor_count_since_flush_ = 0;
    anchor_flush_requested_ = false;
    anchor_flush_boundary_ns_ = 0;
  }

  if (!snapshot.empty()) {
    append_hdf5(snapshot);
  }
}

void RecorderPlugin::append_hdf5(const std::deque<RecordedMessage>& messages) {
  if (messages.empty()) {
    return;
  }

  std::lock_guard<std::mutex> file_lk(file_mutex_);
  if (current_h5_path_.empty()) {
    current_h5_path_ = make_h5_path();
  }

  const bool exists = std::filesystem::exists(current_h5_path_);
  hid_t file_id = exists ? H5Fopen(current_h5_path_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT)
                         : H5Fcreate(current_h5_path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0) {
    std::cerr << "recorder_plugin: failed to open HDF5 file: " << current_h5_path_ << "\n";
    return;
  }

  for (const auto& topic : topics_) {
    std::string group_name = topic;
    std::replace(group_name.begin(), group_name.end(), '/', '_');

    hid_t group_id = OpenOrCreateGroup(file_id, group_name);
    if (group_id < 0) {
      std::cerr << "recorder_plugin: failed to create group: " << group_name << "\n";
      continue;
    }

    auto it_proto = topic_proto_.find(topic);
    const bool has_proto = (it_proto != topic_proto_.end());
    const std::string& proto_name = has_proto ? it_proto->second : std::string{};

    std::vector<RecordedMessage> topic_msgs;
    for (const auto& msg : messages) {
      if (msg.topic == topic) {
        topic_msgs.push_back(msg);
      }
    }

    if (topic_msgs.empty()) {
      H5Gclose(group_id);
      continue;
    }

    std::vector<int64_t> timestamps;
    timestamps.reserve(topic_msgs.size());
    for (const auto& msg : topic_msgs) {
      timestamps.push_back(msg.timestamp_ns);
    }
    AppendDataset1D(group_id, "timestamps_ns", H5T_NATIVE_INT64, timestamps);

    if (proto_name == "franka.RobotObservation") {
      std::vector<uint32_t> type_vec;
      std::vector<uint32_t> sequence_vec;
      std::vector<float> sys_time_vec;
      std::vector<double> joint_pos_vec;
      std::vector<double> joint_vel_vec;
      std::vector<double> joint_eff_vec;

      type_vec.reserve(topic_msgs.size());
      sequence_vec.reserve(topic_msgs.size());
      sys_time_vec.reserve(topic_msgs.size());
      joint_pos_vec.reserve(topic_msgs.size() * 7);
      joint_vel_vec.reserve(topic_msgs.size() * 7);
      joint_eff_vec.reserve(topic_msgs.size() * 7);

      for (const auto& msg : topic_msgs) {
        franka::RobotObservation obs;
        if (!obs.ParseFromString(msg.payload)) {
          type_vec.push_back(0);
          sequence_vec.push_back(0);
          sys_time_vec.push_back(0.0f);
          for (int i = 0; i < 7; ++i) {
            joint_pos_vec.push_back(0.0);
            joint_vel_vec.push_back(0.0);
            joint_eff_vec.push_back(0.0);
          }
          continue;
        }
        type_vec.push_back(static_cast<uint32_t>(obs.type()));
        sequence_vec.push_back(obs.sequence());
        sys_time_vec.push_back(obs.sys_time());

        int joint_count = obs.joints_size();
        for (int i = 0; i < 7; ++i) {
          if (i < joint_count) {
            const auto& j = obs.joints(i);
            joint_pos_vec.push_back(j.position());
            joint_vel_vec.push_back(j.velocity());
            joint_eff_vec.push_back(j.effort());
          } else {
            joint_pos_vec.push_back(0.0);
            joint_vel_vec.push_back(0.0);
            joint_eff_vec.push_back(0.0);
          }
        }
      }

      AppendDataset1D(group_id, "type", H5T_NATIVE_UINT32, type_vec);
      AppendDataset1D(group_id, "sequence", H5T_NATIVE_UINT32, sequence_vec);
      AppendDataset1D(group_id, "sys_time", H5T_NATIVE_FLOAT, sys_time_vec);
      AppendDataset2D(group_id, "joints_position", H5T_NATIVE_DOUBLE, joint_pos_vec, topic_msgs.size(), 7);
      AppendDataset2D(group_id, "joints_velocity", H5T_NATIVE_DOUBLE, joint_vel_vec, topic_msgs.size(), 7);
      AppendDataset2D(group_id, "joints_effort", H5T_NATIVE_DOUBLE, joint_eff_vec, topic_msgs.size(), 7);

    } else if (proto_name == "franka.RobotCommand") {
      std::vector<uint32_t> type_vec;
      std::vector<uint32_t> sequence_vec;
      std::vector<float> sys_time_vec;
      std::vector<double> cmd_pos_vec;
      std::vector<double> cmd_vel_vec;
      std::vector<double> cmd_eff_vec;

      type_vec.reserve(topic_msgs.size());
      sequence_vec.reserve(topic_msgs.size());
      sys_time_vec.reserve(topic_msgs.size());
      cmd_pos_vec.reserve(topic_msgs.size() * 7);
      cmd_vel_vec.reserve(topic_msgs.size() * 7);
      cmd_eff_vec.reserve(topic_msgs.size() * 7);

      for (const auto& msg : topic_msgs) {
        franka::RobotCommand cmd;
        if (!cmd.ParseFromString(msg.payload)) {
          type_vec.push_back(0);
          sequence_vec.push_back(0);
          sys_time_vec.push_back(0.0f);
          for (int i = 0; i < 7; ++i) {
            cmd_pos_vec.push_back(0.0);
            cmd_vel_vec.push_back(0.0);
            cmd_eff_vec.push_back(0.0);
          }
          continue;
        }
        type_vec.push_back(static_cast<uint32_t>(cmd.type()));
        sequence_vec.push_back(cmd.sequence());
        sys_time_vec.push_back(cmd.sys_time());

        int joint_count = cmd.joints_size();
        for (int i = 0; i < 7; ++i) {
          if (i < joint_count) {
            const auto& j = cmd.joints(i);
            cmd_pos_vec.push_back(j.position());
            cmd_vel_vec.push_back(j.velocity());
            cmd_eff_vec.push_back(j.effort());
          } else {
            cmd_pos_vec.push_back(0.0);
            cmd_vel_vec.push_back(0.0);
            cmd_eff_vec.push_back(0.0);
          }
        }
      }

      AppendDataset1D(group_id, "type", H5T_NATIVE_UINT32, type_vec);
      AppendDataset1D(group_id, "sequence", H5T_NATIVE_UINT32, sequence_vec);
      AppendDataset1D(group_id, "sys_time", H5T_NATIVE_FLOAT, sys_time_vec);
      AppendDataset2D(group_id, "joints_position", H5T_NATIVE_DOUBLE, cmd_pos_vec, topic_msgs.size(), 7);
      AppendDataset2D(group_id, "joints_velocity", H5T_NATIVE_DOUBLE, cmd_vel_vec, topic_msgs.size(), 7);
      AppendDataset2D(group_id, "joints_effort", H5T_NATIVE_DOUBLE, cmd_eff_vec, topic_msgs.size(), 7);

    } else if (proto_name == "franka.CartesianDPoseCmd") {
      // Each cartesian delta-pose command stores six floats. We pack them
      // into a single (N, 6) float dataset for compactness and easy slicing
      // alongside the timestamps_ns array written above.
      constexpr int kCols = 6;
      std::vector<float> values_vec;
      values_vec.reserve(topic_msgs.size() * kCols);

      for (const auto& msg : topic_msgs) {
        franka::CartesianDPoseCmd dpose;
        if (!dpose.ParseFromString(msg.payload)) {
          for (int i = 0; i < kCols; ++i) {
            values_vec.push_back(0.0f);
          }
          continue;
        }
        values_vec.push_back(dpose.dx());
        values_vec.push_back(dpose.dy());
        values_vec.push_back(dpose.dz());
        values_vec.push_back(dpose.droll());
        values_vec.push_back(dpose.dpitch());
        values_vec.push_back(dpose.dyaw());
      }

      AppendDataset2D(group_id, "values", H5T_NATIVE_FLOAT, values_vec, topic_msgs.size(), kCols);

    } else if (proto_name == "demo_inference.Observation") {
      constexpr int kValuesCols = 19;
      std::vector<double> values_vec;
      values_vec.reserve(topic_msgs.size() * kValuesCols);

      for (const auto& msg : topic_msgs) {
        demo_inference::Observation obs;
        if (!obs.ParseFromString(msg.payload)) {
          for (int i = 0; i < kValuesCols; ++i) {
            values_vec.push_back(0.0);
          }
          continue;
        }
        for (int i = 0; i < kValuesCols; ++i) {
          if (i < obs.values_size()) {
            values_vec.push_back(static_cast<double>(obs.values(i)));
          } else {
            values_vec.push_back(0.0);
          }
        }
      }

      AppendDataset2D(group_id, "values", H5T_NATIVE_DOUBLE, values_vec, topic_msgs.size(), kValuesCols);

    } else if (proto_name == "kinect.rgbImage") {
      std::vector<int32_t> width_vec;
      std::vector<int32_t> height_vec;
      std::vector<int32_t> channels_vec;
      std::vector<int32_t> type_vec;

      width_vec.reserve(topic_msgs.size());
      height_vec.reserve(topic_msgs.size());
      channels_vec.reserve(topic_msgs.size());
      type_vec.reserve(topic_msgs.size());

      const uint64_t base_image_offset = static_cast<uint64_t>(Dataset1DSize(group_id, "image_data"));
      std::vector<uint64_t> img_offsets;
      std::vector<uint64_t> img_lengths;
      std::vector<char> img_blob;
      uint64_t current_offset = base_image_offset;

      for (const auto& msg : topic_msgs) {
        kinect::rgbImage img;
        if (!img.ParseFromString(msg.payload)) {
          width_vec.push_back(0);
          height_vec.push_back(0);
          channels_vec.push_back(0);
          type_vec.push_back(0);
          img_offsets.push_back(current_offset);
          img_lengths.push_back(0);
          continue;
        }
        width_vec.push_back(img.width());
        height_vec.push_back(img.height());
        channels_vec.push_back(img.channels());
        type_vec.push_back(img.type());

        img_offsets.push_back(current_offset);
        img_lengths.push_back(static_cast<uint64_t>(img.image().size()));
        img_blob.insert(img_blob.end(), img.image().begin(), img.image().end());
        current_offset += img.image().size();
      }

      AppendDataset1D(group_id, "width", H5T_NATIVE_INT32, width_vec);
      AppendDataset1D(group_id, "height", H5T_NATIVE_INT32, height_vec);
      AppendDataset1D(group_id, "channels", H5T_NATIVE_INT32, channels_vec);
      AppendDataset1D(group_id, "format_type", H5T_NATIVE_INT32, type_vec);
      AppendDataset1D(group_id, "image_data", H5T_NATIVE_CHAR, img_blob);
      AppendDataset1D(group_id, "image_offsets", H5T_NATIVE_UINT64, img_offsets);
      AppendDataset1D(group_id, "image_lengths", H5T_NATIVE_UINT64, img_lengths);
    }

    std::cout << "recorder_plugin: wrote " << topic_msgs.size() << " samples for topic '" << topic
              << "' (group " << group_name << ")\n";

    H5Gclose(group_id);
  }

  H5Fflush(file_id, H5F_SCOPE_GLOBAL);
  H5Fclose(file_id);

  std::cout << "recorder_plugin: HDF5 appended: " << current_h5_path_
            << " (messages=" << messages.size() << ")\n";
}

}  // namespace robo_lab

extern "C" {

ROBO_LAB_PLUGIN_EXPORT robo_lab::Plugin* robo_lab_plugin_create() {
  return new robo_lab::RecorderPlugin();
}

ROBO_LAB_PLUGIN_EXPORT void robo_lab_plugin_destroy(robo_lab::Plugin* plugin) {
  delete plugin;
}

}
