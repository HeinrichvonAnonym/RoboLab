#include "plugins/recorder_plugin.h"

#include "plugin_interface.h"

#include "franka.pb.h"
#include "kinect.pb.h"

#include <algorithm>
#include <chrono>
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

void WriteDataset1D(hid_t group_id, const char* name, const std::vector<int64_t>& data) {
  if (data.empty()) return;
  hsize_t dims[1] = {data.size()};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t ds = H5Dcreate2(group_id, name, H5T_NATIVE_INT64, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(ds);
  H5Sclose(space);
}

void WriteDataset1D(hid_t group_id, const char* name, const std::vector<uint32_t>& data) {
  if (data.empty()) return;
  hsize_t dims[1] = {data.size()};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t ds = H5Dcreate2(group_id, name, H5T_NATIVE_UINT32, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(ds, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(ds);
  H5Sclose(space);
}

void WriteDataset1D(hid_t group_id, const char* name, const std::vector<float>& data) {
  if (data.empty()) return;
  hsize_t dims[1] = {data.size()};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t ds = H5Dcreate2(group_id, name, H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(ds);
  H5Sclose(space);
}

void WriteDataset2D(hid_t group_id, const char* name, const std::vector<double>& data, hsize_t nrows, hsize_t ncols) {
  if (data.empty() || nrows == 0 || ncols == 0) return;
  hsize_t dims[2] = {nrows, ncols};
  hid_t space = H5Screate_simple(2, dims, nullptr);
  hid_t ds = H5Dcreate2(group_id, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(ds);
  H5Sclose(space);
}

}  // namespace

void RecorderPlugin::subscribe_callback(const std::string& key, const std::string& payload) {
  const int64_t t_ns = now_ns_wall();

  std::lock_guard<std::mutex> lk(buffer_mutex_);
  if (!accepting_) {
    return;
  }

  const float hz = std::max(record_frequency_, 0.1f);
  const int64_t min_interval_ns = static_cast<int64_t>(1e9f / hz);
  int64_t& last = last_record_ns_by_topic_[key];
  if (last != 0 && (t_ns - last) < min_interval_ns) {
    return;
  }
  last = t_ns;

  RecordedMessage msg;
  msg.topic = key;
  msg.payload = payload;
  msg.timestamp_ns = t_ns;
  buffer_.push_back(std::move(msg));
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

  if (root["save_per_second"]) {
    float sps = root["save_per_second"].as<float>();
    if (sps > 0.0f) {
      save_interval_s_ = sps;
    }
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
            << " save_interval_s=" << save_interval_s_
            << " data_dir=" << data_dir_ << "\n";
  return true;
}

void RecorderPlugin::run() {
  stop_ = false;
  accepting_ = true;
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

  std::cout << "recorder_plugin: recording\n";

  const bool periodic_save = (save_interval_s_ > 0.0f);
  auto last_save = std::chrono::steady_clock::now();

  while (!stop_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (periodic_save) {
      const auto now = std::chrono::steady_clock::now();
      const float elapsed_s = std::chrono::duration<float>(now - last_save).count();
      if (elapsed_s >= save_interval_s_) {
        last_save = now;

        std::deque<RecordedMessage> snapshot;
        {
          std::lock_guard<std::mutex> lk(buffer_mutex_);
          snapshot.swap(buffer_);
        }
        if (!snapshot.empty()) {
          const std::string path = make_h5_path();
          write_hdf5(path, snapshot);
        }
      }
    }
  }

  // No HDF5 write here: shutdown saves were a source of corrupt/partial files under concurrent Zenoh
  // teardown. Data is only persisted on periodic save_interval (see save_per_second in YAML).
  {
    std::lock_guard<std::mutex> lk(buffer_mutex_);
    accepting_ = false;
    buffer_.clear();
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

void RecorderPlugin::write_hdf5(const std::string& h5_path,
                                const std::deque<RecordedMessage>& messages) {
  if (messages.empty()) {
    return;
  }

  // Write to *.h5.part then atomic rename. Prevents readers/tools from opening a truncated file
  // mid-write (common cause of "bad object header version" / deserialize errors).
  const std::string part_path = h5_path + ".part";
  std::error_code fs_ec;
  std::filesystem::remove(part_path, fs_ec);

  hid_t file_id = H5Fcreate(part_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0) {
    std::cerr << "recorder_plugin: failed to create HDF5 temp file: " << part_path << "\n";
    return;
  }

  for (const auto& topic : topics_) {
    std::string group_name = topic;
    std::replace(group_name.begin(), group_name.end(), '/', '_');

    hid_t group_id = H5Gcreate2(file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
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
    WriteDataset1D(group_id, "timestamps_ns", timestamps);

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

      WriteDataset1D(group_id, "type", type_vec);
      WriteDataset1D(group_id, "sequence", sequence_vec);
      WriteDataset1D(group_id, "sys_time", sys_time_vec);
      WriteDataset2D(group_id, "joints_position", joint_pos_vec, topic_msgs.size(), 7);
      WriteDataset2D(group_id, "joints_velocity", joint_vel_vec, topic_msgs.size(), 7);
      WriteDataset2D(group_id, "joints_effort", joint_eff_vec, topic_msgs.size(), 7);

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

      WriteDataset1D(group_id, "type", type_vec);
      WriteDataset1D(group_id, "sequence", sequence_vec);
      WriteDataset1D(group_id, "sys_time", sys_time_vec);
      WriteDataset2D(group_id, "joints_position", cmd_pos_vec, topic_msgs.size(), 7);
      WriteDataset2D(group_id, "joints_velocity", cmd_vel_vec, topic_msgs.size(), 7);
      WriteDataset2D(group_id, "joints_effort", cmd_eff_vec, topic_msgs.size(), 7);

    } else if (proto_name == "kinect.rgbImage") {
      std::vector<int32_t> width_vec;
      std::vector<int32_t> height_vec;
      std::vector<int32_t> channels_vec;
      std::vector<int32_t> type_vec;

      width_vec.reserve(topic_msgs.size());
      height_vec.reserve(topic_msgs.size());
      channels_vec.reserve(topic_msgs.size());
      type_vec.reserve(topic_msgs.size());

      std::vector<uint64_t> img_offsets;
      std::vector<uint64_t> img_lengths;
      std::vector<char> img_blob;
      uint64_t current_offset = 0;

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

      hsize_t dims_n[1] = {topic_msgs.size()};
      hid_t space_n = H5Screate_simple(1, dims_n, nullptr);

      hid_t ds_width = H5Dcreate2(group_id, "width", H5T_NATIVE_INT32, space_n, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(ds_width, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, width_vec.data());
      H5Dclose(ds_width);

      hid_t ds_height = H5Dcreate2(group_id, "height", H5T_NATIVE_INT32, space_n, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(ds_height, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, height_vec.data());
      H5Dclose(ds_height);

      hid_t ds_channels = H5Dcreate2(group_id, "channels", H5T_NATIVE_INT32, space_n, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(ds_channels, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, channels_vec.data());
      H5Dclose(ds_channels);

      hid_t ds_type = H5Dcreate2(group_id, "format_type", H5T_NATIVE_INT32, space_n, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(ds_type, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, type_vec.data());
      H5Dclose(ds_type);

      H5Sclose(space_n);

      hsize_t dims_img[1] = {img_blob.size()};
      hid_t space_img = H5Screate_simple(1, dims_img, nullptr);
      hid_t ds_img = H5Dcreate2(group_id, "image_data", H5T_NATIVE_CHAR, space_img, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (!img_blob.empty()) {
        H5Dwrite(ds_img, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, img_blob.data());
      }
      H5Dclose(ds_img);
      H5Sclose(space_img);

      hsize_t dims_idx[1] = {img_offsets.size()};
      hid_t space_idx = H5Screate_simple(1, dims_idx, nullptr);

      hid_t ds_off = H5Dcreate2(group_id, "image_offsets", H5T_NATIVE_UINT64, space_idx, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(ds_off, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, img_offsets.data());
      H5Dclose(ds_off);

      hid_t ds_len = H5Dcreate2(group_id, "image_lengths", H5T_NATIVE_UINT64, space_idx, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(ds_len, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, img_lengths.data());
      H5Dclose(ds_len);

      H5Sclose(space_idx);
    }

    std::cout << "recorder_plugin: wrote " << topic_msgs.size() << " samples for topic '" << topic
              << "' (group " << group_name << ")\n";

    H5Gclose(group_id);
  }

  if (H5Fclose(file_id) < 0) {
    std::cerr << "recorder_plugin: H5Fclose failed; removing incomplete " << part_path << "\n";
    std::filesystem::remove(part_path, fs_ec);
    return;
  }

  std::filesystem::rename(part_path, h5_path, fs_ec);
  if (fs_ec) {
    std::cerr << "recorder_plugin: rename " << part_path << " -> " << h5_path << ": " << fs_ec.message() << "\n";
    return;
  }

  std::cout << "recorder_plugin: HDF5 saved: " << h5_path << " (messages=" << messages.size() << ")\n";
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
