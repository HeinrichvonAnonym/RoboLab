#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace robo_lab {

// Loads a recorded /franka_state group from an HDF5 file produced by
// recorder_plugin and serves out per-instant joint targets via linear
// interpolation against the original wall-clock timestamps.
//
// Expected datasets (created by recorder_plugin.cpp):
//   franka_state/joints_position : float64, shape [N, 7]
//   franka_state/timestamps_ns   : int64,   shape [N]
//
// All other fields (velocity, effort, sys_time, ...) are ignored: this stage
// only replays joint positions.
class FrankaReplayer {
 public:
  FrankaReplayer() = default;
  ~FrankaReplayer() = default;

  // Open the HDF5 file and load joints_position + timestamps_ns into memory.
  bool load(const std::string& h5_path);

  size_t num_frames() const { return positions_.size(); }
  bool empty() const { return positions_.empty(); }

  // Wall-clock duration of the recording (last - first timestamp), in seconds.
  double duration_s() const;

  // Linearly interpolate the recorded trajectory at progress t in [0, 1],
  // where 0 maps to the first recorded timestamp and 1 to the last.
  // Returns false if the recording is empty.
  bool sample_normalized(double t01, std::array<double, 7>& q_out) const;

  // Linearly interpolate at the given monotonic playback time in nanoseconds
  // measured from the start of the recording (i.e. t_play_ns == 0 returns the
  // first frame). Values past the end clamp to the last frame.
  bool sample_at(int64_t t_play_ns, std::array<double, 7>& q_out) const;

  // Raw access if the caller wants to pace by recorded timestamps directly.
  const std::vector<std::array<double, 7>>& positions() const { return positions_; }
  const std::vector<int64_t>& timestamps_ns() const { return timestamps_ns_; }

 private:
  // joints_position[i] for i in [0, N).
  std::vector<std::array<double, 7>> positions_;
  // Wall-clock recorder timestamps, monotonically increasing within a recording.
  std::vector<int64_t> timestamps_ns_;
};

}  // namespace robo_lab
