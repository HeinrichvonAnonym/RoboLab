#include "plugins/franka_replayer.h"

#include <algorithm>
#include <iostream>

#include <hdf5.h>

namespace robo_lab {

namespace {

// Load a 2D float64 dataset of width 7 into row-major std::array<double,7>s.
// Returns false on any HDF5 error or shape mismatch.
bool read_positions_dataset(hid_t file_id,
                            const char* path,
                            std::vector<std::array<double, 7>>* out) {
  out->clear();
  hid_t ds = H5Dopen2(file_id, path, H5P_DEFAULT);
  if (ds < 0) {
    std::cerr << "franka_replayer: dataset not found: " << path << '\n';
    return false;
  }
  hid_t space = H5Dget_space(ds);
  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank != 2) {
    std::cerr << "franka_replayer: " << path << " has rank " << rank << " (expected 2)\n";
    H5Sclose(space);
    H5Dclose(ds);
    return false;
  }
  hsize_t dims[2] = {0, 0};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  H5Sclose(space);

  if (dims[1] != 7) {
    std::cerr << "franka_replayer: " << path << " has " << dims[1] << " cols (expected 7)\n";
    H5Dclose(ds);
    return false;
  }

  std::vector<double> flat(static_cast<size_t>(dims[0]) * 7);
  if (H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data()) < 0) {
    std::cerr << "franka_replayer: H5Dread failed on " << path << '\n';
    H5Dclose(ds);
    return false;
  }
  H5Dclose(ds);

  out->reserve(static_cast<size_t>(dims[0]));
  for (size_t i = 0; i < dims[0]; ++i) {
    std::array<double, 7> q;
    for (int j = 0; j < 7; ++j) {
      q[j] = flat[i * 7 + j];
    }
    out->push_back(q);
  }
  return true;
}

bool read_int64_dataset(hid_t file_id, const char* path, std::vector<int64_t>* out) {
  out->clear();
  hid_t ds = H5Dopen2(file_id, path, H5P_DEFAULT);
  if (ds < 0) {
    std::cerr << "franka_replayer: dataset not found: " << path << '\n';
    return false;
  }
  hid_t space = H5Dget_space(ds);
  const int rank = H5Sget_simple_extent_ndims(space);
  if (rank != 1) {
    std::cerr << "franka_replayer: " << path << " has rank " << rank << " (expected 1)\n";
    H5Sclose(space);
    H5Dclose(ds);
    return false;
  }
  hsize_t dims[1] = {0};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  H5Sclose(space);

  out->resize(static_cast<size_t>(dims[0]));
  if (H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, out->data()) < 0) {
    std::cerr << "franka_replayer: H5Dread failed on " << path << '\n';
    H5Dclose(ds);
    return false;
  }
  H5Dclose(ds);
  return true;
}

}  // namespace

bool FrankaReplayer::load(const std::string& h5_path) {
  positions_.clear();
  timestamps_ns_.clear();

  hid_t file_id = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    std::cerr << "franka_replayer: failed to open " << h5_path << '\n';
    return false;
  }

  bool ok = read_positions_dataset(file_id, "/franka_state/joints_position", &positions_) &&
            read_int64_dataset(file_id, "/franka_state/timestamps_ns", &timestamps_ns_);
  H5Fclose(file_id);
  if (!ok) {
    return false;
  }

  if (positions_.size() != timestamps_ns_.size()) {
    std::cerr << "franka_replayer: positions (" << positions_.size()
              << ") and timestamps (" << timestamps_ns_.size() << ") size mismatch\n";
    positions_.clear();
    timestamps_ns_.clear();
    return false;
  }
  if (positions_.empty()) {
    std::cerr << "franka_replayer: empty franka_state group in " << h5_path << '\n';
    return false;
  }

  // Recorder timestamps should already be monotonic, but guard against gaps so
  // sample_at()'s upper_bound search is well-defined.
  for (size_t i = 1; i < timestamps_ns_.size(); ++i) {
    if (timestamps_ns_[i] < timestamps_ns_[i - 1]) {
      std::cerr << "franka_replayer: non-monotonic timestamp at index " << i
                << " (will be clamped to previous)\n";
      timestamps_ns_[i] = timestamps_ns_[i - 1];
    }
  }

  std::cout << "franka_replayer: loaded " << positions_.size()
            << " frames from " << h5_path
            << " (duration=" << duration_s() << "s)\n";
  return true;
}

double FrankaReplayer::duration_s() const {
  if (timestamps_ns_.size() < 2) {
    return 0.0;
  }
  const int64_t span_ns = timestamps_ns_.back() - timestamps_ns_.front();
  return static_cast<double>(span_ns) * 1e-9;
}

bool FrankaReplayer::sample_normalized(double t01, std::array<double, 7>& q_out) const {
  if (positions_.empty()) {
    return false;
  }
  t01 = std::clamp(t01, 0.0, 1.0);
  const int64_t total = timestamps_ns_.back() - timestamps_ns_.front();
  const int64_t t_play_ns = static_cast<int64_t>(t01 * static_cast<double>(total));
  return sample_at(t_play_ns, q_out);
}

bool FrankaReplayer::sample_at(int64_t t_play_ns, std::array<double, 7>& q_out) const {
  if (positions_.empty()) {
    return false;
  }
  if (t_play_ns <= 0) {
    q_out = positions_.front();
    return true;
  }
  const int64_t t_abs = timestamps_ns_.front() + t_play_ns;
  if (t_abs >= timestamps_ns_.back()) {
    q_out = positions_.back();
    return true;
  }

  // upper_bound returns first index whose timestamp > t_abs; the bracket is
  // (idx-1, idx). Both are safely inside the array because of the bounds checks
  // above (t_abs strictly between front and back).
  auto it = std::upper_bound(timestamps_ns_.begin(), timestamps_ns_.end(), t_abs);
  const size_t hi = static_cast<size_t>(it - timestamps_ns_.begin());
  const size_t lo = hi - 1;

  const int64_t t_lo = timestamps_ns_[lo];
  const int64_t t_hi = timestamps_ns_[hi];
  const int64_t span = t_hi - t_lo;
  double alpha = 0.0;
  if (span > 0) {
    alpha = static_cast<double>(t_abs - t_lo) / static_cast<double>(span);
  }

  const auto& q_lo = positions_[lo];
  const auto& q_hi = positions_[hi];
  for (int j = 0; j < 7; ++j) {
    q_out[j] = q_lo[j] + alpha * (q_hi[j] - q_lo[j]);
  }
  return true;
}

}  // namespace robo_lab
