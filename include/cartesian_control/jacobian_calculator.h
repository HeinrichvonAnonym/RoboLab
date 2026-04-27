#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

namespace robo_lab {

struct DHParams {
    double theta_offset;  // Joint angle offset
    double d;             // Link offset
    double a;             // Link length
    double alpha;         // Link twist
};

// Forward kinematics for a serial manipulator using Modified DH (Craig)
// convention. Mirrors the tested ROS baseline in
// ros_ws/src/speed_adaptive_control/src/jacobian_calculator.cpp.
class JacobianCalculator {
public:
    JacobianCalculator() = default;
    ~JacobianCalculator() = default;

    void set_dh_params(const std::vector<DHParams>& dh_params);
    void set_dh_params(const std::vector<std::array<double, 4>>& dh_raw);
    void set_franka_default_dh();

    // Cumulative base->link_i transforms for every actuated joint.
    // transforms[i] is the world pose of frame i AFTER applying joint i.
    // Size == num_joints().
    std::vector<Eigen::Matrix4d> compute_link_transforms(
        const std::vector<double>& q) const;

    size_t num_joints() const { return num_joints_; }

private:
    // Modified DH (Craig convention) single-joint transform.
    Eigen::Matrix4d dh_transform(const DHParams& dh, double q) const;

    std::vector<DHParams> dh_params_;
    size_t num_joints_{7};
};

}  // namespace robo_lab
