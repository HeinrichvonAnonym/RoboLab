#include "cartesian_control/jacobian_calculator.h"

#include <cmath>

namespace robo_lab {

void JacobianCalculator::set_franka_default_dh() {
    // Modified DH (Craig) parameters from the tested ROS baseline.
    // Format: [theta_offset, d, a, alpha].
    dh_params_ = {
        {0.0, 0.333,  0.0,     0.0},
        {0.0, 0.0,    0.0,    -M_PI / 2.0},
        {0.0, 0.316,  0.0,     M_PI / 2.0},
        {0.0, 0.0,    0.0825,  M_PI / 2.0},
        {0.0, 0.384, -0.0825, -M_PI / 2.0},
        {0.0, 0.0,    0.0,     M_PI / 2.0},
        {0.0, 0.207,  0.088,   M_PI / 2.0},
    };
    num_joints_ = dh_params_.size();
}

void JacobianCalculator::set_dh_params(const std::vector<DHParams>& dh_params) {
    dh_params_ = dh_params;
    // Baseline only chains the actuated joints; any trailing flange row is
    // ignored here, matching the ROS reference behaviour.
    num_joints_ = std::min(dh_params_.size(), static_cast<size_t>(7));
}

void JacobianCalculator::set_dh_params(const std::vector<std::array<double, 4>>& dh_raw) {
    dh_params_.clear();
    dh_params_.reserve(dh_raw.size());
    for (const auto& row : dh_raw) {
        DHParams dh;
        dh.theta_offset = row[0];
        dh.d = row[1];
        dh.a = row[2];
        dh.alpha = row[3];
        dh_params_.push_back(dh);
    }
    num_joints_ = std::min(dh_params_.size(), static_cast<size_t>(7));
}

Eigen::Matrix4d JacobianCalculator::dh_transform(const DHParams& dh, double q) const {
    const double theta = q + dh.theta_offset;
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const double ca = std::cos(dh.alpha);
    const double sa = std::sin(dh.alpha);

    Eigen::Matrix4d T;
    T <<
        ct,      -st,       0.0,    dh.a,
        st * ca,  ct * ca, -sa,    -dh.d * sa,
        st * sa,  ct * sa,  ca,     dh.d * ca,
        0.0,      0.0,      0.0,    1.0;
    return T;
}

std::vector<Eigen::Matrix4d> JacobianCalculator::compute_link_transforms(
    const std::vector<double>& q) const {
    const size_t n = std::min(num_joints_, q.size());
    std::vector<Eigen::Matrix4d> transforms;
    transforms.reserve(n);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (size_t i = 0; i < n; ++i) {
        T = T * dh_transform(dh_params_[i], q[i]);
        transforms.push_back(T);
    }
    return transforms;
}

Eigen::Matrix4d JacobianCalculator::compute_ee_transform(
    const std::vector<double>& q,
    const Eigen::Matrix4d& tool_offset) const {
    const auto transforms = compute_link_transforms(q);
    if (transforms.empty()) {
        return tool_offset;
    }
    return transforms.back() * tool_offset;
}

Eigen::MatrixXd JacobianCalculator::compute_jacobian(
    const std::vector<double>& q,
    const Eigen::Matrix4d& tool_offset) const {
    const auto transforms = compute_link_transforms(q);
    const size_t n = transforms.size();

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, static_cast<int>(num_joints_));
    if (n == 0) {
        return J;
    }

    // End-effector position in base frame, shifted by the tool offset.
    const Eigen::Vector3d O_e =
        (transforms.back() * tool_offset).block<3, 1>(0, 3);

    for (size_t i = 0; i < n; ++i) {
        // Joint-i axis (z of frame i) and origin (any point on the axis works
        // for the cross product; we use the frame origin).
        const Eigen::Vector3d Z_i = transforms[i].block<3, 1>(0, 2);
        const Eigen::Vector3d O_i = transforms[i].block<3, 1>(0, 3);

        J.block<3, 1>(0, static_cast<int>(i)) = Z_i.cross(O_e - O_i);
        J.block<3, 1>(3, static_cast<int>(i)) = Z_i;
    }
    return J;
}

}  // namespace robo_lab
