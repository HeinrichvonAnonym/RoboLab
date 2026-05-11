#include "cartesian_control/cartesian_controller.h"

#include <algorithm>
#include <cmath>

namespace robo_lab {

CartesianController::CartesianController() {
    set_franka_default_dh();
    // Default per-joint step clamp: 5 cdeg per dpose call. Aggressive enough
    // to follow a key press, gentle enough to never produce a torque spike.
    max_dq_per_step_.assign(num_joints(), 0.05);
}

void CartesianController::set_dh_params(
    const std::vector<std::array<double, 4>>& dh_params) {
    jacobian_calc_.set_dh_params(dh_params);
    if (max_dq_per_step_.size() != num_joints()) {
        max_dq_per_step_.assign(num_joints(), 0.05);
    }
}

void CartesianController::set_franka_default_dh() {
    jacobian_calc_.set_franka_default_dh();
    if (max_dq_per_step_.size() != num_joints()) {
        max_dq_per_step_.assign(num_joints(), 0.05);
    }
}

void CartesianController::set_tool_offset(const Eigen::Matrix4d& T_tool) {
    tool_offset_ = T_tool;
}

void CartesianController::set_damping(double lambda) {
    damping_ = std::max(lambda, 0.0);
}

void CartesianController::set_max_dq_per_step(const std::vector<double>& max_dq) {
    max_dq_per_step_ = max_dq;
    // Pad / trim to match the kinematic chain length so callers can supply a
    // shorter list and we still index safely.
    if (max_dq_per_step_.size() < num_joints()) {
        max_dq_per_step_.resize(num_joints(), 0.05);
    } else if (max_dq_per_step_.size() > num_joints()) {
        max_dq_per_step_.resize(num_joints());
    }
}

void CartesianController::set_joint_limits(const std::vector<double>& lower,
                                           const std::vector<double>& upper) {
    joint_lower_ = lower;
    joint_upper_ = upper;
}

void CartesianController::seed_target(const std::vector<double>& q_current) {
    if (q_current.empty()) {
        return;
    }
    q_target_ = q_current;
    if (q_target_.size() < num_joints()) {
        q_target_.resize(num_joints(), 0.0);
    } else if (q_target_.size() > num_joints()) {
        q_target_.resize(num_joints());
    }
    target_seeded_ = true;
}

std::vector<CartesianPose> CartesianController::compute_link_poses(
    const JointState& state) const {
    std::vector<CartesianPose> poses;
    if (state.position.empty()) {
        return poses;
    }

    const auto transforms = jacobian_calc_.compute_link_transforms(state.position);
    poses.reserve(transforms.size());
    for (const auto& T : transforms) {
        CartesianPose p;
        p.position = T.block<3, 1>(0, 3);
        Eigen::Quaterniond q(T.block<3, 3>(0, 0));
        q.normalize();
        p.orientation = q;
        p.timestamp = state.timestamp;
        poses.push_back(p);
    }
    return poses;
}

CartesianPose CartesianController::compute_ee_pose(
    const std::vector<double>& q_current) const {
    CartesianPose p;
    if (q_current.empty()) {
        return p;
    }
    const Eigen::Matrix4d T = jacobian_calc_.compute_ee_transform(q_current, tool_offset_);
    p.position = T.block<3, 1>(0, 3);
    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    q.normalize();
    p.orientation = q;
    return p;
}

bool CartesianController::apply_dpose(
    const std::array<double, 6>& dpose,
    std::vector<double>* q_target_out) {
    if (!target_seeded_ || q_target_.size() != num_joints()) {
        return false;
    }

    // Build dpose vector. Layout: [dx, dy, dz, droll, dpitch, dyaw].
    Eigen::Matrix<double, 6, 1> dx;
    for (int i = 0; i < 6; ++i) {
        dx(i) = dpose[i];
    }

    // Geometric Jacobian at the *target* configuration. Using the latched
    // target rather than the latest measured q lets each command produce a
    // deterministic dq even if robot tracking lags slightly behind.
    const Eigen::MatrixXd J = jacobian_calc_.compute_jacobian(q_target_, tool_offset_);

    // Damped least-squares pseudo-inverse:
    //   dq = J^T (J J^T + lambda^2 I)^-1 dx
    // The lambda^2 I term keeps the inversion conditioned through singular
    // configurations -- without it a small dx can blow up dq.
    const Eigen::MatrixXd JJt = J * J.transpose();
    Eigen::MatrixXd damped = JJt;
    const double lam2 = damping_ * damping_;
    for (int i = 0; i < damped.rows(); ++i) {
        damped(i, i) += lam2;
    }
    const Eigen::VectorXd dq = J.transpose() * damped.ldlt().solve(dx);

    // Clamp per-joint magnitude. A single key press should never move a
    // joint by more than max_dq_per_step_[i].
    Eigen::VectorXd dq_clamped = dq;
    for (Eigen::Index i = 0; i < dq_clamped.size(); ++i) {
        const double cap = (i < static_cast<Eigen::Index>(max_dq_per_step_.size()))
                               ? max_dq_per_step_[i]
                               : 0.05;
        if (dq_clamped(i) > cap) dq_clamped(i) = cap;
        if (dq_clamped(i) < -cap) dq_clamped(i) = -cap;
    }

    // Integrate into the latched target, then clamp to joint limits.
    for (size_t i = 0; i < q_target_.size(); ++i) {
        q_target_[i] += dq_clamped(static_cast<Eigen::Index>(i));
        if (i < joint_lower_.size() && q_target_[i] < joint_lower_[i]) {
            q_target_[i] = joint_lower_[i];
        }
        if (i < joint_upper_.size() && q_target_[i] > joint_upper_[i]) {
            q_target_[i] = joint_upper_[i];
        }
    }

    if (q_target_out != nullptr) {
        *q_target_out = q_target_;
    }
    return true;
}

}  // namespace robo_lab
