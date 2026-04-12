#include "cartesian_control/cartesian_controller.h"
#include <cmath>
#include <iostream>

namespace robo_lab {

CartesianController::CartesianController() {
    set_franka_default_dh();
}

void CartesianController::set_dh_params(const std::vector<std::array<double, 4>>& dh_params) {
    jacobian_calc_.set_dh_params(dh_params);
    last_dq_.resize(jacobian_calc_.num_joints(), 0.0);
}

void CartesianController::set_franka_default_dh() {
    jacobian_calc_.set_franka_default_dh();
    last_dq_.resize(jacobian_calc_.num_joints(), 0.0);
}

void CartesianController::set_limits(const CartesianLimits& limits) {
    limits_ = limits;
}

void CartesianController::set_current_joint_state(const JointState& state) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_ = state;
    
    if (!state.position.empty()) {
        current_pose_.position = jacobian_calc_.get_position(state.position);
        current_pose_.orientation = jacobian_calc_.get_orientation(state.position);
        current_pose_.timestamp = state.timestamp;
    }
}

void CartesianController::set_target_pose(const CartesianPose& pose) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    target_pose_ = pose;
    has_target_ = true;
}

CartesianPose CartesianController::get_current_pose() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_pose_;
}

CartesianPose CartesianController::get_target_pose() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return target_pose_;
}

JointState CartesianController::get_current_joint_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_;
}

Eigen::Vector3d CartesianController::quaternion_to_axis_angle(const Eigen::Quaterniond& q) {
    Eigen::Quaterniond q_norm = q.normalized();
    if (q_norm.w() < 0) {
        q_norm.coeffs() = -q_norm.coeffs();
    }
    
    double angle = 2.0 * std::acos(std::clamp(q_norm.w(), -1.0, 1.0));
    
    if (std::abs(angle) < 1e-10) {
        return Eigen::Vector3d::Zero();
    }
    
    Eigen::Vector3d axis(q_norm.x(), q_norm.y(), q_norm.z());
    double sin_half = std::sqrt(1.0 - q_norm.w() * q_norm.w());
    
    if (sin_half > 1e-10) {
        axis /= sin_half;
    }
    
    return axis * angle;
}

Eigen::Quaterniond CartesianController::axis_angle_to_quaternion(const Eigen::Vector3d& aa) {
    double angle = aa.norm();
    if (angle < 1e-10) {
        return Eigen::Quaterniond::Identity();
    }
    
    Eigen::Vector3d axis = aa / angle;
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}

std::vector<double> CartesianController::solve_ik_damped_least_squares(
    const CartesianPose& target,
    const std::vector<double>& q_current,
    double damping) const {
    
    const size_t n = q_current.size();
    std::vector<double> dq(n, 0.0);
    
    Eigen::Vector3d pos_current = jacobian_calc_.get_position(q_current);
    Eigen::Quaterniond rot_current = jacobian_calc_.get_orientation(q_current);
    
    // Position error
    Eigen::Vector3d pos_error = target.position - pos_current;
    
    // Orientation error (as axis-angle)
    Eigen::Quaterniond rot_error = target.orientation * rot_current.inverse();
    Eigen::Vector3d ori_error = quaternion_to_axis_angle(rot_error);
    
    // Combined 6D error
    Eigen::Matrix<double, 6, 1> error;
    error.head<3>() = pos_error;
    error.tail<3>() = ori_error;
    
    // Get Jacobian
    Eigen::Matrix<double, 6, Eigen::Dynamic> J = jacobian_calc_.compute_jacobian(q_current);
    
    // Damped Least Squares: dq = J^T * (J * J^T + lambda^2 * I)^-1 * error
    Eigen::Matrix<double, 6, 6> JJT = J * J.transpose();
    Eigen::Matrix<double, 6, 6> JJT_damped = JJT + damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
    
    Eigen::Matrix<double, 6, 1> y = JJT_damped.ldlt().solve(error);
    Eigen::VectorXd dq_eigen = J.transpose() * y;
    
    for (size_t i = 0; i < n; ++i) {
        dq[i] = dq_eigen(i);
    }
    
    return dq;
}

std::vector<double> CartesianController::apply_velocity_limits(
    const std::vector<double>& dq,
    double dt) const {
    
    std::vector<double> dq_limited = dq;
    const size_t n = dq.size();
    
    // Find max velocity scale factor
    double scale = 1.0;
    const double max_joint_vel = 2.0;  // rad/s (conservative for Franka)
    
    for (size_t i = 0; i < n; ++i) {
        double vel = std::abs(dq[i]) / dt;
        if (vel > max_joint_vel) {
            scale = std::min(scale, max_joint_vel / vel);
        }
    }
    
    // Apply scale and acceleration limits
    for (size_t i = 0; i < n; ++i) {
        dq_limited[i] *= scale;
        
        // Limit acceleration
        if (i < last_dq_.size()) {
            double acc = (dq_limited[i] - last_dq_[i]) / dt;
            const double max_acc = 10.0;  // rad/s^2
            if (std::abs(acc) > max_acc) {
                double sign = acc > 0 ? 1.0 : -1.0;
                dq_limited[i] = last_dq_[i] + sign * max_acc * dt;
            }
        }
    }
    
    return dq_limited;
}

std::vector<double> CartesianController::compute_joint_command(double dt) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!has_target_ || current_state_.position.empty()) {
        return current_state_.position;
    }
    
    // Compute joint velocity using IK
    std::vector<double> dq = solve_ik_damped_least_squares(
        target_pose_, current_state_.position, 0.05);
    
    // Apply limits
    dq = apply_velocity_limits(dq, dt);
    
    // Compute new joint positions
    std::vector<double> q_new = current_state_.position;
    for (size_t i = 0; i < q_new.size() && i < dq.size(); ++i) {
        q_new[i] -= dq[i];
    }
    
    // Store for next iteration
    last_dq_ = dq;
    
    return q_new;
}

bool CartesianController::is_target_reached(double pos_tol, double rot_tol) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!has_target_) {
        return true;
    }
    
    double pos_error = (target_pose_.position - current_pose_.position).norm();
    
    Eigen::Quaterniond rot_error = target_pose_.orientation * current_pose_.orientation.inverse();
    double angle_error = 2.0 * std::acos(std::clamp(std::abs(rot_error.w()), 0.0, 1.0));
    
    return (pos_error < pos_tol) && (angle_error < rot_tol);
}

}  // namespace robo_lab
