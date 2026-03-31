#include "cartesian_control/jacobian_calculator.h"
#include <cmath>

namespace robo_lab {

void JacobianCalculator::set_franka_default_dh() {
    // Franka Panda DH parameters (Modified DH / Craig convention)
    // Format: [theta_offset, d, a, alpha]
    // Exact values from reference speed_adaptive_control implementation
    dh_params_ = {
        {0.0,   0.333,   0.0,      0.0},          // Joint 0
        {0.0,   0.0,     0.0,     -M_PI/2},       // Joint 1
        {0.0,   0.316,   0.0,      M_PI/2},       // Joint 2
        {0.0,   0.0,     0.0825,   M_PI/2},       // Joint 3
        {0.0,   0.384,  -0.0825,  -M_PI/2},       // Joint 4
        {0.0,   0.0,     0.0,      M_PI/2},       // Joint 5
        {0.0,   0.207,   0.088,    M_PI/2},       // Joint 6
        {0.0,   0.18,    0.0,     -M_PI},         // Flange (from reference)
    };
    num_joints_ = 7;  // Only 7 actuated joints
}

void JacobianCalculator::set_dh_params(const std::vector<DHParams>& dh_params) {
    dh_params_ = dh_params;
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
    // Modified DH (Craig convention) - same as Franka's convention
    const double theta = q + dh.theta_offset;
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const double ca = std::cos(dh.alpha);
    const double sa = std::sin(dh.alpha);
    
    Eigen::Matrix4d T;
    T << ct,       -st,        0,       dh.a,
         st * ca,   ct * ca,  -sa,     -dh.d * sa,
         st * sa,   ct * sa,   ca,      dh.d * ca,
         0,         0,         0,       1;
    return T;
}

Eigen::Matrix4d JacobianCalculator::forward_kinematics(const std::vector<double>& q) const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const size_t n_joints = std::min(num_joints_, q.size());
    
    // Apply actuated joint transformations
    for (size_t i = 0; i < n_joints; ++i) {
        T = T * dh_transform(dh_params_[i], q[i]);
    }
    
    // Apply flange transformation (if present - row after joints)
    if (dh_params_.size() > num_joints_) {
        T = T * dh_transform(dh_params_[num_joints_], 0.0);
    }
    
    return T;
}

Eigen::Matrix4d JacobianCalculator::forward_kinematics(const std::vector<double>& q, size_t end_joint) const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const size_t n = std::min(end_joint, std::min(num_joints_, q.size()));
    
    for (size_t i = 0; i < n; ++i) {
        T = T * dh_transform(dh_params_[i], q[i]);
    }
    return T;
}

Eigen::Vector3d JacobianCalculator::get_position(const std::vector<double>& q) const {
    Eigen::Matrix4d T = forward_kinematics(q);
    return T.block<3, 1>(0, 3);
}

Eigen::Matrix3d JacobianCalculator::get_rotation_matrix(const std::vector<double>& q) const {
    Eigen::Matrix4d T = forward_kinematics(q);
    return T.block<3, 3>(0, 0);
}

Eigen::Quaterniond JacobianCalculator::get_orientation(const std::vector<double>& q) const {
    Eigen::Matrix3d R = get_rotation_matrix(q);
    Eigen::Quaterniond quat(R);
    quat.normalize();
    return quat;
}

Eigen::Matrix<double, 6, Eigen::Dynamic> JacobianCalculator::compute_jacobian(const std::vector<double>& q) const {
    const size_t n = std::min(num_joints_, q.size());
    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, n);
    J.setZero();
    
    // Compute forward kinematics for all joints (T_dofs style - after each joint)
    std::vector<Eigen::Matrix4d> T_dofs(n);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    
    for (size_t i = 0; i < n; ++i) {
        T = T * dh_transform(dh_params_[i], q[i]);
        T_dofs[i] = T;  // T_dofs[i] = transformation AFTER joint i
    }
    
    // End-effector position (including flange if present)
    Eigen::Matrix4d T_ee = T;
    if (dh_params_.size() > num_joints_) {
        T_ee = T_ee * dh_transform(dh_params_[num_joints_], 0.0);
    }
    Eigen::Vector3d p_ee = T_ee.block<3, 1>(0, 3);
    
    // Base frame z-axis for joint 0
    Eigen::Vector3d z_base(0, 0, 1);
    Eigen::Vector3d p_base(0, 0, 0);
    
    // Compute Jacobian columns (matching reference convention)
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3d z_i, p_i;
        
        if (i == 0) {
            // Joint 0 rotates about base z-axis
            z_i = z_base;
            p_i = p_base;
        } else {
            // Joint i rotates about z-axis of frame i-1 (which is T_dofs[i-1])
            z_i = T_dofs[i - 1].block<3, 1>(0, 2);
            p_i = T_dofs[i - 1].block<3, 1>(0, 3);
        }
        
        // Linear velocity: z_i x (p_ee - p_i)
        J.block<3, 1>(0, i) = z_i.cross(p_ee - p_i);
        
        // Angular velocity: z_i
        J.block<3, 1>(3, i) = z_i;
    }
    
    return J;
}

}  // namespace robo_lab
