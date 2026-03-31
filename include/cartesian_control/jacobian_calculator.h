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

class JacobianCalculator {
public:
    JacobianCalculator() = default;
    ~JacobianCalculator() = default;

    void set_dh_params(const std::vector<DHParams>& dh_params);
    void set_dh_params(const std::vector<std::array<double, 4>>& dh_raw);
    void set_franka_default_dh();

    Eigen::Matrix4d forward_kinematics(const std::vector<double>& q) const;
    Eigen::Matrix4d forward_kinematics(const std::vector<double>& q, size_t end_joint) const;
    
    Eigen::Matrix<double, 6, Eigen::Dynamic> compute_jacobian(const std::vector<double>& q) const;
    
    Eigen::Vector3d get_position(const std::vector<double>& q) const;
    Eigen::Quaterniond get_orientation(const std::vector<double>& q) const;
    Eigen::Matrix3d get_rotation_matrix(const std::vector<double>& q) const;

    size_t num_joints() const { return num_joints_; }

private:
    // Modified DH transform (Craig convention) used by Franka
    Eigen::Matrix4d dh_transform(const DHParams& dh, double q) const;
    
    std::vector<DHParams> dh_params_;
    size_t num_joints_{7};
};

}  // namespace robo_lab
