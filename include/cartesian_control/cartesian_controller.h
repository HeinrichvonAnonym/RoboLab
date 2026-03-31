#pragma once

#include "cartesian_control/jacobian_calculator.h"

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace robo_lab {

struct CartesianPose {
    Eigen::Vector3d position{0, 0, 0};
    Eigen::Quaterniond orientation{1, 0, 0, 0};
    double timestamp{0.0};
    
    CartesianPose() = default;
    CartesianPose(const Eigen::Vector3d& pos, const Eigen::Quaterniond& rot)
        : position(pos), orientation(rot) {}
};

struct JointState {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> effort;
    double timestamp{0.0};
};

struct CartesianLimits {
    double max_pos_error{0.05};       // Max position error to trigger IK
    double max_vel{0.1};              // m/s
    double max_acc{0.2};              // m/s^2
    double max_jerk{0.3};             // m/s^3
    double max_angular_vel{0.5};      // rad/s
    double max_angular_acc{1.0};      // rad/s^2
};

class CartesianController {
public:
    CartesianController();
    ~CartesianController() = default;

    void set_dh_params(const std::vector<std::array<double, 4>>& dh_params);
    void set_franka_default_dh();
    void set_limits(const CartesianLimits& limits);
    
    void set_current_joint_state(const JointState& state);
    void set_target_pose(const CartesianPose& pose);
    
    CartesianPose get_current_pose() const;
    CartesianPose get_target_pose() const;
    JointState get_current_joint_state() const;
    
    std::vector<double> compute_joint_command(double dt);
    
    bool is_target_reached(double pos_tol = 0.001, double rot_tol = 0.01) const;
    
    static Eigen::Vector3d quaternion_to_axis_angle(const Eigen::Quaterniond& q);
    static Eigen::Quaterniond axis_angle_to_quaternion(const Eigen::Vector3d& aa);

private:
    std::vector<double> solve_ik_damped_least_squares(
        const CartesianPose& target,
        const std::vector<double>& q_current,
        double damping = 0.05) const;
    
    std::vector<double> apply_velocity_limits(
        const std::vector<double>& dq,
        double dt) const;
    
    JacobianCalculator jacobian_calc_;
    CartesianLimits limits_;
    
    mutable std::mutex state_mutex_;
    JointState current_state_;
    CartesianPose target_pose_;
    CartesianPose current_pose_;
    
    std::vector<double> last_dq_;
    bool has_target_{false};
};

}  // namespace robo_lab
