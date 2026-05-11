#pragma once

#include "cartesian_control/jacobian_calculator.h"

#include <array>
#include <vector>
#include <Eigen/Dense>

namespace robo_lab {

struct CartesianPose {
    Eigen::Vector3d position{0.0, 0.0, 0.0};
    Eigen::Quaterniond orientation{1.0, 0.0, 0.0, 0.0};
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

// Resolved-rate cartesian controller around JacobianCalculator.
//
// State:
//   - latched joint target q_target_ (must be seeded from a robot
//     observation before apply_dpose() is called).
//
// Per-step pipeline:
//   1. Build the geometric Jacobian J(q_target) at the gripper TCP (frame 7
//      multiplied by tool_offset).
//   2. Solve dq = J^T (J J^T + lambda^2 I)^-1 dpose      (damped LS).
//   3. Clamp |dq_i| to max_dq_per_step_.
//   4. q_target_ += dq  and clamp to joint limits.
//
// This gives a stable mapping from a small Cartesian delta to a small joint
// delta even near singularities, and never lets a single keystroke produce
// a large joint jump.
class CartesianController {
public:
    CartesianController();
    ~CartesianController() = default;

    // ---------- DH / kinematics setup ----------
    void set_dh_params(const std::vector<std::array<double, 4>>& dh_params);
    void set_franka_default_dh();

    // Tool offset applied after the last DH frame (gripper TCP).
    // Defaults to identity (TCP == frame 7).
    void set_tool_offset(const Eigen::Matrix4d& T_tool);

    // ---------- Control configuration ----------
    void set_damping(double lambda);
    void set_max_dq_per_step(const std::vector<double>& max_dq);
    void set_joint_limits(const std::vector<double>& lower,
                          const std::vector<double>& upper);

    // ---------- State ----------
    // Seed (or re-seed) the internal joint target from a current observation.
    // Must be called at least once before apply_dpose().
    void seed_target(const std::vector<double>& q_current);
    bool is_seeded() const { return target_seeded_; }
    const std::vector<double>& q_target() const { return q_target_; }

    // ---------- Diagnostics ----------
    // Returns one CartesianPose per actuated joint. Element i is the pose of
    // frame i in the robot base frame after applying joint i.
    std::vector<CartesianPose> compute_link_poses(const JointState& state) const;

    // World-frame end-effector pose (frame 7 * tool_offset) at q_current.
    CartesianPose compute_ee_pose(const std::vector<double>& q_current) const;

    // ---------- Control step ----------
    // Apply a small Cartesian delta (in base frame) to the latched joint
    // target. dpose = [dx, dy, dz, droll, dpitch, dyaw] -- linear in metres,
    // angular as a small-angle axis-angle vector (rad). On success, writes
    // the new q_target_ into *q_target_out and returns true. Returns false
    // if the controller has not been seeded yet.
    bool apply_dpose(const std::array<double, 6>& dpose,
                     std::vector<double>* q_target_out);

    size_t num_joints() const { return jacobian_calc_.num_joints(); }

private:
    JacobianCalculator jacobian_calc_;
    Eigen::Matrix4d tool_offset_{Eigen::Matrix4d::Identity()};

    // Damped-least-squares damping factor (units of rad). 0.05 is a typical
    // teleop value: large enough to keep dq bounded near singular configs,
    // small enough not to dominate dq in well-conditioned regions.
    double damping_{0.05};

    // Hard per-joint clamp on |dq_i| in a single apply_dpose call.
    std::vector<double> max_dq_per_step_;

    // Joint position limits (rad). Empty means "no clamp".
    std::vector<double> joint_lower_;
    std::vector<double> joint_upper_;

    // Latched joint target. apply_dpose accumulates into this.
    std::vector<double> q_target_;
    bool target_seeded_{false};
};

}  // namespace robo_lab
