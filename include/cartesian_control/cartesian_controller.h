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

// Thin orchestrator around JacobianCalculator: holds DH params and produces
// the world-frame pose of every actuated link from a joint observation.
//
// IK / velocity-limit / control logic intentionally lives elsewhere; this
// stage only validates the FK chain.
class CartesianController {
public:
    CartesianController();
    ~CartesianController() = default;

    void set_dh_params(const std::vector<std::array<double, 4>>& dh_params);
    void set_franka_default_dh();

    // Returns one CartesianPose per actuated joint. Element i is the pose of
    // frame i in the robot base frame after applying joint i.
    std::vector<CartesianPose> compute_link_poses(const JointState& state) const;

    size_t num_joints() const { return jacobian_calc_.num_joints(); }

private:
    JacobianCalculator jacobian_calc_;
};

}  // namespace robo_lab
