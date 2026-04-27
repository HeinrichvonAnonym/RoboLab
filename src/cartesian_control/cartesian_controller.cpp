#include "cartesian_control/cartesian_controller.h"

namespace robo_lab {

CartesianController::CartesianController() {
    set_franka_default_dh();
}

void CartesianController::set_dh_params(
    const std::vector<std::array<double, 4>>& dh_params) {
    jacobian_calc_.set_dh_params(dh_params);
}

void CartesianController::set_franka_default_dh() {
    jacobian_calc_.set_franka_default_dh();
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

}  // namespace robo_lab
