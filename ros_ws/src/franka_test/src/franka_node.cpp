
// COPYRIGHT (c) 2025, Heinrich HE
// mail: 2130238@tongji.edu.cn.
// Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License. | diese Project soll nicht ohne die Lizenz verwendet werden.
// You may obtain a copy of the License at                          | Die kopie der Lizenz kann auf http://www.apache.org/licenses/LICENSE-2.0  
// http://www.apache.org/licenses/LICENSE-2.0                       | oder in der Original-Datei erhalten werden.


#include <ros/ros.h>
#include <franka/robot.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <std_msgs/Float32.h>

#include <franka/robot_state.h>
#include <franka/exception.h>
#include <std_msgs/Float32MultiArray.h>
#include <iostream>
#include <array>
#include <atomic>
#include <thread>
#include <mutex>
#include <sensor_msgs/JointState.h>
#include <Eigen/Dense>

// This is a ros node to excute impedance control of franka panda robot arm.
// the input signals are 
// "/desired_velocity" 
// and 
// "/desired_gripper_width"
// this node will also publish joint state with frequency of 1000 hz and topic 
// "base_feedback/joint_state"
// while using this code, you can replace the catkin_pkg franka_ros

// Dies ist eine ROS Node, die fuer Impetanz-Steuerung von Franka-Panda Roboter durchzufuehren ist.
// Die Eingaebe Signalen sind"
// "/desired_velocity" 
// und
// "/desired_gripper_width"
// Diese Node kann JointState mit dem Frequenz um 1000 Hz und Topic
// "base_feedback/joint_state"
// verbreiten
// Wenn diese Node verwendet ist, kann das Catkin_Pkg Franka_Ros ersetzt werden


std::array<double, 7> desired_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
std::atomic<double> desired_gripper_width(0.08);
std::atomic<double> last_gripper_width(0.08);
std::atomic<bool> new_gripper_command_received(false);
std::mutex velocity_mutex;
std::mutex gripper_mutex;

double gripper_position = 0.04;
double gripper_velocity = 0.0;
double gripper_force = 0.0;

const std::array<double, 7> joint_min = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
const std::array<double, 7> joint_max = { 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973};


void velocityCallback(const std_msgs::Float32MultiArray::ConstPtr& msg){
    if (msg->data.size() < 7){
        ROS_WARN("Reiceived array with less than 7 elements");
        return;
    }

    std::lock_guard<std::mutex> lock(velocity_mutex);
    for (size_t i=0; i<7; i++){   
        // filter 
        desired_velocity[i] = desired_velocity[i]*0.95 + msg->data[i]*0.05;
        // clip
        desired_velocity[i] = static_cast<double>(msg->data[i]);
        desired_velocity[i] = std::max(std::min(desired_velocity[i], 0.3), -0.3);
    }
}

void gripperCallback(const std_msgs::Float32::ConstPtr& msg){
    std::lock_guard<std::mutex> lock(gripper_mutex);
    desired_gripper_width = static_cast<double>(msg->data);
    new_gripper_command_received = true;
}


// core-code
void control_loop(franka::Robot& robot, ros::Publisher& joint_pub, ros::Publisher& eef_pub) {
    ros::Rate r = ros::Rate(10);
    sensor_msgs::JointState joint_msg;
    joint_msg.name = {  "panda_joint1", "panda_joint2", "panda_joint3",
                    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_join1" };
    // std::array<double, 7> q_d = initial_state.q;
    franka::Model model = robot.loadModel();

    // Impertanz Parameter
    std::array<double, 7> k_stiffness = {{50, 50, 50, 40, 40, 40, 40}};
    std::array<double, 7> k_damping;
    for (size_t i = 0; i < 7; i++) {
        k_damping[i] = 2 * sqrt(k_stiffness[i]); // 临界阻尼
            }
    const std::array<double, 7> joint_min = {{-2.8, -1.76, -2.8, -3.07, -2.8, 0.0, -2.8}};
    const std::array<double, 7> joint_max = {{ 2.8,  1.76,  2.8,  0.0,  2.8, 3.75,  2.8}};
    
    
    const double force_threshold_N = 12;  // Complaince threshold
    const double hysteresis_N = 4;        // Hysteresissrc/franka_test/src/franka_node.cpp
    const double lp_alpha = 0.15;         // Filter 
    const double comp_vel_gain = 0.0008;  // m/(N*s) Kraft --> Bewegung
    const double pinv_damping = 0.10;     // Inverse Damping
    const double dq_step_limit = 0.01;    // max dt
    const double max_qoffset_abs = 0.5;
    const double max_tau = 50;
    // const double stiffness_reduction = 0.35;
    // const double stiffness_restore_rate = 0.02;

    Eigen::Vector3d f_lp = Eigen::Vector3d::Zero();

    while(ros::ok())
    {
        try {   
            franka::RobotState init_state = robot.readOnce();
            std::array<double, 7> q_cmd = init_state.q; 
            std::array<double, 7> q_offset = {{0., 0., 0., 0., 0., 0., 0.}};
            // franka::RobotState prev_state = robot.readOnce();
            // std::array<double, 7> prev_q = init_state.q; 
            double stiffness_scale = 1.0;
            std::cout << "Robot state read OK. Ready for control.\n";
            robot.control(
                [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques {

                    // const auto& T_EE = state.O_T_EE;
                    
                    // std_msgs::Float32MultiArray eef_pos;
                    if (!ros::ok()){
                        throw franka::Exception("ROS is shutting down, exiting control loop");
                    }

                    const double dt = period.toSec();

                    joint_msg.header.stamp = ros::Time::now();
                    
                    joint_msg.position.assign(state.q.begin(), state.q.end());
                    joint_msg.velocity.assign(state.dq.begin(), state.dq.end());
                    joint_msg.effort.assign(state.tau_J.begin(), state.tau_J.end());

                    joint_msg.position.push_back(gripper_position);
                    joint_msg.velocity.push_back(gripper_velocity);
                    joint_msg.effort.push_back(gripper_force);

                    // eef_pos.data = {T_EE[12], T_EE[13], T_EE[14]};

                    joint_pub.publish(joint_msg);

                    std::array<double, 7> delta_q_actual;
                    
                    // eef_pub.publish(eef_pos);

                    // jetzige Position
                    

                    // Kraft
                    Eigen::Vector3d f_now(  state.O_F_ext_hat_K[0],
                                            state.O_F_ext_hat_K[1],
                                            state.O_F_ext_hat_K[2]);
                    // printf("force: %f, %f, %f, \n", state.O_F_ext_hat_K[0],state.O_F_ext_hat_K[1],state.O_F_ext_hat_K[2]);
                    f_lp = (1.0 - lp_alpha) * f_lp + lp_alpha * f_now;
                    const double F = f_lp.norm();
                    //Jacobian
                    std::array<double, 42> jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, state);
                    Eigen::Map<const Eigen::Matrix<double, 6, 7>> J_full(jacobian_array.data());
                    Eigen::Matrix<double, 3, 7> Jv = J_full.topRows<3>(); // linear velo

                 
                    if(F > force_threshold_N){
                        // stiffness_scale = std::max(stiffness_scale, stiffness_reduction);
                        // printf(">>");
                        // linear movement
                        Eigen::Vector3d dx = comp_vel_gain * f_lp * dt;

                        // DLS pinv
                        Eigen::Matrix3d JJt = Jv * Jv.transpose();
                        JJt += (pinv_damping * pinv_damping) * Eigen::Matrix3d::Identity();
                        Eigen::Matrix<double, 7, 3> J_pinv = Jv.transpose() * JJt.ldlt().solve(Eigen::Matrix3d::Identity());

                        //Compliance control
                        Eigen::Matrix<double, 7, 1> dq_c = J_pinv * dx;   

                        for (size_t i = 0; i < 7; i++) {
                            double step = std::max(std::min(dq_c[i], dq_step_limit), - dq_step_limit);
                            // q_cmd = state.q;
                    
                            if (q_offset[i] > max_qoffset_abs) q_offset[i] = max_qoffset_abs;
                            if (q_offset[i] < -max_qoffset_abs) q_offset[i] = -max_qoffset_abs;

                            q_cmd[i] = state.q[i];
                            q_cmd[i] += q_offset[i];
                            q_cmd[i] = std::min(std::max(q_cmd[i], joint_min[i] + 0.05), joint_max[i] - 0.05);
                        }
                                 
                    }else{
                        q_offset = {{0., 0., 0., 0., 0., 0., 0.}};
                        // stiffness_scale = 1;
                         std::lock_guard<std::mutex> lock(velocity_mutex);
                        for (size_t i = 0; i < 7; i++) {
                            // q_cmd[i] = state.q[i];
                            q_cmd[i] += desired_velocity[i] * dt; // Differenz 
                        }
                    }

                   

                    for (size_t i = 0; i < 7; i++) {                
                            q_cmd[i] = std::min(std::max(q_cmd[i], joint_min[i] + 0.05), joint_max[i] - 0.05);
                        }             
                    
                    
                    // Rechnung des Kraftmoment
                    std::array<double, 7> tau_d;
                    for (size_t i = 0; i < 7; i++) {
                        // double kd_scales = k_damping[i] * stiffness_scale;
                        // double k_stiffness_scales = k_stiffness[i] * stiffness_scale;
                        double q_err = state.q[i] - q_cmd[i];
                        tau_d[i] = - k_stiffness[i] * q_err - k_damping[i] * state.dq[i];
                    }

                    // Ausgleichung des Gewichtes                    
                    // std::array<double, 7> gravity = model.gravity(state);
                    // for (size_t i = 0; i < 7; i++) {
                    //     tau_d[i] += gravity[i];
                    // }

                    return franka::Torques(tau_d);
                }
            );
        } catch (const franka::Exception& e) {
            std::cerr << "Franka Exception: " << e.what() << std::endl;
        }
        desired_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
        auto state = robot.readOnce();
        joint_msg.position.assign(state.q.begin(), state.q.end());
        joint_msg.velocity.assign(state.dq.begin(), state.dq.end());
        joint_msg.effort.assign(state.tau_J.begin(), state.tau_J.end());
        // eef_pos.data = {T_EE[12], T_EE[13], T_EE[14]};
        joint_pub.publish(joint_msg);
     
        r.sleep();
    }
}

void gripper_control_loop(franka::Gripper& gripper) { 
    try {
        ros::Rate rate(60);
        while (ros::ok()) {
            /*get*/
            franka::GripperState state = gripper.readOnce();
            gripper_position = state.width / 2;
            // gripper_velocity = state.speed;
            // gripper_effort = state.force;
            /*set*/
            bool do_gripper_action = false;
            double target_width = 0.08;
            {
                std::lock_guard<std::mutex> lock(gripper_mutex);
                target_width = desired_gripper_width;
            }

            if (new_gripper_command_received && std::abs(target_width - last_gripper_width) > 1e-4){
                last_gripper_width = target_width;
                do_gripper_action = true;
                new_gripper_command_received = false;
            }

            if (do_gripper_action){
               if (!gripper.grasp(target_width, 0.1, 5.0, 5.0, 0.1)) {
                gripper.move(target_width, 0.1);
            }
            }
            rate.sleep();
        }
    } catch (const franka::Exception& e) {
        std::cerr << "Franka Exception: " << e.what() << std::endl;
    }

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "franka_velocity_control");
    // ros::Subscriber()
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("/desired_velocity", 10, velocityCallback);
    ros::Subscriber gripper_sub = nh.subscribe("/desired_gripper_width", 10, gripperCallback);
    ros::Publisher js_pub = nh.advertise<sensor_msgs::JointState>("/base_feedback/joint_state", 10);
    ros::Publisher ee_pub = nh.advertise<std_msgs::Float32MultiArray>("/eef_position/", 10);
    franka::Robot robot("172.16.0.40");
    franka::Gripper gripper("172.16.0.40");

    // kernel priority
    robot.setCollisionBehavior(
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}
    );

    // robot control
    std::thread control_thread(control_loop, std::ref(robot), std::ref(js_pub), std::ref(ee_pub));
    // gripper control
    std::thread gripper_thread(gripper_control_loop, std::ref(gripper));

    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();

    control_thread.join();
    gripper_thread.join();
    return 0;
}
