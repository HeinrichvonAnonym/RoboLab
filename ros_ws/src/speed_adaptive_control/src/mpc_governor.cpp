#include<speed_adaptive_control/mpc_governor.h>

MPC_Governor::MPC_Governor(ros::NodeHandle nodehandle,
                APF_Controller& apf, 
                ReferenceSelector& ref,
                JacobianCalculator& jac)
{
    nh = nh;
    ac = &apf;
    rf = &ref;
    jc = &jac;
    //fi = &finterface;

    target_pose_sub = nh.subscribe("cube_pose", 10, &MPC_Governor::target_pose_callback, this);
}
MPC_Governor::~MPC_Governor()
{
}

void MPC_Governor::target_pose_callback(const visualization_msgs::Marker::ConstPtr& msg)
{
    target_pose[0] = msg->pose.position.x;
    target_pose[1] = msg->pose.position.y;
    target_pose[2] = msg->pose.position.z;
    target_pose[3] = msg->pose.orientation.x;
    target_pose[4] = msg->pose.orientation.y;
    target_pose[5] = msg->pose.orientation.z;
    target_pose[6] = msg->pose.orientation.w;
    got_target_pose = true;
}

void MPC_Governor::update_kinematic()
{ 
    ac->jacobian = &jc->jacobian_7_dof;
    ac->jacobian_4_dof = &jc->jacobian_4_dof;
    ac->jacobian_5_dof = &jc->jacobian_5_dof;
    ac->jacobian_6_dof = &jc->jacobian_6_dof;
    ac->robot_poses = &jc->pose_array;
}

void MPC_Governor::forward(std::array<double, 7>& query_q)
{   
    rf->get_qr(query_q);
    qr = &(rf->qr); 

    jc->spaceJacobian(*cur_pos);
    update_kinematic();

    ac->cur_pos = *cur_pos;
    ac->cur_vel = *cur_vel;
    vel_cmd = ac->get_vel(*qr);  
    for(int i=0;i<7;i++){
        virtual_q[i] = query_q[i] + vel_cmd[i]*dt;
    }
}

void MPC_Governor::roll_predict()
{ 
    forward(*cur_pos);
    for (int i=0; i<traj_len; i++){
        for(int j=0; j<7; j++){
            virtual_traj[i][j] = virtual_q[j];
        }
        forward(virtual_q);
    }
}

void MPC_Governor::calc_Js()
{ 
    jc->spaceJacobian(virtual_traj[0]);
    cur_pose[0] = jc->pose_array.poses[6].position.x;
    cur_pose[1] = jc->pose_array.poses[6].position.y;
    cur_pose[2] = jc->pose_array.poses[6].position.z;
    cur_pose[3] = jc->pose_array.poses[6].orientation.x;
    cur_pose[4] = jc->pose_array.poses[6].orientation.y;
    cur_pose[5] = jc->pose_array.poses[6].orientation.z;
    cur_pose[6] = jc->pose_array.poses[6].orientation.w;

    jc->spaceJacobian(virtual_traj[traj_len-1]);
    end_pose[0] = jc->pose_array.poses[6].position.x;
    end_pose[1] = jc->pose_array.poses[6].position.y;
    end_pose[2] = jc->pose_array.poses[6].position.z;
    end_pose[3] = jc->pose_array.poses[6].orientation.x;
    end_pose[4] = jc->pose_array.poses[6].orientation.y;
    end_pose[5] = jc->pose_array.poses[6].orientation.z;
    end_pose[6] = jc->pose_array.poses[6].orientation.w;

}

void MPC_Governor::run()
{
    // ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseArray>("/facc/robot_pose", 10);
    ros::Rate loop_rate(rate);
    ros::AsyncSpinner spinner(9);
    spinner.start();

    int print_cnt = 0;
    int roll_cnt = 0;

    int rl_buffer_cnt = 10;

    cur_pos = &(rf->cur_pos);
    cur_vel = &(rf->cur_velo);
    weight_o = &(rf->weight_o);
    weight_r = &(rf->weight_r);

    std::array<double, 7> qr_tmp;

    while (ros::ok())
    {

        roll_predict();

        // temporary
        forward(*cur_pos);
        // need to be modified, while being roll predicted
        ac->update_marker(ac->res_rep_vec, ac->inf_pos, ac->inner_dis, 
                            ac->human_influence_margin);  
        ac->publish_vel(vel_cmd);

        

        if (++print_cnt % 40 == 0)
        {
            printf("rf min ids: %f, apf att: %f, apf rep: %f, num_humanlink: %d, \n",
             rf->min_dis, ac->att_norm, ac->rep_norm, ac->num_human_link);
            print_cnt = 0;
        }
        jc->pose_array_pub.publish(jc->pose_array);
        loop_rate.sleep();
    }
    // fi->join();
    spinner.stop();
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "mpc_governor");
    ros::NodeHandle nh;
    APF_Controller apf_controller(nh);
    ReferenceSelector reference_selector(nh);
    JacobianCalculator jacobian_calculator(nh);
    // Franka_Interface franka_interface(nh);
    MPC_Governor mpc_governor(nh, apf_controller, 
    reference_selector, jacobian_calculator);
    mpc_governor.run();
    return 0;
}