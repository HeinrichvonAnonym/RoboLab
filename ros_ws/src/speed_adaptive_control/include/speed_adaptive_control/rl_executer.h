#include<speed_adaptive_control/apf_controller.h>
#include<speed_adaptive_control/reference_selector.h>
#include<speed_adaptive_control/jacobian_calculator.h>
#include<visualization_msgs/Marker.h>
// #include<speed_adaptive_control/franka_interface.h>

class MPC_Governor
{
    public:
        MPC_Governor(ros::NodeHandle nh, APF_Controller& apf_controller,
         ReferenceSelector& reference_selector,
         JacobianCalculator& jacobian_calculator);
        ~MPC_Governor();

        ros::NodeHandle nh;
        APF_Controller *ac;
        ReferenceSelector *rf;
        JacobianCalculator *jc;
        //Franka_Interface *fi;

        void update_kinematic();
        void forward(std::array<double, 7>& query_q);

        void roll_predict();
        void calc_Js();
        void calc_d_E_att();
        void calc_d_E_rep();
        
        void update();
        void run();

        std::array<double, 7> *qr;
        std::array<double, 7> vel_cmd;
        std::array<double, 7> qv;

        std::array<double, 7> *cur_pos;
        std::array<double, 7> *cur_vel;

        double *weight_o;
        double *weight_r;

        ros::Subscriber target_pose_sub;
        void target_pose_callback(const visualization_msgs::Marker::ConstPtr& msg);
        // (xyz)(xyzw)
        std::array<double, 7> target_pose;
        bool got_target_pose = false;

        ros::Subscriber rl_raw_action_sub;
        void rl_raw_action_callback(const std_msgs::Float32MultiArray::ConstPtr& msg);
        std::array<double, 7> rl_raw_action;
        bool got_rl_raw_action = false;

        

    private:
        const int traj_len = 10;
        const int rate = 100;
        const double dt = 1 / double(rate);
        std::array<double, 7> virtual_q;
        std::array<std::array<double, 7>, 10> virtual_traj;

        // level 1 state
        double cur_eucli_distance;
        double end_eucli_distance;
        std::array<double, 7> cur_pose;
        std::array<double, 7> end_pose;
        double E_att;
        double E_rep;

        // level 2 state
        double OMPL_encli;
        double RL_encli;
        
        // level 3 state
        double d_E_att;
        double d_E_rep;

        
};      