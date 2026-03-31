import rospy
from sensor_msgs.msg import JointState
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
import numpy as np
# import matplotlib.pyplot as plt
import time
from std_msgs.msg import Float32MultiArray
import yaml
import tf



algo_name = "isaac"

def angle_normalize(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class PIDController:
    def __init__(self, kp, ki, kd, alpha = 0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_derivative = 0
        self.alpha = alpha
    
    def compute(self, target_position, current_position, dt):
        target_position = angle_normalize(target_position)
        current_position = angle_normalize(current_position)
        error = target_position - current_position
        err = angle_normalize(error)
        # print(error)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt>0 else 0.0

        derivative = self.prev_derivative * self.alpha + derivative * ( 1 - self.alpha)
        self.prev_derivative = derivative

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output, error

import tf2_ros
class PIDExecuter:
    def __init__(self):
        rospy.init_node("sim_pid_executer", anonymous=True)
        self.joint_velocity_pub = rospy.Publisher("base_feedback/joint_state", JointState, queue_size=10)
    
        # self.initial_position = np.array([0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854])
        self.joint_state = None
        self.init_joint_state()


        rospy.Subscriber("/desired_velocity", Float32MultiArray, self.drive_callback)
        
        self.velocity = [0., 0., 0., 0., 0., 0., 0., 0.]

        self.init_dt = 0.01
        self.dt = 0.01
        self.cur_time = rospy.Time.now()
        self.init_joint_state()

    def init_joint_state(self):
        js = JointState()
        js.name = ['panda_joint1', 
                   'panda_joint2',
                   'panda_joint3',
                   'panda_joint4',
                   'panda_joint5',
                   'panda_joint6',
                   'panda_joint7',
                   'panda_finger_join1']
        js.position = [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.04]
        js.velocity = [0., 0., 0., 0., 0., 0., 0., 0., 0]
        js.effort = [0., 0., 0., 0., 0., 0., 0., 0., 0]
        self.joint_state = js
    
     
    def send_command(self, velocities):
        # print(velocities)
        js = self.joint_state
        js.header.stamp = rospy.Time.now()
        
        for i, vel in enumerate(velocities):
            js.position[i] += float(vel * self.dt)
            # print(type(js.position[i]))
            js.velocity[i] = vel
        rospy.sleep(self.init_dt)
        self.joint_velocity_pub.publish(js)
        
    def drive_callback(self, msg):
        self.velocity = np.array(msg.data)
    
    def update_dt(self, event):
        self.dt = (rospy.Time.now() - self.cur_time).to_sec()
        self.cur_time = rospy.Time.now()
        
    def run(self):
        while not rospy.is_shutdown():
            self.send_command(self.velocity)
            

        
        

if __name__ == "__main__":
    pid_executer = PIDExecuter()
    pid_executer.run()
    rospy.spin()