import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PointStamped
import numpy as np
from pynput.keyboard import Listener, Key


class cubeInterface:
    def __init__(self):
        rospy.init_node('cube_interface', anonymous=True)   
        self.markerA_pub = rospy.Publisher('/cube_pose', Marker, queue_size=10)
        self.markerB_pub = rospy.Publisher('/obs_pose', Marker, queue_size=10)
        self.target_x, self.target_y, self.target_z = 0.4, 0.0, 0.3
        self._init_marker()
        self._init_pose()
        # subscribers
        pass

             

    def _init_pose(self):
        poseA = Pose()
        poseA.position.x = self.target_x
        poseA.position.y = self.target_y
        poseA.position.z = self.target_z

        poseA.orientation.x = 1.0
        poseA.orientation.y = 0.0
        poseA.orientation.z = 0.0
        poseA.orientation.w = 0.0

        poseB = Pose()
        poseB.position.x = 0.41 + np.random.uniform(-0.1,0.05)
        poseB.position.y = -0.1 + np.random.uniform(-0.1,0.1)
        poseB.position.z = 0.31
        poseB.orientation.x = 1.0
        poseB.orientation.y = 0.0
        poseB.orientation.z = 0.0
        poseB.orientation.w = 0.0

        
        self.poseA, self.poseB = poseA, poseB

        self.dt = 0.02

        
    def _init_marker(self):
        markerA = Marker()
        markerA.header.frame_id = "panda_link0"
        markerA.ns = "cube"
        markerA.id = 0
        markerA.type = Marker.CUBE
        markerA.action = Marker.ADD
        markerA.scale.x = 0.05
        markerA.scale.y = 0.05
        markerA.scale.z = 0.05
        markerA.color.a = 1.0
        markerA.color.r = 1.0
        markerA.color.g = 0.0
        markerA.color.b = 0.0
        markerA.pose.orientation.w = 0.0
        markerA.pose.orientation.x = 1.0
        markerA.pose.orientation.y = 0.0
        markerA.pose.orientation.z = 0.0

        markerB = Marker()
        markerB.header.frame_id = "panda_link0"
        markerB.ns = "cube"
        markerB.id = 1
        markerB.type = Marker.CUBE
        markerB.action = Marker.ADD
        markerB.scale.x = 0.075
        markerB.scale.y = 0.075
        markerB.scale.z = 0.075
        markerB.color.a = 1.0
        markerB.color.r = 1.0
        markerB.color.g = 1.0
        markerB.color.b = 0.8
        

        self.markerA = markerA
        self.markerB = markerB
    
    def run(self):
        frame_count = 0
        count = 0
        while(not rospy.is_shutdown()):
            angle = count * np.pi / 180
            self.target_x = 0.41 + np.cos(angle) * 0.1
            self.target_y = -0.1 + np.sin(angle) * 0.2
            self.target_z = 0.31 + np.sin(1.3 * angle) * 0.1
            count += 0.17
            if count > 360 * 1e4:
                count = 0
                frame_count += 1
                if frame_count > 10:
                    frame_count = 0
            # print(">")
            self.markerA.header.stamp = rospy.Time.now()
            self.markerA.pose.position .x = self.target_x
            self.markerA.pose.position .y = self.target_y
            self.markerA.pose.position .z = self.target_z
            self.markerB.header.stamp = rospy.Time.now()
            self.markerB.pose = self.poseB

            self.markerA_pub.publish(self.markerA)
            self.markerB_pub.publish(self.markerB)

            rospy.sleep(self.dt)
       
    

if __name__ == '__main__':
    cube_interface = cubeInterface()
    cube_interface.run()