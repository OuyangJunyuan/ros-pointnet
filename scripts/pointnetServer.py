#!/home/ou/software/anaconda3/envs/dl/bin/python
import sys
import rospy
from rospkg import RosPack

sys.path.append(RosPack().get_path('pointnet_ros'))
import ROSbridge

if __name__ == '__main__':
    classifier = ROSbridge.PointNetServer("pointnet_action")

    # avoid use_sim_time = true and the time-sequence confused
    rate = rospy.Rate(10)
    now = rospy.Time.now()
    while not rospy.is_shutdown():
        last, now = now, rospy.Time.now()
        if now < last:
            classifier.reset_server()
            print("reset")
        try:
            rate.sleep()
        except rospy.ROSTimeMovedBackwardsException:
            pass
