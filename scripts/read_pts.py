import argparse
import numpy as np
import ros_numpy as rnp
import rospy
from sensor_msgs.msg import PointCloud2

data_path = '/home/ou/Documents/dataset/my_dataset/car/points/1.pts'


def ros_start():
    pub = rospy.Publisher('read_pts', PointCloud2, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        point_set = np.loadtxt(data_path, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        print(point_set)
        msg = rnp.msgify(PointCloud2, point_set)
        msg.header.frame_id = 'excavator/base_link'
        pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        ros_start()
    except rospy.ROSInterruptException:
        pass
