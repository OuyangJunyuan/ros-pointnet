# about system
from .tools import *
import os
import numpy as np

# about ros
import rospy
from rospkg import RosPack
from sensor_msgs.msg import PointCloud2 as pc2
from sensor_msgs.point_cloud2 import read_points_list
from visualization_msgs.msg import Marker as mk
# about ros-ActionServer
import actionlib
import pointnet_ros.msg

# about pytorch
import torch
from torch.autograd import Variable
from PointNet.pointnet import model as model


class CloudClassifier(object):

    def __init__(self):
        root = '/home/ou/workspace/ros_ws/ironworks_ws/src/object_recognition/pointnet_ros/PointNet'
        self._path_model = os.path.join(root, 'utils/cls/cls_model_39.pth')

        self._name = 'cloudClassifier'
        self._pc2_sub_topic = '/excavator/lidar_global'
        self._pc2_pub_topic = 'classification'
        self._marker_pub_topic = 'boundingbox'
        self._ns = 'excavator'
        self._frame = 'excavator/base_link'

        self._num_classes = 4
        self._num_points = 2500
        self._confidence_threshold = 0.65
        self._volume_threshold = 0.1

        self._boxes = []
        #
        rospy.init_node(self._name, anonymous=True)
        rospy.Subscriber(self._pc2_sub_topic, pc2, self.callback)
        self._pub = rospy.Publisher(self._pc2_pub_topic, pc2, queue_size=1)
        self._pub3dbox = rospy.Publisher(self._marker_pub_topic, mk, queue_size=1)
        self._pc2_pub_topic_truckpose = rospy.Publisher("trucks_cloud", pc2, queue_size=1)

        self.classifierNet = model.PointNetCls(k=self._num_classes)  # 训练模型为16类
        self.classifierNet.load_state_dict(torch.load(self._path_model))
        self.classifierNet.cuda()
        self.classifierNet.eval()  # 测试模式

        rospy.spin()

    @exetime
    def classifier(self, netdata):
        """
        分类器
        :param netdata: 归一化-中心化-数量标准化 的numpy格式点云数据
        :return: 分类结果list, 不可信分类结果值为-1
        """
        netdata = Variable(torch.from_numpy(np.array(netdata))).float()
        netdata = netdata.transpose(2, 1).cuda()  # 转换为(batch_size, channels, num)

        pred, _, _ = self.classifierNet(netdata)  # 输出log_softmax

        pred = torch.exp(pred.detach().cpu()).topk(1, dim=1)  # detach从计算图中返回一个无gradient的tensor。
        pro, iswhat = pred.values.numpy().reshape(-1), pred.indices.numpy().reshape(-1)  # 转换为行向量

        # confidence threshold
        iswhat[pro < self._confidence_threshold] = -1
        np.set_printoptions(precision=1)

        print("分类结果:", " ".join("{:3d}".format(i) for i in iswhat))
        print("分类概率:", " ".join("{:1.1f}".format(i) for i in pro))
        return iswhat

    def callback(self, sub_msg):
        """
        ros消息回调函数
        :param sub_msg: pc2-xyzl
        :return: none
        """
        t1 = time.time()

        # 分类 data = {"labels": [], "indices": [], "points": [], "netData": []}
        dtype_points, data = ros2net_bridge(sub_msg, self._num_points)
        iswhat = self.classifier(data["netData"])

        # 3dbox体积滤波
        iswhat, self._boxes = volumefilter(data["points"], iswhat, self._volume_threshold)
        msg = get_marker_line_list(self._ns, self._frame, 0)
        for box in self._boxes:
            line_list_append(msg, box["pose"], box["scale"])
        self._pub3dbox.publish(msg)

        # 给点云打分类label并发布
        pub_msg = rendercloud(dtype_points, data["indices"], iswhat)
        pub_msg.header.frame_id = self._frame
        self._pub.publish(pub_msg)

        # 分析车斗体积: 1.split_cloud_numpy 2.boxes 3.iswhat
        trucks = analyze_truck(iswhat, 0, self._boxes, 10)
        indices_np = np.sum(np.array(data["indices"])[trucks], axis=0).astype(bool)
        pub_msg = rnp.msgify(pc2, dtype_points[indices_np])
        pub_msg.header.frame_id = self._frame
        self._pc2_pub_topic_truckpose.publish(pub_msg)

        t2 = time.time()
        print("totoal               : ", (t2 - t1) * 1000)
        print('-------------------------------------')


class PointNetServer:
    def __init__(self, name):
        self._action_name = name
        self._model_name = 'cls_model_39.pth'

        # for NetWork
        self._model_path = os.path.join(RosPack().get_path('pointnet_ros'), 'PointNet/utils/cls', self._model_name)
        self._num_classes = 4
        self._num_points = 2500
        self.classifierNet = model.PointNetCls(k=self._num_classes)  # 训练模型为16类
        self.classifierNet.load_state_dict(torch.load(self._model_path))
        self.classifierNet.cuda()
        self.classifierNet.eval()  # 测试模式

        # for ROS
        rospy.init_node(self._action_name + 'Server')
        self._server = actionlib.SimpleActionServer(self._action_name, pointnet_ros.msg.pointnetAction,
                                                    execute_cb=self.execute_callback, auto_start=False)
        self._result = pointnet_ros.msg.pointnetResult()
        self._server.start()

    def reset_server(self):
        self._server = actionlib.SimpleActionServer(self._action_name, pointnet_ros.msg.pointnetAction,
                                                    execute_cb=self.execute_callback, auto_start=False)
        self._server.start()
        pass

    @exetime
    def classifier(self, netdata):
        """
        分类器
        :param netdata: 归一化-中心化-数量标准化 的numpy格式点云数据(n,2500,3)
        :return: 分类结果list, 不可信分类结果值为-1
        """
        netdata = Variable(torch.from_numpy(np.array(netdata))).float()
        netdata = netdata.transpose(2, 1).cuda()  # 转换为(batch_size, channels, num)

        pred, _, _ = self.classifierNet(netdata)  # 输出log_softmax

        pred = torch.exp(pred.detach().cpu()).topk(1, dim=1)  # detach从计算图中返回一个无gradient的tensor。
        pro, iswhat = pred.values.numpy().reshape(-1), pred.indices.numpy().reshape(-1)  # 转换为行向量

        np.set_printoptions(precision=1)
        print("分类结果:", " ".join("{:3d}".format(i) for i in iswhat))
        print("分类概率:", " ".join("{:1.1f}".format(i) for i in pro))
        return pro, iswhat

    def execute_callback(self, goal):
        """
        ros消息回调函数
        :param goal: pc2-xyzl
        :return: none
        """
        t1 = time.time()

        # 分类 data = {"labels": [], "indices": [], "points": [], "netData": []}
        data = actiongoal2pointnet(goal.clouds, self._num_points)
        prob, labels = self.classifier(data)

        self._result.labels = labels.tolist()
        self._result.prob = prob.tolist()
        self._result.seq = goal.seq
        self._server.set_succeeded(self._result)

        t2 = time.time()
        print("totoal               : ", (t2 - t1) * 1000)
        print('-------------------------------------')
