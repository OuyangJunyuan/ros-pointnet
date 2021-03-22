# about system
import time

# about algorithm
import numpy as np
from scipy.spatial.transform import Rotation

# about ros
import rospy
import ros_numpy as rnp
from visualization_msgs.msg import Marker as mk
from sensor_msgs.msg import PointCloud2 as pc2
from geometry_msgs.msg import Point


def exetime(func):
    def newfunc(*args, **args2):
        _t0 = time.time()
        back = func(*args, **args2)
        _t1 = time.time()
        print("{:20s}".format(func.__name__) + " : " + "{:5.1f}".format((_t1 - _t0) * 1000) + "ms")
        return back

    return newfunc


@exetime
def ros2net_bridge(ros_msg, num_sample=2500):
    """
    将pc2-xyzl数据转换成网络输入
    :param num_sample:
    :param ros_msg:pc2-xyzl 数据
    :return: 输入msg对应np格式点云, 每个聚类对应的网络输入数据list, 每个聚类在原始数据中的PointIndices
    """
    data = rnp.numpify(ros_msg)  # 转换为dtype-numpy  耗时0.05ms
    points, labels = np.array([data['x'], data['y'], data['z']]).T, data['label']  # dypte-array转换为非结构化数组,耗时忽略不计
    # points(2500,3)
    output_dict = {"labels": [], "indices": [], "points": [], "netData": []}
    labels_range = range(1, int(max(labels) + 1))  # 输出和,有效聚类范围
    for i in labels_range:  # i: [1,label_max] 只读取大于大于0的label,0为小物体
        output_dict["labels"].append(i)
        output_dict["indices"].append((labels == int(i)))
        output_dict["points"].append(points[output_dict["indices"][-1]])
        obj = output_dict["points"][-1]  # 抽取label-cloud，只要0.0006
        c_obj = obj - np.expand_dims(np.mean(obj, axis=0), 0)  # 中心化
        i_obj = c_obj / np.max(np.sqrt(np.sum(c_obj ** 2, axis=1)), 0)  # 单位化x
        output_dict["netData"].append(i_obj[np.random.choice(len(i_obj), num_sample)])  # 重采样至2500个 (2500,3)
    print("shape:({},{},{})".format(int(max(labels)), num_sample, 3))
    # 使用np.copy 否则raw_data和ros_msg是共享内存的，其.flags.writable无法被修改。
    return np.copy(data), output_dict


@exetime
def actiongoal2pointnet(clouds, num_sample=2500):
    data = []
    for cloud in clouds:
        temp = rnp.numpify(cloud)
        data.append(np.array([temp['x'], temp['y'], temp['z']]).T[np.random.choice(len(temp), num_sample)])
    output = np.array(data)
    return output


# https://blog.csdn.net/suyunzzz/article/details/105962331
def get3dbox(w_cloud):
    """

    :param w_cloud: 单个物体点云按格式(3,n)
    :return:返回主成分，形心，尺寸(l,w,h)
    """
    cov = np.cov(w_cloud)  # temp.transpose(1, 0))  # cov输入数据行数表示变量数d，列数表示样本量n 得到cov=dxd

    vals, vecs = np.linalg.eig(cov)  # 求特征值和特征向量->主成分PC
    R_wc = np.array([vecs[0], vecs[1], np.cross(vecs[0], vecs[1])])  # rotation matrix respect to world
    t_wc = np.mean(w_cloud, axis=1)  # translation respect to world

    R_cw = R_wc.transpose(1, 0)
    t_cw = -1.0 * R_cw.dot(t_wc)

    c_cloud = R_cw.dot(w_cloud) + t_cw.reshape(-1, 1)
    pmax, pmin = np.max(c_cloud, axis=1), np.min(c_cloud, axis=1)

    t_wc += R_wc.dot(0.5 * (pmin + pmax))  # 变换到形心坐标系

    # T_cw = np.eye(4)  # Tcw: 世界系到点云质心系的变换
    # T_cw[0:3, 0:3] = R_cw
    # T_cw[0:3, 3] = -1.0 * R_cw.dot(t_wc)

    T_wc = np.eye(4)
    T_wc[0:3, 0:3] = R_wc
    T_wc[0:3, 3] = t_wc

    scale = pmax - pmin
    return T_wc, scale


def init_marker():
    msg = mk()
    msg.id = 0
    msg.action = mk.ADD
    msg.pose.orientation.w = 1

    msg.scale.x = 0.1
    msg.color.a = 1
    msg.color.r = 0
    msg.color.g = 1
    msg.color.b = 0
    msg.lifetime = rospy.Duration()
    return msg


def get_marker_delete_list(ns, frame, box_id):
    msg = init_marker()

    msg.ns = ns
    msg.id = box_id
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time.now()

    msg.type = mk.CUBE
    msg.action = mk.DELETE
    return msg


def get_marker_cube(T_wc, scale, ns, frame, box_id):
    msg = init_marker()

    msg.ns = ns
    msg.id = box_id
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time.now()

    msg.type = mk.CUBE
    msg.action = mk.ADD

    msg.pose.position.x = T_wc[0, 3]
    msg.pose.position.y = T_wc[1, 3]
    msg.pose.position.z = T_wc[2, 3]

    quat = Rotation.from_matrix(T_wc[0:3, 0:3]).as_quat()
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]

    msg.scale.x = scale[0]
    msg.scale.y = scale[1]
    msg.scale.z = scale[2]
    msg.color.a = 0.5
    msg.lifetime = rospy.Duration()

    return msg


def get_marker_line_list(ns, frame, box_id):
    msg = init_marker()

    msg.ns = ns
    msg.id = box_id
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time.now()

    msg.type = mk.LINE_LIST
    msg.action = mk.ADD
    msg.scale.x = 0.05

    return msg


def line_list_append(msg, T_wc, scale):
    vertex = np.zeros([8, 3])  # 为了不写8次
    for i in range(8):
        vertex[i] = np.array([-1 if (i >> j) & 0x01 == 0 else 1 for j in range(3)])

    vertex = T_wc[0:3, 0:3].dot((vertex * 0.5 * scale).transpose(1, 0)) + T_wc[0:3, 3].reshape(-1, 1)
    pts = [Point(vertex[0][i], vertex[1][i], vertex[2][i]) for i in range(8)]

    table = [0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7]
    for idx in table:
        msg.points.append(pts[idx])


@exetime
def volumefilter(split_cloud_numpy, iswhat, volume_threshold):
    boxes = []
    for i, cloud in enumerate(split_cloud_numpy):  # enumerate 用来获取索引和元素对
        if iswhat[i] < 0:
            continue
        T_cw, scale = get3dbox(cloud.transpose(1, 0))
        if np.prod(scale) < volume_threshold:
            iswhat[i] = -2
        else:
            boxes.append({"pose": T_cw, "scale": scale, "idx": i})
    return iswhat, boxes


@exetime
def rendercloud(np_pc, idx, iswhat):
    """
    打label
    :param np_pc: numpy 数据原始点云
    :param idx: 每个类别的下标
    :param iswhat: 分类结果
    :return: sensor_msg::PointCloud2 消息类型数据
    """
    np_pc.setflags(write=1)
    for i in range(len(iswhat)):
        np_pc['label'][idx[i]] = iswhat[i] + 1 if i != -1 else 0
    data = rnp.msgify(pc2, np_pc)
    return data


@exetime
def analyze_truck(iswhat, car_id, boxes, valume):
    """
    根据识别信息和3dbox获取车辆
    :param valume:
    :param car_id:
    :param cloud:
    :param iswhat:
    :param boxes:
    :return:
    """
    car_idx = [box["idx"] for i, box in enumerate(boxes) if
               iswhat[box["idx"]] == car_id and np.prod(box["scale"]) > valume]
    return car_idx

# trucks = analyze_truck(data["points"], iswhat, 0, self._boxes, 10)
# for box_idx in trucks:
#     pose = Pose()
#     pose.header.frame_id = self._frame
#     pose.header.stamp = rospy.Time.now()
#     pose.pose.position.x = self._boxes[box_idx]["pose"][0, 3]
#     pose.pose.position.y = self._boxes[box_idx]["pose"][1, 3]
#     pose.pose.position.z = self._boxes[box_idx]["pose"][2, 3]
#     quat = Rotation.from_matrix(self._boxes[box_idx]["pose"][0:3, 0:3]).as_quat()
#     pose.pose.orientation.x = quat[0]
#     pose.pose.orientation.y = quat[1]
#     pose.pose.orientation.z = quat[2]
#     pose.pose.orientation.w = quat[3]
#     self._truckpose.publish(pose)
