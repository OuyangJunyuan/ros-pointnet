//
// Created by ou on 2021/4/17.
//

#ifndef POINTNET_ROS_POINTNET_ROS_H
#define POINTNET_ROS_POINTNET_ROS_H

namespace pointnet_ros {
    enum PointNetLabel {
        TOO_SMALL_VOLUME = -2,
        TOO_SMALL_PROB = -1,
        CAR = 0,
        IRON_PILE = 1,
        PEDESTRIAN = 2,
        TREE = 3
    };
};

#endif //POINTNET_ROS_POINTNET_ROS_H
