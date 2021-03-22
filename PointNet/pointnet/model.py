from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


# T-Net: is a pointnet itself.获取3x3的变换矩阵，校正点云姿态；效果一般，后续的改进并没有再加入这部分
# 经过全连接层映射到9个数据，最后调整为3x3矩阵
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        # mlp
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  # in out kernel_size
        # mlp
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # fc
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        #
        self.relu = nn.ReLU()
        # bn
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # Variable已被弃用，之前的版本中，pytorch的tensor只能在CPU计算，Variable将tensor转换成variable，具有三个属性（data\grad\grad_fn）
        # 现在二者已经融合，Variable返回tensor
        # iden生成单位变换矩阵
        # repeat(batchsize, 1)，重复batchsize次，生成batchsize x 9的tensor
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        # view()相当于numpy中的resize(),重构tensor维度，-1表示缺省参数由系统自动计算（为batchsize大小）
        # 返回结果为 batchsize x 3 x 3
        x = x.view(-1, 3, 3)
        return x


# 数据为k维，用于mlp之后的高维特征，同上
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# backbone
class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        # mlp之后的64高维数据，feature transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # x = (batch_size, dim, n)
        n_pts = x.size()[2]  # size()返回张量各个维度的尺度
        trans = self.stn(x)  # STN3网络 调整姿态
        x = x.transpose(2, 1)  # 交换tensor的两个维度，将点云数据转换为3xn形式，便于和旋转矩阵计算
        x = torch.bmm(x, trans)  # 两个batch矩阵乘法
        x = x.transpose(2, 1)  # 计算完成，转换为原始形式
        # cnn中经典 conv - bn - relu 结构
        x = F.relu(self.bn1(self.conv1(x)))  # 第一次mlp，每个点由3维升为64维

        # 是否进行feature_transform
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x  # 保留经过第一次mlp的特征，便于后续分割进行特征拼接融合
        x = F.relu(self.bn2(self.conv2(x)))  # 第二次mlp的第一层，64->128
        x = self.bn3(self.conv3(x))  # 第二次mlp的第二层，128->1024
        x = torch.max(x, 2, keepdim=True)[0]  # pointnet的核心操作，最大池化操作保证了点云的置换不变性（最大池化操作为对称函数）
        x = x.view(-1, 1024)  # resize池化结果的形状，获得全局1024维特征
        if self.global_feat:  # 全局特征，true：不进行局部特征的连接，用于分类；false进行局部特征的连接，用于分割
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


# 分类网络
class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)  # k为类别数目
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    # 在执行model(data)时，forward()函数自动调用
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)  # backbone
        x = F.relu(self.bn1(self.fc1(x)))  # 第三次mlp的第一层：1024->512
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))  # 第三次mlp的第二层：512->256
        x = self.fc3(x)  # 全连接得到k维
        return F.log_softmax(x, dim=1), trans, trans_feat  # log_softmax分类，解决softmax在计算e的次方时容易造成的上溢出和下溢出问题


# 分割网络
class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]  # size()返回张量各个维度的尺度
        n_pts = x.size()[2]  # 每个物体的点数
        x, trans, trans_feat = self.feat(x)  # backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]  # （b,c,n） 这里d是维度即channel
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]  # 丢去batch_size维度
    if trans.is_cuda:
        I = I.cuda()
    #  正则，惩罚项。  |TT'-I|为惩罚项，用来使得学习得到的变换矩阵接近正交矩阵，并且求去batch中所有trans的均值，得到的结果丢弃维度1和2，保留列batch_size。
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k=5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k=3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
