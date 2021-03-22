from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

# 优化器：adam-Adaptive Moment Estimation(自适应矩估计)，利用梯度的一阶矩和二阶矩动态调整每个参数的学习率
# betas：用于计算梯度一阶矩和二阶矩的系数
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
# 学习率调整：每个step_size次epoch后，学习率x0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# Moves all model parameters and buffers to the GPU.
classifier.cuda()

num_batch = len(dataset) / opt.batchSize  # 计算batch的数量

for epoch in range(opt.nepoch):
    scheduler.step()
    # 将一个可遍历对象组合为一个索引序列，同时列出数据和数据下标,(0, seq[0])...
    # __init__(self, iterable, start=0)，参数为可遍历对象及起始位置
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]  # 取所有行的第0列
        points = points.transpose(2, 1)  # 维度交换
        points, target = points.cuda(), target.cuda()  # tensor转到cuda上
        optimizer.zero_grad()  # 梯度清除，避免backward时梯度累加
        classifier = classifier.train()  # 训练模式，使能BN和dropout
        pred, trans, trans_feat = classifier(points)  # 网络结果预测输出
        loss = F.nll_loss(pred, target)  # 损失函数：负log似然损失，在分类网络中使用了log_softmax，二者结合其实就是交叉熵损失函数
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()  # loss反向传播
        optimizer.step()  # 梯度下降，参数优化
        pred_choice = pred.data.max(1)[1]  # max(1)返回每一行中的最大值及索引,[1]取出索引（代表着类别）
        correct = pred_choice.eq(target.data).cpu().sum()  # 判断和target是否匹配，并计算匹配的数量
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        # 每10次batch之后，进行一次测试
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()  # 测试模式，固定住BN和dropout
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))