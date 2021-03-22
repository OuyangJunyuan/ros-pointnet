from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement


def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))


def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')  # 路径拼接
        self.cat = {}
        self.data_augmentation = data_augmentation  # 数据扩充
        self.classification = classification
        self.seg_classes = {}

        # with expression [as target]: expression-需要执行的表达式；target-变量或元祖，存储expression执行的结果
        with open(self.catfile, 'r') as f:  # 打开目录txt文件，'r':open for reading
            for line in f:
                # strip():移除字符串头尾指定的字符（默认为空格或换行符）
                # split():指定分隔符对字符串进行切片，返回分割后的字符串列表(默认为所有的空字符，包括空格、换行\n、制表符\t等)
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]  # cat为字典，通过[键]索引。键：类别；值：文件夹名称
        # print(self.cat)
        if not class_choice is None:  # 类别选择，对那些种类物体进行分类
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}  # key和value互换

        self.meta = {}
        # json文件类似xml文件，可存储键值对和数组等
        # split=train
        # format()：字符串格式化函数，使用{}代替之前的%
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        # for item in self.cat：item为键
        # for item in self.cat.values():item为值
        # for item in self.cat.items():item为键值对（元组的形式）
        # for k, v in self.cat.items():更为规范的键值对读取方式
        for item in self.cat:
            self.meta[item] = []  # meta为字典，键为类别，键值为空

        for file in filelist:  # 读取shuffled_train_file_list.json
            _, category, uuid = file.split('/')  # category为某一类别所在文件夹，uuid为某一类别的某一个
            if category in self.cat.values():
                # points_label路径生成，包括原始点云及用于分割的标签
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label',
                                                                      uuid + '.seg')))

        self.datapath = []
        for item in self.cat:  # cat存储类别及其所在文件夹，item访问键，即类别
            for fn in self.meta[item]:  # meta为字典，fn访问值，即路径
                self.datapath.append((item, fn[0], fn[1]))  # item为类别，fn[0]为点云路径，fn[1]为用于分割的标签路径
        # sorted():对所有可迭代兑现进行排序，默认为升序；sorted(self.cat)对字典cat中的键（种类）进行排序
        # zip():  函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
        # dict(): 创建字典。dict(zip(['one', 'two'], [1, 2])) -> {'two': 2, 'one': 1}
        # 下列操作实现了对类别进行数字编码表示
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    # 该方法的实例对象可通过索引取值，自动调用该方法
    def __getitem__(self, index):
        fn = self.datapath[index]  # 获取类别、点云路径、分割标签路径元组
        cls = self.classes[self.datapath[index][0]]  # 获取数字编码的类别标签
        point_set = np.loadtxt(fn[1]).astype(np.float32)  # 读取pts点云
        seg = np.loadtxt(fn[2]).astype(np.int64)  # 读取分割标签
        # print(point_set.shape, seg.shape)
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # 数据增强
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.fns)


class MydataSet(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')  # 路径拼接
        self.cat = {}
        self.data_augmentation = data_augmentation  # 数据扩充
        self.classification = classification
        self.seg_classes = {}

        # with expression [as target]: expression-需要执行的表达式；target-变量或元祖，存储expression执行的结果
        with open(self.catfile, 'r') as f:  # 打开目录txt文件，'r':open for reading
            for line in f:
                # strip():移除字符串头尾指定的字符（默认为空格或换行符）
                # split():指定分隔符对字符串进行切片，返回分割后的字符串列表(默认为所有的空字符，包括空格、换行\n、制表符\t等)
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]  # cat为字典，通过[键]索引。键：类别；值：文件夹名称
        # print(self.cat)
        if not class_choice is None:  # 类别选择，对那些种类物体进行分类
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}  # key和value互换

        self.meta = {}
        # json文件类似xml文件，可存储键值对和数组等
        # split=train
        # format()：字符串格式化函数，使用{}代替之前的%
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        # for item in self.cat：item为键
        # for item in self.cat.values():item为值
        # for item in self.cat.items():item为键值对（元组的形式）
        # for k, v in self.cat.items():更为规范的键值对读取方式
        for item in self.cat:
            self.meta[item] = []  # meta为字典，键为类别，键值为空

        for file in filelist:  # 读取shuffled_train_file_list.json
            _, category, uuid = file.split('/')  # category为某一类别所在文件夹，uuid为某一类别的某一个
            if category in self.cat.values():
                # points_label路径生成，包括原始点云及用于分割的标签
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid)))

        self.datapath = []
        for item in self.cat:  # cat存储类别及其所在文件夹，item访问键，即类别
            for fn in self.meta[item]:  # meta为字典，fn访问值，即路径
                self.datapath.append((item, fn))  # item为类别，fn[0]为点云路径，fn[1]为用于分割的标签路径
        # sorted():对所有可迭代兑现进行排序，默认为升序；sorted(self.cat)对字典cat中的键（种类）进行排序
        # zip():  函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
        # dict(): 创建字典。dict(zip(['one', 'two'], [1, 2])) -> {'two': 2, 'one': 1}
        # 下列操作实现了对类别进行数字编码表示
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)

    # 该方法的实例对象可通过索引取值，自动调用该方法
    def __getitem__(self, index):
        fn = self.datapath[index]  # 获取类别、点云路径、分割标签路径元组
        cls = self.classes[self.datapath[index][0]]  # 获取数字编码的类别标签
        point_set = np.loadtxt(fn[1]).astype(np.float32)  # 读取pts点云
        # print(point_set.shape, seg.shape)
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:  # 数据增强
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set, cls

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root=datapath, class_choice=['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(), seg.type())

        d = ShapeNetDataset(root=datapath, classification=True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(), cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])
