"""
数据集模块
支持 MS1M (rec格式) 和常见验证集的加载
"""

import os
import numbers
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
from typing import Tuple, List, Optional
import cv2


class FaceDataset(Dataset):
    """
    通用人脸数据集类
    支持从文件夹加载图像
    """

    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        """
        初始化数据集

        Args:
            root_dir: 数据根目录
            transform: 图像变换
            is_train: 是否为训练集
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # 收集所有图像和标签
        self.samples = []
        self.labels = []
        self.label_to_idx = {}

        self._scan_directory()

    def _scan_directory(self):
        """扫描目录结构，收集图像路径和标签"""
        idx = 0
        for identity in sorted(os.listdir(self.root_dir)):
            identity_path = os.path.join(self.root_dir, identity)
            if not os.path.isdir(identity_path):
                continue

            self.label_to_idx[identity] = idx

            for img_name in os.listdir(identity_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(identity_path, img_name)
                    self.samples.append(img_path)
                    self.labels.append(idx)

            idx += 1

        self.num_classes = idx
        print(f"加载数据集: {len(self.samples)} 张图片, {self.num_classes} 个身份")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        label = self.labels[idx]

        # 加载图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class MXFaceDataset(Dataset):
    """
    MS1M 数据集加载器 (MXNet RecordIO 格式)
    兼容 InsightFace 提供的数据集格式
    """

    def __init__(self, root_dir: str, transform=None):
        """
        初始化 MXNet 格式数据集

        Args:
            root_dir: 数据目录，包含 train.rec, train.idx
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform

        # 加载索引
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')

        try:
            import mxnet as mx
            self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        except ImportError:
            print("警告: mxnet 未安装，无法加载 rec 格式数据")
            print("请运行: pip install mxnet")
            self.imgrec = None
            return

        # 读取头信息
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)

        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        self.num_classes = int(header.label[0]) if header.flag > 0 else len(set(self.labels))
        print(f"加载 MS1M 数据集: {len(self.imgidx)} 张图片")

    def __len__(self):
        if self.imgrec is None:
            return 0
        return len(self.imgidx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        import mxnet as mx

        s = self.imgrec.read_idx(self.imgidx[idx])
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = int(label)

        # 解码图像
        sample = mx.image.imdecode(img).asnumpy()
        sample = Image.fromarray(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class LFWDataset(Dataset):
    """
    LFW 验证数据集
    用于评估模型在人脸验证任务上的性能
    """

    def __init__(self, root_dir: str, pairs_file: str, transform=None):
        """
        初始化 LFW 数据集

        Args:
            root_dir: LFW 图像目录
            pairs_file: pairs.txt 文件路径
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform

        # 解析 pairs 文件
        self.pairs = []
        self.labels = []
        self._parse_pairs(pairs_file)

    def _parse_pairs(self, pairs_file: str):
        """解析 LFW pairs 文件"""
        with open(pairs_file, 'r') as f:
            lines = f.readlines()[1:]  # 跳过第一行

        for line in lines:
            parts = line.strip().split('\t')

            if len(parts) == 3:
                # 同一人
                name = parts[0]
                idx1, idx2 = int(parts[1]), int(parts[2])
                path1 = os.path.join(self.root_dir, name, f"{name}_{idx1:04d}.jpg")
                path2 = os.path.join(self.root_dir, name, f"{name}_{idx2:04d}.jpg")
                self.pairs.append((path1, path2))
                self.labels.append(1)

            elif len(parts) == 4:
                # 不同人
                name1, idx1 = parts[0], int(parts[1])
                name2, idx2 = parts[2], int(parts[3])
                path1 = os.path.join(self.root_dir, name1, f"{name1}_{idx1:04d}.jpg")
                path2 = os.path.join(self.root_dir, name2, f"{name2}_{idx2:04d}.jpg")
                self.pairs.append((path1, path2))
                self.labels.append(0)

        print(f"LFW 数据集: {len(self.pairs)} 对图像")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path1, path2 = self.pairs[idx]
        label = self.labels[idx]

        # 加载图像
        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


def get_train_transforms(image_size: int = 112) -> transforms.Compose:
    """
    获取训练数据增强变换

    Args:
        image_size: 图像大小

    Returns:
        transforms.Compose 对象
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transforms(image_size: int = 112) -> transforms.Compose:
    """
    获取验证/测试数据变换

    Args:
        image_size: 图像大小

    Returns:
        transforms.Compose 对象
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_train_dataloader(data_dir: str, batch_size: int, num_workers: int = 4,
                         image_size: int = 112, use_mxnet: bool = True) -> DataLoader:
    """
    获取训练数据加载器

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像大小
        use_mxnet: 是否使用 MXNet 格式数据

    Returns:
        DataLoader 对象
    """
    transform = get_train_transforms(image_size)

    if use_mxnet and os.path.exists(os.path.join(data_dir, 'train.rec')):
        dataset = MXFaceDataset(data_dir, transform=transform)
    else:
        dataset = FaceDataset(data_dir, transform=transform, is_train=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


def get_val_dataloader(data_dir: str, pairs_file: str, batch_size: int,
                       num_workers: int = 4, image_size: int = 112) -> DataLoader:
    """
    获取验证数据加载器

    Args:
        data_dir: 数据目录
        pairs_file: pairs 文件路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像大小

    Returns:
        DataLoader 对象
    """
    transform = get_val_transforms(image_size)
    dataset = LFWDataset(data_dir, pairs_file, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == '__main__':
    print("数据集模块测试")
    print("=" * 50)

    # 测试 transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    print("训练变换:", train_transform)
    print("验证变换:", val_transform)
