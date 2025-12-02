"""
评估工具
用于验证集评估和指标计算
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
from typing import Tuple, List, Optional
from tqdm import tqdm


def extract_features(model: nn.Module, dataloader, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    从数据集中提取特征

    Args:
        model: 特征提取模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        features: 特征数组 (N, embedding_size)
        magnitudes: 幅度数组 (N,)
    """
    model.eval()
    features_list = []
    magnitudes_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='提取特征'):
            images = batch[0].to(device)

            # 提取特征
            embeddings = model(images)

            # 计算幅度
            magnitudes = torch.norm(embeddings, dim=1)

            # L2 归一化
            features = F.normalize(embeddings, dim=1)

            features_list.append(features.cpu().numpy())
            magnitudes_list.append(magnitudes.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    magnitudes = np.concatenate(magnitudes_list, axis=0)

    return features, magnitudes


def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.ndarray:
    """
    计算余弦相似度

    Args:
        feat1: 特征1 (N, D) 或 (D,)
        feat2: 特征2 (M, D) 或 (D,)

    Returns:
        相似度矩阵 (N, M) 或标量
    """
    if feat1.ndim == 1:
        feat1 = feat1.reshape(1, -1)
    if feat2.ndim == 1:
        feat2 = feat2.reshape(1, -1)

    # 归一化
    feat1 = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True)
    feat2 = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True)

    similarity = np.dot(feat1, feat2.T)

    return similarity


def evaluate_lfw(
    model: nn.Module,
    dataloader,
    device: str = 'cuda',
    flip: bool = True
) -> Tuple[float, float, float]:
    """
    在 LFW 数据集上评估

    Args:
        model: 模型
        dataloader: LFW 数据加载器
        device: 设备
        flip: 是否使用水平翻转增强

    Returns:
        accuracy: 准确率
        best_threshold: 最佳阈值
        auc_score: AUC 分数
    """
    model.eval()

    embeddings1_list = []
    embeddings2_list = []
    labels_list = []

    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc='LFW 评估'):
            img1, img2 = img1.to(device), img2.to(device)

            # 提取特征
            emb1 = model(img1)
            emb2 = model(img2)

            if flip:
                # 水平翻转增强
                emb1_flip = model(torch.flip(img1, dims=[3]))
                emb2_flip = model(torch.flip(img2, dims=[3]))
                emb1 = (emb1 + emb1_flip) / 2
                emb2 = (emb2 + emb2_flip) / 2

            # 归一化
            emb1 = F.normalize(emb1, dim=1)
            emb2 = F.normalize(emb2, dim=1)

            embeddings1_list.append(emb1.cpu().numpy())
            embeddings2_list.append(emb2.cpu().numpy())
            labels_list.append(label.numpy())

    embeddings1 = np.concatenate(embeddings1_list, axis=0)
    embeddings2 = np.concatenate(embeddings2_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # 计算相似度
    similarities = np.sum(embeddings1 * embeddings2, axis=1)

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    auc_score = auc(fpr, tpr)

    # 找到最佳阈值
    best_threshold, best_accuracy = find_best_threshold(similarities, labels)

    return best_accuracy, best_threshold, auc_score


def find_best_threshold(
    similarities: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 1000
) -> Tuple[float, float]:
    """
    寻找最佳阈值

    Args:
        similarities: 相似度数组
        labels: 标签数组 (1=同一人, 0=不同人)
        num_thresholds: 搜索的阈值数量

    Returns:
        best_threshold: 最佳阈值
        best_accuracy: 最佳准确率
    """
    thresholds = np.linspace(similarities.min(), similarities.max(), num_thresholds)

    best_accuracy = 0.0
    best_threshold = 0.0

    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)
        accuracy = np.mean(predictions == labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def evaluate_verification(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    labels: np.ndarray,
    nfolds: int = 10
) -> Tuple[float, float, float, List[float]]:
    """
    交叉验证评估人脸验证性能

    Args:
        embeddings1: 第一组特征 (N, D)
        embeddings2: 第二组特征 (N, D)
        labels: 标签
        nfolds: 交叉验证折数

    Returns:
        accuracy: 平均准确率
        std: 标准差
        best_threshold: 最佳阈值
        accuracies: 每折准确率列表
    """
    assert len(embeddings1) == len(embeddings2) == len(labels)

    n = len(labels)
    fold_size = n // nfolds

    # 计算所有相似度
    similarities = np.sum(embeddings1 * embeddings2, axis=1)

    accuracies = []
    thresholds = []

    for fold in range(nfolds):
        # 划分验证集和训练集
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size

        val_indices = np.arange(val_start, val_end)
        train_indices = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])

        # 在训练集上找最佳阈值
        train_similarities = similarities[train_indices]
        train_labels = labels[train_indices]
        threshold, _ = find_best_threshold(train_similarities, train_labels)

        # 在验证集上评估
        val_similarities = similarities[val_indices]
        val_labels = labels[val_indices]
        predictions = (val_similarities >= threshold).astype(int)
        accuracy = np.mean(predictions == val_labels)

        accuracies.append(accuracy)
        thresholds.append(threshold)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_threshold = np.mean(thresholds)

    return mean_accuracy, std_accuracy, mean_threshold, accuracies


def calculate_tar_far(
    similarities: np.ndarray,
    labels: np.ndarray,
    far_target: float = 0.001
) -> Tuple[float, float]:
    """
    计算特定 FAR 下的 TAR

    Args:
        similarities: 相似度数组
        labels: 标签数组
        far_target: 目标 FAR

    Returns:
        tar: True Accept Rate
        threshold: 对应的阈值
    """
    fpr, tpr, thresholds = roc_curve(labels, similarities)

    # 找到最接近目标 FAR 的点
    idx = np.argmin(np.abs(fpr - far_target))

    tar = tpr[idx]
    threshold = thresholds[idx]

    return tar, threshold


class MagnitudeDistributionAnalyzer:
    """
    特征幅度分布分析器
    用于监控 MagFace 的训练效果
    """

    def __init__(self):
        self.magnitudes = []
        self.labels = []

    def add(self, magnitudes: np.ndarray, labels: np.ndarray):
        """添加数据"""
        self.magnitudes.extend(magnitudes.tolist())
        self.labels.extend(labels.tolist())

    def reset(self):
        """重置"""
        self.magnitudes = []
        self.labels = []

    def get_statistics(self) -> dict:
        """
        获取统计信息

        Returns:
            包含均值、标准差、分位数等的字典
        """
        magnitudes = np.array(self.magnitudes)

        return {
            'mean': np.mean(magnitudes),
            'std': np.std(magnitudes),
            'min': np.min(magnitudes),
            'max': np.max(magnitudes),
            'median': np.median(magnitudes),
            'q25': np.percentile(magnitudes, 25),
            'q75': np.percentile(magnitudes, 75),
        }

    def quality_distribution(self, quality_bins: List[float] = [0, 20, 40, 60, 80, 110]) -> dict:
        """
        按质量分数分布

        Args:
            quality_bins: 分箱边界

        Returns:
            各区间的样本比例
        """
        magnitudes = np.array(self.magnitudes)
        distribution = {}

        for i in range(len(quality_bins) - 1):
            low, high = quality_bins[i], quality_bins[i + 1]
            count = np.sum((magnitudes >= low) & (magnitudes < high))
            ratio = count / len(magnitudes)
            distribution[f'{low}-{high}'] = {
                'count': int(count),
                'ratio': float(ratio)
            }

        return distribution
