"""
MagFace Loss 实现
核心创新：利用特征向量的幅度（magnitude）来度量样本质量，
高质量样本使用更大的margin，低质量样本使用更小的margin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Tuple, Optional


class MagFaceLoss(nn.Module):
    """
    MagFace Loss

    核心公式:
    1. 幅度限制: a_i = clamp(||f_i||, l_a, u_a)
    2. Margin 函数: m(a_i) = (u_m - l_m) / (u_a - l_a) * (a_i - l_a) + l_m
    3. 正则项: g(a_i) = 1/a_i + a_i/u_a^2
    4. 总 Loss: L = L_cls + lambda_g * g(a_i)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        scale: float = 64.0,
        l_a: float = 10.0,      # 幅度下界
        u_a: float = 110.0,     # 幅度上界
        l_m: float = 0.45,      # Margin 下界
        u_m: float = 0.8,       # Margin 上界
        lambda_g: float = 35.0, # 正则项系数
        easy_margin: bool = False
    ):
        """
        初始化 MagFace Loss

        Args:
            num_classes: 类别数（身份数）
            embedding_size: 特征维度
            scale: 缩放因子 s
            l_a: 幅度下界 (论文默认 10)
            u_a: 幅度上界 (论文默认 110)
            l_m: Margin 下界 (论文默认 0.45)
            u_m: Margin 上界 (论文默认 0.8)
            lambda_g: 正则项系数 (论文默认 35)
            easy_margin: 是否使用 easy margin
        """
        super().__init__()

        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        self.lambda_g = lambda_g
        self.easy_margin = easy_margin

        # 分类权重矩阵 W (num_classes, embedding_size)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # 预计算 margin 斜率
        self.margin_slope = (u_m - l_m) / (u_a - l_a)

        print(f"MagFace Loss 初始化:")
        print(f"  - 类别数: {num_classes}")
        print(f"  - 特征维度: {embedding_size}")
        print(f"  - Scale: {scale}")
        print(f"  - 幅度范围: [{l_a}, {u_a}]")
        print(f"  - Margin 范围: [{l_m}, {u_m}]")
        print(f"  - Lambda_g: {lambda_g}")

    def calc_margin(self, x_norm: Tensor) -> Tensor:
        """
        计算 margin m(a_i)

        公式: m(a_i) = (u_m - l_m) / (u_a - l_a) * (a_i - l_a) + l_m

        Args:
            x_norm: 特征幅度 (B,)

        Returns:
            对应的 margin 值 (B,)
        """
        margin = self.margin_slope * (x_norm - self.l_a) + self.l_m
        return margin

    def calc_regularizer(self, x_norm: Tensor) -> Tensor:
        """
        计算正则项 g(a_i)

        公式: g(a_i) = 1/a_i + a_i/u_a^2

        这是一个凸函数，鼓励特征幅度向最优值靠拢

        Args:
            x_norm: 特征幅度 (B,)

        Returns:
            正则项值 (B,)
        """
        g = 1.0 / x_norm + x_norm / (self.u_a ** 2)
        return g

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        前向传播

        Args:
            embeddings: 特征向量 (B, embedding_size)，注意：不要预先归一化
            labels: 标签 (B,)

        Returns:
            loss: 总损失
            x_norm: 特征幅度 (用于监控)
            g: 正则项值 (用于监控)
        """
        # Step 1: 计算特征幅度并限制范围
        x_norm = torch.norm(embeddings, dim=1, keepdim=True)  # (B, 1)
        x_norm_clamp = torch.clamp(x_norm, self.l_a, self.u_a)  # (B, 1)

        # Step 2: L2 归一化特征和权重
        x_normalized = embeddings / (x_norm + 1e-10)  # (B, embedding_size)
        w_normalized = F.normalize(self.weight, dim=1)  # (num_classes, embedding_size)

        # Step 3: 计算余弦相似度
        cos_theta = F.linear(x_normalized, w_normalized)  # (B, num_classes)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # Step 4: 计算每个样本的 margin
        ada_margin = self.calc_margin(x_norm_clamp.squeeze(1))  # (B,)

        # Step 5: 计算 cos(theta + m)
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        cos_m = torch.cos(ada_margin)  # (B,)
        sin_m = torch.sin(ada_margin)  # (B,)

        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)  # (B, num_classes)

        # 只对真实标签位置应用 margin
        # cos(theta_yi + m_i)
        cos_theta_m = cos_theta * cos_m.unsqueeze(1) - sin_theta * sin_m.unsqueeze(1)

        if self.easy_margin:
            # easy margin: 当 cos(theta) > 0 时才应用 margin
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            # 标准 margin: 保证单调性
            # 当 theta + m > pi 时，使用 cos(theta) - mm
            mm = sin_m * ada_margin  # m * sin(m)
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta > threshold.unsqueeze(1),
                cos_theta_m,
                cos_theta - mm.unsqueeze(1)
            )

        # Step 6: 创建 one-hot 标签，只在真实类别位置应用 margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # 只对正确类别应用 margin
        output = cos_theta * (1 - one_hot) + cos_theta_m * one_hot

        # Step 7: 缩放并计算交叉熵损失
        output = output * self.scale
        cls_loss = F.cross_entropy(output, labels)

        # Step 8: 计算正则项
        g = self.calc_regularizer(x_norm_clamp.squeeze(1))  # (B,)
        reg_loss = self.lambda_g * g.mean()

        # 总损失
        total_loss = cls_loss + reg_loss

        return total_loss, x_norm.squeeze(1), g


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (对比基准)

    固定 margin，不考虑样本质量
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        # L2 归一化
        x_normalized = F.normalize(embeddings, dim=1)
        w_normalized = F.normalize(self.weight, dim=1)

        # 余弦相似度
        cos_theta = F.linear(x_normalized, w_normalized)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # cos(theta + m)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)

        # one-hot
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = cos_theta * (1 - one_hot) + cos_theta_m * one_hot
        output = output * self.scale

        return F.cross_entropy(output, labels)


class MagFaceModel(nn.Module):
    """
    完整的 MagFace 模型
    包含 Backbone + MagFace Loss
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        embedding_size: int = 512,
        scale: float = 64.0,
        l_a: float = 10.0,
        u_a: float = 110.0,
        l_m: float = 0.45,
        u_m: float = 0.8,
        lambda_g: float = 35.0
    ):
        super().__init__()

        self.backbone = backbone
        self.head = MagFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            scale=scale,
            l_a=l_a,
            u_a=u_a,
            l_m=l_m,
            u_m=u_m,
            lambda_g=lambda_g
        )

    def forward(self, x: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        训练时的前向传播

        Args:
            x: 输入图像 (B, 3, 112, 112)
            labels: 标签 (B,)

        Returns:
            loss: 总损失
            x_norm: 特征幅度
            g: 正则项
        """
        embeddings = self.backbone(x)
        return self.head(embeddings, labels)

    def extract_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        推理时提取特征

        Args:
            x: 输入图像 (B, 3, 112, 112)

        Returns:
            features: 归一化特征向量 (B, embedding_size)
            magnitude: 特征幅度/质量分数 (B,)
        """
        with torch.no_grad():
            embeddings = self.backbone(x)
            magnitude = torch.norm(embeddings, dim=1)
            features = F.normalize(embeddings, dim=1)

        return features, magnitude


if __name__ == '__main__':
    # 测试代码
    print("MagFace Loss 测试")
    print("=" * 50)

    # 创建 Loss
    loss_fn = MagFaceLoss(
        num_classes=85742,
        embedding_size=512,
        scale=64.0,
        l_a=10.0,
        u_a=110.0,
        l_m=0.45,
        u_m=0.8,
        lambda_g=35.0
    )

    # 模拟输入
    batch_size = 32
    embeddings = torch.randn(batch_size, 512) * 50  # 模拟不同幅度的特征
    labels = torch.randint(0, 85742, (batch_size,))

    # 计算 Loss
    total_loss, x_norm, g = loss_fn(embeddings, labels)

    print(f"总损失: {total_loss.item():.4f}")
    print(f"特征幅度 - 均值: {x_norm.mean().item():.2f}, 范围: [{x_norm.min().item():.2f}, {x_norm.max().item():.2f}]")
    print(f"正则项 - 均值: {g.mean().item():.4f}")

    # 测试 margin 计算
    print("\n" + "=" * 50)
    print("Margin 计算测试:")
    test_norms = torch.tensor([10.0, 50.0, 80.0, 110.0])
    margins = loss_fn.calc_margin(test_norms)
    for n, m in zip(test_norms, margins):
        print(f"  幅度 {n:.1f} -> Margin {m:.3f}")
