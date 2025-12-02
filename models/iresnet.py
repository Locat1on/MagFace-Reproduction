"""
iResNet (Improved ResNet) 骨干网络
用于人脸识别特征提取，输出 512 维特征向量
注意：与标准 ResNet 不同，最后没有 ReLU 激活
"""

import torch
import torch.nn as nn
from typing import List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 卷积，带 padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """
    iResNet 基本残差块
    结构: BN -> Conv -> BN -> PReLU -> Conv -> BN -> SE (可选)
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        use_se: bool = False
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IBasicBlock 只支持 groups=1 和 base_width=64')
        if dilation > 1:
            raise NotImplementedError('IBasicBlock 不支持 dilation > 1')

        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-05, momentum=0.9)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.downsample = downsample
        self.stride = stride

        # SE 模块 (Squeeze-and-Excitation)
        self.use_se = use_se
        if use_se:
            self.se = SEModule(planes, reduction=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation 模块
    用于通道注意力机制
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class IResNet(nn.Module):
    """
    Improved ResNet 用于人脸识别

    特点:
    1. 使用 PReLU 而非 ReLU
    2. BN 在卷积之前
    3. 最后没有激活函数，直接输出特征
    4. 可选 SE 模块
    """

    fc_scale: int = 7 * 7

    def __init__(
        self,
        block: type,
        layers: List[int],
        dropout: float = 0.0,
        num_features: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        use_se: bool = False
    ):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation 长度必须为 3")

        # 输入层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(self.inplanes)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, use_se=use_se)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], use_se=use_se)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], use_se=use_se)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], use_se=use_se)

        # 输出层
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=2e-05, momentum=0.9)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False  # 固定 BN 权重为 1

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        use_se: bool = False
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=2e-05, momentum=0.9),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, use_se=use_se
        ))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation, use_se=use_se
            ))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_embedding_before_bn: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, 112, 112)
            return_embedding_before_bn: 是否返回 BN 前的原始特征 (用于质量分数)

        Returns:
            特征向量 (B, 512)，注意：不进行 L2 归一化
            或 (B, 512), (B, 512) 若 return_embedding_before_bn=True
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        embedding_before_bn = x.clone()  # BN 前的原始特征
        x = self.features(x)

        if return_embedding_before_bn:
            return x, embedding_before_bn
        else:
            return x


# ============== 预定义模型 ==============

def iresnet18(dropout: float = 0.0, num_features: int = 512, **kwargs) -> IResNet:
    """iResNet-18"""
    return IResNet(IBasicBlock, [2, 2, 2, 2], dropout=dropout, num_features=num_features, **kwargs)


def iresnet34(dropout: float = 0.0, num_features: int = 512, **kwargs) -> IResNet:
    """iResNet-34"""
    return IResNet(IBasicBlock, [3, 4, 6, 3], dropout=dropout, num_features=num_features, **kwargs)


def iresnet50(dropout: float = 0.0, num_features: int = 512, **kwargs) -> IResNet:
    """iResNet-50"""
    return IResNet(IBasicBlock, [3, 4, 14, 3], dropout=dropout, num_features=num_features, **kwargs)


def iresnet100(dropout: float = 0.0, num_features: int = 512, **kwargs) -> IResNet:
    """iResNet-100 (论文默认使用)"""
    return IResNet(IBasicBlock, [3, 13, 30, 3], dropout=dropout, num_features=num_features, **kwargs)


def iresnet200(dropout: float = 0.0, num_features: int = 512, **kwargs) -> IResNet:
    """iResNet-200"""
    return IResNet(IBasicBlock, [6, 26, 60, 6], dropout=dropout, num_features=num_features, **kwargs)


# SE 版本
def iresnet50_se(dropout: float = 0.0, num_features: int = 512, **kwargs) -> IResNet:
    """iResNet-50 with SE"""
    return IResNet(IBasicBlock, [3, 4, 14, 3], dropout=dropout, num_features=num_features, use_se=True, **kwargs)


def iresnet100_se(dropout: float = 0.0, num_features: int = 512, **kwargs) -> IResNet:
    """iResNet-100 with SE"""
    return IResNet(IBasicBlock, [3, 13, 30, 3], dropout=dropout, num_features=num_features, use_se=True, **kwargs)


def get_backbone(name: str, **kwargs) -> IResNet:
    """
    获取骨干网络

    Args:
        name: 网络名称，如 'iresnet50', 'iresnet100'
        **kwargs: 其他参数

    Returns:
        IResNet 实例
    """
    backbone_dict = {
        'iresnet18': iresnet18,
        'iresnet34': iresnet34,
        'iresnet50': iresnet50,
        'iresnet100': iresnet100,
        'iresnet200': iresnet200,
        'iresnet50_se': iresnet50_se,
        'iresnet100_se': iresnet100_se,
    }

    if name not in backbone_dict:
        raise ValueError(f"未知的骨干网络: {name}, 可选: {list(backbone_dict.keys())}")

    return backbone_dict[name](**kwargs)


if __name__ == '__main__':
    # 测试代码
    print("iResNet 骨干网络测试")
    print("=" * 50)

    # 创建模型
    model = iresnet100(dropout=0.0, num_features=512)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        features = model(x)

    print(f"输入大小: {x.shape}")
    print(f"输出大小: {features.shape}")
    print(f"特征幅度: {torch.norm(features, dim=1)}")
