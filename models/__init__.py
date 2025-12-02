# models/__init__.py
from .iresnet import (
    IResNet, IBasicBlock,
    iresnet18, iresnet34, iresnet50, iresnet100, iresnet200,
    iresnet50_se, iresnet100_se,
    get_backbone
)
from .magface import MagFaceLoss, ArcFaceLoss, MagFaceModel

__all__ = [
    'IResNet',
    'IBasicBlock',
    'iresnet18',
    'iresnet34',
    'iresnet50',
    'iresnet100',
    'iresnet200',
    'iresnet50_se',
    'iresnet100_se',
    'get_backbone',
    'MagFaceLoss',
    'ArcFaceLoss',
    'MagFaceModel'
]
