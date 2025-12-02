"""
通用工具函数
"""

import os
import random
import numpy as np
import torch
from torch import nn
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime


def setup_seed(seed: int = 42):
    """
    设置随机种子，保证可复现性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, name: str = 'train') -> logging.Logger:
    """
    设置日志

    Args:
        log_dir: 日志目录
        name: 日志名称

    Returns:
        logger 对象
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 文件 handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    head: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    best_acc: float = 0.0,
    extra_info: Optional[Dict] = None
):
    """
    保存检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        save_path: 保存路径
        head: 分类头
        scheduler: 学习率调度器
        best_acc: 最佳准确率
        extra_info: 额外信息
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }

    if head is not None:
        checkpoint['head_state_dict'] = head.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if extra_info is not None:
        checkpoint.update(extra_info)

    torch.save(checkpoint, save_path)
    print(f"检查点已保存到: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    head: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True
) -> Dict:
    """
    加载检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器
        head: 分类头
        scheduler: 学习率调度器
        strict: 是否严格匹配

    Returns:
        检查点信息
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if head is not None and 'head_state_dict' in checkpoint:
        head.load_state_dict(checkpoint['head_state_dict'], strict=strict)

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """
    计算和存储平均值和当前值
    用于训练过程中的指标记录
    """

    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """
    进度显示器
    """

    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数量

    Args:
        model: PyTorch 模型

    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def warm_up_lr(optimizer: torch.optim.Optimizer, epoch: int, batch_idx: int,
               num_batches: int, warmup_epochs: int, base_lr: float):
    """
    学习率 warmup

    Args:
        optimizer: 优化器
        epoch: 当前轮数
        batch_idx: 当前批次
        num_batches: 每轮批次数
        warmup_epochs: warmup 轮数
        base_lr: 基础学习率
    """
    if epoch >= warmup_epochs:
        return

    total_steps = warmup_epochs * num_batches
    current_step = epoch * num_batches + batch_idx
    lr = base_lr * (current_step + 1) / total_steps

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int,
                         milestones: List[int], gamma: float, base_lr: float):
    """
    手动调整学习率

    Args:
        optimizer: 优化器
        epoch: 当前轮数
        milestones: 衰减节点
        gamma: 衰减因子
        base_lr: 基础学习率
    """
    lr = base_lr
    for milestone in milestones:
        if epoch >= milestone:
            lr *= gamma

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
