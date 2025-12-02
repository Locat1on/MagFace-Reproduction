"""
MagFace 训练脚本
完整的训练流程，包含数据加载、模型训练、验证、检查点保存等
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from models.iresnet import get_backbone
from models.magface import MagFaceLoss
from data.dataset import get_train_dataloader
from utils.utils import (
    setup_seed, setup_logging, save_checkpoint, load_checkpoint,
    AverageMeter, ProgressMeter, count_parameters, get_lr, warm_up_lr
)
from utils.evaluation import MagnitudeDistributionAnalyzer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MagFace 训练')

    # 数据
    parser.add_argument('--data_dir', type=str, default='data/ms1m-retinaface-t1',
                        help='训练数据目录')
    parser.add_argument('--num_classes', type=int, default=85742,
                        help='类别数（身份数）')

    # 模型
    parser.add_argument('--backbone', type=str, default='iresnet100',
                        choices=['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100'],
                        help='骨干网络')
    parser.add_argument('--embedding_size', type=int, default=512,
                        help='特征维度')

    # MagFace 参数
    parser.add_argument('--scale', type=float, default=64.0, help='缩放因子')
    parser.add_argument('--l_a', type=float, default=10.0, help='幅度下界')
    parser.add_argument('--u_a', type=float, default=110.0, help='幅度上界')
    parser.add_argument('--l_m', type=float, default=0.45, help='Margin 下界')
    parser.add_argument('--u_m', type=float, default=0.8, help='Margin 上界')
    parser.add_argument('--lambda_g', type=float, default=35.0, help='正则项系数')

    # 训练
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.1, help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[10, 18, 22],
                        help='学习率衰减节点')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Warmup 轮数')

    # 设备
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--fp16', action='store_true', help='使用混合精度训练')

    # 保存
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--save_freq', type=int, default=1, help='保存频率')
    parser.add_argument('--print_freq', type=int, default=100, help='打印频率')
    parser.add_argument('--resume', type=str, default=None, help='断点续训路径')

    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str,
    args,
    scaler: GradScaler = None,
    writer: SummaryWriter = None,
    mag_analyzer: MagnitudeDistributionAnalyzer = None
):
    """
    训练一个 epoch

    Args:
        model: 骨干网络
        head: MagFace Loss 头
        dataloader: 数据加载器
        optimizer: 优化器
        epoch: 当前轮数
        device: 设备
        args: 参数
        scaler: 混合精度 scaler
        writer: TensorBoard writer
        mag_analyzer: 幅度分析器
    """
    model.train()
    head.train()

    # 指标记录
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    mag_mean = AverageMeter('Mag', ':.2f')
    reg_mean = AverageMeter('Reg', ':.4f')

    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, mag_mean, reg_mean],
        prefix=f"Epoch: [{epoch}]"
    )

    end = time.time()

    for batch_idx, (images, labels) in enumerate(dataloader):
        data_time.update(time.time() - end)

        # Warmup
        if epoch < args.warmup_epochs:
            warm_up_lr(optimizer, epoch, batch_idx, len(dataloader),
                      args.warmup_epochs, args.lr)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        if args.fp16 and scaler is not None:
            with autocast():
                embeddings = model(images)
                loss, x_norm, g = head(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            loss, x_norm, g = head(embeddings, labels)

            loss.backward()
            optimizer.step()

        # 更新指标
        losses.update(loss.item(), images.size(0))
        mag_mean.update(x_norm.mean().item(), images.size(0))
        reg_mean.update(g.mean().item(), images.size(0))

        # 幅度分析
        if mag_analyzer is not None:
            mag_analyzer.add(x_norm.detach().cpu().numpy(),
                           labels.detach().cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

        # 打印进度
        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

            # TensorBoard
            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/Magnitude', x_norm.mean().item(), global_step)
                writer.add_scalar('Train/Regularizer', g.mean().item(), global_step)
                writer.add_scalar('Train/LR', get_lr(optimizer), global_step)

    return losses.avg, mag_mean.avg


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    setup_seed(args.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'magface_{args.backbone}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # 设置日志
    logger = setup_logging(output_dir, 'train')
    logger.info(f"参数: {args}")

    # 设备
    device = torch.device(f'cuda:{args.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建模型
    logger.info(f"创建模型: {args.backbone}")
    model = get_backbone(args.backbone, dropout=0.0, num_features=args.embedding_size)
    model = model.to(device)

    # 多 GPU
    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)

    logger.info(f"模型参数量: {count_parameters(model):,}")

    # MagFace Loss
    head = MagFaceLoss(
        num_classes=args.num_classes,
        embedding_size=args.embedding_size,
        scale=args.scale,
        l_a=args.l_a,
        u_a=args.u_a,
        l_m=args.l_m,
        u_m=args.u_m,
        lambda_g=args.lambda_g
    ).to(device)

    # 优化器
    optimizer = SGD(
        [{'params': model.parameters()}, {'params': head.parameters()}],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)

    # 混合精度
    scaler = GradScaler() if args.fp16 else None

    # 断点续训
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, head, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"从 epoch {start_epoch} 继续训练")

    # 数据加载
    logger.info("加载训练数据...")
    train_loader = get_train_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=112,
        use_mxnet=os.path.exists(os.path.join(args.data_dir, 'train.rec'))
    )

    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

    # 幅度分析器
    mag_analyzer = MagnitudeDistributionAnalyzer()

    # 训练循环
    logger.info("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}, LR: {get_lr(optimizer):.6f}")

        # 重置幅度分析器
        mag_analyzer.reset()

        # 训练
        train_loss, train_mag = train_one_epoch(
            model, head, train_loader, optimizer, epoch, device, args,
            scaler=scaler, writer=writer, mag_analyzer=mag_analyzer
        )

        # 更新学习率
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # 幅度统计
        mag_stats = mag_analyzer.get_statistics()
        logger.info(f"幅度统计: 均值={mag_stats['mean']:.2f}, "
                   f"范围=[{mag_stats['min']:.2f}, {mag_stats['max']:.2f}]")

        # 质量分布
        quality_dist = mag_analyzer.quality_distribution()
        logger.info(f"质量分布: {quality_dist}")

        # TensorBoard - epoch 级别
        writer.add_scalar('Epoch/Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Magnitude_Mean', mag_stats['mean'], epoch)
        writer.add_scalar('Epoch/Magnitude_Std', mag_stats['std'], epoch)

        # 保存检查点
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(output_dir, 'checkpoints', f'epoch_{epoch + 1}.pth')
            save_checkpoint(
                model, optimizer, epoch, save_path,
                head=head, scheduler=scheduler,
                extra_info={'mag_stats': mag_stats}
            )

    # 保存最终模型（仅骨干网络，用于推理）
    final_model_path = os.path.join(output_dir, 'magface_final.pth')
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logger.info(f"最终模型已保存到: {final_model_path}")

    writer.close()
    logger.info("训练完成!")


if __name__ == '__main__':
    main()
