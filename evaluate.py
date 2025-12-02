"""
MagFace 评估脚本
在 LFW, CFP-FP, AgeDB 等验证集上评估模型性能
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.iresnet import get_backbone
from data.dataset import LFWDataset, get_val_transforms
from torch.utils.data import DataLoader
from utils.evaluation import evaluate_verification, find_best_threshold, calculate_tar_far


def parse_args():
    parser = argparse.ArgumentParser(description='MagFace 评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--backbone', type=str, default='iresnet100', help='骨干网络')
    parser.add_argument('--data_dir', type=str, required=True, help='验证数据目录')
    parser.add_argument('--pairs_file', type=str, required=True, help='pairs 文件路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--flip', action='store_true', help='是否使用翻转增强')
    return parser.parse_args()


def evaluate(args):
    """评估主函数"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {args.backbone}")
    model = get_backbone(args.backbone)

    state_dict = torch.load(args.model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # 加载数据
    print("加载验证数据...")
    transform = get_val_transforms(112)
    dataset = LFWDataset(args.data_dir, args.pairs_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 提取特征
    embeddings1_list = []
    embeddings2_list = []
    magnitudes1_list = []
    magnitudes2_list = []
    labels_list = []

    print("提取特征...")
    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader):
            img1, img2 = img1.to(device), img2.to(device)

            # 提取特征
            emb1 = model(img1)
            emb2 = model(img2)

            # 翻转增强
            if args.flip:
                emb1_flip = model(torch.flip(img1, dims=[3]))
                emb2_flip = model(torch.flip(img2, dims=[3]))
                emb1 = (emb1 + emb1_flip) / 2
                emb2 = (emb2 + emb2_flip) / 2

            # 计算幅度
            mag1 = torch.norm(emb1, dim=1)
            mag2 = torch.norm(emb2, dim=1)

            # 归一化
            emb1 = F.normalize(emb1, dim=1)
            emb2 = F.normalize(emb2, dim=1)

            embeddings1_list.append(emb1.cpu().numpy())
            embeddings2_list.append(emb2.cpu().numpy())
            magnitudes1_list.append(mag1.cpu().numpy())
            magnitudes2_list.append(mag2.cpu().numpy())
            labels_list.append(label.numpy())

    embeddings1 = np.concatenate(embeddings1_list, axis=0)
    embeddings2 = np.concatenate(embeddings2_list, axis=0)
    magnitudes1 = np.concatenate(magnitudes1_list, axis=0)
    magnitudes2 = np.concatenate(magnitudes2_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # 评估
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)

    # 10-fold 交叉验证
    acc, std, threshold, accs = evaluate_verification(embeddings1, embeddings2, labels)
    print(f"准确率: {acc * 100:.2f}% ± {std * 100:.2f}%")
    print(f"最佳阈值: {threshold:.4f}")

    # 计算相似度
    similarities = np.sum(embeddings1 * embeddings2, axis=1)

    # 不同 FAR 下的 TAR
    print("\n不同 FAR 下的 TAR:")
    for far in [1e-1, 1e-2, 1e-3, 1e-4]:
        tar, th = calculate_tar_far(similarities, labels, far)
        print(f"  FAR={far:.0e}: TAR={tar * 100:.2f}%")

    # 幅度统计
    print("\n幅度统计:")
    all_magnitudes = np.concatenate([magnitudes1, magnitudes2])
    print(f"  均值: {np.mean(all_magnitudes):.2f}")
    print(f"  标准差: {np.std(all_magnitudes):.2f}")
    print(f"  范围: [{np.min(all_magnitudes):.2f}, {np.max(all_magnitudes):.2f}]")

    # 按质量分组的准确率
    print("\n按质量分组的准确率:")
    min_mags = np.minimum(magnitudes1, magnitudes2)

    for low, high in [(0, 30), (30, 50), (50, 70), (70, 110)]:
        mask = (min_mags >= low) & (min_mags < high)
        if mask.sum() > 0:
            group_acc = np.mean((similarities[mask] >= threshold) == labels[mask])
            print(f"  幅度 [{low}, {high}): {group_acc * 100:.2f}% ({mask.sum()} 对)")


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
