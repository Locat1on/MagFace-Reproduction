"""
MagFace 配置文件
包含训练、推理所需的所有超参数设置
"""

from easydict import EasyDict as edict

# 创建配置对象
cfg = edict()

# ==================== 数据配置 ====================
cfg.data = edict()
cfg.data.train_root = 'data/ms1m-retinaface-t1'  # MS1M-V2 训练数据路径
cfg.data.val_root = 'data/faces_emore'           # 验证数据路径
cfg.data.num_classes = 85742                      # MS1M-V2 身份数量
cfg.data.image_size = 112                         # 图像大小 112x112
cfg.data.num_workers = 8                          # 数据加载线程数

# ==================== 模型配置 ====================
cfg.model = edict()
cfg.model.backbone = 'iresnet100'                 # 骨干网络: iresnet50, iresnet100
cfg.model.embedding_size = 512                    # 特征向量维度
cfg.model.dropout = 0.0                           # Dropout 比例

# ==================== MagFace Loss 配置 ====================
cfg.magface = edict()
cfg.magface.scale = 64                            # Scale factor (s)
# 幅度限制参数
cfg.magface.l_a = 10                              # 幅度下界 (lower bound of magnitude)
cfg.magface.u_a = 110                             # 幅度上界 (upper bound of magnitude)
# Margin 参数
cfg.magface.l_m = 0.45                            # Margin 下界
cfg.magface.u_m = 0.8                             # Margin 上界
# 正则化参数
cfg.magface.lambda_g = 35                         # 正则项系数

# ==================== 训练配置 ====================
cfg.train = edict()
cfg.train.batch_size = 512                        # 批次大小 (论文使用 8x1080Ti)
cfg.train.epochs = 25                             # 总训练轮数
cfg.train.start_epoch = 0                         # 起始轮数 (用于断点续训)

# 优化器配置
cfg.train.optimizer = 'sgd'                       # 优化器类型
cfg.train.lr = 0.1                                # 初始学习率
cfg.train.momentum = 0.9                          # SGD 动量
cfg.train.weight_decay = 5e-4                     # 权重衰减

# 学习率调度
cfg.train.lr_scheduler = 'multistep'              # 学习率调度策略
cfg.train.lr_milestones = [10, 18, 22]            # 学习率衰减的 epoch
cfg.train.lr_gamma = 0.1                          # 学习率衰减因子

# 梯度累积 (显存不足时使用)
cfg.train.gradient_accumulation_steps = 1         # 梯度累积步数

# ==================== 验证配置 ====================
cfg.val = edict()
cfg.val.batch_size = 256                          # 验证批次大小
cfg.val.frequency = 1                             # 每几个 epoch 验证一次

# ==================== 推理配置 ====================
cfg.inference = edict()
cfg.inference.quality_threshold = 23              # 质量阈值 (低于此值认为质量差)
cfg.inference.similarity_threshold = 0.4          # 相似度阈值 (用于识别)
cfg.inference.detector = 'mtcnn'                  # 人脸检测器: mtcnn, retinaface

# ==================== 检查点与日志 ====================
cfg.checkpoint = edict()
cfg.checkpoint.save_dir = 'checkpoints'           # 模型保存路径
cfg.checkpoint.save_frequency = 1                 # 保存频率 (每几个 epoch 保存一次)
cfg.checkpoint.resume = None                      # 断点续训的模型路径

cfg.log = edict()
cfg.log.dir = 'logs'                              # 日志保存路径
cfg.log.print_frequency = 100                     # 打印频率 (每几个 batch 打印一次)

# ==================== 设备配置 ====================
cfg.device = edict()
cfg.device.gpu_ids = [0]                          # 使用的 GPU ID 列表
cfg.device.fp16 = False                           # 是否使用混合精度训练


def get_config():
    """获取配置"""
    return cfg


def print_config(config, indent=0):
    """打印配置"""
    for key, value in config.items():
        if isinstance(value, edict):
            print(' ' * indent + f'{key}:')
            print_config(value, indent + 2)
        else:
            print(' ' * indent + f'{key}: {value}')


if __name__ == '__main__':
    print("MagFace 配置:")
    print("=" * 50)
    print_config(cfg)
