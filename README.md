# MagFace 人脸识别系统

基于 MagFace 论文 "MagFace: A Universal Representation for Face Recognition and Quality Assessment" 的完整复现与人脸识别系统实现。

## 项目特点

- **幅度感知 (Magnitude-aware)**：利用特征向量的幅度作为质量度量，无需额外的质量检测网络
- **统一框架**：同时支持人脸识别和质量评估
- **完整流程**：从数据预处理到 Web 应用的端到端实现
- **预训练模型**：提供官方预训练权重，可直接使用

## 预训练模型

项目已包含官方预训练模型：
- **路径**: `pretrain/magface_epoch_00025.pth`
- **Backbone**: iResNet-100
- **训练数据**: MS1M-V2
- **特征维度**: 512

## 项目结构

```
MagFace/
├── configs/                 # 配置文件
│   └── config.py           # 训练和推理配置
├── data/                    # 数据处理
│   ├── dataset.py          # 数据集类
│   └── preprocess.py       # 人脸检测、对齐、裁剪
├── models/                  # 模型定义
│   ├── iresnet.py          # iResNet backbone
│   └── magface.py          # MagFace Loss
├── utils/                   # 工具函数
│   ├── utils.py            # 通用工具
│   └── evaluation.py       # 评估指标
├── inference/               # 推理模块
│   └── pipeline.py         # 完整推理流程
├── pretrain/                # 预训练模型
│   └── magface_epoch_00025.pth
├── web/                     # Web 应用
│   └── app.py              # Streamlit 界面
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── demo.py                  # 快速开始示例
├── requirements.txt         # 依赖包
└── README.md
```

## 环境配置

### 1. 创建虚拟环境

```bash
conda create -n magface python=3.8
conda activate magface
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 PyTorch (根据 CUDA 版本选择)

```bash
# CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch torchvision
```

## 快速开始 (使用预训练模型)

```python
from inference.pipeline import FaceRecognitionPipeline

# 初始化 (使用官方预训练模型)
pipeline = FaceRecognitionPipeline(
    model_path='pretrain/magface_epoch_00025.pth',
    backbone='iresnet100',
    device='cuda',  # 或 'cpu'
    detector='mtcnn'
)

# 注册人脸
result = pipeline.register('张三', 'path/to/zhangsan.jpg')
print(f"质量分: {result['quality']['magnitude']:.2f}")

# 识别人脸
result = pipeline.recognize('path/to/test.jpg')
if result['is_recognized']:
    print(f"识别结果: {result['best_match']['name']}")
    print(f"相似度: {result['best_match']['similarity']:.2%}")

# 1:1 验证
result = pipeline.verify('face1.jpg', 'face2.jpg')
print(f"是否同一人: {result['is_same_person']}")
```

## 运行 Demo

```bash
python demo.py
```

## 启动 Web 应用

```bash
streamlit run web/app.py
```

## 数据准备 (如需训练)

### 训练数据
- **MS1M-V2**: 5.8M 张图片，85k 个身份
- 下载地址：[InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)

### 验证数据
- LFW, CFP-FP, AgeDB-30, IJB-C
- 所有图片需对齐并裁剪为 112x112

### 数据预处理

```python
from data.preprocess import FacePreprocessor

preprocessor = FacePreprocessor(detector='mtcnn')
aligned_face = preprocessor.process('path/to/image.jpg')
```

## 训练 (可选)

```bash
python train.py --config configs/config.py
```

### 主要超参数
- Optimizer: SGD, Momentum=0.9, Weight Decay=5e-4
- Batch Size: 512 (8x1080Ti) 或根据显存调整
- Learning Rate: 0.1, 在 epoch 10, 18, 22 衰减
- Epochs: 25

## 推理 API

```python
from inference.pipeline import FaceRecognitionPipeline

pipeline = FaceRecognitionPipeline(model_path='pretrain/magface_epoch_00025.pth')

# 注册人脸
pipeline.register('张三', 'path/to/zhangsan.jpg')

# 识别人脸
result = pipeline.recognize('path/to/test.jpg')
if result['success'] and result['is_recognized']:
    print(f"识别结果: {result['best_match']['name']}")
    print(f"相似度: {result['best_match']['similarity']:.2%}")
    print(f"质量分: {result['quality']['magnitude']:.2f}")
```

## Web 应用

```bash
streamlit run web/app.py
```

### 功能
1. **注册模式**: 上传人脸照片 + 输入姓名 → 存入底库
2. **识别模式**: 上传人脸照片 → 显示识别结果 + 质量评分

## MagFace 核心公式

### 幅度限制
$$a_i = \text{clamp}(\|f_i\|, l_a, u_a)$$
其中 $l_a=10, u_a=110$

### Margin 函数
$$m(a_i) = \frac{u_m-l_m}{u_a-l_a}(a_i-l_a) + l_m$$
其中 $l_m=0.45, u_m=0.8$

### 正则项
$$g(a_i) = \frac{1}{a_i} + \frac{a_i}{u_a^2}$$

### 总损失
$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_g \cdot g(a_i)$$
其中 $\lambda_g=35$

## 质量评估

MagFace 的特征幅度与图像质量正相关：
- 幅度高 (>50): 高质量正脸
- 幅度中 (25-50): 一般质量
- 幅度低 (<25): 低质量（模糊、遮挡、极端角度）

建议质量阈值：20-25

## 参考

- [MagFace 论文](https://arxiv.org/abs/2103.06627)
- [MagFace 官方实现](https://github.com/IrvingMeng/MagFace)
- [InsightFace](https://github.com/deepinsight/insightface)

## License

MIT License
