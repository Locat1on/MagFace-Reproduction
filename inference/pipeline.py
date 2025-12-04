"""
MagFace 推理 Pipeline
完整的人脸识别流程：检测 -> 对齐 -> 特征提取 -> 质量评估 -> 比对
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
import pickle
import cv2
from PIL import Image

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.iresnet import get_backbone
from data.preprocess import FacePreprocessor, normalize_image


class FaceRecognitionPipeline:
    """
    人脸识别推理 Pipeline

    功能:
    1. 人脸检测与对齐
    2. 特征提取
    3. 质量评估 (基于 MagFace 幅度)
    4. 人脸比对
    5. 底库管理
    """

    def __init__(
        self,
        model_path: str = None,
        backbone: str = 'iresnet100',
        embedding_size: int = 512,
        device: str = 'cuda',
        detector: str = 'mtcnn',
        quality_threshold: float = 23.0,
        similarity_threshold: float = 0.4
    ):
        """
        初始化 Pipeline

        Args:
            model_path: 预训练模型路径
            backbone: 骨干网络类型
            embedding_size: 特征维度
            device: 设备
            detector: 人脸检测器类型
            quality_threshold: 质量阈值 (低于此值认为质量差)
            similarity_threshold: 相似度阈值 (用于识别)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.quality_threshold = quality_threshold
        self.similarity_threshold = similarity_threshold
        self.debug = os.environ.get('MAGFACE_DEBUG', '1') != '0'

        # 初始化人脸检测器
        print("初始化人脸检测器...")
        self.preprocessor = FacePreprocessor(detector=detector, device=str(self.device))

        # 初始化特征提取模型
        print(f"初始化特征提取模型: {backbone}...")
        self.model = get_backbone(backbone, dropout=0.0, num_features=embedding_size)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载预训练权重
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("警告: 未加载预训练权重，模型使用随机初始化")

        # 人脸底库
        self._reset_database()

    def _load_model(self, model_path: str):
        """加载模型权重（兼容官方预训练模型格式）"""
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # 官方预训练模型格式: checkpoint['state_dict'] 包含完整模型
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"检测到官方预训练模型格式")
        else:
            state_dict = checkpoint

        # 提取 backbone 权重，处理各种前缀格式
        # 官方模型格式: features.module.conv1.weight -> conv1.weight
        backbone_state_dict = {}

        for k, v in state_dict.items():
            new_key = k
            is_top_level_classifier = k in {'fc.weight', 'fc.bias'}

            # 移除各种可能的前缀
            prefixes_to_remove = [
                'features.module.',  # 官方模型格式
                'module.features.',
                'backbone.module.',
                'module.backbone.',
                'features.',
                'backbone.',
                'module.',
            ]

            for prefix in prefixes_to_remove:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break

            if is_top_level_classifier:
                continue

            if 'header' in new_key or 'classifier' in new_key:
                continue

            backbone_state_dict[new_key] = v

        # DEBUG: 检查 fc.weight 是否存在于 backbone_state_dict
        if 'fc.weight' in backbone_state_dict:
            print(f"DEBUG: fc.weight found in backbone_state_dict with shape {backbone_state_dict['fc.weight'].shape}")
            print(f"DEBUG: fc.weight mean: {backbone_state_dict['fc.weight'].float().mean():.6f}")
        else:
            print("DEBUG: fc.weight NOT found in backbone_state_dict")

        print(f"提取到 {len(backbone_state_dict)} 个 backbone 参数")

        # 加载权重，使用 strict=False 允许部分加载
        missing_keys, unexpected_keys = self.model.load_state_dict(backbone_state_dict, strict=False)

        if missing_keys:
            print(f"警告: 缺失的参数 ({len(missing_keys)}): {missing_keys[:10]}...")
        if unexpected_keys:
            print(f"警告: 多余的参数 ({len(unexpected_keys)}): {unexpected_keys[:10]}...")

        if self.debug:
            print(f"DEBUG: load_state summary | missing={len(missing_keys)} | unexpected={len(unexpected_keys)}")

        if len(missing_keys) > 20:
             print("严重警告: 大量参数缺失，模型可能未正确加载！")

        self._debug_model_stats()

        print(f"模型加载完成")

    def detect_and_align(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        检测和对齐人脸

        Args:
            image: 图像路径或 numpy 数组 (BGR)

        Returns:
            对齐后的人脸图像 (112x112 BGR)，失败返回 None
        """
        return self.preprocessor.process(image)

    def extract_features(self, face_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        提取人脸特征

        Args:
            face_image: 对齐后的人脸图像 (112x112 BGR)

        Returns:
            features: 归一化特征向量 (512,)
            magnitude: 特征幅度 (质量分数)
        """
        # BGR -> RGB (MagFace 模型通常使用 RGB 训练)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # 预处理
        face = normalize_image(face_image)  # 归一化到 [-1, 1]

        if self.debug:
            print(
                f"DEBUG: Face input stats - Min: {face.min():.2f}, Max: {face.max():.2f}, Mean: {face.mean():.2f}, Std: {face.std():.2f}"
            )

        face = np.transpose(face, (2, 0, 1))  # HWC -> CHW
        face_tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)  # (1, 3, 112, 112)

        if self.debug:
            print(
                f"DEBUG: Torch tensor stats - Min: {face_tensor.min().item():.2f}, Max: {face_tensor.max().item():.2f}, Mean: {face_tensor.mean().item():.2f}, Std: {face_tensor.std().item():.2f}"
            )

        # 提取特征
        with torch.no_grad():
            model_output = self.model(face_tensor, return_embedding_before_bn=True)

            if isinstance(model_output, tuple):
                embedding, embedding_before_bn = model_output
            else:
                embedding = model_output
                embedding_before_bn = model_output

            # NOTE: MagFace 论文中的质量分数取自 BN 之后的特征幅度
            magnitude = torch.norm(embedding, dim=1).item()

            if self.debug:
                emb_before = embedding_before_bn.detach().cpu()
                emb_after = embedding.detach().cpu()
                print(
                    f"DEBUG: Embedding_before_bn stats - min: {emb_before.min().item():.4f}, max: {emb_before.max().item():.4f}, mean: {emb_before.mean().item():.4f}, std: {emb_before.std().item():.4f}, L2(before BN): {torch.norm(embedding_before_bn, dim=1).item():.4f}"
                )
                print(
                    f"DEBUG: Embedding (after BN) stats - min: {emb_after.min().item():.4f}, max: {emb_after.max().item():.4f}, mean: {emb_after.mean().item():.4f}, std: {emb_after.std().item():.4f}, L2: {torch.norm(embedding, dim=1).item():.4f}"
                )

            feature_tensor = F.normalize(embedding, dim=1)
            feature = feature_tensor.squeeze(0).cpu().numpy()

            if self.debug:
                print(
                    f"DEBUG: Final feature stats - min: {feature.min():.4f}, max: {feature.max():.4f}, mean: {feature.mean():.4f}, L2: {np.linalg.norm(feature):.4f}"
                )
                print("DEBUG: MagFace Magnitude (after BN): {:.4f}".format(magnitude))

        return feature, magnitude

    def assess_quality(self, magnitude: float) -> Dict:
        """
        评估人脸质量

        Args:
            magnitude: 特征幅度

        Returns:
            质量评估结果
        """
        # 质量等级划分
        if magnitude >= 60:
            level = 'excellent'
            description = '优秀 - 高质量正脸'
        elif magnitude >= 40:
            level = 'good'
            description = '良好 - 质量较好'
        elif magnitude >= self.quality_threshold:
            level = 'acceptable'
            description = '可接受 - 质量一般'
        else:
            level = 'poor'
            description = '较差 - 建议重新采集'

        return {
            'magnitude': magnitude,
            'level': level,
            'description': description,
            'is_acceptable': magnitude >= self.quality_threshold
        }

    def register(self, name: str, image: Union[str, np.ndarray],
                 save_image: bool = False) -> Dict:
        """
        注册人脸到底库

        Args:
            name: 姓名
            image: 图像路径或 numpy 数组
            save_image: 是否保存原图

        Returns:
            注册结果
        """
        # 检测和对齐
        face = self.detect_and_align(image)
        if face is None:
            return {
                'success': False,
                'message': '未检测到人脸，请重新上传'
            }

        # 提取特征
        feature, magnitude = self.extract_features(face)

        # 质量检查
        quality = self.assess_quality(magnitude)
        if not quality['is_acceptable']:
            return {
                'success': False,
                'message': f"照片质量过低 (分数: {magnitude:.2f})，请上传更清晰的照片",
                'quality': quality
            }

        # 检查是否已存在
        if name in self.database['names']:
            # 更新已有人员
            idx = self.database['names'].index(name)
            self.database['features'][idx] = feature
            self.database['magnitudes'][idx] = magnitude
            if save_image:
                self.database['images'][idx] = face
            message = f"更新成功: {name}"
        else:
            # 添加新人员
            self.database['names'].append(name)
            self.database['magnitudes'].append(magnitude)
            if self.database['features'] is None:
                self.database['features'] = feature.reshape(1, -1)
            else:
                self.database['features'] = np.vstack([self.database['features'], feature])
            if save_image:
                self.database['images'].append(face)
            message = f"注册成功: {name}"

        return {
            'success': True,
            'message': message,
            'quality': quality,
            'database_size': len(self.database['names'])
        }

    def recognize(self, image: Union[str, np.ndarray], top_k: int = 1) -> Dict:
        """
        识别人脸

        Args:
            image: 图像路径或 numpy 数组
            top_k: 返回前 k 个匹配结果

        Returns:
            识别结果
        """
        # 检测和对齐
        face = self.detect_and_align(image)
        if face is None:
            return {
                'success': False,
                'message': '未检测到人脸'
            }

        # 提取特征
        feature, magnitude = self.extract_features(face)

        # 质量检查
        quality = self.assess_quality(magnitude)
        if not quality['is_acceptable']:
            return {
                'success': False,
                'message': f"照片质量过低 (分数: {magnitude:.2f})，请重新上传",
                'quality': quality
            }

        # 检查底库是否为空
        if self.database['features'] is None or len(self.database['names']) == 0:
            return {
                'success': False,
                'message': '底库为空，请先注册人脸',
                'quality': quality
            }

        # 计算相似度
        similarities = np.dot(self.database['features'], feature)

        # 获取 top-k 结果
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = similarities[idx]
            name = self.database['names'][idx]

            if sim >= self.similarity_threshold:
                results.append({
                    'name': name,
                    'similarity': float(sim),
                    'is_match': True
                })
            else:
                results.append({
                    'name': name,
                    'similarity': float(sim),
                    'is_match': False
                })

        # 最佳匹配
        best_match = results[0] if results else None

        return {
            'success': True,
            'best_match': best_match,
            'all_results': results,
            'quality': quality,
            'is_recognized': best_match['is_match'] if best_match else False
        }

    def verify(self, image1: Union[str, np.ndarray],
               image2: Union[str, np.ndarray]) -> Dict:
        """
        1:1 人脸验证

        Args:
            image1: 第一张图像
            image2: 第二张图像

        Returns:
            验证结果
        """
        # 处理第一张图
        face1 = self.detect_and_align(image1)
        if face1 is None:
            return {'success': False, 'message': '第一张图未检测到人脸'}

        # 处理第二张图
        face2 = self.detect_and_align(image2)
        if face2 is None:
            return {'success': False, 'message': '第二张图未检测到人脸'}

        # 提取特征
        feat1, mag1 = self.extract_features(face1)
        feat2, mag2 = self.extract_features(face2)

        # 计算相似度
        similarity = float(np.dot(feat1, feat2))

        return {
            'success': True,
            'similarity': similarity,
            'is_same_person': similarity >= self.similarity_threshold,
            'quality_1': self.assess_quality(mag1),
            'quality_2': self.assess_quality(mag2)
        }

    def save_database(self, path: str):
        """保存底库"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.database, f)
        print(f"底库已保存到: {path}")

    def load_database(self, path: str):
        """加载底库"""
        if not os.path.exists(path):
            print(f"底库文件不存在: {path}")
            return

        with open(path, 'rb') as f:
            self.database = pickle.load(f)
        print(f"底库加载成功，共 {len(self.database['names'])} 人")

    def _reset_database(self):
        self.database = {
            'names': [],
            'features': None,
            'magnitudes': [],
            'images': []
        }

    def clear_database(self, path: Optional[str] = None, remove_file: bool = True) -> Dict:
        """清空内存中的底库，并可选删除持久化文件"""
        previous_count = len(self.database.get('names', []))
        self._reset_database()

        removed_file = False
        if path and remove_file:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    removed_file = True
            except OSError as exc:
                return {
                    'success': False,
                    'message': f"删除文件失败: {exc}",
                    'removed_file': False,
                    'previous_count': previous_count
                }

        return {
            'success': True,
            'message': f"已清空底库，共删除 {previous_count} 条记录",
            'removed_file': removed_file,
            'previous_count': previous_count
        }

    def list_registered(self) -> List[str]:
        """列出所有已注册人员"""
        return self.database['names'].copy()

    def remove(self, name: str) -> bool:
        """
        从底库删除人员

        Args:
            name: 姓名

        Returns:
            是否删除成功
        """
        if name not in self.database['names']:
            return False

        idx = self.database['names'].index(name)
        self.database['names'].pop(idx)
        self.database['magnitudes'].pop(idx)
        self.database['features'] = np.delete(self.database['features'], idx, axis=0)
        if self.database['images']:
            self.database['images'].pop(idx)

        return True

    def _debug(self, message: str):
        if self.debug:
            print(message)

    def _debug_model_stats(self):
        if not self.debug:
            return
        try:
            fc_weight = self.model.state_dict().get('fc.weight')
            if fc_weight is not None:
                self._debug(
                    f"DEBUG: Model fc.weight stats | shape={tuple(fc_weight.shape)} | mean={fc_weight.float().mean():.6f} | std={fc_weight.float().std():.6f}"
                )
            bn2_running_mean = getattr(self.model.bn2, 'running_mean', None)
            bn2_running_var = getattr(self.model.bn2, 'running_var', None)
            if bn2_running_mean is not None and bn2_running_var is not None:
                self._debug(
                    f"DEBUG: Model bn2 running stats | mean_mean={bn2_running_mean.mean().item():.6f} | var_mean={bn2_running_var.mean().item():.6f}"
                )
            features_weight = getattr(self.model.features, 'weight', None)
            if features_weight is not None:
                self._debug(
                    f"DEBUG: Model features BN weight stats | mean={features_weight.float().mean():.6f} | std={features_weight.float().std():.6f}"
                )
        except Exception as exc:
            self._debug(f"DEBUG: Failed to dump model stats: {exc}")


class BatchFeatureExtractor:
    """
    批量特征提取器
    用于大规模数据的特征提取
    """

    def __init__(
        self,
        model_path: str,
        backbone: str = 'iresnet100',
        device: str = 'cuda',
        batch_size: int = 64
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # 加载模型
        self.model = get_backbone(backbone)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量提取特征

        Args:
            images: 对齐后的人脸图像列表 (112x112 BGR)

        Returns:
            features: 特征矩阵 (N, 512)
            magnitudes: 幅度数组 (N,)
        """
        features_list = []
        magnitudes_list = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]

            # 预处理
            batch_tensors = []
            for img in batch:
                # BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = normalize_image(img)
                img = np.transpose(img, (2, 0, 1))
                batch_tensors.append(img)

            batch_tensor = torch.from_numpy(np.stack(batch_tensors)).to(self.device)

            # 提取特征
            with torch.no_grad():
                embeddings = self.model(batch_tensor)
                mags = torch.norm(embeddings, dim=1)
                feats = F.normalize(embeddings, dim=1)

            features_list.append(feats.cpu().numpy())
            magnitudes_list.append(mags.cpu().numpy())

        features = np.concatenate(features_list, axis=0)
        magnitudes = np.concatenate(magnitudes_list, axis=0)

        return features, magnitudes
