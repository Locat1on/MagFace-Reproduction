"""
InsightFace 人脸识别 Pipeline
使用 InsightFace 库实现检测、对齐、特征提取、质量评估
"""

import os
import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple, Union
import pickle


class InsightFacePipeline:
    """
    基于 InsightFace 的人脸识别 Pipeline

    功能:
    1. 人脸检测与对齐
    2. 特征提取 (512维)
    3. 质量评估 (基于检测置信度和特征幅度)
    4. 人脸比对
    5. 底库管理
    """

    def __init__(
        self,
        model_name: str = 'buffalo_l',
        device: str = 'cuda',
        det_size: Tuple[int, int] = (640, 640),
        quality_threshold: float = 0.5,
        similarity_threshold: float = 0.4
    ):
        """
        初始化 Pipeline

        Args:
            model_name: InsightFace 模型名称 ('buffalo_l', 'buffalo_s', 'buffalo_sc')
            device: 设备 ('cuda' 或 'cpu')
            det_size: 检测输入尺寸
            quality_threshold: 质量阈值 (0~1, 基于检测置信度)
            similarity_threshold: 相似度阈值 (用于识别)
        """
        self.quality_threshold = quality_threshold
        self.similarity_threshold = similarity_threshold
        self.det_size = det_size

        # 初始化 InsightFace
        print("初始化 InsightFace...")
        try:
            from insightface.app import FaceAnalysis
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.app = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=det_size)
            print(f"InsightFace 初始化成功: {model_name}")
        except ImportError:
            raise ImportError("请安装 insightface: pip install insightface onnxruntime-gpu")

        # 人脸底库
        self.database = {
            'names': [],           # 姓名列表
            'features': None,      # 特征矩阵 (N, 512)
            'qualities': [],       # 质量分数列表
            'images': []           # 对齐后的人脸图像 (可选)
        }

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的人脸

        Args:
            image: BGR 格式的图像

        Returns:
            检测到的人脸列表，每个元素包含 bbox, landmarks, embedding, det_score 等
        """
        faces = self.app.get(image)
        results = []
        for face in faces:
            results.append({
                'bbox': face.bbox.astype(int),
                'landmarks': face.kps.astype(np.float32) if face.kps is not None else None,
                'embedding': face.embedding,
                'det_score': float(face.det_score),
                'age': getattr(face, 'age', None),
                'gender': getattr(face, 'gender', None),
            })
        return results

    def extract_features(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        """
        从图像中检测人脸并提取特征

        Args:
            image: BGR 格式的图像

        Returns:
            embedding: 512维特征向量，失败返回 None
            quality: 质量分数 (0~1)
            aligned_face: 对齐后的人脸图像 (112x112)，失败返回 None
        """
        faces = self.app.get(image)
        if len(faces) == 0:
            return None, 0.0, None

        # 取置信度最高的人脸
        face = max(faces, key=lambda x: x.det_score)
        embedding = face.embedding
        quality = float(face.det_score)

        # 获取对齐后的人脸图像（如果有）
        aligned_face = None
        if hasattr(face, 'normed_embedding'):
            # InsightFace 内部已对齐，尝试获取
            pass

        # 用 landmark 做简单裁剪作为展示
        if face.kps is not None:
            aligned_face = self._align_face(image, face.kps)

        return embedding, quality, aligned_face

    def _align_face(self, image: np.ndarray, landmarks: np.ndarray, output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        根据关键点对齐人脸 (简化版)

        Args:
            image: 原始图像 (BGR)
            landmarks: 5个关键点坐标 (5, 2)
            output_size: 输出图像大小

        Returns:
            对齐后的人脸图像
        """
        from skimage import transform as trans

        # ArcFace 标准对齐模板
        dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, dst)
        M = tform.params[0:2, :]
        aligned = cv2.warpAffine(image, M, output_size, borderValue=0)
        return aligned

    def assess_quality(self, det_score: float) -> Dict:
        """
        评估人脸质量

        Args:
            det_score: 检测置信度 (0~1)

        Returns:
            质量评估结果
        """
        if det_score >= 0.9:
            level = 'excellent'
            description = '优秀 - 高质量正脸'
        elif det_score >= 0.7:
            level = 'good'
            description = '良好 - 质量较好'
        elif det_score >= self.quality_threshold:
            level = 'acceptable'
            description = '可接受 - 质量一般'
        else:
            level = 'poor'
            description = '较差 - 建议重新采集'

        return {
            'magnitude': det_score,
            'level': level,
            'description': description,
            'is_acceptable': det_score >= self.quality_threshold
        }

    def register(self, name: str, image: Union[str, np.ndarray], save_image: bool = False) -> Dict:
        """
        注册人脸到底库

        Args:
            name: 姓名
            image: 图像路径或 numpy 数组 (BGR)
            save_image: 是否保存对齐后的人脸图像

        Returns:
            注册结果
        """
        # 加载图像
        if isinstance(image, str):
            if not os.path.exists(image):
                return {'success': False, 'message': f'图像文件不存在: {image}'}
            image = cv2.imread(image)
            if image is None:
                return {'success': False, 'message': f'无法读取图像: {image}'}

        # 提取特征
        embedding, quality, aligned_face = self.extract_features(image)
        if embedding is None:
            return {'success': False, 'message': '未检测到人脸，请重新上传'}

        # 质量检查
        quality_info = self.assess_quality(quality)
        if not quality_info['is_acceptable']:
            return {
                'success': False,
                'message': f"照片质量过低 (分数: {quality:.2f})，请上传更清晰的照片",
                'quality': quality_info
            }

        # 检查是否已存在
        if name in self.database['names']:
            idx = self.database['names'].index(name)
            self.database['features'][idx] = embedding
            self.database['qualities'][idx] = quality
            if save_image and aligned_face is not None:
                self.database['images'][idx] = aligned_face
            message = f"更新成功: {name}"
        else:
            self.database['names'].append(name)
            self.database['qualities'].append(quality)
            if self.database['features'] is None:
                self.database['features'] = embedding.reshape(1, -1)
            else:
                self.database['features'] = np.vstack([self.database['features'], embedding])
            if save_image and aligned_face is not None:
                self.database['images'].append(aligned_face)
            message = f"注册成功: {name}"

        return {
            'success': True,
            'message': message,
            'quality': quality_info,
            'database_size': len(self.database['names'])
        }

    def recognize(self, image: Union[str, np.ndarray], top_k: int = 1) -> Dict:
        """
        识别人脸

        Args:
            image: 图像路径或 numpy 数组 (BGR)
            top_k: 返回前 k 个匹配结果

        Returns:
            识别结果
        """
        # 加载图像
        if isinstance(image, str):
            if not os.path.exists(image):
                return {'success': False, 'message': f'图像文件不存在: {image}'}
            image = cv2.imread(image)
            if image is None:
                return {'success': False, 'message': f'无法读取图像: {image}'}

        # 提取特征
        embedding, quality, aligned_face = self.extract_features(image)
        if embedding is None:
            return {'success': False, 'message': '未检测到人脸'}

        # 质量检查
        quality_info = self.assess_quality(quality)
        if not quality_info['is_acceptable']:
            return {
                'success': False,
                'message': f"照片质量过低 (分数: {quality:.2f})，请重新上传",
                'quality': quality_info
            }

        # 底库为空
        if self.database['features'] is None or len(self.database['names']) == 0:
            return {
                'success': False,
                'message': '底库为空，请先注册人脸',
                'quality': quality_info
            }

        # 计算相似度 (余弦相似度)
        embedding = embedding / np.linalg.norm(embedding)
        db_features = self.database['features'] / np.linalg.norm(self.database['features'], axis=1, keepdims=True)
        similarities = np.dot(db_features, embedding)

        # 排序
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            sim = float(similarities[idx])
            results.append({
                'name': self.database['names'][idx],
                'similarity': sim,
                'is_match': sim >= self.similarity_threshold
            })

        # 最佳匹配
        best = results[0]
        is_recognized = best['is_match']

        return {
            'success': True,
            'is_recognized': is_recognized,
            'best_match': best,
            'all_results': results,
            'quality': quality_info
        }

    def verify(self, image1: Union[str, np.ndarray], image2: Union[str, np.ndarray]) -> Dict:
        """
        1:1 人脸验证

        Args:
            image1: 第一张图像
            image2: 第二张图像

        Returns:
            验证结果
        """
        # 加载图像
        if isinstance(image1, str):
            image1 = cv2.imread(image1)
        if isinstance(image2, str):
            image2 = cv2.imread(image2)

        if image1 is None or image2 is None:
            return {'success': False, 'message': '无法读取图像'}

        # 提取特征
        emb1, q1, _ = self.extract_features(image1)
        emb2, q2, _ = self.extract_features(image2)

        if emb1 is None:
            return {'success': False, 'message': '照片1未检测到人脸'}
        if emb2 is None:
            return {'success': False, 'message': '照片2未检测到人脸'}

        # 质量检查
        quality_1 = self.assess_quality(q1)
        quality_2 = self.assess_quality(q2)

        # 计算相似度
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        similarity = float(np.dot(emb1, emb2))

        return {
            'success': True,
            'similarity': similarity,
            'is_same_person': similarity >= self.similarity_threshold,
            'quality_1': quality_1,
            'quality_2': quality_2
        }

    def list_registered(self) -> List[str]:
        """列出所有已注册人员"""
        return self.database['names'].copy()

    def remove(self, name: str) -> bool:
        """从底库中移除人员"""
        if name not in self.database['names']:
            return False

        idx = self.database['names'].index(name)
        self.database['names'].pop(idx)
        self.database['qualities'].pop(idx)
        self.database['features'] = np.delete(self.database['features'], idx, axis=0)
        if self.database['images']:
            self.database['images'].pop(idx)
        return True

    def save_database(self, path: str):
        """保存底库到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.database, f)
        print(f"底库已保存到: {path}")

    def load_database(self, path: str):
        """从文件加载底库"""
        if not os.path.exists(path):
            print(f"底库文件不存在: {path}")
            return
        with open(path, 'rb') as f:
            self.database = pickle.load(f)
        print(f"底库已加载，共 {len(self.database['names'])} 人")

    def detect_and_align(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        检测和对齐人脸（兼容原接口）

        Args:
            image: 图像路径或 numpy 数组 (BGR)

        Returns:
            对齐后的人脸图像 (112x112 BGR)，失败返回 None
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return None

        _, _, aligned_face = self.extract_features(image)
        return aligned_face


if __name__ == '__main__':
    print("InsightFace Pipeline 测试")
    print("=" * 50)

    pipeline = InsightFacePipeline(model_name='buffalo_l', device='cuda')
    print(f"Pipeline 初始化成功!")

    # 测试质量评估
    print("\n质量评估测试:")
    for score in [0.3, 0.5, 0.7, 0.9, 0.99]:
        quality = pipeline.assess_quality(score)
        print(f"  分数 {score:.2f}: {quality['level']} - {quality['description']}")
