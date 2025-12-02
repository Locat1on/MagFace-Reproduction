"""
数据预处理模块
包含人脸检测、对齐、裁剪功能
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Union
import os
from skimage import transform as trans


# 标准人脸关键点坐标 (用于对齐到 112x112)
# 5个关键点: 左眼、右眼、鼻尖、左嘴角、右嘴角
ARCFACE_REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],   # 左眼
    [73.5318, 51.5014],   # 右眼
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 左嘴角
    [70.7299, 92.2041]    # 右嘴角
], dtype=np.float32)


class FacePreprocessor:
    """
    人脸预处理器
    支持 MTCNN 和 RetinaFace 两种检测器
    """

    def __init__(self, detector: str = 'mtcnn', device: str = 'cuda'):
        """
        初始化预处理器

        Args:
            detector: 检测器类型 'mtcnn' 或 'retinaface'
            device: 设备类型 'cuda' 或 'cpu'
        """
        self.detector_type = detector
        self.device = device
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """初始化人脸检测器"""
        if self.detector_type == 'mtcnn':
            try:
                from facenet_pytorch import MTCNN
                self.detector = MTCNN(
                    image_size=112,
                    margin=0,
                    min_face_size=20,
                    thresholds=[0.5, 0.6, 0.6],  # 降低阈值以提高检测率
                    factor=0.709,
                    post_process=False,
                    select_largest=True,
                    keep_all=True,  # 改为返回所有检测到的人脸
                    device=self.device
                )
                print("MTCNN 检测器初始化成功")
            except ImportError:
                print("警告: facenet-pytorch 未安装，请运行: pip install facenet-pytorch")

        elif self.detector_type == 'retinaface':
            try:
                from insightface.app import FaceAnalysis
                self.detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.detector.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
                print("RetinaFace 检测器初始化成功")
            except ImportError:
                print("警告: insightface 未安装，请运行: pip install insightface")

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        检测图像中的人脸

        Args:
            image: BGR 格式的图像 (OpenCV 格式)

        Returns:
            检测到的人脸列表，每个元素包含 bbox 和 landmarks
        """
        faces = []

        if self.detector is None:
            print("错误: 检测器未初始化")
            return faces

        if self.detector_type == 'mtcnn':
            # MTCNN 需要 RGB 格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # 检测人脸和关键点
            boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)

            if boxes is not None:
                for i, (box, prob, lms) in enumerate(zip(boxes, probs, landmarks)):
                    if prob > 0.8:  # 置信度阈值
                        faces.append({
                            'bbox': box.astype(int),
                            'confidence': float(prob),
                            'landmarks': lms.astype(np.float32)
                        })

        elif self.detector_type == 'retinaface':
            # RetinaFace
            results = self.detector.get(image)
            for face in results:
                faces.append({
                    'bbox': face.bbox.astype(int),
                    'confidence': float(face.det_score),
                    'landmarks': face.kps.astype(np.float32)
                })

        return faces

    def align_face(self, image: np.ndarray, landmarks: np.ndarray,
                   output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        根据关键点对齐人脸
        使用仿射变换将检测到的关键点对齐到标准位置

        Args:
            image: 原始图像 (BGR)
            landmarks: 5个关键点坐标 (5, 2)
            output_size: 输出图像大小

        Returns:
            对齐后的人脸图像
        """
        # 计算仿射变换矩阵
        src_pts = landmarks.astype(np.float32)
        dst_pts = ARCFACE_REFERENCE_POINTS.copy()

        # 使用相似变换估计
        transform_matrix = self._estimate_affine_2d(src_pts, dst_pts)

        # 应用仿射变换
        aligned_face = cv2.warpAffine(
            image,
            transform_matrix,
            output_size,
            borderValue=(0, 0, 0)
        )

        return aligned_face

    def _estimate_affine_2d(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        估计2D仿射变换矩阵 (相似变换)
        使用 skimage.transform.SimilarityTransform 确保与标准实现一致

        Args:
            src_pts: 源点坐标
            dst_pts: 目标点坐标

        Returns:
            2x3 仿射变换矩阵
        """
        tform = trans.SimilarityTransform()
        tform.estimate(src_pts, dst_pts)
        return tform.params[0:2, :]

    def process(self, image_path: Union[str, np.ndarray],
                return_all: bool = False) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """
        完整的人脸预处理流程

        Args:
            image_path: 图像路径或图像数组
            return_all: 是否返回所有检测到的人脸

        Returns:
            对齐后的人脸图像 (112x112 BGR)，如果失败返回 None
        """
        # 加载图像
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                print(f"错误: 图像文件不存在: {image_path}")
                return None
            image = cv2.imread(image_path)
            if image is None:
                print(f"错误: 无法读取图像: {image_path}")
                return None
        else:
            image = image_path

        # 检测人脸
        faces = self.detect_faces(image)

        if len(faces) == 0:
            print("警告: 未检测到人脸")
            return None

        # 对齐人脸
        aligned_faces = []
        for face in faces:
            aligned = self.align_face(image, face['landmarks'])
            aligned_faces.append(aligned)

        if return_all:
            return aligned_faces
        else:
            return aligned_faces[0]  # 返回第一个（最大的）人脸

    def process_batch(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        """
        批量处理图像

        Args:
            image_paths: 图像路径列表

        Returns:
            对齐后的人脸图像列表
        """
        results = []
        for path in image_paths:
            result = self.process(path)
            results.append(result)
        return results


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    归一化图像到 [-1, 1] 范围（保持 BGR 顺序以兼容 InsightFace/MagFace 预训练模型）

    Args:
        image: BGR 图像 (0-255)

    Returns:
        归一化后的图像 (BGR)
    """
    image = image.astype(np.float32)
    image = (image - 127.5) / 127.5
    return image


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    反归一化图像到 [0, 255] 范围

    Args:
        image: 归一化后的图像 (-1 到 1)

    Returns:
        原始范围的图像
    """
    image = image * 127.5 + 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def load_and_preprocess(image_path: str, preprocessor: FacePreprocessor = None) -> Optional[np.ndarray]:
    """
    便捷函数：加载并预处理图像

    Args:
        image_path: 图像路径
        preprocessor: 预处理器实例

    Returns:
        预处理后的图像 tensor (C, H, W)，归一化到 [-1, 1]
    """
    if preprocessor is None:
        preprocessor = FacePreprocessor(detector='mtcnn', device='cpu')

    # 获取对齐的人脸
    aligned_face = preprocessor.process(image_path)

    if aligned_face is None:
        return None

    # 归一化
    normalized = normalize_image(aligned_face)

    # HWC -> CHW
    normalized = np.transpose(normalized, (2, 0, 1))

    return normalized


if __name__ == '__main__':
    # 测试代码
    import sys

    print("人脸预处理模块测试")
    print("=" * 50)

    # 创建预处理器
    preprocessor = FacePreprocessor(detector='mtcnn', device='cpu')

    # 测试处理
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        result = preprocessor.process(test_image)

        if result is not None:
            print(f"成功处理图像，输出大小: {result.shape}")
            cv2.imwrite('aligned_face.jpg', result)
            print("对齐后的人脸已保存到 aligned_face.jpg")
        else:
            print("处理失败")
    else:
        print("使用方法: python preprocess.py <image_path>")
