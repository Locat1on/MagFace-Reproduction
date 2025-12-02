import os
import cv2
import numpy as np
from inference.pipeline import FaceRecognitionPipeline
from data.preprocess import FacePreprocessor

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

pipeline = FaceRecognitionPipeline(
    model_path='pretrain/magface_epoch_00025.pth',
    backbone='iresnet100',
    device='cuda',
    detector='mtcnn',
    quality_threshold=15.0,
    similarity_threshold=0.40,
)

image_path = 'photo/lfw/Abdoulaye_Wade/Abdoulaye_Wade_0001.jpg'       # 用于注册
image_path2 = 'photo/lfw/Abdoulaye_Wade/Abdoulaye_Wade_0002.jpg'    # 用于识别
image_path3 = 'photo/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'  # 陌生人测试
human_name = 'Abdoulaye Wade'

# ========== 1. 保存原始图片 ==========
print("\n[步骤1] 读取原始图片...")
original_image = cv2.imread(image_path)
cv2.imwrite(os.path.join(output_dir, '1_original.jpg'), original_image)
print(f"  注册图片大小: {original_image.shape}")

original_image2 = cv2.imread(image_path2)
cv2.imwrite(os.path.join(output_dir, '1_original_query.jpg'), original_image2)
print(f"  待识别图片大小: {original_image2.shape}")

# ========== 2. 人脸检测结果 ==========
print("\n[步骤2] 人脸检测...")
faces = pipeline.preprocessor.detect_faces(original_image)
print(f"  检测到 {len(faces)} 个人脸")

# 绘制检测框和关键点
detected_image = original_image.copy()
for i, face in enumerate(faces):
    # 绘制边界框
    x1, y1, x2, y2 = face['bbox']
    cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 绘制置信度
    conf = face['confidence']
    cv2.putText(detected_image, f'{conf:.4f}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 绘制5个关键点
    landmarks = face['landmarks']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    labels = ['左眼', '右眼', '鼻尖', '左嘴角', '右嘴角']
    for j, (lm, color) in enumerate(zip(landmarks, colors)):
        x, y = int(lm[0]), int(lm[1])
        cv2.circle(detected_image, (x, y), 8, color, -1)

cv2.imwrite(os.path.join(output_dir, '2_detected.jpg'), detected_image)
print(f"  保存检测结果: {output_dir}/2_detected.jpg")

# ========== 3. 对齐后的人脸 ==========
print("\n[步骤3] 人脸对齐...")
aligned_face = pipeline.preprocessor.process(image_path)
if aligned_face is not None:
    cv2.imwrite(os.path.join(output_dir, '3_aligned_112x112.jpg'), aligned_face)
    print(f"  对齐后大小: {aligned_face.shape}")
    print(f"  保存对齐结果: {output_dir}/3_aligned_112x112.jpg")

# ========== 4. 注册人脸 ==========
print("\n[步骤4] 注册人脸...")
result = pipeline.register(human_name, image_path)
print(f"  注册结果: {result['message']}")
print(f"  质量评分: {result['quality']['magnitude']:.2f}")
print(f"  质量等级: {result['quality']['description']}")

# ========== 5. 识别人脸（使用第二张照片）==========
print("\n[步骤5] 识别人脸（使用第二张照片）...")

# 先检测第二张图片的人脸
faces2 = pipeline.preprocessor.detect_faces(original_image2)
print(f"  检测到 {len(faces2)} 个人脸")

result = pipeline.recognize(image_path2)

if result['success']:
    best_match = result['best_match']
    print(f"  识别成功！")
    print(f"  匹配人员: {best_match['name']}")
    print(f"  相似度: {best_match['similarity']:.4f}")

    # 创建识别结果可视化（在第二张图片上标注）
    result_image = original_image2.copy()
    if len(faces2) > 0:
        x1, y1, x2, y2 = faces2[0]['bbox']
        # 绘制绿色框表示识别成功
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        # 添加文字信息
        text = f"{best_match['name']} ({best_match['similarity']:.2%})"
        cv2.putText(result_image, text, (x1, y1-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(output_dir, '4_recognized.jpg'), result_image)
    print(f"  保存识别结果: {output_dir}/4_recognized.jpg")
else:
    print(f"  识别失败: {result.get('message', '未知错误')}")

# ========== 6. 陌生人验证（应该识别失败）==========
print("\n[步骤6] 陌生人验证（使用完全不同的人）...")
stranger_path = image_path3
stranger_image = cv2.imread(stranger_path)
print(f"  陌生人图片大小: {stranger_image.shape}")

# 检测人脸
faces_stranger = pipeline.preprocessor.detect_faces(stranger_image)
print(f"  检测到 {len(faces_stranger)} 个人脸")

result_stranger = pipeline.recognize(stranger_path)

if result_stranger['success']:
    best_match = result_stranger['best_match']
    print(f"  ⚠️ 匹配到人员: {best_match['name']}")
    print(f"  相似度: {best_match['similarity']:.4f}")
    print(f"  是否通过阈值: {best_match['is_match']}")

    if best_match['is_match']:
        print("  ❌ 警告：陌生人被错误识别为已注册用户！")
        color = (0, 0, 255)  # 红色表示错误匹配
    else:
        print("  ✅ 正确：相似度低于阈值，拒绝识别")
        color = (0, 165, 255)  # 橙色表示低相似度
else:
    print(f"  ✅ 正确：识别失败 - {result_stranger.get('message', '未知')}")
    color = (128, 128, 128)  # 灰色

# 可视化陌生人识别结果
stranger_result_image = stranger_image.copy()
if len(faces_stranger) > 0:
    x1, y1, x2, y2 = faces_stranger[0]['bbox']
    cv2.rectangle(stranger_result_image, (x1, y1), (x2, y2), color, 4)
    if result_stranger['success']:
        sim = result_stranger['best_match']['similarity']
        text = f"Stranger ({sim:.2%})"
    else:
        text = "Unknown"
    cv2.putText(stranger_result_image, text, (x1, y1-15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

cv2.imwrite(os.path.join(output_dir, '6_stranger_test.jpg'), stranger_result_image)
print(f"  保存陌生人测试结果: {output_dir}/6_stranger_test.jpg")

# ========== 7. 特征可视化 ==========
print("\n[步骤7] 特征可视化...")
feature, magnitude = pipeline.extract_features(aligned_face)

# 将512维特征重塑为图像形式 (16x32)
feature_vis = feature.reshape(16, 32)
# 归一化到 0-255
feature_vis = ((feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min()) * 255).astype(np.uint8)
# 放大显示
feature_vis = cv2.resize(feature_vis, (320, 160), interpolation=cv2.INTER_NEAREST)
# 应用颜色映射
feature_vis_color = cv2.applyColorMap(feature_vis, cv2.COLORMAP_JET)

cv2.imwrite(os.path.join(output_dir, '5_feature_map.jpg'), feature_vis_color)
print(f"  特征维度: {feature.shape}")
print(f"  特征幅度(质量分数): {magnitude:.2f}")
print(f"  保存特征图: {output_dir}/5_feature_map.jpg")

# ========== 汇总 ==========
print("\n" + "=" * 50)
print("处理完成！所有输出图片保存在:", output_dir)
print("=" * 50)
print(f"  1_original.jpg        - 注册用原图")
print(f"  1_original_query.jpg  - 识别用原图")
print(f"  2_detected.jpg        - 人脸检测结果(框+关键点)")
print(f"  3_aligned_112x112.jpg - 对齐后的人脸(112x112)")
print(f"  4_recognized.jpg      - 同一人识别结果")
print(f"  5_feature_map.jpg     - 512维特征可视化")
print(f"  6_stranger_test.jpg   - 陌生人测试结果")
