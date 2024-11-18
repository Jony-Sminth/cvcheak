import os
import cv2
import json
import albumentations as A
import numpy as np
from tqdm import tqdm

class DataPreprocessing:
    def __init__(self, image_dir, label_path):
        self.image_dir = image_dir  # 图像文件夹
        self.label_path = label_path  # 标签文件
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(height=256, width=256, p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.Resize(height=512, width=512, p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.labels = self._load_labels()

    def _load_labels(self):
        # 从 JSON 文件加载标签
        with open(self.label_path, 'r') as f:
            labels = json.load(f)
        return labels

    def load_image(self, image_id):
        # 加载图像
        image_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image {image_id} not found in directory {self.image_dir}")
        # OpenCV 默认加载的是 BGR 格式，将其转换为 RGB 格式
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def preprocess_image(self, image_id):
        # 加载图像
        image = self.load_image(image_id)

        # 获取该图像的标签
        regions = []
        for label in self.labels:
            if label["id"] == image_id:
                regions = label["region"]
                break

        # 为了让模型了解篡改的区域，我们为这些区域创建一个掩膜
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for region in regions:
            x1, y1, x2, y2 = map(int, region)
            mask[y1:y2, x1:x2] = 1

        # 使用数据增强库进行数据增强
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        return image, mask

    def create_dataset(self, save_dir, limit=None):
        # 创建一个增强后的数据集
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, label in enumerate(tqdm(self.labels, desc="Processing images")):
            if limit is not None and idx >= limit:
                break
            image_id = label["id"]
            try:
                image, mask = self.preprocess_image(image_id)
                
                # 保存处理后的图像
                image_save_path = os.path.join(save_dir, image_id)
                mask_save_path = os.path.join(save_dir, f"mask_{image_id}")

                cv2.imwrite(image_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(mask_save_path, mask * 255)  # 保存为单通道的掩膜图像
            except FileNotFoundError as e:
                print(e)
