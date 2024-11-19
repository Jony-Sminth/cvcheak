import os
import cv2
import json
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

class DataPreprocessing:
    def __init__(self, image_dir, label_path):
        self.image_dir = image_dir  # 图像文件夹
        self.label_path = label_path  # 标签文件

        # 使用 torchvision.transforms 进行图像变换，调整图像尺寸并转换为张量
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # 将所有图像大小调整为 512x512
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 加载标签文件
        self.labels = self._load_labels()

    def _load_labels(self):
        # 从 JSON 文件加载标签
        with open(self.label_path, 'r') as f:
            labels = json.load(f)
        return labels

    def load_image(self, image_id):
        # 加载图像，使用 PIL 读取，并保持 RGB 格式
        image_path = os.path.join(self.image_dir, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_id} not found in directory {self.image_dir}")
        image = Image.open(image_path).convert("RGB")
        return image

    def preprocess_image(self, image_id):
        # 加载图像
        image = self.load_image(image_id)
        width, height = image.size

        # 获取该图像的标签
        regions = []
        for label in self.labels:
            if label["id"] == image_id:
                regions = label["region"]
                break

        # 为了让模型了解篡改的区域，我们为这些区域创建一个掩膜
        mask = np.zeros((height, width), dtype=np.uint8)  # mask 的形状为 (height, width)
        for region in regions:
            x1, y1, x2, y2 = map(int, region)
            # 确保坐标不越界
            x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))

            if x1 < x2 and y1 < y2:
                mask[y1:y2, x1:x2] = 255  # 使用 255 而不是 1 提高对比度

        # 使用 torchvision.transforms 进行图像增强和预处理
        image = self.transform(image)  # 转换为张量，并标准化

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
                
                # 保存处理后的图像和掩膜
                image_save_path = os.path.join(save_dir, f"{image_id}.pt")  # 保存为 PyTorch 的张量格式
                mask_save_path = os.path.join(save_dir, f"mask_{image_id}.png")

                torch.save(image, image_save_path)  # 保存图像为 PyTorch 的张量文件
                cv2.imwrite(mask_save_path, mask)  # 保存掩膜为单通道的 PNG 图像
            except FileNotFoundError as e:
                print(e)

# 创建数据预处理对象并生成数据集
if __name__ == "__main__":
    # 训练集处理
    train_image_dir = 'data/train/images/'
    train_label_path = 'data/train/label_train_split.json'
    train_save_dir = 'data/preprocessed_train/'

    train_preprocessor = DataPreprocessing(image_dir=train_image_dir, label_path=train_label_path)
    train_preprocessor.create_dataset(save_dir=train_save_dir)

    # 验证集处理
    val_image_dir = 'data/val/images/'
    val_label_path = 'data/train/label_val_split.json'
    val_save_dir = 'data/preprocessed_val/'

    val_preprocessor = DataPreprocessing(image_dir=val_image_dir, label_path=val_label_path)
    val_preprocessor.create_dataset(save_dir=val_save_dir)

    print("Data preprocessing complete.")