from PIL import Image  # 从 PIL 导入 Image
from torchvision import transforms  # 从 torchvision 导入 transforms
import matplotlib.pyplot as plt  # 用于图像可视化
import numpy as np
import json
import os


class DataPreprocessing:
    def __init__(self, image_dir, label_path):
        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.labels = self._load_labels()

    def _load_labels(self):
        with open(self.label_path, 'r') as f:
            labels = json.load(f)
        return labels

    def load_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_id} not found in directory {self.image_dir}")
        image = Image.open(image_path).convert("RGB")
        return image

    def preprocess_image(self, image_id):
        # 加载图像
        image = self.load_image(image_id)
        width, height = image.size

        # 显示原始图像
        plt.imshow(image)
        plt.title(f"Original Image - {image_id}")
        plt.show()

        # 获取图像的标签
        regions = []
        for label in self.labels:
            if label["id"] == image_id:
                regions = label["region"]
                break

        # 打印标签信息
        print(f"Regions for {image_id}: {regions}")

        # 为了让模型了解篡改的区域，我们为这些区域创建一个掩膜
        mask = np.zeros((height, width), dtype=np.uint8)  # (height, width)
        for region in regions:
            x1, y1, x2, y2 = map(int, region)
            # 确保坐标不越界
            x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))

            if x1 < x2 and y1 < y2:
                mask[y1:y2, x1:x2] = 255  # 使用 255 而不是 1 提高对比度
                # 显示掩膜的当前状态
                plt.imshow(mask, cmap='hot')
                plt.title(f"Mask after applying region {region}")
                plt.show()

        # 确保掩膜不为空
        if np.all(mask == 0):
            print("Warning: Mask is completely empty, no tampered regions detected.")
        else:
            print("Mask created successfully.")

        # 进行图像增强和预处理
        image = self.transform(image)

        # 将图像转为 numpy 以便显示
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 0.5) + 0.5  # 反归一化，恢复为 [0, 1] 范围
        image_np = np.clip(image_np, 0, 1)  # 确保所有值在 [0, 1] 范围内

        # 显示增强后的图像
        plt.imshow(image_np)
        plt.title(f"Transformed Image - {image_id}")
        plt.show()

        return image, mask

if __name__ == "__main__":
    # 设置图像和标签路径
    image_dir = 'data/train/images/'  # 这里替换为你的图像所在的文件夹路径
    label_path = 'data/train/label_train.json'  # 这里替换为你的标签文件的路径

    # 初始化 DataPreprocessing 类
    preprocessor = DataPreprocessing(image_dir=image_dir, label_path=label_path)

    # 指定你要试验的图像 ID

    for image_id in ['train_1001.jpg', 'train_2001.jpg', 'train_3001.jpg','train_13817.jpg']:
        try:
            preprocessed_image, mask = preprocessor.preprocess_image(image_id)
        except FileNotFoundError as e:
            print(e)

