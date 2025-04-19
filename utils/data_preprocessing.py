import os
import cv2
import json
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

class DataPreprocessing:
    def __init__(self, image_dir, label_path, target_size=(800, 800), debug=False):
        self.image_dir = image_dir  # 图像文件夹
        self.label_path = label_path  # 标签文件
        self.target_size = target_size
        self.debug = debug

        # 使用 torchvision.transforms 进行图像变换，调整图像尺寸并转换为张量
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),  # 添加固定尺寸调整
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 加载标签文件，如果提供了标签路径
        self.labels = self._load_labels() if os.path.exists(label_path) else []

    def _load_labels(self):
        # 从 JSON 文件加载标签
        with open(self.label_path, 'r') as f:
            labels = json.load(f)
            
        # 添加统计信息
        if self.debug:
            total_images = len(labels)
            images_with_regions = sum(1 for label in labels if label['region'])
            print(f"Label Stats: Total images: {total_images}, Images with regions: {images_with_regions} ({images_with_regions/total_images*100:.2f}%)")
            
            # 打印几个样本示例
            for i in range(min(3, len(labels))):
                print(f"Sample label {i}: {labels[i]}")
                
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

        # 调整图像尺寸并转换为张量
        # 注意：此方法将首先将图像调整为固定大小，然后标准化
        image_tensor = self.transform(image)

        # 调整掩码大小以匹配调整后的图像
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        return image_tensor, mask, (height, width)

    def preprocess_for_prediction(self, image_path, label_data=None):
        """
        用于预测的统一预处理接口
        
        Args:
            image_path: 图像路径
            label_data: 可选的标签数据，用于创建掩码
            
        Returns:
            处理后的图像张量和尺寸信息
        """
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {str(e)}")
            
        original_width, original_height = image.size
        
        # 创建原始尺寸的掩码
        mask = np.zeros((original_height, original_width), dtype=np.uint8)
        
        # 如果有标签数据，创建掩码
        if label_data:
            regions = label_data.get("region", [])
            for region in regions:
                x1, y1, x2, y2 = map(int, region)
                # 确保坐标不越界
                x1, x2 = max(0, min(x1, original_width)), max(0, min(x2, original_width))
                y1, y2 = max(0, min(y1, original_height)), max(0, min(y2, original_height))
                
                if x1 < x2 and y1 < y2:
                    mask[y1:y2, x1:x2] = 255
        
        # 应用图像变换 (包括调整大小和标准化)
        image_tensor = self.transform(image)
        
        # 调整掩码大小以匹配调整后的图像
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 创建掩码张量
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        
        # 合并为4通道张量
        combined_tensor = torch.cat([image_tensor, mask_tensor.unsqueeze(0)], dim=0)
        
        return combined_tensor, (original_height, original_width)

    def create_dataset(self, save_dir, limit=None):
        # 创建一个增强后的数据集
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 用于保存原始图像尺寸信息
        sizes_dict = {}

        for idx, label in enumerate(tqdm(self.labels, desc="Processing images")):
            if limit is not None and idx >= limit:
                break
            image_id = label["id"]
            try:
                file_base_name = os.path.splitext(image_id)[0]  # 去掉原始扩展名

                # 预处理图像和掩膜
                image, mask, original_size = self.preprocess_image(image_id)
                
                # 保存原始尺寸信息
                sizes_dict[image_id] = original_size
                
                # 在标签中添加原始尺寸信息
                label["original_size"] = original_size

                # 保存处理后的图像
                image_save_path = os.path.join(save_dir, f"{file_base_name}.pt")
                torch.save(image, image_save_path)

                # 保存掩膜
                mask_save_path = os.path.join(save_dir, f"{file_base_name}_mask.png")
                if mask is not None and mask.size > 0:
                    cv2.imwrite(mask_save_path, mask)
                else:
                    print(f"Warning: Mask is empty for {image_id}")
                    
                if self.debug and idx % 100 == 0:
                    # 输出调试信息
                    print(f"Processed image {idx}: {image_id}")
                    print(f"Original size: {original_size}, Target size: {self.target_size}")
                    print(f"Regions: {label['region']}")
                    
            except Exception as e:
                print(f"Failed to process image {image_id}: {e}")
                if self.debug:
                    import traceback
                    print(traceback.format_exc())
                    
        # 保存更新后的标签文件
        updated_label_path = os.path.join(save_dir, 'updated_labels.json')
        with open(updated_label_path, 'w') as f:
            json.dump(self.labels, f, indent=2)
            
        if self.debug:
            print(f"Updated labels saved to {updated_label_path}")


# 创建数据预处理对象并生成数据集
if __name__ == "__main__":
    # 训练集处理
    train_image_dir = 'data/train/images/'
    train_label_path = 'data/train/label_train_split.json'
    train_save_dir = 'data/preprocessed_train/'

    train_preprocessor = DataPreprocessing(
        image_dir=train_image_dir, 
        label_path=train_label_path,
        target_size=(800, 800),
        debug=True
    )
    train_preprocessor.create_dataset(save_dir=train_save_dir)

    # 验证集处理
    val_image_dir = 'data/train/images/'
    val_label_path = 'data/train/label_val_split.json'
    val_save_dir = 'data/preprocessed_val/'

    val_preprocessor = DataPreprocessing(
        image_dir=val_image_dir, 
        label_path=val_label_path,
        target_size=(800, 800),
        debug=True
    )
    val_preprocessor.create_dataset(save_dir=val_save_dir)

    print("Data preprocessing complete.")