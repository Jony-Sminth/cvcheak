import os
import cv2
import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None, debug=False):
        """
        初始化数据集

        Args:
            image_dir (str): 图像和掩膜的存储路径
            label_path (str): 标签文件路径（JSON）
            transform (callable, optional): 图像变换函数
            debug (bool): 是否打印调试信息
        """
        self.image_dir = image_dir
        self.transform = transform
        self.debug = debug

        # 加载标签文件
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
            
        # 统计正负样本比例
        if self.debug:
            total_images = len(self.labels)
            images_with_regions = sum(1 for label in self.labels if label['region'])
            print(f"Dataset Stats: Total images: {total_images}, Images with regions: {images_with_regions} ({images_with_regions/total_images*100:.2f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            label = self.labels[idx]
            image_id = label['id']
            file_base_name = os.path.splitext(image_id)[0]

            # 加载图像和掩膜
            image_path = os.path.join(self.image_dir, f"{file_base_name}.pt")
            mask_path = os.path.join(self.image_dir, f"{file_base_name}_mask.png")

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                if self.debug:
                    print(f"Warning: Missing files for {image_id}, image_path exists: {os.path.exists(image_path)}, mask_path exists: {os.path.exists(mask_path)}")
                # 返回空数据
                return None, None

            image = torch.load(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                if self.debug:
                    print(f"Warning: Failed to load mask for {image_id}")
                # 返回空数据
                return None, None
                
            # 获取原始和预处理后的图像尺寸
            original_height, original_width = label.get('original_size', (0, 0))
            current_height, current_width = image.shape[1], image.shape[2]
            
            # 计算缩放比例
            width_ratio = current_width / original_width if original_width > 0 else 1.0
            height_ratio = current_height / original_height if original_height > 0 else 1.0
            
            # 将掩码转换为张量并归一化
            mask = torch.from_numpy(mask).float() / 255.0
            
            # 将图像和掩码组合为4通道输入
            combined_tensor = torch.cat([image, mask.unsqueeze(0)], dim=0)

            # 检查目标框并根据调整后的尺寸比例更新坐标
            regions = label['region']
            
            if not regions:
                # 没有篡改区域的情况
                target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64)
                }
            else:
                # 根据图像缩放比例调整边界框坐标
                valid_regions = []
                
                for region in regions:
                    x1, y1, x2, y2 = map(float, region)
                    
                    # 根据缩放比例调整坐标
                    if original_height > 0 and original_width > 0:
                        x1 = x1 * width_ratio
                        x2 = x2 * width_ratio
                        y1 = y1 * height_ratio
                        y2 = y2 * height_ratio
                    
                    # 检查并修复无效的框
                    if x1 == x2:
                        x2 += 1  # 如果宽度为0，加1个像素
                    if y1 == y2:
                        y2 += 1  # 如果高度为0，加1个像素
                    if x1 > x2:
                        x1, x2 = x2, x1  # 如果左边界大于右边界，交换
                    if y1 > y2:
                        y1, y2 = y2, y1  # 如果上边界大于下边界，交换
                        
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, current_width - 1))
                    x2 = max(0, min(x2, current_width - 1))
                    y1 = max(0, min(y1, current_height - 1))
                    y2 = max(0, min(y2, current_height - 1))
                    
                    if x2 > x1 and y2 > y1:
                        valid_regions.append([x1, y1, x2, y2])

                if self.debug and idx % 100 == 0:
                    print(f"Image {idx} - Original regions: {regions}")
                    print(f"Image {idx} - Adjusted regions: {valid_regions}")
                    print(f"Image {idx} - Scale factors: width={width_ratio}, height={height_ratio}")

                # 创建目标字典
                target = {
                    "boxes": torch.tensor(valid_regions, dtype=torch.float32) if valid_regions else torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.ones((len(valid_regions),), dtype=torch.int64) if valid_regions else torch.zeros((0,), dtype=torch.int64)
                }

            return combined_tensor, target

        except Exception as e:
            if self.debug:
                print(f"Error processing index {idx}: {str(e)}")
                import traceback
                print(traceback.format_exc())
            return None, None

def collate_fn(batch):
    """
    批处理函数，用于 DataLoader。

    Args:
        batch (list): 一个批次的样本。

    Returns:
        tuple: (图像列表, 目标列表) 或 (None, None)（如果批次为空）。
    """
    batch = [sample for sample in batch if sample[0] is not None and sample[1] is not None]
    if len(batch) == 0:
        return None, None
    return tuple(zip(*batch))