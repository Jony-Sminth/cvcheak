import os
import cv2
import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        """
        初始化自定义数据集。

        Args:
            image_dir (str): 图像文件路径。
            label_path (str): 标签文件路径。
            transform (callable, optional): 图像预处理操作。
        """
        self.image_dir = image_dir
        self.label_path = label_path  # 保存 label_path 为实例属性
        self.transform = transform
        self.labels = self._load_labels()
        self.warning_count = 0  # 添加警告计数器

    def _load_labels(self):
        """
        加载标签文件。

        Returns:
            list: 标签列表。
        """
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file not found: {self.label_path}")
        with open(self.label_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取数据样本。

        Args:
            idx (int): 样本索引。

        Returns:
            tuple: (图像, 目标) 或 (None, None)（如果无效）。
        """
        try:
            label = self.labels[idx]
            image_id = label['id']
            file_base_name = os.path.splitext(image_id)[0]

            # 加载图像和掩膜
            image_path = os.path.join(self.image_dir, f"{file_base_name}.pt")
            mask_path = os.path.join(self.image_dir, f"{file_base_name}_mask.png")

            image = torch.load(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = torch.from_numpy(mask).float() / 255.0
            image = torch.cat([image, mask.unsqueeze(0)], dim=0)

            # 检查目标框
            regions = label['region']
            if not regions:
                if self.warning_count < 10:  # 只打印前 10 条警告
                    print(f"Warning: No regions found for image {image_id}")
                self.warning_count += 1
                return None, None

            target = {
                "boxes": torch.tensor([region for region in regions], dtype=torch.float32),
                "labels": torch.ones((len(regions),), dtype=torch.int64)
            }

            return image, target

        except Exception as e:
            print(f"Error processing index {idx}: {str(e)}")
            return None, None


def collate_fn(batch):
    """
    批处理函数，用于 DataLoader。

    Args:
        batch (list): 一个批次的样本。

    Returns:
        tuple: (图像列表, 目标列表) 或 (None, None)（如果批次为空）。
    """
    # 过滤掉无效样本
    batch = [sample for sample in batch if sample[0] is not None]

    if len(batch) == 0:
        return None, None

    return tuple(zip(*batch))
