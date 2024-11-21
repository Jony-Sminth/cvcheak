import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        """
        自定义数据集类

        Args:
            image_dir (str): 图像文件夹路径，包含预处理后的 .pt 文件。
            label_path (str): 标签文件路径，JSON 格式。
            transform (callable, optional): 图像变换操作。
        """
        self.image_dir = image_dir
        self.transform = transform

        # 加载标签
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_id = label['id']
        regions = label['region']

        # 加载 .pt 格式的图像
        file_base_name = os.path.splitext(image_id)[0]
        image_path = os.path.join(self.image_dir, f"{file_base_name}.pt")
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            return None, None
        image = torch.load(image_path)

        # 检查目标框是否为空
        if len(regions) == 0:
            print(f"Warning: No target regions for image {image_id}")
            return None, None

        # 转换标签为目标格式
        target = {
            "boxes": torch.tensor([region for region in regions], dtype=torch.float32),
            "labels": torch.ones((len(regions),), dtype=torch.int64),  # 假设所有目标都属于一个类
        }

        return image, target


def collate_fn(batch):
    # 过滤掉那些为 None 的样本
    batch = [sample for sample in batch if sample[0] is not None]

    if len(batch) == 0:
        return None, None

    return tuple(zip(*batch))
