import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        """
        Args:
            image_dir (str): 图像文件的路径
            label_path (str): 标签文件的路径 (JSON 格式)
            transform (callable, optional): 对图像进行的转换
        """
        self.image_dir = image_dir
        self.transform = transform
        with open(label_path, 'r') as f:
            self.labels = json.load(f)  # 加载标签

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取图像信息和标签信息
        label_info = self.labels[idx]
        image_id = label_info["id"]
        regions = label_info["region"]

        # 构造图像路径
        image_path = os.path.join(self.image_dir, image_id)
        
        # 检查图像是否存在，如果不存在则跳过或抛出警告
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            # 你可以选择重新尝试或直接抛出异常，这里选择返回空(None)，在 DataLoader 时进一步处理
            return None, None

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 创建标签
        boxes = []
        for region in regions:
            x1, y1, x2, y2 = map(int, region)
            boxes.append([x1, y1, x2, y2])

        # 转换为 tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # 假设所有的目标类别都是 1（篡改区域）

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # 应用图像的变换
        if self.transform:
            image = self.transform(image)

        return image, target


def collate_fn(batch):
    # 过滤掉那些为 None 的样本
    batch = [sample for sample in batch if sample[0] is not None]

    if len(batch) == 0:
        return None, None

    return tuple(zip(*batch))
