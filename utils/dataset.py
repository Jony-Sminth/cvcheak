import os
import cv2
import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        """
        初始化数据集

        Args:
            image_dir (str): 图像和掩膜的存储路径
            label_path (str): 标签文件路径（JSON）
            transform (callable, optional): 图像变换函数
        """
        self.image_dir = image_dir
        self.transform = transform

        # 加载标签文件
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

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

            image = torch.load(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = torch.from_numpy(mask).float() / 255.0
            image = torch.cat([image, mask.unsqueeze(0)], dim=0)

            # 检查目标框
            regions = label['region']
            if not regions:
                target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64)
                }
            else:
                # 添加框的验证
                valid_regions = []
                for region in regions:
                    x1, y1, x2, y2 = map(float, region)
                    # 检查并修复无效的框
                    if x1 == x2:
                        x2 += 1  # 如果宽度为0，加1个像素
                    if y1 == y2:
                        y2 += 1  # 如果高度为0，加1个像素
                    if x1 > x2:
                        x1, x2 = x2, x1  # 如果左边界大于右边界，交换
                    if y1 > y2:
                        y1, y2 = y2, y1  # 如果上边界大于下边界，交换
                    valid_regions.append([x1, y1, x2, y2])

                target = {
                    "boxes": torch.tensor(valid_regions, dtype=torch.float32),
                    "labels": torch.ones((len(valid_regions),), dtype=torch.int64)
                }

            return image, target

        except Exception as e:
            print(f"Error processing index {idx}: {str(e)}")
            raise

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