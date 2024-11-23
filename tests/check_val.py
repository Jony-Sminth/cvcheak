import os
import json
import torch
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset, collate_fn
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 配置路径
val_image_dir = os.path.abspath('data/preprocessed_val/')
val_label_path = os.path.abspath('data/train/label_val_split.json')

# 检查路径是否存在
if not os.path.exists(val_image_dir):
    raise FileNotFoundError(f"Validation image directory not found: {val_image_dir}")
if not os.path.exists(val_label_path):
    raise FileNotFoundError(f"Validation label file not found: {val_label_path}")

# 打印目录内容以确认可访问
print(f"Validation image directory exists: {os.path.exists(val_image_dir)}")
print(f"Files in validation image directory: {os.listdir(val_image_dir)[:10]}")  # 仅打印前 10 个文件

# 检查标签文件内容
with open(val_label_path, 'r') as f:
    labels = json.load(f)
print(f"Total labels: {len(labels)}")

# 验证标签与图片是否匹配
for img_name, annotations in labels.items():
    img_path = os.path.join(val_image_dir, img_name)
    if not os.path.exists(img_path):
        print(f"Missing image file: {img_path}")
    if not annotations:
        print(f"No annotations for image: {img_name}")

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 初始化验证集
val_dataset = CustomDataset(image_dir=val_image_dir, label_path=val_label_path, transform=transform)

# 检查 CustomDataset 输出
for idx in range(min(len(val_dataset), 5)):  # 只检查前 5 个样本
    try:
        image, target = val_dataset[idx]
        print(f"Sample {idx}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Target: {target}")
        if 'boxes' not in target or 'labels' not in target:
            print(f"  Missing keys in target for sample {idx}")
    except Exception as e:
        print(f"Error in sample {idx}: {e}")

# 创建验证集加载器
val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# 检查 collate_fn 输出
for batch_idx, (images, targets) in enumerate(val_loader):
    print(f"Batch {batch_idx}:")
    print(f"  Number of images: {len(images)}")
    print(f"  Targets: {targets}")
    for target in targets:
        if not isinstance(target, dict):
            print(f"  Target is not a dict: {target}")
        elif 'boxes' not in target or 'labels' not in target:
            print(f"  Missing keys in target: {target}")
    if batch_idx == 2:  # 只检查前 3 个批次
        break

# 检查模型前向传播
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ...  # 替换为你的模型实例
model.to(device)
model.eval()

# 测试模型前向传播
with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        try:
            loss_dict = model(images, targets)
            print(f"Batch {batch_idx}: Forward pass successful")
            print(f"Loss dict: {loss_dict}")
        except Exception as e:
            print(f"Batch {batch_idx}: Forward pass failed with error: {e}")
        break  # 只测试一个批次

# 数据可视化
def visualize_sample(image, target):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    image = image.permute(1, 2, 0).cpu().numpy()  # 从 (C, H, W) 转换为 (H, W, C)
    ax.imshow(image)
    # 绘制边界框
    for box in target['boxes']:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# 可视化第一个样本
image, target = val_dataset[0]
visualize_sample(image, target)
