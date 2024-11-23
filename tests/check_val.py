import os
import json
import torch
import multiprocessing
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset, collate_fn
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def run_validation_check():
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

    # 统计没有标注的图像数量
    no_annotation_count = 0
    for label in labels:
        img_id = label["id"]
        regions = label.get("region", [])
        if not regions:
            no_annotation_count += 1
            print(f"No annotations for image: {img_id}")
    print(f"\nTotal images without annotations: {no_annotation_count}")

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 初始化验证集
    print("\nInitializing validation dataset...")
    val_dataset = CustomDataset(image_dir=val_image_dir, label_path=val_label_path, transform=transform)
    print(f"Dataset size: {len(val_dataset)}")

    # 检查 CustomDataset 输出
    print("\nChecking CustomDataset outputs:")
    for idx in range(min(len(val_dataset), 5)):  # 只检查前 5 个样本
        try:
            image, target = val_dataset[idx]
            print(f"\nSample {idx}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Target boxes shape: {target['boxes'].shape if 'boxes' in target else 'No boxes'}")
            print(f"  Target labels shape: {target['labels'].shape if 'labels' in target else 'No labels'}")
        except Exception as e:
            print(f"Error in sample {idx}: {e}")

    # 创建验证集加载器
    print("\nCreating DataLoader...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # 设置为0以避免多进程问题
        collate_fn=collate_fn
    )

    # 检查 DataLoader 输出
    print("\nChecking DataLoader outputs:")
    try:
        for batch_idx, (images, targets) in enumerate(val_loader):
            print(f"\nBatch {batch_idx}:")
            if images is None or targets is None:
                print("  Received None batch, skipping...")
                continue
                
            print(f"  Number of images in batch: {len(images)}")
            print(f"  Image shapes: {[img.shape for img in images]}")
            print(f"  Number of targets: {len(targets)}")
            
            for i, target in enumerate(targets):
                print(f"  Target {i} boxes shape: {target['boxes'].shape}")
                print(f"  Target {i} labels shape: {target['labels'].shape}")
            
            if batch_idx == 2:  # 只检查前 3 个批次
                break
    except Exception as e:
        print(f"Error during DataLoader iteration: {str(e)}")

    print("\nValidation data check complete.")

    # 可视化函数
    def visualize_sample(image, target, save_path=None):
        try:
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # 显示原始图像（前3通道）
            img_display = image[:3].permute(1, 2, 0).cpu().numpy()
            img_display = (img_display + 1) / 2  # 反归一化
            ax1.imshow(img_display)
            ax1.set_title("Original Image with Boxes")
            
            # 在原始图像上绘制边界框
            for box in target['boxes']:
                x1, y1, x2, y2 = box.tolist()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                    linewidth=2, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)
            
            # 显示mask通道
            mask_display = image[3].cpu().numpy()
            ax2.imshow(mask_display, cmap='gray')
            ax2.set_title("Mask Channel")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        except Exception as e:
            print(f"Error during visualization: {str(e)}")

    # 可视化几个样本
    print("\nVisualizing samples...")
    os.makedirs('visualization', exist_ok=True)
    for i in range(min(5, len(val_dataset))):
        try:
            image, target = val_dataset[i]
            save_path = f'visualization/sample_{i}.png'
            visualize_sample(image, target, save_path)
            print(f"Sample {i} visualization saved to {save_path}")
        except Exception as e:
            print(f"Error visualizing sample {i}: {e}")

    print("\nVisualization complete.")

if __name__ == '__main__':
    # Windows平台下使用多进程可能会出现问题，因此需要调用freeze_support()来解决
    multiprocessing.freeze_support()
    run_validation_check()