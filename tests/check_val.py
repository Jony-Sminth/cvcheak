from torch.utils.data import DataLoader
from utils.dataset import CustomDataset, collate_fn

val_image_dir = 'data/preprocessed_val/'
val_label_path = 'data/train/label_val_split.json'

# 定义验证集的图像预处理
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 实例化验证集
val_dataset = CustomDataset(image_dir=val_image_dir, label_path=val_label_path, transform=transform)

# 假设你已经定义了 val_dataset
val_loader = DataLoader(
    val_dataset,
    batch_size=2,  # 和训练时相同的 batch size
    shuffle=False,
    num_workers=4,  # 和训练时保持一致
    collate_fn=collate_fn  # 使用自定义的 collate 函数
)

# 遍历验证集并打印内容
for batch_idx, (images, targets) in enumerate(val_loader):
    print(f"Batch {batch_idx}:")
    print(f"Images: {len(images)}")  # 检查图像数量
    print(f"Targets: {targets}")  # 打印目标的内容

    # 可进一步检查 targets 的格式
    for target in targets:
        if not isinstance(target, dict):
            print(f"Target is not a dict: {target}")
        elif 'boxes' not in target or 'labels' not in target:
            print(f"Missing keys in target: {target}")
