import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.Faster_R_CNN import get_faster_rcnn_model
from utils.dataset import CustomDataset  # 自定义 Dataset 类在 dataset.py 文件中
from utils.dataset import collate_fn  

# 定义训练函数
def train_model():
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载训练和验证数据集
    train_dataset = CustomDataset(image_dir='data/preprocessed_train/', label_path='data/train/label_train_split.json', transform=transform)
    val_dataset = CustomDataset(image_dir='data/preprocessed_val/', label_path='data/train/label_val_split.json', transform=transform)

    # 创建 DataLoader，使用自定义 collate_fn
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 加载模型
    model = get_faster_rcnn_model(num_classes=2)
    model = model.to('cuda') if torch.cuda.is_available() else model.to('cpu')

    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            # 检查是否为有效批次
            if images is None or targets is None:
                continue  # 跳过这个无效的批次

            # 将图像和标签转移到 GPU（如果可用）
            images = [img.to('cuda') if torch.cuda.is_available() else img.to('cpu') for img in images]
            targets = [{k: v.to('cuda') if torch.cuda.is_available() else v for k, v in t.items()} for t in targets]

            # 前向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to('cuda') if torch.cuda.is_available() else img.to('cpu') for img in images]
                targets = [{k: v.to('cuda') if torch.cuda.is_available() else v for k, v in t.items()} for t in targets]

                # 计算验证集损失
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        print(f"Validation Loss after Epoch {epoch + 1}: {val_loss / len(val_loader)}")


# 运行训练
if __name__ == '__main__':
    train_model()
