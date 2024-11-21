import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.Faster_R_CNN import get_faster_rcnn_model
from utils.dataset import CustomDataset
from utils.dataset import collate_fn


class ModelTrainer:
    def __init__(self, model_name='faster_rcnn', num_classes=2, device=None):
        """
        初始化模型训练器

        Args:
            model_name (str): 模型名称，目前支持 'faster_rcnn'。
            num_classes (int): 类别数量（包含背景类）。
            device (str, optional): 设备类型，'cuda' 或 'cpu'，默认根据环境自动选择。
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        # 初始化模型
        self.model = self._load_model()
        self.model = self.model.to(self.device)

    def _load_model(self):
        """
        加载指定的模型。

        Returns:
            torch.nn.Module: 已初始化的模型实例。
        """
        if self.model_name == 'faster_rcnn':
            return get_faster_rcnn_model(num_classes=self.num_classes)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

    def train(self, train_image_dir, val_image_dir, train_label_path, val_label_path,
              batch_size=4, num_epochs=10, learning_rate=0.005, momentum=0.9, weight_decay=0.0005,
              save_dir="models/checkpoints/"):
        """
        开始模型训练。

        Args:
            train_image_dir (str): 训练图像文件夹路径。
            val_image_dir (str): 验证图像文件夹路径。
            train_label_path (str): 训练标签路径。
            val_label_path (str): 验证标签路径。
            batch_size (int): 每批训练的图像数量。
            num_epochs (int): 训练的总轮数。
            learning_rate (float): 优化器的学习率。
            momentum (float): SGD 优化器的动量参数。
            weight_decay (float): 权重衰减参数，用于防止过拟合。
            save_dir (str): 模型保存目录路径。
        """
        # 图像预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 加载数据集
        train_dataset = CustomDataset(image_dir=train_image_dir, label_path=train_label_path, transform=transform)
        val_dataset = CustomDataset(image_dir=val_image_dir, label_path=val_label_path, transform=transform)

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

        # 定义优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        # 创建模型保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 开始训练
        for epoch in range(num_epochs):
            self._train_one_epoch(train_loader, optimizer, epoch)
            self._validate_one_epoch(val_loader, epoch)

            # 保存每个 epoch 后的模型
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        # 最终保存完整模型
        final_model_path = os.path.join(save_dir, "model_final.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")

    def _train_one_epoch(self, train_loader, optimizer, epoch):
        """
        训练单个 epoch。

        Args:
            train_loader (DataLoader): 训练集的 DataLoader。
            optimizer (Optimizer): 优化器。
            epoch (int): 当前训练轮数。
        """
        self.model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            if images is None or targets is None:
                continue

            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # 前向传播
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_loader)}")

    def _validate_one_epoch(self, val_loader, epoch):
        """
        验证单个 epoch。

        Args:
            val_loader (DataLoader): 验证集的 DataLoader。
            epoch (int): 当前验证轮数。
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # 计算验证集损失
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        print(f"Validation Loss after Epoch {epoch + 1}: {val_loss / len(val_loader)}")
        