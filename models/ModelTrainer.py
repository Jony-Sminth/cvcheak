import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.Faster_R_CNN import get_faster_rcnn_model
from utils.dataset import CustomDataset
from utils.dataset import collate_fn
from tqdm import tqdm
import GPUtil  # GPU 监控模块
import time

class ModelTrainer:
    def __init__(self, model_name='faster_rcnn', num_classes=2, device=None, log_frequency=100):
        """
        初始化模型训练器

        Args:
            model_name (str): 模型名称，目前支持 'faster_rcnn'
            num_classes (int): 类别数量（包含背景类）
            device (str, optional): 设备类型，'cuda' 或 'cpu'，默认根据环境自动选择
            log_frequency (int): 每多少个 batch 打印一次日志
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.log_frequency = log_frequency

        # 初始化模型
        self.model = self._load_model()
        self.model = self.model.to(self.device)

    def set_log_frequency(self, log_frequency):
        """
        设置日志打印频率

        Args:
            log_frequency (int): 每多少个 batch 打印一次日志
        """
        self.log_frequency = log_frequency

    def log_gpu_status(self):
        """
        打印当前 GPU 使用状态
        """
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU ID: {gpu.id}, Name: {gpu.name}, Load: {gpu.load * 100:.2f}%, "
                  f"Free Memory: {gpu.memoryFree} MB, Total Memory: {gpu.memoryTotal} MB")

    def _load_model(self):
        """
        加载指定的模型

        Returns:
            torch.nn.Module: 已初始化的模型实例
        """
        if self.model_name == 'faster_rcnn':
            return get_faster_rcnn_model(num_classes=self.num_classes)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

    def _train_one_epoch(self, train_loader, optimizer, epoch):
        """
        训练单个 epoch。
        """
        self.model.train()
        epoch_loss = 0
        batch_count = len(train_loader)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', total=batch_count)

        for batch_idx, (images, targets) in enumerate(pbar):
            if images is None or targets is None:
                print(f"Skipping invalid batch {batch_idx}")
                continue

            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            # 每 log_frequency 批次打印一次日志
            if (batch_idx + 1) % self.log_frequency == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Batch [{batch_idx + 1}/{batch_count}], Loss: {losses.item():.4f}")
                self.log_gpu_status()  # 打印 GPU 使用状态

            # 更新进度条
            pbar.set_postfix({'loss': f'{losses.item():.4f}', 'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'})

        return epoch_loss / batch_count

    def _validate_one_epoch(self, val_loader, epoch):
        """
        验证单个 epoch。
        """
        self.model.eval()
        val_loss = 0
        batch_count = len(val_loader)

        pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}', total=batch_count)

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(pbar):
                if images is None or targets is None:
                    continue

                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

                if (batch_idx + 1) % self.log_frequency == 0:
                    print(f"Validation Epoch [{epoch + 1}], Batch [{batch_idx + 1}/{batch_count}], Loss: {losses.item():.4f}")
                    self.log_gpu_status()  # 打印 GPU 使用状态

                pbar.set_postfix({'val_loss': f'{losses.item():.4f}', 'avg_val_loss': f'{val_loss / (batch_idx + 1):.4f}'})

        return val_loss / batch_count

    def train(self, train_image_dir, val_image_dir, train_label_path, val_label_path,
              batch_size=2, num_epochs=10, learning_rate=0.005, momentum=0.9, 
              weight_decay=0.0005, save_dir="models/checkpoints/"):
        """
        开始模型训练。
        """
        print("\nInitializing training...")
        print(f"Training with following parameters:")
        print(f"- Batch size: {batch_size}")
        print(f"- Number of epochs: {num_epochs}")
        print(f"- Learning rate: {learning_rate}")
        print(f"- Momentum: {momentum}")
        print(f"- Weight decay: {weight_decay}")
        print(f"- Device: {self.device}")

        self.num_epochs = num_epochs  # 保存总 epoch 数用于显示进度

        # 图像预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 加载数据集
        print("\nLoading datasets...")
        train_dataset = CustomDataset(image_dir=train_image_dir, 
                                    label_path=train_label_path, 
                                    transform=transform)
        val_dataset = CustomDataset(image_dir=val_image_dir, 
                                  label_path=val_label_path, 
                                  transform=transform)
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4, 
                                collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4, 
                              collate_fn=collate_fn)

        # 定义优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, 
                                  momentum=momentum, weight_decay=weight_decay)

        # 创建模型保存目录
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nCheckpoints will be saved to: {save_dir}")

        # 记录训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }

        print("\nStarting training loop...")
        training_start_time = time.time()

        try:
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                
                # 训练一个 epoch
                train_loss = self._train_one_epoch(train_loader, optimizer, epoch)
                history['train_loss'].append(train_loss)
                
                # 验证
                val_loss = self._validate_one_epoch(val_loader, epoch)
                history['val_loss'].append(val_loss)

                # 保存模型
                checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path}")

                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
        finally:
            # 保存最终模型
            final_model_path = os.path.join(save_dir, "model_final.pth")
            torch.save(self.model.state_dict(), final_model_path)
            
            total_time = time.time() - training_start_time
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Final model saved at {final_model_path}")

        return history
