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
    def __init__(self, model_name='faster_rcnn', num_classes=2, device=None, log_frequency=100, debug=True):
        """
        初始化时添加debug参数
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.log_frequency = log_frequency
        self.debug = debug  # 添加debug标志
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        
        if self.debug:
            print(f"\nDebug Mode Enabled")
            print(f"Model Architecture:")
            print(self.model)
            print(f"Model Device: {self.device}")

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

    def _print_debug_info(self, phase, batch_idx, images, targets, loss_dict=None):
        """
        用于打印调试信息的辅助方法
        """
        if not self.debug:
            return

        print(f"\n{'='*50}")
        print(f"Debug Info for {phase} - Batch {batch_idx}")
        print(f"{'='*50}")
        
        # 打印图像信息
        print("\nImage Information:")
        print(f"Number of images in batch: {len(images)}")
        for i, img in enumerate(images):
            print(f"Image {i} shape: {img.shape}")
            print(f"Image {i} device: {img.device}")
            print(f"Image {i} dtype: {img.dtype}")
            print(f"Image {i} value range: [{img.min():.4f}, {img.max():.4f}]")

        # 打印目标信息
        print("\nTarget Information:")
        print(f"Number of targets: {len(targets)}")
        for i, target in enumerate(targets):
            print(f"\nTarget {i} content:")
            for k, v in target.items():
                print(f"  {k}: shape={v.shape}, device={v.device}, dtype={v.dtype}")
                print(f"  {k} values: {v}")

        # 如果有损失字典，打印损失信息
        if loss_dict is not None:
            print("\nLoss Information:")
            print(f"Loss dict type: {type(loss_dict)}")
            if isinstance(loss_dict, dict):
                print("Loss dict content:")
                for k, v in loss_dict.items():
                    print(f"  {k}: {v.item():.4f}")
            else:
                print(f"Single loss value: {loss_dict.item():.4f}")

        print(f"\nCurrent GPU Status:")
        self.log_gpu_status()
        print(f"{'='*50}\n")

    def _train_one_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        epoch_loss = 0
        batch_count = len(train_loader)

        if self.debug:
            print(f"\nStarting training epoch {epoch + 1}")
            print(f"Number of batches: {batch_count}")

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', total=batch_count)

        for batch_idx, (images, targets) in enumerate(pbar):
            try:
                if images is None or targets is None:
                    print(f"Skipping invalid batch {batch_idx}")
                    if self.debug:
                        print(f"Debug: images is None: {images is None}")
                        print(f"Debug: targets is None: {targets is None}")
                    continue

                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                if self.debug and batch_idx % self.log_frequency == 0:
                    self._print_debug_info("Training", batch_idx, images, targets)

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                
                if self.debug and batch_idx % self.log_frequency == 0:
                    print(f"\nForward Pass Output:")
                    print(f"Loss dict type: {type(loss_dict)}")
                    if isinstance(loss_dict, dict):
                        print(f"Loss dict keys: {loss_dict.keys()}")
                        print(f"Loss values: {[v.item() for v in loss_dict.values()]}")
                
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                elif isinstance(loss_dict, list):
                    losses = sum(loss_item for loss_item in loss_dict)
                else:
                    losses = loss_dict

                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()

                if (batch_idx + 1) % self.log_frequency == 0:
                    print(f"Epoch [{epoch + 1}/{self.num_epochs}], Batch [{batch_idx + 1}/{batch_count}], Loss: {losses.item():.4f}")
                    self.log_gpu_status()

                pbar.set_postfix({'loss': f'{losses.item():.4f}', 'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'})

            except Exception as e:
                print(f"\nError in training batch {batch_idx}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                if self.debug:
                    import traceback
                    print("\nFull traceback:")
                    print(traceback.format_exc())
                continue
        return epoch_loss / batch_count
        
    def _validate_one_epoch(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0
        batch_count = len(val_loader)

        if self.debug:
            print(f"\nStarting validation epoch {epoch + 1}")
            print(f"Number of batches: {batch_count}")

        pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}', total=batch_count)

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    if images is None or targets is None:
                        if self.debug:
                            print(f"Debug: Skipping invalid validation batch {batch_idx}")
                            print(f"Debug: images is None: {images is None}")
                            print(f"Debug: targets is None: {targets is None}")
                        continue

                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    if self.debug and batch_idx % self.log_frequency == 0:
                        self._print_debug_info("Validation", batch_idx, images, targets)

                    loss_dict = self.model(images, targets)
                    
                    if self.debug and batch_idx % self.log_frequency == 0:
                        print(f"\nValidation Forward Pass Output:")
                        print(f"Loss dict type: {type(loss_dict)}")
                        if isinstance(loss_dict, dict):
                            print(f"Loss dict keys: {loss_dict.keys()}")
                            print(f"Loss values: {[v.item() for v in loss_dict.values()]}")
                    
                    if isinstance(loss_dict, dict):
                        losses = sum(loss for loss in loss_dict.values())
                    else:
                        losses = loss_dict

                    val_loss += losses.item()

                    if (batch_idx + 1) % self.log_frequency == 0:
                        print(f"Validation Epoch [{epoch + 1}], Batch [{batch_idx + 1}/{batch_count}], Loss: {losses.item():.4f}")
                        self.log_gpu_status()

                    pbar.set_postfix({'val_loss': f'{losses.item():.4f}', 'avg_val_loss': f'{val_loss / (batch_idx + 1):.4f}'})

                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    if self.debug:
                        import traceback
                        print("\nFull traceback:")
                        print(traceback.format_exc())
                    continue

        return val_loss / batch_count

    def train(self, train_image_dir, val_image_dir, train_label_path, val_label_path,
              batch_size=2, num_epochs=10, learning_rate=0.005, momentum=0.9, 
              weight_decay=0.0005, save_dir="models/checkpoints/"):
        print("\nInitializing training...")
        print(f"Training with following parameters:")
        print(f"- Batch size: {batch_size}")
        print(f"- Number of epochs: {num_epochs}")
        print(f"- Learning rate: {learning_rate}")
        print(f"- Momentum: {momentum}")
        print(f"- Weight decay: {weight_decay}")
        print(f"- Device: {self.device}")
        print(f"- Debug mode: {self.debug}")

        self.num_epochs = num_epochs

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        print("\nLoading datasets...")
        train_dataset = CustomDataset(image_dir=train_image_dir, 
                                    label_path=train_label_path, 
                                    transform=transform)
        val_dataset = CustomDataset(image_dir=val_image_dir, 
                                  label_path=val_label_path, 
                                  transform=transform)
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        if self.debug:
            print("\nChecking first few samples from datasets:")
            for i in range(min(3, len(train_dataset))):
                img, target = train_dataset[i]
                print(f"\nTrain Sample {i}:")
                print(f"Image shape: {img.shape}")
                print(f"Target: {target}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4, 
                                collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4, 
                              collate_fn=collate_fn)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, 
                                  momentum=momentum, weight_decay=weight_decay)

        os.makedirs(save_dir, exist_ok=True)
        print(f"\nCheckpoints will be saved to: {save_dir}")

        history = {
            'train_loss': [],
            'val_loss': []
        }

        print("\nStarting training loop...")
        training_start_time = time.time()

        try:
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                
                train_loss = self._train_one_epoch(train_loader, optimizer, epoch)
                history['train_loss'].append(train_loss)
                
                val_loss = self._validate_one_epoch(val_loader, epoch)
                history['val_loss'].append(val_loss)

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

                if self.debug:
                    print(f"\nEpoch {epoch + 1} Summary:")
                    print(f"Training Loss: {train_loss:.4f}")
                    print(f"Validation Loss: {val_loss:.4f}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            if self.debug:
                import traceback
                print("\nFull traceback:")
                print(traceback.format_exc())
            raise
        finally:
            final_model_path = os.path.join(save_dir, "model_final.pth")
            torch.save(self.model.state_dict(), final_model_path)
            
            total_time = time.time() - training_start_time
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Final model saved at {final_model_path}")

        return history