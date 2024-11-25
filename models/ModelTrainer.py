import math
import os
import traceback
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.Faster_R_CNN import get_faster_rcnn_model
from utils.dataset import CustomDataset
from utils.dataset import collate_fn
from tqdm import tqdm
import GPUtil  # GPU 监控模块
import time
import onnx
import torch.onnx
from torchvision.ops import box_iou
import traceback

class ModelTrainer:
    def __init__(self, model_name='faster_rcnn', num_classes=2, device=None, log_frequency=100, debug=True, train_dataset_limit=None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.log_frequency = log_frequency
        self.debug = debug
        self.train_dataset_limit = train_dataset_limit
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
                
                losses, loss_value = self._calculate_total_loss(loss_dict)
                losses.backward()
                optimizer.step()

                epoch_loss += loss_value

                if (batch_idx + 1) % self.log_frequency == 0:
                    print(f"Epoch [{epoch + 1}/{self.num_epochs}], "
                        f"Batch [{batch_idx + 1}/{batch_count}], Loss: {loss_value:.4f}")
                    self.log_gpu_status()

                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
                })

            except Exception as e:
                print(f"\nError in training batch {batch_idx}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                if self.debug:
                    print("\nFull traceback:")
                    print(traceback.format_exc())
                continue

        return epoch_loss / batch_count

    def _calculate_total_loss(self, loss_dict):
        """
        计算总损失，支持列表格式的损失
        """
        try:
            if isinstance(loss_dict, dict):
                # 处理字典格式的损失
                total_loss = sum(loss for loss in loss_dict.values() if isinstance(loss, torch.Tensor))
                return total_loss, total_loss.item()
            elif isinstance(loss_dict, list):
                # 处理列表格式的损失
                # 假设列表中第一个元素是损失字典
                if loss_dict and isinstance(loss_dict[0], dict):
                    return self._calculate_total_loss(loss_dict[0])
                else:
                    print(f"Warning: Unexpected list format in loss: {loss_dict}")
                    return torch.tensor(0.0, device=self.device), 0.0
            elif isinstance(loss_dict, torch.Tensor):
                # 处理张量格式的损失
                if loss_dict.numel() == 1:
                    return loss_dict, loss_dict.item()
                else:
                    total_loss = loss_dict.sum()
                    return total_loss, total_loss.item()
            elif isinstance(loss_dict, (int, float)):
                # 处理数值格式的损失
                return torch.tensor(loss_dict, device=self.device), float(loss_dict)
            else:
                print(f"Warning: Unexpected loss format: {type(loss_dict)}")
                return torch.tensor(0.0, device=self.device), 0.0
        except Exception as e:
            print(f"Error in loss calculation: {str(e)}")
            return torch.tensor(0.0, device=self.device), 0.0

    def _compute_correct_predictions(self, gt_boxes, gt_labels, pred):
        """
        计算预测框与真实框的IoU，并使用更严格的标准统计正确预测的数量
        
        Args:
            gt_boxes (Tensor): 真实边界框
            gt_labels (Tensor): 真实标签
            pred (dict): 包含 'boxes', 'labels' 和 'scores' 的预测结果字典
        
        Returns:
            int: 正确预测的数量
        """
        try:    
            # 提取预测结果
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred.get('scores', torch.ones_like(pred_labels))  # 如果没有scores，默认为1
            
            # 如果没有预测框或真实框，直接返回0
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                return 0
                
            # 确保张量在同一设备上
            pred_boxes = pred_boxes.to(self.device)
            gt_boxes = gt_boxes.to(self.device)
            pred_labels = pred_labels.to(self.device)
            gt_labels = gt_labels.to(self.device)
            pred_scores = pred_scores.to(self.device)
            
            # 设置阈值
            IOU_THRESHOLD = 0.5
            SCORE_THRESHOLD = 0.5
            
            # 只考虑置信度高于阈值的预测框
            high_conf_mask = pred_scores > SCORE_THRESHOLD
            pred_boxes = pred_boxes[high_conf_mask]
            pred_labels = pred_labels[high_conf_mask]
            pred_scores = pred_scores[high_conf_mask]
            
            if len(pred_boxes) == 0:
                return 0
            
            # 计算IoU矩阵
            ious = box_iou(pred_boxes, gt_boxes)  # shape: [num_pred, num_gt]
            
            # 初始化计数器
            correct_count = 0
            matched_gt_indices = set()
            
            # 按置信度降序处理预测框
            conf_sort = torch.argsort(pred_scores, descending=True)
            for pred_idx in conf_sort:
                # 找到最佳匹配的真实框
                iou_with_gt = ious[pred_idx]
                best_gt_iou, best_gt_idx = iou_with_gt.max(dim=0)
                best_gt_idx = best_gt_idx.item()
                
                # 如果这个真实框已经被匹配过，跳过
                if best_gt_idx in matched_gt_indices:
                    continue
                    
                # 检查是否满足条件：IoU大于阈值且类别匹配
                if (best_gt_iou > IOU_THRESHOLD and 
                    pred_labels[pred_idx] == gt_labels[best_gt_idx]):
                    correct_count += 1
                    matched_gt_indices.add(best_gt_idx)
            
            if self.debug and (correct_count > 0 or len(gt_boxes) > 0):
                print(f"\nAccuracy Debug Info:")
                print(f"GT boxes: {len(gt_boxes)}, Pred boxes: {len(pred_boxes)}")
                print(f"Correct predictions: {correct_count}")
                print(f"GT Labels: {gt_labels}")
                print(f"Pred Labels: {pred_labels}")
                print(f"Best IoUs: {ious.max(dim=0)[0]}")
            
            return correct_count
            
        except Exception as e:
            print(f"Error in compute_correct_predictions: {str(e)}")
            print(traceback.format_exc())
            return 0

    def _validate_one_epoch(self, val_loader, epoch):
        """
        验证一个epoch，包含更详细的评估指标
        """
        self.model.eval()
        val_loss = 0
        batch_count = len(val_loader)
        total_gt_objects = 0     # 真实框总数
        total_pred_objects = 0   # 预测框总数
        correct_predictions = 0   # 正确预测数
        
        if self.debug:
            print(f"\nStarting validation epoch {epoch + 1}")
            print(f"Number of batches: {batch_count}")

        pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}', total=batch_count)

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    if images is None or targets is None:
                        continue

                    # 将数据移到正确的设备上
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    try:
                        # 获取预测结果
                        predictions = self.model(images)
                        
                        # 计算损失
                        self.model.train()
                        loss_dict = self.model(images, targets)
                        self.model.eval()
                        
                        _, loss_value = self._calculate_total_loss(loss_dict)
                        if loss_value is not None and not math.isnan(loss_value):
                            val_loss += loss_value

                        # 统计预测结果
                        for target, pred in zip(targets, predictions):
                            total_gt_objects += len(target["boxes"])
                            total_pred_objects += len(pred["boxes"])
                            
                            correct = self._compute_correct_predictions(
                                target["boxes"], 
                                target["labels"], 
                                pred
                            )
                            correct_predictions += correct

                        # 计算各项指标
                        precision = correct_predictions / max(total_pred_objects, 1)
                        recall = correct_predictions / max(total_gt_objects, 1)
                        f1_score = 2 * (precision * recall) / max((precision + recall), 1e-6)
                        
                        # 更新进度条
                        pbar.set_postfix({
                            'val_loss': f'{loss_value:.4f}',
                            'avg_loss': f'{val_loss/(batch_idx+1):.4f}',
                            'precision': f'{precision:.3f}',
                            'recall': f'{recall:.3f}',
                            'f1': f'{f1_score:.3f}'
                        })

                    except Exception as e:
                        print(f"Error in batch processing: {str(e)}")
                        continue

                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}: {str(e)}")
                    if self.debug:
                        print(traceback.format_exc())
                    continue

            # 计算最终指标
            final_loss = val_loss / batch_count if batch_count > 0 else float('inf')
            final_precision = correct_predictions / max(total_pred_objects, 1)
            final_recall = correct_predictions / max(total_gt_objects, 1)
            final_f1 = 2 * (final_precision * final_recall) / max((final_precision + final_recall), 1e-6)

            if self.debug:
                print(f"\nValidation Summary:")
                print(f"Average Loss: {final_loss:.4f}")
                print(f"Total GT Objects: {total_gt_objects}")
                print(f"Total Predictions: {total_pred_objects}")
                print(f"Correct Predictions: {correct_predictions}")
                print(f"Precision: {final_precision:.4f}")
                print(f"Recall: {final_recall:.4f}")
                print(f"F1 Score: {final_f1:.4f}")

            return final_loss, final_f1  # 返回F1分数作为主要评估指标

    def _handle_irregular_loss_dict(self, loss_dict):
        losses = 0
        loss_value = 0
        for _, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                losses += v
                loss_value += v.item()
            else:
                print(f"Warning: Found non-Tensor value in loss_dict: {v}")
        return losses, loss_value

    def _convert_loss_dict_to_dict(self, loss_list):
        """
        将列表类型的 loss_dict 转换为字典类型。
        """
        loss_dict = {}
        for i, loss in enumerate(loss_list):
            loss_dict[f"loss_{i}"] = loss
        return loss_dict

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
        
        if self.train_dataset_limit is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, range(self.train_dataset_limit))
        
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
                print("\nFull traceback:")
                print(traceback.format_exc())
            raise
        finally:
            try:
                # 保存 PyTorch 模型（用于继续训练）
                self.model = self.model.cpu()
                final_model_path = os.path.join(save_dir, "model_final.pth")
                torch.save(self.model.state_dict(), final_model_path)
                print(f"\nPyTorch model saved at {final_model_path}")
                
                # 导出 ONNX 模型（用于部署）
                try:
                    # 清理 GPU 缓存（如果使用的是 GPU）
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    onnx_path = os.path.join(save_dir, "model_final.onnx")
                    print("\nStarting ONNX export...")
                    self.export_model(self.model, None, onnx_path)
                    print("ONNX export completed successfully!")
                    
                except Exception as export_error:
                    print(f"\nWarning: Failed to export ONNX model:")
                    print(f"Error type: {type(export_error).__name__}")
                    print(f"Error message: {str(export_error)}")
                    if self.debug:
                        print("\nFull export error traceback:")
                        print(traceback.format_exc())
                
                total_time = time.time() - training_start_time
                print(f"\nTraining completed in {total_time:.2f}s")

                return history
            
            except Exception as e:
                print(f"\nError during finalization:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                if self.debug:
                    print("\nFull traceback:")
                    print(traceback.format_exc())
                raise
    def _export_transform(self, images_batch):
        """
        转换输入格式，从批次张量转换为张量列表
        """
        if isinstance(images_batch, torch.Tensor):
            if images_batch.ndim == 4:  # [B, C, H, W]
                return [img for img in images_batch]
        return images_batch

    def export_model(self, model, sample_input, save_path):
        """
        将模型导出为 ONNX 格式，使用 JIT trace 来处理 Faster R-CNN
        
        Args:
            model: 要导出的模型
            sample_input: 样例输入
            save_path: 保存路径
        """
        try:
            # 确保模型处于评估模式
            model.eval()
            
            # 确保模型在CPU上
            model = model.to('cpu')
            
            class FasterRCNNWrapper(torch.nn.Module):
                def __init__(self, model):
                    super(FasterRCNNWrapper, self).__init__()
                    self.model = model
                    self.model.eval()

                def forward(self, images):
                    # 预处理输入
                    if isinstance(images, torch.Tensor):
                        if images.dim() == 4:  # [batch_size, channels, height, width]
                            images = [img for img in images]  # 转换为图像列表
                        else:  # [channels, height, width]
                            images = [images]  # 单张图像转换为列表
                    
                    # 使用 no_grad 来确保不计算梯度
                    with torch.no_grad():
                        # 获取模型预测结果
                        detections = self.model(images)
                        
                        if len(detections) == 0:
                            return (
                                torch.zeros((0, 4), dtype=torch.float32),
                                torch.zeros((0,), dtype=torch.int64),
                                torch.zeros((0,), dtype=torch.float32)
                            )
                        
                        # 获取第一个预测结果（因为我们每次只处理一张图片）
                        detection = detections[0]
                        
                        # 提取预测框、标签和分数
                        boxes = detection.get('boxes', torch.zeros((0, 4), dtype=torch.float32))
                        labels = detection.get('labels', torch.zeros((0,), dtype=torch.int64))
                        scores = detection.get('scores', torch.zeros((0,), dtype=torch.float32))
                        
                        # 确保所有输出都是张量
                        if not isinstance(boxes, torch.Tensor):
                            boxes = torch.tensor(boxes)
                        if not isinstance(labels, torch.Tensor):
                            labels = torch.tensor(labels)
                        if not isinstance(scores, torch.Tensor):
                            scores = torch.tensor(scores)
                        
                        return boxes, labels, scores

            # 创建模型包装器
            wrapped_model = FasterRCNNWrapper(model)
            
            # 创建示例输入
            dummy_input = torch.randn(3, 800, 800)
            
            # 使用 JIT trace 来捕获模型
            traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
            
            # 动态轴配置
            dynamic_axes = {
                'input': {
                    1: 'height',
                    2: 'width'
                },
                'boxes': {0: 'num_detections'},
                'labels': {0: 'num_detections'},
                'scores': {0: 'num_detections'}
            }
            
            # 导出为 ONNX
            torch.onnx.export(
                traced_model,               # 使用 traced 模型
                dummy_input,                # 示例输入
                save_path,                  # 保存路径
                export_params=True,         # 导出模型参数
                opset_version=11,           # ONNX 算子集版本
                do_constant_folding=True,   # 常量折叠优化
                input_names=['input'],      # 输入名称
                output_names=[              # 输出名称
                    'boxes',
                    'labels', 
                    'scores'
                ],
                dynamic_axes=dynamic_axes,  # 动态轴
                verbose=self.debug
            )
            
            if self.debug:
                print(f"\nModel exported to ONNX format: {save_path}")
                
            # 验证导出的模型
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            
            if self.debug:
                print("ONNX model checked successfully!")
                
        except Exception as e:
            print(f"\nError exporting model to ONNX:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            import traceback
            print(traceback.format_exc())