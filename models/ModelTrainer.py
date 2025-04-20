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
import matplotlib.pyplot as plt
import numpy as np
import logging
import datetime

class ModelTrainer:
    def __init__(self, model_name='faster_rcnn', num_classes=2, device=None, log_frequency=100, debug=True, train_dataset_limit=None, log_to_file=False):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.log_frequency = log_frequency
        self.debug = debug
        self.train_dataset_limit = train_dataset_limit
        self.log_to_file = log_to_file
        self.logger = None
        self.model = self._load_model()
        self.model = self.model.to(self.device)

        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'precision': [], 'recall': [], 'mAP': [], 'mAR': []}
    
        
        if self.debug:
            print(f"\nDebug Mode Enabled")
            print(f"Model Architecture:")
            print(self.model)
            print(f"Model Device: {self.device}")

    def _setup_logger(self, save_dir):
        """
        设置日志记录器
        
        Args:
            save_dir (str): 保存日志的目录
        """
        if not self.log_to_file:
            return
            
        # 创建日志目录
        log_dir = os.path.join(save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件名，包含时间戳
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        
        # 配置日志记录器
        self.logger = logging.getLogger('model_trainer')
        self.logger.setLevel(logging.INFO)
        
        # 添加文件处理器，指定 UTF-8 编码
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 清除之前的handlers，避免重复日志
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"===== 训练开始于 {timestamp} =====")
        self.logger.info(f"模型名称: {self.model_name}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"调试模式: {self.debug}")
        
        return log_file

    def log_message(self, message, level='info'):
        """
        记录消息到日志文件
        
        Args:
            message (str): 要记录的消息
            level (str): 日志级别 ('info', 'warning', 'error', 'debug')
        """
        if not self.log_to_file or self.logger is None:
            return
            
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'debug':
            self.logger.debug(message)

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
        gpu_info = []
        for gpu in gpus:
            info = f"GPU ID: {gpu.id}, Name: {gpu.name}, Load: {gpu.load * 100:.2f}%, Free Memory: {gpu.memoryFree} MB, Total Memory: {gpu.memoryTotal} MB"
            print(info)
            gpu_info.append(info)
            
        if self.log_to_file and self.logger:
            self.logger.info("GPU状态: " + " | ".join(gpu_info))

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
        
    def _visualize_batch(self, images, targets, predictions=None, phase="training", batch_idx=0):
        """
        可视化批次中的图像、真实框和预测框
        """
        if not self.debug:
            return
            
        debug_dir = os.path.join("debug_output/new_debug", phase)
        os.makedirs(debug_dir, exist_ok=True)
        
        # 只可视化批次中的前2张图像
        num_images = min(2, len(images))
        
        for i in range(num_images):
            try:
                # 获取图像张量（4通道）并转换为RGB进行可视化
                img_tensor = images[i].cpu().clone()
                
                # 分离RGB通道和掩码通道
                rgb_tensor = img_tensor[:3]
                mask_tensor = img_tensor[3] if img_tensor.shape[0] > 3 else None
                
                # 反标准化RGB通道
                for c in range(3):
                    rgb_tensor[c] = rgb_tensor[c] * 0.5 + 0.5
                
                # 转换为numpy进行可视化
                rgb_np = rgb_tensor.permute(1, 2, 0).numpy()
                
                # 创建图形
                if predictions is not None and mask_tensor is not None:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                elif mask_tensor is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    axes = [ax]
                
                # 显示RGB图像
                axes[0].imshow(rgb_np)
                axes[0].set_title(f"RGB Image")
                axes[0].axis('off')
                
                # 显示掩码（如果有）
                if mask_tensor is not None:
                    mask_np = mask_tensor.numpy()
                    axes[1].imshow(mask_np, cmap='gray')
                    axes[1].set_title(f"Mask Channel")
                    axes[1].axis('off')
                
                # 显示RGB图像和目标/预测框
                if predictions is not None:
                    if mask_tensor is not None:
                        ax_idx = 2
                    else:
                        ax_idx = 1
                    axes[ax_idx].imshow(rgb_np)
                    axes[ax_idx].set_title(f"Detections")
                    axes[ax_idx].axis('off')
                    
                    # 绘制真实框（蓝色）
                    if 'boxes' in targets[i] and len(targets[i]['boxes']) > 0:
                        boxes = targets[i]['boxes'].cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            width, height = x2-x1, y2-y1
                            rect = plt.Rectangle((x1, y1), width, height, 
                                              fill=False, edgecolor='blue', linewidth=2)
                            axes[ax_idx].add_patch(rect)
                            axes[ax_idx].text(x1, y1-5, 'GT', color='blue', fontsize=8,
                                           bbox=dict(facecolor='white', alpha=0.7))
                    
                    # 绘制预测框（红色）
                    if 'boxes' in predictions[i] and len(predictions[i]['boxes']) > 0:
                        pred_boxes = predictions[i]['boxes'].cpu().numpy()
                        pred_scores = predictions[i]['scores'].cpu().numpy()
                        
                        for box, score in zip(pred_boxes, pred_scores):
                            if score > 0.3:  # 只显示置信度较高的预测
                                x1, y1, x2, y2 = box
                                width, height = x2-x1, y2-y1
                                rect = plt.Rectangle((x1, y1), width, height, 
                                                  fill=False, edgecolor='red', linewidth=2)
                                axes[ax_idx].add_patch(rect)
                                axes[ax_idx].text(x1, y1-5, f'{score:.2f}', 
                                               color='red', fontsize=8, 
                                               bbox=dict(facecolor='white', alpha=0.7))
                
                # 保存图像
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, f"batch_{batch_idx}_img_{i}.png"))
                plt.close(fig)
                
            except Exception as e:
                print(f"Error visualizing image {i}: {str(e)}")
                continue

    def _train_one_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        epoch_loss = 0
        batch_count = len(train_loader)

        if self.debug:
            print(f"\nStarting training epoch {epoch + 1}")
            print(f"Number of batches: {batch_count}")
            
        self.log_message(f"\n===== 开始训练 Epoch {epoch + 1}/{self.num_epochs} =====")
        self.log_message(f"批次数量: {batch_count}")

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
                    # 添加可视化函数调用
                    self._visualize_batch(images, targets, phase="training", batch_idx=batch_idx)

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
                    log_msg = f"Epoch [{epoch + 1}/{self.num_epochs}], Batch [{batch_idx + 1}/{batch_count}], Loss: {loss_value:.4f}"
                    print(log_msg)
                    self.log_message(log_msg)
                    self.log_gpu_status()

                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
                })

            except Exception as e:
                error_msg = f"\nError in training batch {batch_idx}: {type(e).__name__} - {str(e)}"
                print(error_msg)
                self.log_message(error_msg, level='error')
                if self.debug:
                    print("\nFull traceback:")
                    print(traceback.format_exc())
                    self.log_message(traceback.format_exc(), level='error')
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
            
            # 设置阈值 - 使用更严格的IOU阈值
            IOU_THRESHOLD = 0.5  # 提高IoU阈值，使评估更严格
            SCORE_THRESHOLD = 0.5  # 分数阈值保持不变
            
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
                print(f"Using IOU threshold: {IOU_THRESHOLD}")
            
            return correct_count
            
        except Exception as e:
            print(f"Error in compute_correct_predictions: {str(e)}")
            print(traceback.format_exc())
            return 0

    def _compute_ap_and_ar(self, all_predictions, all_targets, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
        """
        计算平均精度(AP)和平均召回率(AR)
        
        Args:
            all_predictions: 所有预测结果的列表
            all_targets: 所有真实目标的列表
            iou_thresholds: 用于计算AP和AR的IOU阈值列表
        
        Returns:
            ap: 平均精度
            ar: 平均召回率
        """
        aps = []
        ars = []
        
        # 对每个IOU阈值计算AP和AR
        for iou_threshold in iou_thresholds:
            # 收集所有预测的置信度和是否为TP
            all_scores = []
            all_tp = []
            
            total_gt = 0
            
            # 处理每张图像
            for predictions, targets in zip(all_predictions, all_targets):
                # 获取预测框、分数和标签
                pred_boxes = predictions['boxes'].cpu()
                pred_scores = predictions['scores'].cpu()
                pred_labels = predictions['labels'].cpu()
                
                # 获取真实框和标签
                gt_boxes = targets['boxes'].cpu()
                gt_labels = targets['labels'].cpu()
                
                total_gt += len(gt_boxes)
                
                # 如果没有预测或者没有真实目标，跳过
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue
                    
                # 按置信度降序排序预测
                sorted_indices = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_indices]
                pred_scores = pred_scores[sorted_indices]
                pred_labels = pred_labels[sorted_indices]
                
                # 记录每个真实框是否被匹配
                gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
                
                # 对每个预测框计算IOU并确定是否为TP
                for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                    # 只考虑与预测相同类别的真实框
                    same_class_mask = gt_labels == label
                    if not same_class_mask.any():
                        all_scores.append(score.item())
                        all_tp.append(0)  # 没有匹配的真实框，为FP
                        continue
                    
                    # 计算与相同类别的真实框的IOU
                    class_gt_boxes = gt_boxes[same_class_mask]
                    class_gt_indices = torch.where(same_class_mask)[0]
                    
                    if len(class_gt_boxes) > 0:
                        # 计算当前预测框与所有相同类别的真实框的IOU
                        ious = box_iou(box.unsqueeze(0), class_gt_boxes)[0]
                        max_iou, max_idx = ious.max(dim=0)
                        max_gt_idx = class_gt_indices[max_idx]
                        
                        # 如果IOU大于阈值且该真实框未被匹配
                        if max_iou >= iou_threshold and not gt_matched[max_gt_idx]:
                            gt_matched[max_gt_idx] = True
                            all_scores.append(score.item())
                            all_tp.append(1)  # 是TP
                        else:
                            all_scores.append(score.item())
                            all_tp.append(0)  # 是FP
                    else:
                        all_scores.append(score.item())
                        all_tp.append(0)  # 是FP
            
            # 如果没有预测或者没有真实目标，AP和AR都设为0
            if len(all_scores) == 0 or total_gt == 0:
                aps.append(0)
                ars.append(0)
                continue
            
            # 将列表转换为numpy数组
            scores = np.array(all_scores)
            tp = np.array(all_tp)
            
            # 按置信度降序排序
            sort_indices = np.argsort(scores)[::-1]
            tp = tp[sort_indices]
            
            # 计算累积TP和FP
            cumulative_tp = np.cumsum(tp)
            cumulative_fp = np.cumsum(1 - tp)
            
            # 计算精度和召回率
            precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-10)
            recall = cumulative_tp / (total_gt + 1e-10)
            
            # 计算AP (使用11点插值法)
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11
            
            # 计算AR (最大召回率)
            ar = np.max(recall) if len(recall) > 0 else 0
            
            aps.append(ap)
            ars.append(ar)
        
        # 计算平均值
        mAP = np.mean(aps)
        mAR = np.mean(ars)
        
        return mAP, mAR
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
        
        # 收集所有预测和真实目标用于计算AP和AR
        all_predictions = []
        all_targets = []
        
        if self.debug:
            print(f"\nStarting validation epoch {epoch + 1}")
            print(f"Number of batches: {batch_count}")
            
        self.log_message(f"\n===== 开始验证 Epoch {epoch + 1}/{self.num_epochs} =====")
        self.log_message(f"批次数量: {batch_count}")

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
                        
                        # 收集预测和真实目标用于AP和AR计算
                        all_predictions.extend(predictions)
                        all_targets.extend([{k: v.clone() for k, v in t.items()} for t in targets])
                        
                        # 添加可视化函数调用
                        if self.debug and batch_idx % self.log_frequency == 0:
                            self._visualize_batch(images, targets, predictions, phase="validation", batch_idx=batch_idx)
                        
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
                    error_msg = f"\nError in validation batch {batch_idx}: {str(e)}"
                    print(error_msg)
                    self.log_message(error_msg, level='error')
                    if self.debug:
                        print(traceback.format_exc())
                        self.log_message(traceback.format_exc(), level='error')
                    continue

            # 计算最终指标
            final_loss = val_loss / batch_count if batch_count > 0 else float('inf')
            final_precision = correct_predictions / max(total_pred_objects, 1)
            final_recall = correct_predictions / max(total_gt_objects, 1)
            final_f1 = 2 * (final_precision * final_recall) / max((final_precision + final_recall), 1e-6)

            # 计算AP和AR
            try:
                mAP, mAR = self._compute_ap_and_ar(all_predictions, all_targets)
                
                # 添加到历史记录中用于绘图
                if 'mAP' not in self.history:
                    self.history['mAP'] = []
                    self.history['mAR'] = []
                self.history['mAP'].append(mAP)
                self.history['mAR'].append(mAR)
                
            except Exception as e:
                error_msg = f"计算AP和AR时出错: {str(e)}"
                print(error_msg)
                self.log_message(error_msg, level='error')
                if self.debug:
                    print(traceback.format_exc())
                mAP, mAR = 0.0, 0.0

            # 记录验证结果
            validation_summary = f"\nValidation Summary - Epoch {epoch + 1}:"
            validation_summary += f"\nAverage Loss: {final_loss:.4f}"
            validation_summary += f"\nTotal GT Objects: {total_gt_objects}"
            validation_summary += f"\nTotal Predictions: {total_pred_objects}"
            validation_summary += f"\nCorrect Predictions: {correct_predictions}"
            validation_summary += f"\nPrecision: {final_precision:.4f}"
            validation_summary += f"\nRecall: {final_recall:.4f}"
            validation_summary += f"\nF1 Score: {final_f1:.4f}"
            validation_summary += f"\nMean Average Precision (mAP): {mAP:.4f}"
            validation_summary += f"\nMean Average Recall (mAR): {mAR:.4f}"
            
            if self.debug:
                print(validation_summary)
                
            self.log_message(validation_summary)

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
            batch_size=2, num_epochs=10, learning_rate=0.001, momentum=0.9, 
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
        print(f"- Log to file: {self.log_to_file}")
        
        # 设置日志记录器
        log_file = self._setup_logger(save_dir)
        if log_file:
            print(f"Logs will be saved to: {log_file}")
        
        # 记录训练参数到日志
        self.log_message("\n===== 训练参数 =====")
        self.log_message(f"批大小: {batch_size}")
        self.log_message(f"训练轮数: {num_epochs}")
        self.log_message(f"学习率: {learning_rate}")
        self.log_message(f"动量: {momentum}")
        self.log_message(f"权重衰减: {weight_decay}")

        self.num_epochs = num_epochs

        # 不使用transform，因为数据已经预处理过
        transform = None

        print("\nLoading datasets...")
        self.log_message("\n===== 加载数据集 =====")
        
        train_dataset = CustomDataset(image_dir=train_image_dir, 
                                    label_path=train_label_path, 
                                    transform=transform,
                                    debug=self.debug)
        val_dataset = CustomDataset(image_dir=val_image_dir, 
                                label_path=val_label_path, 
                                transform=transform,
                                debug=self.debug)
        
        if self.train_dataset_limit is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, range(self.train_dataset_limit))
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        self.log_message(f"训练数据集大小: {len(train_dataset)}")
        self.log_message(f"验证数据集大小: {len(val_dataset)}")

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
                                  
        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )

        os.makedirs(save_dir, exist_ok=True)
        # 创建调试输出目录
        debug_dir = os.path.join("debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        
        print(f"\nCheckpoints will be saved to: {save_dir}")
        print(f"Debug visualizations will be saved to: {debug_dir}")
        
        self.log_message(f"模型检查点保存路径: {save_dir}")
        self.log_message(f"调试可视化保存路径: {debug_dir}")

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'precision': [],
            'recall': []
        }

        print("\nStarting training loop...")
        self.log_message("\n===== 开始训练循环 =====")
        training_start_time = time.time()

        try:
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                
                train_loss = self._train_one_epoch(train_loader, optimizer, epoch)
                history['train_loss'].append(train_loss)
                
                val_loss, val_f1 = self._validate_one_epoch(val_loader, epoch)
                history['val_loss'].append(val_loss)
                history['val_f1'].append(val_f1)

                # 根据F1分数调整学习率
                scheduler.step(val_f1)
                
                checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path}")
                self.log_message(f"模型检查点已保存至: {checkpoint_path}")

                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
                
                # 记录每个epoch的结果
                epoch_summary = f"\nEpoch {epoch + 1} 总结:"
                epoch_summary += f"\n- 训练损失: {train_loss:.4f}"
                epoch_summary += f"\n- 验证损失: {val_loss:.4f}"
                epoch_summary += f"\n- 验证F1分数: {val_f1:.4f}"
                epoch_summary += f"\n- 耗时: {epoch_time:.2f}s"
                self.log_message(epoch_summary)

                if self.debug:
                    print(f"\nEpoch {epoch + 1} Summary:")
                    print(f"Training Loss: {train_loss:.4f}")
                    print(f"Validation Loss: {val_loss:.4f}")
                    print(f"Validation F1: {val_f1:.4f}")
                    
                # 可视化训练进程
                # 可视化训练进程
                if self.debug and epoch > 0:
                    try:
                        plt.figure(figsize=(15, 10))
                        
                        # 第一行图表
                        plt.subplot(2, 2, 1)
                        plt.plot(history['train_loss'], label='Train Loss')
                        plt.plot(history['val_loss'], label='Val Loss')
                        plt.legend()
                        plt.title('Loss Curves')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        
                        plt.subplot(2, 2, 2)
                        plt.plot(history['val_f1'], label='F1 Score')
                        plt.legend()
                        plt.title('F1 Score')
                        plt.xlabel('Epoch')
                        plt.ylabel('F1')
                        
                        # 第二行图表 - AP和AR
                        if 'mAP' in history and len(history['mAP']) > 0:
                            plt.subplot(2, 2, 3)
                            plt.plot(history['mAP'], label='mAP', marker='o')
                            plt.plot(history['mAR'], label='mAR', marker='x')
                            plt.legend()
                            plt.title('mAP and mAR')
                            plt.xlabel('Epoch')
                            plt.ylabel('Score')
                            
                            # Precision和Recall
                            plt.subplot(2, 2, 4)
                            if 'precision' in history and 'recall' in history:
                                plt.plot(history['precision'], label='Precision', marker='o')
                                plt.plot(history['recall'], label='Recall', marker='x')
                                plt.legend()
                                plt.title('Precision and Recall')
                                plt.xlabel('Epoch')
                                plt.ylabel('Score')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(debug_dir, f"training_progress_epoch_{epoch+1}.png"))
                        plt.close()
                    except Exception as e:
                        print(f"Error plotting training progress: {str(e)}")
                        self.log_message(f"绘制训练进度图时出错: {str(e)}", level='error')
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.log_message("\n训练被用户中断", level='warning')
        except Exception as e:
            error_msg = f"\nError during training: {type(e).__name__} - {str(e)}"
            print(error_msg)
            self.log_message(error_msg, level='error')
            if self.debug:
                print("\nFull traceback:")
                print(traceback.format_exc())
                self.log_message(traceback.format_exc(), level='error')
            raise
        finally:
            # 记录总训练时间
            total_training_time = time.time() - training_start_time
            print(f"\nTotal training time: {total_training_time:.2f}s")
            self.log_message(f"\n总训练时间: {total_training_time:.2f}s")
            
            try:
                # 保存 PyTorch 模型（用于继续训练）
                self.model = self.model.cpu()
                final_model_path = os.path.join(save_dir, "model_final.pth")
                torch.save(self.model.state_dict(), final_model_path)
                print(f"\nPyTorch model saved at {final_model_path}")
                self.log_message(f"最终PyTorch模型已保存至: {final_model_path}")
                
                # 保存训练历史记录
                history_path = os.path.join(save_dir, "training_history.npy")
                np.save(history_path, history)
                print(f"Training history saved at {history_path}")
                self.log_message(f"训练历史记录已保存至: {history_path}")

            except Exception as e:
                error_msg = f"\nError saving final model: {type(e).__name__} - {str(e)}"
                print(error_msg)
                self.log_message(error_msg, level='error')
                if self.debug:
                    print("\nFull traceback:")
                    print(traceback.format_exc())
                    self.log_message(traceback.format_exc(), level='error')
                
            # 如果需要导出ONNX模型，可以取消下面的注释
            # try:
            #     # 清理 GPU 缓存（如果使用的是 GPU）
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()
            #     
            #     onnx_path = os.path.join(save_dir, "model_final.onnx")
            #     dummy_input = torch.randn(1, 3, 800, 800) # 根据实际输入调整
            #     torch.onnx.export(self.model, dummy_input, onnx_path, 
            #                        export_params=True, opset_version=11, 
            #                        do_constant_folding=True, 
            #                        input_names=['input'], output_names=['output'])
            #     print(f"\nONNX model saved at {onnx_path}")
            #     self.log_message(f"ONNX模型已保存至: {onnx_path}")
            # except Exception as e:
            #     print(f"Error exporting ONNX model: {str(e)}")
            #     self.log_message(f"导出ONNX模型时出错: {str(e)}", level='error')