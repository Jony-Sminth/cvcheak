import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import cv2
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# 添加当前目录到路径以导入本地模块
# 导入数据预处理类以确保一致性
from data_preprocessing import DataPreprocessing

@dataclass
class PredictionConfig:
    """预测配置类，用于存储预测相关的参数"""
    confidence_threshold: float = 0.5  # 降低默认阈值以提高检测率
    device: Optional[str] = None
    batch_size: int = 1
    num_classes: int = 2
    normalize_mean: List[float] = None
    normalize_std: List[float] = None
    debug_mode: bool = False  # 是否启用调试模式

    def __post_init__(self):
        if self.normalize_mean is None:
            # 确保使用与模型一致的4通道标准化参数
            self.normalize_mean = [0.5, 0.5, 0.5, 0.0]  # 与模型定义中保持一致
        if self.normalize_std is None:
            self.normalize_std = [0.5, 0.5, 0.5, 1.0]   # 与模型定义中保持一致

class ImageProcessor:
    """图像预处理类，使用与训练一致的预处理逻辑"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        # 创建一个预处理器实例，使用与训练时相同的预处理
        self.preprocessor = DataPreprocessing("", "", target_size=(800, 800), debug=config.debug_mode)
        
    def preprocess_image(self, image: Union[str, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """使用与训练一致的预处理方法"""
        if isinstance(image, str):
            image_path = image
            if self.config.debug_mode:
                print(f"处理图像文件: {image_path}")
        else:
            # 如果是PIL图像，先保存为临时文件
            tmp_dir = "tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, "tmp_image.jpg")
            image.save(tmp_path)
            image_path = tmp_path
            if self.config.debug_mode:
                print(f"保存临时图像到: {tmp_path}")
        
        try:
            # 使用与训练相同的预处理方法
            image_tensor, original_size = self.preprocessor.preprocess_for_prediction(image_path)
            
            if self.config.debug_mode:
                print(f"图像预处理完成: 张量形状={image_tensor.shape}, 原始尺寸={original_size}")
                
            return image_tensor, original_size
        except Exception as e:
            print(f"图像预处理错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
        
    def load_image(self, image_path: str) -> Image.Image:
        """加载图像文件"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图像文件：{image_path}")
        return Image.open(image_path).convert('RGB')

class TamperingPredictor:
    """图像篡改检测预测器类"""
    
    def __init__(self, model_path: str, config: Optional[PredictionConfig] = None):
        """
        初始化预测器
        
        参数:
            model_path: 模型权重文件路径
            config: 预测配置参数
        """
        self.config = config or PredictionConfig()
        self.device = (self.config.device if self.config.device 
                      else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        if self.config.debug_mode:
            print(f"使用设备: {self.device}")
            print(f"置信度阈值: {self.config.confidence_threshold}")
            print(f"加载模型: {model_path}")
        
        # 初始化图像处理器
        self.processor = ImageProcessor(self.config)
        
        # 初始化模型
        self.model = self._initialize_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _initialize_model(self, model_path: str) -> torch.nn.Module:
        """初始化并加载模型"""
        try:
            from models.Faster_R_CNN import get_faster_rcnn_model
            
            if self.config.debug_mode:
                print("创建Faster R-CNN模型...")
                
            model = get_faster_rcnn_model(num_classes=self.config.num_classes)
            
            if self.config.debug_mode:
                print("加载模型权重...")
                
            # 修改加载方式，使用map_location并指定strict=False
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
                if self.config.debug_mode:
                    print("从state_dict加载权重")
            
            # 检查模型参数
            if self.config.debug_mode:
                model_keys = set(model.state_dict().keys())
                checkpoint_keys = set(checkpoint.keys())
                missing_keys = model_keys - checkpoint_keys
                extra_keys = checkpoint_keys - model_keys
                print(f"模型参数数量: {len(model_keys)}")
                print(f"检查点参数数量: {len(checkpoint_keys)}")
                print(f"缺失参数数量: {len(missing_keys)}")
                print(f"额外参数数量: {len(extra_keys)}")
            
            model.load_state_dict(checkpoint, strict=False)
            
            if self.config.debug_mode:
                print("模型加载成功")
                
            return model
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            raise RuntimeError(error_msg)

    def predict_image(self, image: Union[str, Image.Image]) -> List[List[float]]:
        """预测单个图像中的篡改区域"""
        if self.config.debug_mode:
            print(f"\n开始预测图像...")
            
        # 预处理图像
        image_tensor, original_size = self.processor.preprocess_image(image)
        original_height, original_width = original_size
        
        if self.config.debug_mode:
            print(f"预处理完成: 原始尺寸 = {original_width}x{original_height}")
        
        # 将张量放到正确的设备上
        image_tensor = image_tensor.to(self.device)
        
        # 模型预测
        with torch.no_grad():
            try:
                if self.config.debug_mode:
                    print("执行模型预测...")
                    
                predictions = self.model([image_tensor])[0]
                
                if self.config.debug_mode:
                    print(f"获得预测结果: {len(predictions['boxes'])} 个检测框")
                    if len(predictions['boxes']) > 0:
                        print(f"预测分数: {predictions['scores'].cpu().numpy()}")
                        print(f"预测框: {predictions['boxes'].cpu().numpy()}")
            except Exception as e:
                print(f"模型预测时出错：{str(e)}")
                print(f"张量统计信息：\n最小值: {image_tensor.min()}\n最大值: {image_tensor.max()}\n均值: {image_tensor.mean()}\n形状: {image_tensor.shape}")
                raise

        # 根据置信度阈值筛选预测框
        keep = predictions['scores'] > self.config.confidence_threshold
        # 应用非极大值抑制
        from torchvision.ops import nms
        boxes = predictions['boxes'][keep].cpu()
        scores = predictions['scores'][keep].cpu()
        
        # 如果有多个框
        if len(boxes) > 1:
            # 应用NMS，设置IoU阈值
            keep_indices = nms(boxes, scores, iou_threshold=0.5)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
        
        if self.config.debug_mode:
            print(f"应用置信度阈值 {self.config.confidence_threshold} 后保留 {len(boxes)} 个框")
        
        # 计算缩放比例，将坐标从800x800映射回原始尺寸
        width_ratio = original_width / 800.0
        height_ratio = original_height / 800.0
        
        if self.config.debug_mode:
            print(f"坐标缩放比例: 宽度 = {width_ratio}, 高度 = {height_ratio}")
        
        # 转换预测框格式并四舍五入到小数点后1位
        regions = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # 映射回原始尺寸
            orig_x1 = x1 * width_ratio
            orig_y1 = y1 * height_ratio
            orig_x2 = x2 * width_ratio
            orig_y2 = y2 * height_ratio
            
            # 四舍五入到小数点后1位
            region = [
                float(round(orig_x1, 1)), 
                float(round(orig_y1, 1)),
                float(round(orig_x2, 1)), 
                float(round(orig_y2, 1))
            ]
            regions.append(region)
            
            if self.config.debug_mode and len(scores) > i:
                print(f"检测框 {i+1}: 坐标={region}, 分数={scores[i]:.4f}")
        
        return regions
        
    def process_and_predict(self, 
                          input_path: str, 
                          output_path: Optional[str] = None,
                          save_visualization: bool = False) -> List[Dict]:
        """
        处理输入（可以是单个文件或目录）并生成预测结果
        
        参数:
            input_path: 输入图像或目录的路径
            output_path: 输出JSON文件的保存路径
            save_visualization: 是否保存可视化结果
            
        返回:
            预测结果列表
        """
        results = []
        
        if os.path.isfile(input_path):
            # 处理单个文件
            if self.config.debug_mode:
                print(f"处理单个文件: {input_path}")
                
            regions = self.predict_image(input_path)
            result = {
                "id": os.path.basename(input_path),
                "region": regions
            }
            results.append(result)
            
            if save_visualization:
                viz_path = os.path.splitext(output_path)[0] + '_viz.jpg' if output_path else 'result_viz.jpg'
                self._save_visualization(input_path, regions, viz_path)
                
                if self.config.debug_mode:
                    print(f"保存可视化结果到: {viz_path}")
        
        elif os.path.isdir(input_path):
            # 处理目录
            if self.config.debug_mode:
                print(f"处理目录: {input_path}")
                
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [f for f in os.listdir(input_path) 
                         if any(f.lower().endswith(fmt) for fmt in supported_formats)]
            
            if self.config.debug_mode:
                print(f"找到 {len(image_files)} 个图像文件")
            
            for image_file in tqdm(image_files, desc="处理图像"):
                image_path = os.path.join(input_path, image_file)
                try:
                    if self.config.debug_mode:
                        print(f"\n正在处理图像: {image_file}")
                        
                    regions = self.predict_image(image_path)
                    result = {
                        "id": image_file,
                        "region": regions
                    }
                    results.append(result)
                    
                    if save_visualization:
                        viz_dir = os.path.join(os.path.dirname(output_path), 'visualizations') if output_path else 'visualizations'
                        os.makedirs(viz_dir, exist_ok=True)
                        viz_path = os.path.join(viz_dir, f'{os.path.splitext(image_file)[0]}_viz.jpg')
                        self._save_visualization(image_path, regions, viz_path)
                        
                        if self.config.debug_mode:
                            print(f"保存可视化结果到: {viz_path}")
                
                except Exception as e:
                    print(f"处理图像 {image_file} 时出错: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
            if self.config.debug_mode:
                print(f"保存预测结果到: {output_path}")
        
        return results
        
    def _save_visualization(self, image_path: str, regions: List[List[float]], output_path: str):
        """保存预测结果的可视化图像"""
        try:
            # 读取原始图像
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            if self.config.debug_mode:
                print(f"可视化: 图像尺寸 {w}x{h}, 绘制 {len(regions)} 个区域")
            
            # 在图像上绘制检测框
            for region in regions:
                x1, y1, x2, y2 = map(int, region)
                # 确保坐标在图像范围内
                x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
                y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
                
                if x1 < x2 and y1 < y2:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 添加文字标签
                    cv2.putText(image, "Tampering", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 添加检测到的区域数量信息
            text = f"Detected: {len(regions)}"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 保存图像
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            
        except Exception as e:
            print(f"保存可视化结果时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())

def demo_usage():
    config = PredictionConfig(
        confidence_threshold=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=1,
        # 使用与模型定义一致的4通道标准化参数
        normalize_mean=[0.5, 0.5, 0.5, 0.0],
        normalize_std=[0.5, 0.5, 0.5, 1.0]
    )

    # 创建预测器
    predictor = TamperingPredictor(
        model_path="output/faster_rcnn/model_epoch_1.pth",
        config=config
    )

    # 处理整个验证集目录
    results = predictor.process_and_predict(
        input_path="data/train/images",
        output_path="output/results.json",
        save_visualization=True  # 启用可视化结果保存
    )

    print("预测完成，结果已保存")

if __name__ == '__main__':
    demo_usage()