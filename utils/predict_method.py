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

@dataclass
class PredictionConfig:
    """预测配置类，用于存储预测相关的参数"""
    confidence_threshold: float = 0.5
    device: Optional[str] = None
    batch_size: int = 1
    num_classes: int = 2
    normalize_mean: List[float] = None
    normalize_std: List[float] = None

    def __post_init__(self):
        if self.normalize_mean is None:
            # 更新为4通道的标准化参数
            self.normalize_mean = [0.485, 0.456, 0.406, 0.0]  # 第4通道为mask通道
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225, 1.0]   # 第4通道为mask通道

class ImageProcessor:
    """图像预处理类，负责数据的加载和预处理"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.transform = self._get_transform()

    def _get_transform(self) -> transforms.Compose:
        """创建图像转换pipeline"""
        return transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            # 注意：标准化将在preprocess_image中手动处理
        ])

    def preprocess_image(self, image: Union[str, Image.Image]) -> Tuple[torch.Tensor, np.ndarray]:
        if isinstance(image, str):
            image = self.load_image(image)
        
        # 将图像转换为 numpy 数组
        image_array = np.array(image)
        
        # 检查并处理图像通道数
        if len(image_array.shape) == 2:  # 灰度图像
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[-1] == 4:  # RGBA 图像
            image_array = image_array[:, :, :3]  # 只保留 RGB 通道
        elif image_array.shape[-1] != 3:  # 其他情况
            raise ValueError(f"不支持的图像格式：通道数为 {image_array.shape[-1]}")
        
        # 转回 PIL 图像
        image = Image.fromarray(image_array)
        
        # 获取图像尺寸
        width, height = image.size
        
        # 应用转换（不包括标准化）
        image_tensor = self.transform(image)
        
        # 创建第四个通道（mask通道）
        mask_channel = torch.zeros_like(image_tensor[0]).unsqueeze(0)
        image_tensor = torch.cat([image_tensor, mask_channel], dim=0)
        
        # 手动执行标准化
        for c in range(4):
            image_tensor[c] = (image_tensor[c] - self.config.normalize_mean[c]) / self.config.normalize_std[c]
        
        return image_tensor, (height, width)

    def load_image(self, image_path: str) -> Image.Image:
        """
        加载图像文件
        
        参数:
            image_path: 图像文件路径
            
        返回:
            PIL.Image对象
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图像文件：{image_path}")
        return Image.open(image_path).convert('RGB')

    # def preprocess_image(self, image: Union[str, Image.Image]) -> Tuple[torch.Tensor, np.ndarray]:
    #     if isinstance(image, str):
    #         image = self.load_image(image)
        
    #     # 将图像转换为 numpy 数组
    #     image_array = np.array(image)
        
    #     # 检查并处理图像通道数
    #     if len(image_array.shape) == 2:  # 灰度图像
    #         image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    #     elif image_array.shape[-1] == 4:  # RGBA 图像
    #         image_array = image_array[:, :, :3]  # 只保留 RGB 通道
    #     elif image_array.shape[-1] != 3:  # 其他情况
    #         raise ValueError(f"不支持的图像格式：通道数为 {image_array.shape[-1]}")
        
    #     # 转回 PIL 图像
    #     image = Image.fromarray(image_array)
        
    #     # 获取图像尺寸
    #     width, height = image.size
        
    #     # 应用变换
    #     image_tensor = self.transform(image)
        
    #     return image_tensor, (height, width)
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
            model = get_faster_rcnn_model(num_classes=self.config.num_classes)
            # 修改加载方式，使用map_location并指定strict=False
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint, strict=False)
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def predict_image(self, image: Union[str, Image.Image]) -> List[List[float]]:
        # 预处理图像
        image_tensor, original_size = self.processor.preprocess_image(image)
        
        # 重要：确保使用3通道的标准化参数
        if len(self.config.normalize_mean) > 3:
            self.config.normalize_mean = self.config.normalize_mean[:3]
        if len(self.config.normalize_std) > 3:
            self.config.normalize_std = self.config.normalize_std[:3]
        
        # 确保图像张量是3D的 [C, H, W]
        if image_tensor.dim() == 4:  # 如果是4D，去掉批次维度
            image_tensor = image_tensor.squeeze(0)
        
        # 将张量放到正确的设备上
        image_tensor = image_tensor.to(self.device)
        
        # 模型预测 - 注意这里传入列表
        with torch.no_grad():
            try:
                predictions = self.model([image_tensor])[0]
            except Exception as e:
                print(f"模型预测时出错：{str(e)}")
                print(f"张量统计信息：\n最小值: {image_tensor.min()}\n最大值: {image_tensor.max()}\n均值: {image_tensor.mean()}")
                raise
    
        # 根据置信度阈值筛选预测框
        keep = predictions['scores'] > self.config.confidence_threshold
        boxes = predictions['boxes'][keep].cpu().numpy()
        
        # 转换预测框格式并四舍五入到小数点后1位
        regions = [[float(round(x1, 1)), float(round(y1, 1)), 
                float(round(x2, 1)), float(round(y2, 1))] 
                for x1, y1, x2, y2 in boxes]
        
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
            regions = self.predict_image(input_path)
            result = {
                "id": os.path.basename(input_path),
                "region": regions
            }
            results.append(result)
            
            if save_visualization:
                self._save_visualization(input_path, regions, 
                                      os.path.splitext(output_path)[0] + '_viz.jpg')
        
        elif os.path.isdir(input_path):
            # 处理目录
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [f for f in os.listdir(input_path) 
                         if any(f.lower().endswith(fmt) for fmt in supported_formats)]
            
            for image_file in tqdm(image_files, desc="处理图像"):
                image_path = os.path.join(input_path, image_file)
                try:
                    regions = self.predict_image(image_path)
                    result = {
                        "id": image_file,
                        "region": regions
                    }
                    results.append(result)
                    
                    if save_visualization:
                        viz_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
                        os.makedirs(viz_dir, exist_ok=True)
                        viz_path = os.path.join(viz_dir, f'{os.path.splitext(image_file)[0]}_viz.jpg')
                        self._save_visualization(image_path, regions, viz_path)
                
                except Exception as e:
                    print(f"处理图像 {image_file} 时出错: {str(e)}")
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        
        return results

    def _save_visualization(self, image_path: str, regions: List[List[float]], output_path: str):
        """保存预测结果的可视化图像"""
        image = cv2.imread(image_path)
        for region in regions:
            x1, y1, x2, y2 = map(int, region)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(output_path, image)

def demo_usage():
    """演示如何使用改进后的预测器"""
    # 创建配置
    config = PredictionConfig(
        confidence_threshold=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=1,
        normalize_mean=[0.5, 0.5, 0.5],
        normalize_std=[0.5, 0.5, 0.5]
    )
    
    # 初始化预测器
    predictor = TamperingPredictor(
        model_path="F:/learning/CVwork/output/model_final.pth",
        config=config
    )
    
    # 示例：处理整个测试目录
    results = predictor.process_and_predict(
        input_path="data/val/images/val_2199.png",
        output_path="output/results.json",
        save_visualization=True  # 启用可视化结果保存
    )
    
    print("预测完成，结果已保存")