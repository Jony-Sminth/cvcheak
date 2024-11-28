import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
import os
import json
from tqdm import tqdm
import cv2

@dataclass
class PreprocessedConfig:
    """预处理图像的配置类"""
    confidence_threshold: float = 0.5
    device: Optional[str] = None
    batch_size: int = 1
    num_classes: int = 2

class PreprocessedPredictor:
    def __init__(self, model_path: str, config: Optional[PreprocessedConfig] = None):
        self.config = config or PreprocessedConfig()
        self.device = (self.config.device if self.config.device 
                      else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.model = self._initialize_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _initialize_model(self, model_path: str) -> torch.nn.Module:
        try:
            from models.Faster_R_CNN import get_faster_rcnn_model
            model = get_faster_rcnn_model(num_classes=self.config.num_classes)
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint, strict=False)
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def predict_preprocessed(self, preprocessed_tensor: torch.Tensor, mask_tensor: Optional[torch.Tensor] = None) -> List[List[float]]:
        """
        使用图像和mask进行预测
        
        参数:
            preprocessed_tensor: 预处理后的RGB图像张量 [3, H, W]
            mask_tensor: mask张量 [1, H, W]，如果为None则使用零张量
        """
        preprocessed_tensor = preprocessed_tensor.to(self.device)
        
        if preprocessed_tensor.dim() == 4:
            preprocessed_tensor = preprocessed_tensor.squeeze(0)
            
        # 如果没有提供mask，创建一个全零mask
        if mask_tensor is None:
            mask_tensor = torch.zeros((1, preprocessed_tensor.shape[1], preprocessed_tensor.shape[2]), 
                                    device=self.device)
        else:
            mask_tensor = mask_tensor.to(self.device)
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor[0].unsqueeze(0)  # 确保形状是 [1, H, W]
            
        # 组合RGB和mask通道
        input_tensor = torch.cat([preprocessed_tensor, mask_tensor], dim=0)
        
        with torch.no_grad():
            try:
                predictions = self.model([input_tensor])[0]
            except Exception as e:
                print(f"模型预测时出错：{str(e)}")
                print(f"输入张量形状: {input_tensor.shape}")
                print(f"张量统计信息：\n最小值: {input_tensor.min()}\n最大值: {input_tensor.max()}\n均值: {input_tensor.mean()}")
                raise
        
        keep = predictions['scores'] > self.config.confidence_threshold
        boxes = predictions['boxes'][keep].cpu().numpy()
        
        regions = [[float(round(x1, 1)), float(round(y1, 1)), 
                   float(round(x2, 1)), float(round(y2, 1))] 
                  for x1, y1, x2, y2 in boxes]
        
        return regions

    def process_preprocessed_folder(self,
                                  preprocessed_dir: str,
                                  output_path: Optional[str] = None,
                                  batch_size: int = 1) -> List[Dict]:
        """处理预处理后的图像文件夹"""
        pt_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.pt') and not f.endswith('_mask.pt')]
        results = []
        
        for i in tqdm(range(0, len(pt_files), batch_size), desc="处理预处理图像"):
            batch_files = pt_files[i:i + batch_size]
            
            for pt_file in batch_files:
                try:
                    # 加载预处理后的RGB张量
                    tensor_path = os.path.join(preprocessed_dir, pt_file)
                    tensor = torch.load(tensor_path, map_location=self.device)
                    
                    # 尝试加载对应的mask
                    base_name = os.path.splitext(pt_file)[0]
                    mask_path = os.path.join(preprocessed_dir, f"{base_name}_mask.png")
                    
                    mask_tensor = None
                    if os.path.exists(mask_path):
                        # 读取并转换mask为张量
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            mask = mask.astype(np.float32) / 255.0  # 归一化到 [0,1]
                            mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # 添加通道维度
                    
                    # 预测
                    regions = self.predict_preprocessed(tensor, mask_tensor)
                    
                    result = {
                        "id": f"{base_name}.png",
                        "region": regions
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"处理文件 {pt_file} 时出错: {str(e)}")
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        
        return results