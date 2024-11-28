# 创建配置
import torch
from utils.predict_method import PredictionConfig, TamperingPredictor
config = PredictionConfig(
    confidence_threshold=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=1,
    # 使用4通道的标准化参数
    normalize_mean=[0.485, 0.456, 0.406, 0.0],
    normalize_std=[0.229, 0.224, 0.225, 1.0]
)

predictor = TamperingPredictor(
    model_path="output/model_final.pth",
    config=config
)
# 示例：处理单个图像文件
results = predictor.process_and_predict(
    input_path="data/train/images/train_13999.jpg",
    output_path="output/results.json",
    save_visualization=True
)

print("预测完成，结果已保存")