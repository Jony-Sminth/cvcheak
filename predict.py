# 创建配置
import torch
from utils.predict_method import PredictionConfig, TamperingPredictor
from utils.data_preprocessing import DataPreprocessing

# 注释掉演示用法，避免误运行
# dem0 = demo_usage()

# 创建正确的配置
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
    input_path="data/val/images",
    output_path="output/results.json",
    save_visualization=False  # 启用可视化结果保存
)

print("预测完成，结果已保存")