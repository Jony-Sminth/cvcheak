from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torchvision
import torch.nn as nn

def get_faster_rcnn_model(num_classes):
    # 加载预训练 Faster R-CNN 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 修改 backbone，支持 4 通道输入
    model.backbone.body.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 修改 transform，支持 4 通道图像
    model.transform = GeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=[0.5, 0.5, 0.5, 0.0],  # 第 4 通道为 mask，均值为 0.0
        image_std=[0.5, 0.5, 0.5, 1.0]    # 第 4 通道标准差为 1.0
    )

    # 修改分类头，适应自定义类别数量
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
