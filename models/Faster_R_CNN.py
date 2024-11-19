import torchvision
import torch
import torchvision.transforms as T

def get_faster_rcnn_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the head of the model with a new one (num_classes is 2, assuming "tampered" and "not tampered")
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

# 使用2个类别，一个是背景，一个是篡改区域
# model = get_faster_rcnn_model(num_classes=2)
