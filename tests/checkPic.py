import os
import json

def check_dataset(image_dir, label_path):
    # 加载标签
    with open(label_path, 'r') as f:
        labels = json.load(f)

    missing_images = []

    # 检查每个标签是否有对应的图像文件
    for label in labels:
        image_id = label["id"]
        image_path = os.path.join(image_dir, image_id)
        if not os.path.exists(image_path):
            missing_images.append(image_path)
    
    # 输出结果
    if len(missing_images) == 0:
        print("All images are present.")
    else:
        print(f"Missing {len(missing_images)} images:")
        for missing in missing_images[:10]:  # 只打印前 10 个缺失的
            print(missing)

# 使用函数检查数据集
check_dataset(image_dir='data/preprocessed_train/', label_path='data/train/label_train_split.json')
