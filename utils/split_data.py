import os
import json
from sklearn.model_selection import train_test_split

# 数据路径
label_path = "data/train/label_train.json"
image_dir = "data/train/images"

# 载入标签
with open(label_path, 'r') as f:
    labels = json.load(f)

# 获取图像文件名列表
image_files = [label["id"] for label in labels]

# 使用 sklearn 进行数据集划分（80%训练，20%验证）
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# 将标签也分为训练和验证集
train_labels = [label for label in labels if label["id"] in train_files]
val_labels = [label for label in labels if label["id"] in val_files]

# 保存划分后的标签（可选）
with open("data/train/label_train_split.json", 'w') as f:
    json.dump(train_labels, f)

with open("data/train/label_val_split.json", 'w') as f:
    json.dump(val_labels, f)

# 输出结果
print(f"训练集图像数量: {len(train_files)}")
print(f"验证集图像数量: {len(val_files)}")
