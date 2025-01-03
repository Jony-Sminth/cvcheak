import os
import shutil
import json

def classify_images(preprocessed_image_dir, label_path):
    """
    根据标签文件将图像分类到篡改和未篡改文件夹中。
    Args:
        preprocessed_image_dir (str): 预处理后的图像存储路径
        label_path (str): 标签文件的路径
    """
    tampered_dir = os.path.join(preprocessed_image_dir, "tampered")
    not_tampered_dir = os.path.join(preprocessed_image_dir, "not_tampered")

    os.makedirs(tampered_dir, exist_ok=True)
    os.makedirs(not_tampered_dir, exist_ok=True)

    with open(label_path, 'r') as f:
        labels = json.load(f)

    for label in labels:
        image_id = label['id']
        regions = label['region']
        # 文件扩展名为 .pt（预处理后的文件格式）
        image_id = image_id.replace('.pt', '.jpg', '.png')  # 适配文件扩展名
        src_path = os.path.join(preprocessed_image_dir, image_id)

        if len(regions) > 0:
            dst_path = os.path.join(tampered_dir, image_id)
        else:
            dst_path = os.path.join(not_tampered_dir, image_id)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: Source image {src_path} not found.")

    print(f"Images from {preprocessed_image_dir} have been successfully classified into tampered and not_tampered folders.")

if __name__ == "__main__":
    # 分类训练集图像
    train_preprocessed_image_dir = "data/preprocessed_train/"
    train_label_path = "data/train/label_train_split.json"
    classify_images(train_preprocessed_image_dir, train_label_path)

    # 分类验证集图像
    val_preprocessed_image_dir = "data/preprocessed_val/"
    val_label_path = "data/train/label_val_split.json"
    classify_images(val_preprocessed_image_dir, val_label_path)