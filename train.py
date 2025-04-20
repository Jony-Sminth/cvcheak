from models.ModelTrainer import ModelTrainer

if __name__ == '__main__':
    trainer = ModelTrainer(
        model_name='faster_rcnn', 
        num_classes=2, 
        log_frequency=650,  # 数值小就更频繁的日志记录
        debug=False,        # 启用调试模式
        train_dataset_limit=None,  # 不限制训练数据量，使用全部数据
        log_to_file=True  # 日志输出到文件
    )

    trainer.train(
        # 使用预处理后的数据目录
        train_image_dir='data/preprocessed_train/',
        val_image_dir='data/preprocessed_val/',
        # 使用包含原始尺寸信息的更新标签
        train_label_path='data/preprocessed_train/updated_labels.json',
        val_label_path='data/preprocessed_val/updated_labels.json',
        batch_size=4,
        num_epochs=10,
        learning_rate=0.001,  # 降低学习率
        momentum=0.90,
        weight_decay=0.0005,  # 增加权重衰减
        save_dir="output/faster_rcnn_two"
    )