from models.ModelTrainer import ModelTrainer

if __name__ == '__main__':
    trainer = ModelTrainer(model_name='faster_rcnn', num_classes=2, log_frequency= 1300)  # 在初始化时定义类别数量

    trainer.train(
        train_image_dir='data/preprocessed_train/',
        val_image_dir='data/preprocessed_val/',
        train_label_path='data/train/label_train_split.json',
        val_label_path='data/train/label_val_split.json',
        batch_size=2,
        num_epochs=20,
        learning_rate=0.001,
        momentum=0.95,
        weight_decay=0.0001,
        save_dir="output/faster_rcnn/"
    )
