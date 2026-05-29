# data_loader.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, CLASS_NAMES, RANDOM_SEED

tf.random.set_seed(RANDOM_SEED)


def create_generators():
    """创建训练、验证、测试集的DataFrame迭代器"""
    # 训练数据增强
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 验证/测试：仅归一化
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True
    )

    valid_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'valid'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )

    return train_gen, valid_gen, test_gen


if __name__ == "__main__":
    # 测试数据加载
    train, valid, test = create_generators()
    print("类别映射:", train.class_indices)