# data_loader.py
import os
import tensorflow as tf
from keras.applications import efficientnet
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, CLASS_NAMES, RANDOM_SEED, TRAIN_FOLDER_NAMES, TEST_FOLDER_NAMES

tf.random.set_seed(RANDOM_SEED)


def create_generators():
    """创建训练、验证、测试集的DataFrame迭代器"""
    # 训练数据增强（增强版）
    train_datagen = ImageDataGenerator(
        preprocessing_function=efficientnet.preprocess_input,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # 验证/测试：仅归一化
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=efficientnet.preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=TRAIN_FOLDER_NAMES,
        shuffle=True
    )

    valid_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'valid'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=TRAIN_FOLDER_NAMES,
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=TEST_FOLDER_NAMES,
        shuffle=False
    )

    return train_gen, valid_gen, test_gen


if __name__ == "__main__":
    # 测试数据加载
    train, valid, test = create_generators()
    print("类别映射:", train.class_indices)