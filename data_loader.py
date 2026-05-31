# data_loader.py
import os
import glob
import numpy as np
import pandas as pd
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


# ============================================================
# K-Fold 交叉验证数据工具
# ============================================================

def _get_train_datagen():
    """返回训练数据增强器（与 create_generators 保持一致）"""
    return ImageDataGenerator(
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


def _get_val_datagen():
    """返回验证数据增强器（仅归一化）"""
    return ImageDataGenerator(
        preprocessing_function=efficientnet.preprocess_input
    )


def collect_all_image_paths():
    """
    收集 train/ 和 valid/ 目录下所有图像路径与标签。

    用于 K-Fold 交叉验证：将训练集和验证集合并后重新划分。

    返回:
        all_paths:  np.array, 所有图像文件的绝对路径
        all_labels: np.array, 对应的类别索引 (0~3)
    """
    all_paths = []
    all_labels = []

    for split in ['train', 'valid']:
        for class_idx, folder_name in enumerate(TRAIN_FOLDER_NAMES):
            folder_path = os.path.join(DATA_DIR, split, folder_name)
            if not os.path.exists(folder_path):
                continue
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']:
                for img_path in glob.glob(os.path.join(folder_path, ext)):
                    all_paths.append(img_path)
                    all_labels.append(class_idx)

    return np.array(all_paths), np.array(all_labels)


def create_fold_generators(train_paths, train_labels, val_paths, val_labels):
    """
    基于给定的文件路径划分创建当前折的训练/验证生成器。

    使用 flow_from_dataframe 而非 flow_from_directory，
    因为 K-Fold 后文件夹结构不再对应原始划分。

    参数:
        train_paths:  当前折训练集图像路径列表
        train_labels: 当前折训练集标签 (整数 0~3)
        val_paths:    当前折验证集图像路径列表
        val_labels:   当前折验证集标签 (整数 0~3)

    返回:
        train_gen, val_gen
    """
    # 构建 DataFrame（flow_from_dataframe 要求 class 列值为类别名字符串）
    train_df = pd.DataFrame({
        'filename': train_paths,
        'class': [CLASS_NAMES[lbl] for lbl in train_labels]
    })
    val_df = pd.DataFrame({
        'filename': val_paths,
        'class': [CLASS_NAMES[lbl] for lbl in val_labels]
    })

    train_gen = _get_train_datagen().flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True
    )

    val_gen = _get_val_datagen().flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )

    return train_gen, val_gen


if __name__ == "__main__":
    # 测试数据加载
    train, valid, test = create_generators()
    print("类别映射:", train.class_indices)