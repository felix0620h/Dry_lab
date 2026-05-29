# model.py
import tensorflow as tf
from keras import layers, models, regularizers
from keras.applications import EfficientNetB0
from config import IMG_SIZE, NUM_CLASSES, DROPOUT_RATE, L2_REG


def build_model(freeze_backbone=True):
    """构建EfficientNetB0迁移学习模型（增强正则化）"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = not freeze_backbone

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax',
                           kernel_regularizer=regularizers.l2(L2_REG))(x)

    model = models.Model(inputs, outputs)
    return model, base_model


if __name__ == "__main__":
    model, _ = build_model(freeze_backbone=True)
    model.summary()