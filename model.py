# model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from config import IMG_SIZE, NUM_CLASSES


def build_model(freeze_backbone=True):
    """构建EfficientNetB0迁移学习模型"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = not freeze_backbone  # 根据参数冻结或解冻

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)  # 保持BatchNorm行为
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model, base_model


if __name__ == "__main__":
    model, _ = build_model(freeze_backbone=True)
    model.summary()