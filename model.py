# model.py
import tensorflow as tf
from keras import layers, models, regularizers
from keras.applications import EfficientNetB0
from config import IMG_SIZE, NUM_CLASSES, DROPOUT_RATE, L2_REG, USE_ATTENTION, CBAM_RATIO


def cbam_block(input_tensor, ratio=8):
    """
    CBAM (Convolutional Block Attention Module)
    通道注意力 + 空间注意力，帮助模型聚焦判别性区域。
    对缓解腺癌↔大细胞癌混淆特别有效：模型会学习关注细胞形态的细微差异。
    """
    channels = input_tensor.shape[-1]

    # ===== 1. 通道注意力 (Channel Attention) =====
    # "哪些特征通道更重要？" — 对每个通道赋予不同权重
    # 同时使用 AvgPool 和 MaxPool 两个视角
    ca_avg = layers.GlobalAveragePooling2D()(input_tensor)
    ca_avg = layers.Reshape((1, 1, channels))(ca_avg)
    ca_avg = layers.Conv2D(channels // ratio, kernel_size=1, activation='relu')(ca_avg)
    ca_avg = layers.Conv2D(channels, kernel_size=1, activation='sigmoid')(ca_avg)

    ca_max = layers.GlobalMaxPooling2D()(input_tensor)
    ca_max = layers.Reshape((1, 1, channels))(ca_max)
    ca_max = layers.Conv2D(channels // ratio, kernel_size=1, activation='relu')(ca_max)
    ca_max = layers.Conv2D(channels, kernel_size=1, activation='sigmoid')(ca_max)

    channel_att = layers.Add()([ca_avg, ca_max])
    x = layers.Multiply()([input_tensor, channel_att])

    # ===== 2. 空间注意力 (Spatial Attention) =====
    # "图像的哪些空间位置更重要？" — 聚焦关键区域
    sa_avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    sa_max = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    sa_concat = layers.Concatenate()([sa_avg, sa_max])
    spatial_att = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(sa_concat)

    output = layers.Multiply()([x, spatial_att])
    return output


def build_model(freeze_backbone=True, use_attention=True):
    """
    构建 EfficientNetB0 迁移学习模型。

    参数:
        freeze_backbone: True=冻结骨干（第一阶段），False=可训练（第二阶段微调）
        use_attention:   是否在骨干后插入 CBAM 注意力模块
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = not freeze_backbone

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)

    # ---- CBAM 注意力（在 GAP 之前，保留空间维度） ----
    if use_attention:
        x = cbam_block(x, ratio=CBAM_RATIO)

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