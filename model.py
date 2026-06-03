# model.py
import tensorflow as tf
from keras import layers, models, regularizers, saving
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from config import IMG_SIZE, NUM_CLASSES, DROPOUT_RATE, L2_REG, USE_ATTENTION, CBAM_RATIO, BACKBONE


# ============================================================
# 骨干网络工厂
# ============================================================
EFFICIENTNET_MAP = {
    'b0': EfficientNetB0,
    'b1': EfficientNetB1,
    'b2': EfficientNetB2,
    'b3': EfficientNetB3,
}


def get_backbone(backbone_name='b0', input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """根据名称获取 EfficientNet 骨干网络"""
    if backbone_name not in EFFICIENTNET_MAP:
        raise ValueError(f"不支持的骨干网络: {backbone_name}，可选: {list(EFFICIENTNET_MAP.keys())}")
    net_fn = EFFICIENTNET_MAP[backbone_name]
    return net_fn(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )


# ============================================================
# CBAM — 自定义 Keras Layer，确保可序列化保存/加载
# ============================================================

@saving.register_keras_serializable(package="CBAM")
class SpatialPooling(layers.Layer):
    """
    沿通道维度池化，保留空间维度。用于 CBAM 空间注意力分支。
    
    用法: SpatialPooling('mean') 等价于 reduce_mean(x, axis=-1, keepdims=True)
          SpatialPooling('max')  等价于 reduce_max(x, axis=-1, keepdims=True)
    """
    def __init__(self, pooling='mean', **kwargs):
        super().__init__(**kwargs)
        self.pooling = pooling

    def call(self, x):
        if self.pooling == 'mean':
            return tf.reduce_mean(x, axis=-1, keepdims=True)
        elif self.pooling == 'max':
            return tf.reduce_max(x, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 1)

    def get_config(self):
        config = super().get_config()
        config['pooling'] = self.pooling
        return config


def cbam_block(input_tensor, ratio=8):
    """
    CBAM (Convolutional Block Attention Module)
    通道注意力 + 空间注意力，帮助模型聚焦判别性区域。
    对缓解腺癌↔大细胞癌混淆特别有效：模型会学习关注细胞形态的细微差异。

    可安全保存/加载（不使用 lambda，提供 output_shape）。
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
    # 使用自定义 SpatialPooling Layer（可序列化）
    sa_avg = SpatialPooling(pooling='mean', name='spatial_avg_pool')(x)
    sa_max = SpatialPooling(pooling='max', name='spatial_max_pool')(x)
    sa_concat = layers.Concatenate(axis=-1, name='spatial_concat')([sa_avg, sa_max])
    spatial_att = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', name='spatial_att')(sa_concat)

    output = layers.Multiply()([x, spatial_att])
    return output


def build_model(freeze_backbone=True, use_attention=True, backbone_name=None):
    """
    构建 EfficientNet 迁移学习模型。

    参数:
        freeze_backbone: True=冻结骨干（第一阶段），False=可训练（第二阶段微调）
        use_attention:   是否在骨干后插入 CBAM 注意力模块
        backbone_name:   骨干网络名称 ('b0','b1','b2','b3')，默认使用 config.BACKBONE
    """
    if backbone_name is None:
        backbone_name = BACKBONE

    base_model = get_backbone(backbone_name)
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
    import sys
    test_backbone = sys.argv[1] if len(sys.argv) > 1 else BACKBONE
    print(f"构建 EfficientNet{test_backbone.upper()} 模型...")
    model, base_model = build_model(freeze_backbone=True, backbone_name=test_backbone)
    model.summary()
    print(f"\n骨干网络层数: {len(base_model.layers)}")
    print(f"模型总参数量: {model.count_params():,}")