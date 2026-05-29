# train.py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config import (
    EPOCHS_INITIAL, EPOCHS_FINETUNE, INITIAL_LR, FINETUNE_LR,
    MODEL_INITIAL_PATH, MODEL_FINETUNE_PATH, RANDOM_SEED
)
from data_loader import create_generators
from model import build_model

tf.random.set_seed(RANDOM_SEED)


def train():
    # 加载数据
    train_gen, valid_gen, _ = create_generators()

    # ---------- 第一阶段：冻结骨干 ----------
    print("===== 第一阶段：训练分类头（冻结EfficientNet）=====")
    model, base_model = build_model(freeze_backbone=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_INITIAL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]

    history_initial = model.fit(
        train_gen,
        epochs=EPOCHS_INITIAL,
        validation_data=valid_gen,
        callbacks=callbacks,
        verbose=1
    )

    # ---------- 第二阶段：解冻顶层进行微调 ----------
    print("===== 第二阶段：微调模型（解冻部分层）=====")
    # 解冻骨干网络的后一部分（例如最后30层，也可全部解冻）
    base_model.trainable = True
    # 可选：保持前100层冻结（加快微调，减少过拟合）
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_finetune = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_FINETUNE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    ]

    history_finetune = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=valid_gen,
        callbacks=callbacks_finetune,
        verbose=1
    )

    # 保存两个阶段的历史记录（可选，便于后续分析）
    return history_initial, history_finetune


if __name__ == "__main__":
    train()