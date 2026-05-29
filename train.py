# train.py
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from config import (
    EPOCHS_INITIAL, EPOCHS_FINETUNE, INITIAL_LR, FINETUNE_LR, MIN_LR,
    MODEL_INITIAL_PATH, MODEL_FINETUNE_PATH, RANDOM_SEED,
    LABEL_SMOOTHING, CLASS_WEIGHTS
)
from data_loader import create_generators
from model import build_model

tf.random.set_seed(RANDOM_SEED)


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """带 warmup 的余弦退火学习率调度"""
    def __init__(self, initial_lr, target_lr, warmup_steps, total_steps):
        super().__init__()
        self._initial_lr = initial_lr
        self._target_lr = target_lr
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        # Warmup 阶段：线性上升
        warmup_lr = self._initial_lr * step_f / self._warmup_steps
        # 余弦退火阶段
        progress = (step_f - self._warmup_steps) / (self._total_steps - self._warmup_steps)
        cosine_lr = self._target_lr + 0.5 * (self._initial_lr - self._target_lr) * (
            1 + tf.cos(tf.constant(3.1415926) * tf.clip_by_value(progress, 0.0, 1.0))
        )
        return tf.where(step < self._warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'initial_lr': self._initial_lr,
            'target_lr': self._target_lr,
            'warmup_steps': self._warmup_steps,
            'total_steps': self._total_steps,
        }


def train():
    # 加载数据
    train_gen, valid_gen, _ = create_generators()

    steps_per_epoch = train_gen.samples // train_gen.batch_size
    total_steps_initial = steps_per_epoch * EPOCHS_INITIAL
    total_steps_finetune = steps_per_epoch * EPOCHS_FINETUNE

    # ---------- 第一阶段：冻结骨干 ----------
    print("===== 第一阶段：训练分类头（冻结EfficientNet）=====")
    model, base_model = build_model(freeze_backbone=True)

    lr_schedule_initial = WarmupCosineDecay(
        initial_lr=INITIAL_LR,
        target_lr=MIN_LR,
        warmup_steps=steps_per_epoch,  # 1 个 epoch 的 warmup
        total_steps=total_steps_initial
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_initial),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_INITIAL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    history_initial = model.fit(
        train_gen,
        epochs=EPOCHS_INITIAL,
        validation_data=valid_gen,
        callbacks=callbacks,
        class_weight=CLASS_WEIGHTS,
        verbose=1
    )

    # ---------- 第二阶段：解冻顶层进行微调 ----------
    print("===== 第二阶段：微调模型（解冻EfficientNet后半部分）=====")
    base_model.trainable = True
    # 冻结前 150 层（保留底层特征），解冻后层
    for layer in base_model.layers[:150]:
        layer.trainable = False

    lr_schedule_finetune = WarmupCosineDecay(
        initial_lr=FINETUNE_LR,
        target_lr=MIN_LR,
        warmup_steps=max(1, steps_per_epoch // 2),
        total_steps=total_steps_finetune
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_finetune),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )

    callbacks_finetune = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_FINETUNE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    history_finetune = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=valid_gen,
        callbacks=callbacks_finetune,
        class_weight=CLASS_WEIGHTS,
        verbose=1
    )

    return history_initial, history_finetune


if __name__ == "__main__":
    train()