# train.py
import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from config import (
    EPOCHS_INITIAL, EPOCHS_FINETUNE, INITIAL_LR, FINETUNE_LR, MIN_LR,
    MODEL_INITIAL_PATH, MODEL_FINETUNE_PATH, RANDOM_SEED,
    LABEL_SMOOTHING, CLASS_WEIGHTS, CLASS_NAMES,
    N_FOLDS, CV_EPOCHS_INITIAL, CV_EPOCHS_FINETUNE, USE_ATTENTION
)
from data_loader import create_generators, collect_all_image_paths, create_fold_generators
from model import build_model

tf.random.set_seed(RANDOM_SEED)


@keras.saving.register_keras_serializable(package="CustomLR")
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
    model, base_model = build_model(freeze_backbone=True, use_attention=USE_ATTENTION)

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


def _train_single_fold(train_paths, train_labels, val_paths, val_labels, fold_idx, total_folds):
    """
    在单折数据上执行两阶段训练，返回最终的 val_accuracy。

    这是 train_cross_validation 的内部辅助函数。
    """
    train_gen, val_gen = create_fold_generators(
        train_paths, train_labels, val_paths, val_labels
    )

    steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
    total_steps_initial = steps_per_epoch * CV_EPOCHS_INITIAL
    total_steps_finetune = steps_per_epoch * CV_EPOCHS_FINETUNE

    # ---- 第一阶段：冻结骨干 ----
    model, base_model = build_model(freeze_backbone=True, use_attention=USE_ATTENTION)

    lr_schedule = WarmupCosineDecay(
        initial_lr=INITIAL_LR, target_lr=MIN_LR,
        warmup_steps=steps_per_epoch, total_steps=total_steps_initial
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    model.fit(
        train_gen, epochs=CV_EPOCHS_INITIAL, validation_data=val_gen,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0),
            ModelCheckpoint(f"cv_fold{fold_idx}_initial.keras", monitor='val_accuracy',
                            save_best_only=True, verbose=0),
        ],
        class_weight=CLASS_WEIGHTS, verbose=0
    )

    # ---- 第二阶段：微调 ----
    base_model.trainable = True
    for layer in base_model.layers[:150]:
        layer.trainable = False

    lr_schedule_ft = WarmupCosineDecay(
        initial_lr=FINETUNE_LR, target_lr=MIN_LR,
        warmup_steps=max(1, steps_per_epoch // 2), total_steps=total_steps_finetune
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_ft),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen, epochs=CV_EPOCHS_FINETUNE, validation_data=val_gen,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0),
            ModelCheckpoint(f"cv_fold{fold_idx}_finetune.keras", monitor='val_accuracy',
                            save_best_only=True, verbose=0),
        ],
        class_weight=CLASS_WEIGHTS, verbose=0
    )

    # 记录最佳验证准确率
    best_val_acc = max(history.history['val_accuracy'])
    return best_val_acc


def train_cross_validation():
    """
    K-Fold 分层交叉验证训练。

    将 train/ + valid/ 的所有样本合并，按类别比例分层划分为 K 折。
    每折独立进行两阶段训练，最终输出所有折的平均准确率 ± 标准差。

    返回:
        fold_accuracies: 每折的验证最佳准确率列表
        mean_acc:        平均准确率
        std_acc:         标准差
    """
    print(f"\n{'='*65}")
    print(f"  🔬 K-Fold 分层交叉验证 (K={N_FOLDS})")
    print(f"  注意力机制: {'✅ 启用 CBAM' if USE_ATTENTION else '❌ 禁用'}")
    print(f"{'='*65}\n")

    # 收集所有 train+valid 图像路径与标签
    all_paths, all_labels = collect_all_image_paths()
    print(f"  合并后总样本数: {len(all_paths)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {np.sum(all_labels == i)} 张")
    print()

    # 分层 K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        print(f"  ┌{'─'*59}┐")
        print(f"  │  Fold {fold+1}/{N_FOLDS}  |  Train: {len(train_idx)}  |  Val: {len(val_idx)}")
        print(f"  └{'─'*59}┘")

        train_paths, train_labels = all_paths[train_idx], all_labels[train_idx]
        val_paths, val_labels = all_paths[val_idx], all_labels[val_idx]

        acc = _train_single_fold(
            train_paths, train_labels, val_paths, val_labels,
            fold_idx=fold + 1, total_folds=N_FOLDS
        )
        fold_accuracies.append(acc)
        print(f"  ✅ Fold {fold+1} 最佳 val_accuracy: {acc:.4f}\n")

    # 汇总统计
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print(f"  {'='*65}")
    print(f"  📊 交叉验证结果汇总")
    print(f"  {'='*65}")
    for i, acc in enumerate(fold_accuracies):
        print(f"    Fold {i+1}: {acc:.4f}")
    print(f"  {'─'*35}")
    print(f"    平均准确率: {mean_acc:.4f}")
    print(f"    标准差:     {std_acc:.4f}")
    print(f"    95% CI:     [{mean_acc - 1.96*std_acc:.4f}, {mean_acc + 1.96*std_acc:.4f}]")
    print(f"  {'='*65}\n")

    return fold_accuracies, mean_acc, std_acc


if __name__ == "__main__":
    train()