# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import keras
from config import MODEL_FINETUNE_PATH, MODEL_INITIAL_PATH, CLASS_NAMES
from data_loader import create_generators
from train import WarmupCosineDecay  # 注册自定义学习率调度器


def plot_history(history, title_suffix=""):
    """绘制训练曲线"""
    if not history:
        return
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Acc')
    plt.plot(epochs_range, val_acc, label='Validation Acc')
    plt.legend(loc='lower right')
    plt.title(f'Accuracy {title_suffix}')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Loss {title_suffix}')
    plt.tight_layout()
    plt.savefig(f'training_curves_{title_suffix}.png')
    plt.show()


def evaluate():
    # 加载测试数据
    _, _, test_gen = create_generators()

    # 加载最佳微调模型（如果不存在则尝试初始模型）
    try:
        model = load_model(MODEL_FINETUNE_PATH, compile=False)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"加载模型: {MODEL_FINETUNE_PATH}")
    except:
        print(f"未找到 {MODEL_FINETUNE_PATH}，尝试加载初始模型")
        try:
            model = load_model(MODEL_INITIAL_PATH, compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except:
            model = load_model("best_model_initial.h5")

    # 测试集评估
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\n测试集准确率: {test_acc:.4f}")

    # 预测
    test_gen.reset()
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes

    # 分类报告
    print("\n分类报告:")
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES)

    # 返回评估结果（用于日志记录）
    from log_training import save_evaluation_results
    return save_evaluation_results(test_loss, test_acc, report_dict, cm)


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """绘制美观的混淆矩阵"""
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # 生成美观的显示标签（替换点号为空格，首字母大写）
    display_names = [name.replace('.', ' ').title() for name in class_names]

    # 计算总体准确率
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    accuracy = total_correct / total_samples * 100

    plt.figure(figsize=(9, 7.5))

    # 使用更美观的色板
    cmap = sns.color_palette("Blues", as_cmap=True)

    # 绘制热力图
    ax = sns.heatmap(cm, annot=False, fmt='d', cmap=cmap,
                     xticklabels=display_names, yticklabels=display_names,
                     linewidths=1, linecolor='white', cbar_kws={'shrink': 0.8})

    # 在每个格子中手动标注：数量 + 百分比
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_percent[i, j]
            color = 'white' if cm_percent[i, j] > 50 else 'black'
            text = f'{count}\n({pct:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    # 设置标签和标题
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix  (Overall Accuracy: {accuracy:.1f}%)',
              fontsize=14, fontweight='bold', pad=15)

    # 旋转标签
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"\n混淆矩阵已保存至: {save_path}")


if __name__ == "__main__":
    evaluate()