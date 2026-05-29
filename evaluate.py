# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import keras
from config import MODEL_FINETUNE_PATH, MODEL_INITIAL_PATH, CLASS_NAMES
from data_loader import create_generators


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
        model = load_model(MODEL_FINETUNE_PATH)
        print(f"加载模型: {MODEL_FINETUNE_PATH}")
    except:
        print(f"未找到 {MODEL_FINETUNE_PATH}，尝试加载初始模型")
        try:
            model = load_model(MODEL_INITIAL_PATH)
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
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == "__main__":
    evaluate()