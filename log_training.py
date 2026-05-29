"""
训练日志工具 — 记录训练配置、历程和评估结果
"""
import json
import os
from datetime import datetime
from collections import defaultdict

LOG_DIR = "training_logs"


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_training_log(history_initial, history_finetune, eval_results=None):
    """
    保存训练历史到 JSON 日志文件

    参数:
        history_initial: 第一阶段 fit() 返回的 History 对象
        history_finetune: 第二阶段 fit() 返回的 History 对象
        eval_results: dict，包含评估结果（测试准确率、分类报告等）
    """
    _ensure_log_dir()

    # 导入配置用于记录
    import config

    # 构建日志条目
    log_entry = {
        "timestamp": _timestamp(),
        "config": {
            "img_size": config.IMG_SIZE,
            "batch_size": config.BATCH_SIZE,
            "class_names": config.CLASS_NAMES,
            "epochs_initial": config.EPOCHS_INITIAL,
            "epochs_finetune": config.EPOCHS_FINETUNE,
            "initial_lr": config.INITIAL_LR,
            "finetune_lr": config.FINETUNE_LR,
            "label_smoothing": config.LABEL_SMOOTHING,
            "dropout_rate": config.DROPOUT_RATE,
            "l2_reg": config.L2_REG,
            "class_weights": config.CLASS_WEIGHTS,
        },
        "phase1_history": _extract_history(history_initial),
        "phase2_history": _extract_history(history_finetune),
        "evaluation": eval_results or {},
    }

    # 写入文件
    filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    # 同时更新 latest 文件
    latest_path = os.path.join(LOG_DIR, "latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    print(f"\n📝 训练日志已保存: {filepath}")
    return filepath


def _extract_history(history):
    """从 Keras History 对象中提取可序列化的指标记录"""
    if history is None:
        return {}

    epochs = range(1, len(history.history.get('loss', [])) + 1)
    records = []
    for i, epoch in enumerate(epochs):
        record = {"epoch": epoch}
        for metric_name, values in history.history.items():
            if i < len(values):
                record[metric_name] = round(float(values[i]), 4)
        records.append(record)
    return records


def save_evaluation_results(test_loss, test_acc, class_report, cm):
    """
    保存评估结果

    参数:
        test_loss: 测试集 loss
        test_acc: 测试集准确率
        class_report: classification_report 返回的 dict
        cm: 混淆矩阵 (numpy array)
    """
    return {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_acc), 4),
        "classification_report": class_report,
        "confusion_matrix": cm.tolist() if hasattr(cm, 'tolist') else cm,
    }


def list_logs():
    """列出所有训练日志"""
    _ensure_log_dir()
    logs = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.json') and f != 'latest.json'])
    if not logs:
        print("暂无训练日志")
        return

    print(f"\n{'='*60}")
    print(f"{'训练日志列表':^60}")
    print(f"{'='*60}")
    for log in logs:
        filepath = os.path.join(LOG_DIR, log)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ts = data.get('timestamp', 'unknown')
        acc = data.get('evaluation', {}).get('test_accuracy', 'N/A')
        print(f"  📄 {log}")
        print(f"     时间: {ts}")
        print(f"     测试准确率: {acc}")
        print()


if __name__ == "__main__":
    list_logs()
