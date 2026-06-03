"""
可视化训练日志 — 从 latest.json 读取数据，生成多面板综合图表
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============ 配置 ============
LOG_PATH = os.path.join("training_logs", "latest.json")
SAVE_PATH = "visualization_summary.png"
CLASS_NAMES = ['Adenocarcinoma', 'Large Cell\nCarcinoma', 'Squamous Cell\nCarcinoma', 'Normal']

# 调色板
COLORS = {
    'train_acc': '#2196F3',
    'val_acc': '#FF9800',
    'train_loss': '#4CAF50',
    'val_loss': '#F44336',
    'phase1_bg': '#E3F2FD',
    'phase2_bg': '#FFF3E0',
    'precision': '#42A5F5',
    'recall': '#FFA726',
    'f1': '#66BB6A',
}


def load_log(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics(history):
    """从历史记录列表提取指标数组"""
    epochs = [h['epoch'] for h in history]
    acc = [h['accuracy'] for h in history]
    val_acc = [h['val_accuracy'] for h in history]
    loss = [h['loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    return epochs, acc, val_acc, loss, val_loss


def plot_training_curves(ax, p1_hist, p2_hist, title="Training Curves"):
    """子图1: 两阶段训练曲线（准确率 + 损失）"""
    # 提取数据
    e1, acc1, val_acc1, loss1, val_loss1 = extract_metrics(p1_hist)
    e2, acc2, val_acc2, loss2, val_loss2 = extract_metrics(p2_hist)

    # 阶段分界点
    phase_boundary = len(p1_hist)

    # 合并数据
    epochs_all = e1 + [phase_boundary + e for e in e2]
    acc_all = acc1 + acc2
    val_acc_all = val_acc1 + val_acc2
    loss_all = loss1 + loss2
    val_loss_all = val_loss1 + val_loss2

    # 背景色标识阶段
    ax.axvspan(0.5, phase_boundary + 0.5, alpha=0.08, color=COLORS['phase1_bg'], label='Phase 1 (Frozen)')
    ax.axvspan(phase_boundary + 0.5, len(epochs_all) + 0.5, alpha=0.08, color=COLORS['phase2_bg'],
               label='Phase 2 (Fine-tune)')

    # 分界线
    ax.axvline(x=phase_boundary + 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # 准确率曲线
    l1, = ax.plot(epochs_all, acc_all, 'o-', color=COLORS['train_acc'], markersize=2.5, linewidth=1.2,
                  label='Train Accuracy')
    l2, = ax.plot(epochs_all, val_acc_all, 's-', color=COLORS['val_acc'], markersize=2.5, linewidth=1.2,
                  label='Val Accuracy')

    # 在阶段分界处标记最佳 validation accuracy
    best_val1_idx = np.argmax(val_acc1)
    best_val2_idx = np.argmax(val_acc2) + phase_boundary
    ax.annotate(f'Best: {val_acc1[best_val1_idx]:.1%}',
                xy=(e1[best_val1_idx], val_acc1[best_val1_idx]),
                xytext=(5, 15), textcoords='offset points', fontsize=7,
                color=COLORS['val_acc'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['val_acc'], lw=0.8))
    ax.annotate(f'Best: {val_acc2[best_val2_idx - phase_boundary]:.1%}',
                xy=(best_val2_idx, val_acc2[best_val2_idx - phase_boundary]),
                xytext=(5, -20), textcoords='offset points', fontsize=7,
                color=COLORS['val_acc'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['val_acc'], lw=0.8))

    # 损失曲线（用第二个 y 轴）
    ax2 = ax.twinx()
    l3, = ax2.plot(epochs_all, loss_all, 'o-', color=COLORS['train_loss'], markersize=2.5, linewidth=1.2,
                   alpha=0.7, label='Train Loss')
    l4, = ax2.plot(epochs_all, val_loss_all, 's-', color=COLORS['val_loss'], markersize=2.5, linewidth=1.2,
                   alpha=0.7, label='Val Loss')
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_ylim(bottom=0)

    # 轴标签
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, len(epochs_all) + 0.5)

    # 阶段标注
    mid1 = phase_boundary / 2
    mid2 = phase_boundary + (len(e2) / 2)
    ax.text(mid1, 1.02, 'Phase 1: Frozen Backbone', ha='center', fontsize=8,
            color='#1565C0', fontweight='bold', style='italic')
    ax.text(mid2, 1.02, 'Phase 2: Fine-tuning', ha='center', fontsize=8,
            color='#E65100', fontweight='bold', style='italic')

    # 合并图例
    lines = [l1, l2, l3, l4]
    labels = ['Train Acc', 'Val Acc', 'Train Loss', 'Val Loss']
    ax.legend(lines, labels, loc='lower left', fontsize=7, ncol=2, framealpha=0.9)

    ax.grid(True, alpha=0.3)
    return ax


def plot_confusion_matrix(ax, cm, class_names):
    """子图2: 混淆矩阵（数量 + 百分比）"""
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    overall_acc = np.trace(cm) / np.sum(cm) * 100

    # 使用美观的蓝调色图
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(cm, annot=False, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor='white',
                cbar_kws={'shrink': 0.75, 'label': 'Count'},
                ax=ax)

    # 手动标注每个格子：数量 + 百分比
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_percent[i, j]
            color = 'white' if cm_percent[i, j] > 50 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{count}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=8, color=color, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    ax.set_title(f'Confusion Matrix  (Overall Acc: {overall_acc:.1f}%)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    return ax


def plot_classification_report(ax, report, class_names):
    """子图3: 分类报告柱状图"""
    metrics = ['precision', 'recall', 'f1-score']
    values = {m: [] for m in metrics}
    for cls in ['adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma', 'normal']:
        for m in metrics:
            values[m].append(report[cls][m])

    x = np.arange(len(class_names))
    width = 0.22

    bars1 = ax.bar(x - width, values['precision'], width, color=COLORS['precision'],
                   edgecolor='white', linewidth=0.5, label='Precision')
    bars2 = ax.bar(x, values['recall'], width, color=COLORS['recall'],
                   edgecolor='white', linewidth=0.5, label='Recall')
    bars3 = ax.bar(x + width, values['f1-score'], width, color=COLORS['f1'],
                   edgecolor='white', linewidth=0.5, label='F1-Score')

    # 在柱上标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha='right', fontsize=7.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Classification Report by Class', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    return ax


def plot_metrics_radar(ax, report):
    """子图4: 雷达图 — 各类别的 Precision / Recall / F1 对比"""
    categories = ['Adeno-\ncarcinoma', 'Large Cell\nCarcinoma', 'Squamous\nCarcinoma', 'Normal']
    metrics = ['precision', 'recall', 'f1-score']
    colors = [COLORS['precision'], COLORS['recall'], COLORS['f1']]
    labels = ['Precision', 'Recall', 'F1-Score']

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for i, metric in enumerate(metrics):
        vals = [report[cls][metric] for cls in ['adenocarcinoma', 'large.cell.carcinoma',
                                                 'squamous.cell.carcinoma', 'normal']]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', color=colors[i], linewidth=1.5, markersize=4, label=labels[i])
        ax.fill(angles, vals, alpha=0.05, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title('Metrics by Class (Radar)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    return ax


def add_summary_box(fig, report, test_acc, test_loss):
    """在图表底部添加摘要信息"""
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    summary_text = (
        f"Test Accuracy: {test_acc:.1%}  |  "
        f"Test Loss: {test_loss:.4f}  |  "
        f"Macro Avg F1: {macro_f1:.3f}  |  "
        f"Weighted Avg F1: {weighted_f1:.3f}"
    )
    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#BDBDBD'))


def main():
    # 加载数据
    log = load_log(LOG_PATH)
    cfg = log['config']
    p1_hist = log['phase1_history']
    p2_hist = log['phase2_history']
    eval_data = log['evaluation']
    cm = np.array(eval_data['confusion_matrix'])
    report = eval_data['classification_report']
    test_acc = eval_data['test_accuracy']
    test_loss = eval_data['test_loss']

    # 创建画布
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.28, wspace=0.25,
                          height_ratios=[1, 1, 0.8],
                          left=0.06, right=0.95, top=0.93, bottom=0.06)

    # 标题
    fig.suptitle(f'Chest Cancer Classification — EfficientNetB0\n'
                 f'Trained: {log["timestamp"]}  |  '
                 f'Batch Size: {cfg["batch_size"]}  |  '
                 f'LR: {cfg["initial_lr"]} → {cfg["finetune_lr"]}  |  '
                 f'Label Smoothing: {cfg["label_smoothing"]}',
                 fontsize=14, fontweight='bold', y=0.975)

    # === 子图 1: 训练曲线 ===
    ax1 = fig.add_subplot(gs[0, :])
    plot_training_curves(ax1, p1_hist, p2_hist)

    # === 子图 2: 混淆矩阵 ===
    ax2 = fig.add_subplot(gs[1, 0])
    plot_confusion_matrix(ax2, cm, CLASS_NAMES)

    # === 子图 3: 分类报告柱状图 ===
    ax3 = fig.add_subplot(gs[1, 1])
    plot_classification_report(ax3, report, CLASS_NAMES)

    # === 子图 4: 雷达图 ===
    ax4 = fig.add_subplot(gs[2, 0], projection='polar')
    plot_metrics_radar(ax4, report)

    # === 子图 5: 数据分布 + 关键配置 ===
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # 数据集分布
    support_vals = [report['adenocarcinoma']['support'],
                    report['large.cell.carcinoma']['support'],
                    report['squamous.cell.carcinoma']['support'],
                    report['normal']['support']]
    short_names = ['Adeno', 'Large Cell', 'Squamous', 'Normal']

    # 训练集样本数
    train_samples = [195, 115, 155, 148]
    valid_samples = [23, 21, 15, 13]

    bar_width = 0.25
    x = np.arange(len(short_names))

    ax5_sub = ax5.inset_axes([0.05, 0.45, 0.9, 0.5])
    bars1 = ax5_sub.bar(x - bar_width, train_samples, bar_width, color='#1976D2', label='Train', edgecolor='white')
    bars2 = ax5_sub.bar(x, valid_samples, bar_width, color='#FFA000', label='Valid', edgecolor='white')
    bars3 = ax5_sub.bar(x + bar_width, support_vals, bar_width, color='#388E3C', label='Test', edgecolor='white')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax5_sub.text(bar.get_x() + bar.get_width() / 2., h + 1,
                         f'{int(h)}', ha='center', va='bottom', fontsize=6.5)

    ax5_sub.set_xticks(x)
    ax5_sub.set_xticklabels(short_names, fontsize=7.5)
    ax5_sub.set_ylabel('Sample Count', fontsize=9)
    ax5_sub.set_title('Dataset Distribution', fontsize=10, fontweight='bold')
    ax5_sub.legend(fontsize=7, loc='upper right')
    ax5_sub.grid(True, axis='y', alpha=0.3)

    # 配置信息
    config_lines = [
        "Training Configuration",
        "─" * 28,
        f"Image Size:     {cfg['img_size']}×{cfg['img_size']}",
        f"Batch Size:     {cfg['batch_size']}",
        f"Epochs:         {cfg['epochs_initial']} (P1) + {cfg['epochs_finetune']} (P2)",
        f"Initial LR:     {cfg['initial_lr']}",
        f"Fine-tune LR:   {cfg['finetune_lr']}",
        f"Dropout:        {cfg['dropout_rate']}",
        f"L2 Reg:         {cfg['l2_reg']}",
        f"Label Smooth:   {cfg['label_smoothing']}",
        f"Class Weights:  {cfg['class_weights']}",
    ]
    text_y = 0.38
    for line in config_lines:
        ax5.text(0.05, text_y, line, fontsize=7, fontfamily='monospace',
                 transform=ax5.transAxes, va='top')
        text_y -= 0.05

    # 底部摘要
    add_summary_box(fig, report, test_acc, test_loss)

    # 保存
    plt.savefig(SAVE_PATH, dpi=200, bbox_inches='tight')
    print(f"✅ 可视化已保存至: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
