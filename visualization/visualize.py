"""
可视化训练日志 — 从 latest.json 读取数据，分别生成多张独立清晰图表

输出目录: visualization/

输出文件:
  - training_curves.png       两阶段训练曲线（准确率 + 损失）
  - confusion_matrix.png      混淆矩阵（数量 + 百分比）
  - classification_report.png  分类指标柱状图（Precision / Recall / F1）
  - metrics_radar.png          雷达图（各类指标多维对比）
  - dataset_distribution.png   数据集分布 + 超参配置
"""
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============ 全局字体配置（防止 Windows 下乱码） ============
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'Segoe UI', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {"font.sans-serif": ['Arial', 'Microsoft YaHei', 'Segoe UI', 'DejaVu Sans']})

# ============ 基本配置 ============
LOG_PATH = os.path.join("training_logs", "latest.json")
OUTPUT_DIR = "visualization"
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


def plot_confusion_matrix(ax, cm, class_names, annot_fontsize=8):
    """子图2: 混淆矩阵（数量 + 百分比）

    参数:
        annot_fontsize: 格子内标注字号
    """
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    overall_acc = np.trace(cm) / np.sum(cm) * 100

    # 使用美观的蓝调色图
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(cm, annot=False, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor='white',
                cbar_kws={'shrink': 0.75, 'label': 'Count'},
                ax=ax)

    # 标注每个格子：数量 + 百分比
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_percent[i, j]
            color = 'white' if cm_percent[i, j] > 50 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{count}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=annot_fontsize, color=color, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    ax.set_title(f'Confusion Matrix  (Overall Acc: {overall_acc:.1f}%)',
                 fontsize=13, fontweight='bold', pad=10)
    tick_fs = max(7, annot_fontsize - 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=tick_fs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs)
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


def _make_title(cfg, log):
    """生成统一的标题字符串"""
    return (f'Chest Cancer Classification — EfficientNetB0\n'
            f'{log["timestamp"]}  |  '
            f'Batch {cfg["batch_size"]}  |  '
            f'LR {cfg["initial_lr"]}→{cfg["finetune_lr"]}  |  '
            f'Label Smooth: {cfg["label_smoothing"]}')


def save_fig(fig, filename, dpi=200):
    """保存图表到 OUTPUT_DIR 并打印路径"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ saved: {path}")


# ============================================================
# 独立生成函数（每张图单独输出）
# ============================================================

def gen_training_curves(log):
    """图1: 两阶段训练曲线（准确率 + 损失）"""
    p1_hist = log['phase1_history']
    p2_hist = log['phase2_history']
    cfg = log['config']

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(_make_title(cfg, log), fontsize=13, fontweight='bold', y=1.02)

    plot_training_curves(ax, p1_hist, p2_hist, title="Two-Phase Training Curves")

    # 单独图例更大更清晰
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.tick_params(labelsize=10)

    # 调整第二个 y 轴标签大小
    ax2 = fig.axes[1]
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.tick_params(labelsize=10)

    # 图例调整
    lines = ax.get_lines() + ax2.get_lines()
    labels = ['Train Accuracy', 'Val Accuracy', 'Train Loss', 'Val Loss']
    ax.legend(lines, labels, loc='lower left', fontsize=10, ncol=2, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, 'training_curves.png')


def gen_confusion_matrix(log):
    """图2: 混淆矩阵"""
    cm = np.array(log['evaluation']['confusion_matrix'])
    cfg = log['config']

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.suptitle(_make_title(cfg, log), fontsize=13, fontweight='bold', y=1.02)

    # 通过 annot_fontsize 统一控制字号，避免重复标注
    plot_confusion_matrix(ax, cm, CLASS_NAMES, annot_fontsize=13)

    # 调大轴标签
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')

    fig.tight_layout()
    save_fig(fig, 'confusion_matrix.png')


def gen_classification_report(log):
    """图3: 分类指标柱状图"""
    report = log['evaluation']['classification_report']
    cfg = log['config']

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(_make_title(cfg, log), fontsize=13, fontweight='bold', y=1.02)

    plot_classification_report(ax, report, CLASS_NAMES)

    # 调大字体和数值标注
    ax.set_ylabel('Score', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)

    # 重新标注数值（更大）
    for bar_group in ax.containers:
        for bar in bar_group:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.tight_layout()
    save_fig(fig, 'classification_report.png')


def gen_radar(log):
    """图4: 雷达图"""
    report = log['evaluation']['classification_report']
    cfg = log['config']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    fig.suptitle(_make_title(cfg, log), fontsize=13, fontweight='bold', y=1.05)

    plot_metrics_radar(ax, report)

    # 调大字体
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='lower right', bbox_to_anchor=(1.3, 0))

    fig.tight_layout()
    save_fig(fig, 'metrics_radar.png')


def gen_dataset_distribution(log):
    """图5: 数据集分布 + 超参配置"""
    report = log['evaluation']['classification_report']
    cfg = log['config']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    fig.suptitle(_make_title(cfg, log), fontsize=13, fontweight='bold', y=1.02)

    # ---- 左图: 数据集分布 ----
    support_vals = [report['adenocarcinoma']['support'],
                    report['large.cell.carcinoma']['support'],
                    report['squamous.cell.carcinoma']['support'],
                    report['normal']['support']]
    short_names = ['Adeno', 'Large Cell', 'Squamous', 'Normal']
    train_samples = [195, 115, 155, 148]
    valid_samples = [23, 21, 15, 13]

    bar_width = 0.25
    x = np.arange(len(short_names))

    bars1 = ax1.bar(x - bar_width, train_samples, bar_width, color='#1976D2', label='Train', edgecolor='white')
    bars2 = ax1.bar(x, valid_samples, bar_width, color='#FFA000', label='Valid', edgecolor='white')
    bars3 = ax1.bar(x + bar_width, support_vals, bar_width, color='#388E3C', label='Test', edgecolor='white')

    # 标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., h + 2,
                     f'{int(h)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=11)
    ax1.set_ylabel('Sample Count', fontsize=12)
    ax1.set_title('Dataset Distribution', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)

    # ---- 右图: 超参配置 ----
    ax2.axis('off')
    config_lines = [
        "Training Configuration",
        "=" * 32,
        f"Image Size         {cfg['img_size']} × {cfg['img_size']}",
        f"Batch Size         {cfg['batch_size']}",
        f"Epochs             {cfg['epochs_initial']} (P1) + {cfg['epochs_finetune']} (P2)",
        f"Initial LR         {cfg['initial_lr']}",
        f"Fine-tune LR       {cfg['finetune_lr']}",
        f"Min LR             1e-7",
        f"Dropout            {cfg['dropout_rate']}",
        f"L2 Regularization  {cfg['l2_reg']}",
        f"Label Smoothing    {cfg['label_smoothing']}",
        "",
        "Class Weights:",
        f"  Adenocarcinoma        {cfg['class_weights']['0']}",
        f"  Large Cell Carcinoma  {cfg['class_weights']['1']}",
        f"  Squamous Cell Carcinoma {cfg['class_weights']['2']}",
        f"  Normal                {cfg['class_weights']['3']}",
        "",
        "Model Architecture:",
        f"  Backbone:     EfficientNetB0",
        f"  Attention:    CBAM (ratio={cfg.get('cbam_ratio', 8)})",
        f"  Dropout:      {cfg['dropout_rate']}",
        "",
        "Test Results:",
        f"  Accuracy: {log['evaluation']['test_accuracy']:.2%}",
        f"  Loss:     {log['evaluation']['test_loss']:.4f}",
    ]

    text_y = 0.92
    for line in config_lines:
        ax2.text(0.05, text_y, line, fontsize=11, fontfamily='monospace',
                 transform=ax2.transAxes, va='top',
                 color='#333333')
        text_y -= 0.045

    fig.tight_layout()
    save_fig(fig, 'dataset_distribution.png')


def main():
    # 加载数据
    log = load_log(LOG_PATH)
    cfg = log['config']

    print("=" * 55)
    print("  Generating visualizations from training logs...")
    print("=" * 55)
    print()

    gen_training_curves(log)
    gen_confusion_matrix(log)
    gen_classification_report(log)
    gen_radar(log)
    gen_dataset_distribution(log)

    print()
    print("=" * 55)
    print(f"  ✅ All 5 charts saved to current directory")
    print("=" * 55)


if __name__ == "__main__":
    main()
