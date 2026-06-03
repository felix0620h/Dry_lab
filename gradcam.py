"""
Grad-CAM 可视化 — 分析模型关注区域，诊断分类混淆原因

使用方法:
    python gradcam.py                          # 在测试集上批量分析
    python gradcam.py --image path/to/img.jpg  # 分析单张图片
    python gradcam.py --sample 10              # 展示 N 张典型样本

依赖:
    pip install opencv-python
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'Segoe UI', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from keras.models import load_model, Model
from keras.applications import efficientnet
from config import (
    MODEL_FINETUNE_PATH, MODEL_INITIAL_PATH, CLASS_NAMES,
    IMG_SIZE, DATA_DIR, TEST_FOLDER_NAMES
)
from train import WarmupCosineDecay
from model import cbam_block, SpatialPooling

# 显示用类别名（短格式）
DISPLAY_NAMES = ['Adeno', 'Large Cell', 'Squamous', 'Normal']
# 颜色映射（真实类别 → 正确/错误用不同颜色边框）
CORRECT_COLOR = '#2E7D32'  # 绿色 — 正确
WRONG_COLOR = '#C62828'    # 红色 — 错误


# ============================================================
# Grad-CAM 核心
# ============================================================

def find_last_conv_layer(model):
    """
    找到模型中最后一个 Conv2D 层的名字。
    优先找 backbone 中的最后一层，再找整体模型中的最后一层。
    """
    last_conv = None
    for layer in reversed(model.layers):
        # 跳过输入层、池化层、全连接层
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        raise ValueError("模型中未找到 Conv2D 层！")
    return last_conv


def grad_cam(model, img_array, layer_name=None):
    """
    计算 Grad-CAM 热力图。

    参数:
        model:       Keras 模型
        img_array:   预处理后的图像 (1, H, W, 3)
        layer_name:  目标卷积层名（None 自动找最后一层）

    返回:
        heatmap:     (H, W) 归一化热力图
        pred_idx:    预测类别索引
        pred_probs:  各类别概率
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    # 构建梯度计算图
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array, training=False)
        pred_idx = tf.argmax(predictions[0])
        loss = predictions[:, pred_idx]

    # 计算梯度
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # 加权特征图
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads[:, tf.newaxis, tf.newaxis, :], axis=-1)

    # 归一化到 [0, 1]
    heatmap = tf.maximum(heatmap, 0)  # ReLU: 只保留正贡献
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(pred_idx.numpy()), predictions[0].numpy()


# ============================================================
# 可视化
# ============================================================

def overlay_heatmap(heatmap, img_array, alpha=0.45):
    """
    将热力图叠加到原图上。

    参数:
        heatmap:    (H, W) 归一化热力图
        img_array:  原始图像 (H, W, 3), dtype=uint8
        alpha:      热力图透明度

    返回:
        overlay:    叠加后的图像 (H, W, 3), dtype=uint8
    """
    # 调整热力图尺寸匹配原图
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], (img_array.shape[0], img_array.shape[1])
    ).numpy().squeeze()

    # 应用颜色映射 (jet)
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # (H, W, 3), range [0,1]

    # 叠加
    img_normalized = img_array / 255.0
    overlay = (1 - alpha) * img_normalized + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay, heatmap_resized


def plot_gradcam_sample(original_img, overlay_img, heatmap, true_label, pred_label, probs, save_path=None):
    """
    绘制单张 Grad-CAM 分析图：
    左: 原始图像 | 中: Grad-CAM 叠加 | 右: 类别概率条

    参数:
        original_img:  原始图像 uint8 (H, W, 3)
        overlay_img:   叠加图像 uint8 (H, W, 3)
        heatmap:       热力图 (H, W)
        true_label:    真实类别索引
        pred_label:    预测类别索引
        probs:         各类别概率 (4,)
        save_path:     保存路径，None 则显示
    """
    is_correct = (true_label == pred_label)
    border_color = CORRECT_COLOR if is_correct else WRONG_COLOR

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f'{DISPLAY_NAMES[true_label]}  →  {DISPLAY_NAMES[pred_label]}  '
        f'({"✅ Correct" if is_correct else "❌ Wrong"})',
        fontsize=14, fontweight='bold',
        color=border_color, y=1.05
    )

    # --- 左: 原始图像 ---
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original\nTrue: {DISPLAY_NAMES[true_label]}', fontsize=11)
    axes[0].axis('off')

    # --- 中: Grad-CAM 叠加 ---
    axes[1].imshow(overlay_img)
    axes[1].set_title(f'Grad-CAM\nPred: {DISPLAY_NAMES[pred_label]}', fontsize=11, color=border_color)
    axes[1].axis('off')

    # 添加 colorbar
    im = axes[1].imshow(heatmap, cmap='jet', alpha=0)  # 仅用于 colorbar
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Attention')

    # --- 右: 类别概率条 ---
    colors = [CORRECT_COLOR if i == true_label else '#BDBDBD' for i in range(4)]
    bars = axes[2].barh(DISPLAY_NAMES, probs, color=colors, edgecolor='white', height=0.6)
    axes[2].axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel('Probability', fontsize=10)
    axes[2].set_title('Class Probabilities', fontsize=11)

    # 标注数值
    for bar, prob in zip(bars, probs):
        axes[2].text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{prob:.1%}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_grid_comparison(samples, title="Grad-CAM Analysis", save_path=None):
    """
    绘制对比网格：每行一个样本（原始 | Grad-CAM 叠加 | 概率条）。

    参数:
        samples: list of dicts, 每个包含:
            {'original': uint8, 'overlay': uint8, 'heatmap': ndarray,
             'true_label': int, 'pred_label': int, 'probs': ndarray}
        title:   总标题
        save_path: 保存路径
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.5 * n))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, s in enumerate(samples):
        is_correct = (s['true_label'] == s['pred_label'])
        border_color = CORRECT_COLOR if is_correct else WRONG_COLOR

        # 左: 原始
        axes[i, 0].imshow(s['original'])
        axes[i, 0].set_title(f'[{i+1}] Original\nTrue: {DISPLAY_NAMES[s["true_label"]]}', fontsize=10)
        axes[i, 0].axis('off')

        # 中: Grad-CAM
        axes[i, 1].imshow(s['overlay'])
        axes[i, 1].set_title(f'[{i+1}] Grad-CAM\nPred: {DISPLAY_NAMES[s["pred_label"]]}',
                             fontsize=10, color=border_color)
        axes[i, 1].axis('off')

        # 右: 概率
        colors = [CORRECT_COLOR if j == s['true_label'] else '#BDBDBD' for j in range(4)]
        axes[i, 2].barh(DISPLAY_NAMES, s['probs'], color=colors, edgecolor='white', height=0.6)
        axes[i, 2].axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[i, 2].set_xlim(0, 1)
        axes[i, 2].set_xlabel('Probability', fontsize=9)
        for bar, prob in zip(axes[i, 2].containers[0], s['probs']):
            axes[i, 2].text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{prob:.1%}', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✅ Grid saved: {save_path}")
        plt.close()
    else:
        plt.show()


# ============================================================
# 批量分析
# ============================================================

def load_test_images(num_samples=None):
    """
    从测试集加载图像，返回列表。

    返回:
        images:   list of (H, W, 3) uint8
        labels:   list of int
        paths:    list of str
    """
    images, labels, paths = [], [], []
    for class_idx, folder_name in enumerate(TEST_FOLDER_NAMES):
        folder = os.path.join(DATA_DIR, 'test', folder_name)
        if not os.path.exists(folder):
            continue
        for fname in sorted(os.listdir(folder))[:num_samples]:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                fpath = os.path.join(folder, fname)
                img = tf.keras.utils.load_img(fpath, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = tf.keras.utils.img_to_array(img).astype(np.uint8)
                images.append(img_array)
                labels.append(class_idx)
                paths.append(fpath)
    return images, labels, paths


def analyze_test_set(model, num_samples=None, save_grid=True):
    """
    在测试集上运行 Grad-CAM 分析。

    参数:
        model:       已加载的 Keras 模型
        num_samples: 分析的样本数（None=全部）
        save_grid:   是否保存对比网格图
    """
    print("加载测试集图像...")
    images, true_labels, paths = load_test_images(num_samples)
    print(f"  共 {len(images)} 张图像")

    results = {'correct': [], 'wrong': [], 'all': []}

    for idx, (img, true_label, path) in enumerate(zip(images, true_labels, paths)):
        # 预处理
        img_preprocessed = efficientnet.preprocess_input(img.astype(np.float32))
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # Grad-CAM
        heatmap, pred_label, probs = grad_cam(model, img_batch)

        # 叠加
        overlay, heatmap_resized = overlay_heatmap(heatmap, img)

        sample = {
            'original': img,
            'overlay': overlay,
            'heatmap': heatmap_resized,
            'true_label': true_label,
            'pred_label': pred_label,
            'probs': probs,
            'path': path,
        }
        results['all'].append(sample)

        if true_label == pred_label:
            results['correct'].append(sample)
        else:
            results['wrong'].append(sample)

        if (idx + 1) % 20 == 0:
            print(f"  已分析 {idx + 1}/{len(images)} 张...")

    print(f"\n📊 分析完成:")
    print(f"  总样本: {len(results['all'])}")
    print(f"  正确:   {len(results['correct'])} ({len(results['correct'])/max(len(results['all']),1):.1%})")
    print(f"  错误:   {len(results['wrong'])} ({len(results['wrong'])/max(len(results['all']),1):.1%})")

    # 保存网格图
    if save_grid:
        # 保存错误样本对比
        if results['wrong']:
            max_wrong = min(12, len(results['wrong']))
            plot_grid_comparison(
                results['wrong'][:max_wrong],
                title=f"Grad-CAM: Misclassified Samples ({max_wrong} of {len(results['wrong'])})",
                save_path=os.path.join("visualization", "gradcam_wrong.png")
            )
        # 保存正确样本对比
        if results['correct']:
            max_correct = min(12, len(results['correct']))
            plot_grid_comparison(
                results['correct'][:max_correct],
                title=f"Grad-CAM: Correctly Classified Samples ({max_correct} of {len(results['correct'])})",
                save_path=os.path.join("visualization", "gradcam_correct.png")
            )

    return results


# ============================================================
# 单张图片分析
# ============================================================

def analyze_single_image(model, image_path):
    """分析单张图片并显示 Grad-CAM"""
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img).astype(np.uint8)

    img_preprocessed = efficientnet.preprocess_input(img_array.astype(np.float32))
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    heatmap, pred_label, probs = grad_cam(model, img_batch)
    overlay, heatmap_resized = overlay_heatmap(heatmap, img_array)

    plot_gradcam_sample(
        img_array, overlay, heatmap_resized,
        true_label=-1, pred_label=pred_label, probs=probs,
        save_path=os.path.join("visualization", "gradcam_single.png")
    )

    print(f"\n🔍 单张分析: {os.path.basename(image_path)}")
    print(f"  预测类别: {CLASS_NAMES[pred_label]} ({probs[pred_label]:.1%})")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        print(f"    {name}: {prob:.1%}")
    return pred_label, probs


# ============================================================
# 混淆诊断报告
# ============================================================

def generate_diagnosis_report(results):
    """基于 Grad-CAM 结果生成混淆诊断报告"""
    from collections import Counter

    wrong = results['wrong']

    # 统计混淆对
    confusion_pairs = Counter()
    confusion_details = {}  # (true, pred) -> [sample, ...]

    for s in wrong:
        pair = (CLASS_NAMES[s['true_label']], CLASS_NAMES[s['pred_label']])
        confusion_pairs[pair] += 1
        if pair not in confusion_details:
            confusion_details[pair] = []
        confusion_details[pair].append(s)

    print("\n" + "=" * 60)
    print("  🩺 混淆诊断报告")
    print("=" * 60)

    if not confusion_pairs:
        print("\n  🎉 无错误样本！")
        return

    print(f"\n  最常见的混淆对:")
    for (true_cls, pred_cls), count in confusion_pairs.most_common(5):
        pct = count / max(len(wrong), 1) * 100
        print(f"    {true_cls:25s} → {pred_cls:25s} : {count:3d} 张 ({pct:.1f}%)")

    print("\n  可能的原因分析:")
    for (true_cls, pred_cls), samples in confusion_details.items():
        print(f"\n  ⚠️  {true_cls} 被误判为 {pred_cls} ({len(samples)} 张):")
        print(f"      - 模型在预测 {true_cls} 时 P({true_cls})={np.mean([s['probs'][CLASS_NAMES.index(true_cls)] for s in samples]):.1%}")
        print(f"      - P({pred_cls})={np.mean([s['probs'][CLASS_NAMES.index(pred_cls)] for s in samples]):.1%}")
        print(f"      - 建议检查 Grad-CAM 热力图是否聚焦在正确的组织区域")


# ============================================================
# 主入口
# ============================================================

def load_best_model():
    """加载最佳模型（优先微调模型）"""
    try:
        model = load_model(MODEL_FINETUNE_PATH, compile=False)
        print(f"加载模型: {MODEL_FINETUNE_PATH}")
    except:
        print(f"未找到 {MODEL_FINETUNE_PATH}，尝试加载初始模型")
        try:
            model = load_model(MODEL_INITIAL_PATH, compile=False)
        except:
            model = load_model("best_model_initial.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM 可视化分析工具")
    parser.add_argument('--image', '-i', type=str, default=None,
                        help='分析单张图片的路径')
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='批量分析的样本数量')
    parser.add_argument('--grid', action='store_true', default=True,
                        help='保存对比网格图')
    parser.add_argument('--diagnose', action='store_true', default=True,
                        help='生成混淆诊断报告')
    args = parser.parse_args()

    # 加载模型
    print("=" * 55)
    print("  Grad-CAM 可视化分析")
    print("=" * 55)
    model = load_best_model()

    # 找到目标卷积层
    layer_name = find_last_conv_layer(model)
    print(f"  目标卷积层: {layer_name}")

    if args.image:
        # 单张分析
        analyze_single_image(model, args.image)
    else:
        # 批量分析
        results = analyze_test_set(model, num_samples=args.sample, save_grid=args.grid)

        if args.diagnose:
            generate_diagnosis_report(results)


if __name__ == "__main__":
    main()
