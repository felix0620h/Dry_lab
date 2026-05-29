---
name: data-preprocessing
description: '数据预处理工作流：图像加载、数据增强、tf.data管道构建、类别不平衡处理、数据验证与可视化。Use when: 构建训练数据管道、配置ImageDataGenerator、处理类别不平衡、调试数据加载问题、构建高性能tf.data流水线、自定义数据集加载、数据质量检查与可视化。'
user-invocable: true
---

# 数据预处理工作流

## 适用场景
- 构建图像分类/分割/检测模型前的数据准备
- 配置 `ImageDataGenerator` 数据增强与归一化
- 使用 `tf.data` 构建高性能数据管道
- 处理类别不平衡问题（过采样/欠采样/类别权重）
- 自定义数据集加载（非标准目录结构）
- 数据质量检查、样本可视化、分布分析

---

## 1. 数据目录结构规范

### 标准结构（`flow_from_directory` 兼容）
```
Data/
├── train/
│   ├── class_1/   # 每个类别一个文件夹
│   ├── class_2/
│   └── ...
├── valid/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

### 配置参数（参考 `config.py`）
```python
# 数据集根目录
DATA_DIR = r"D:\path\to\Data"

# 图像尺寸与批次大小
IMG_SIZE = 224
BATCH_SIZE = 32

# 类别名称（按文件夹名，顺序必须一致）
CLASS_NAMES = ['class_a', 'class_b', 'class_c']
NUM_CLASSES = len(CLASS_NAMES)

# 随机种子（保证可重复性）
RANDOM_SEED = 42
```

---

## 2. ImageDataGenerator 配置

### 基础用法
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# 训练集：数据增强 + 归一化
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # 模型对应的预处理器
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 验证/测试集：仅归一化，不做增强
val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
```

### 关键参数详解

| 参数 | 范围 | 说明 |
|------|------|------|
| `preprocessing_function` | - | **必须**使用与骨干网络匹配的预处理器 |
| `rotation_range` | 10~30 | 随机旋转角度，过大可能产生不合理图像 |
| `width_shift_range` | 0.1~0.2 | 水平平移比例 |
| `height_shift_range` | 0.1~0.2 | 垂直平移比例 |
| `shear_range` | 0.1~0.2 | 剪切变换强度 |
| `zoom_range` | 0.1~0.3 | 随机缩放范围 |
| `horizontal_flip` | True/False | 水平翻转（医学图像谨慎使用） |
| `fill_mode` | 'nearest' | 填充模式，常用 'nearest'/'constant'/'reflect' |
| `brightness_range` | [0.8,1.2] | 亮度调整（可选） |
| `channel_shift_range` | 10~30 | 通道偏移（可选） |

### 不同模型的预处理器对照

| 模型 | 预处理器 |
|------|----------|
| EfficientNet 系列 | `tf.keras.applications.efficientnet.preprocess_input` |
| ResNet / VGG / DenseNet | `tf.keras.applications.resnet50.preprocess_input` |
| MobileNet | `tf.keras.applications.mobilenet.preprocess_input` |
| Xception | `tf.keras.applications.xception.preprocess_input` |

> ⚠️ **重要**：预处理器与模型不匹配会导致训练不收敛或准确率异常低下。

---

## 3. 高性能 tf.data 管道

当数据集较大时，`ImageDataGenerator` 可能成为性能瓶颈。推荐使用 `tf.data` 构建管道。

### 从目录加载（推荐替代 `flow_from_directory`）
```python
import tensorflow as tf

def create_tfdata_pipeline(
    data_dir, img_size, batch_size, class_names,
    is_training=True, buffer_size=1000
):
    """构建高性能 tf.data 管道"""
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=is_training,
        seed=RANDOM_SEED
    )

    # 归一化到 [0,1]
    normalization = tf.keras.layers.Rescaling(1./255)

    if is_training:
        # 数据增强层（在GPU上运行更快）
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])

        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 性能优化
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    return dataset
```

### 性能优化要点
```python
# 必须添加 prefetch 避免 GPU 等待
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# 并行数据加载
dataset = dataset.map(process_fn, num_parallel_calls=tf.data.AUTOTUNE)

# 缓存预处理结果（如果数据能放入内存）
dataset = dataset.cache()
```

---

## 4. 类别不平衡处理

### 方法一：计算类别权重（最简单）
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 统计每个类别的样本数
class_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in CLASS_NAMES]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.arange(len(CLASS_NAMES))),
    y=np.repeat(np.arange(len(CLASS_NAMES)), class_counts)
)
class_weight_dict = dict(enumerate(class_weights))

# 训练时传入
model.fit(
    train_gen,
    class_weight=class_weight_dict,  # 让少数类获得更高权重
    ...
)
```

### 方法二：自定义过采样（tf.data）
```python
def balance_dataset(dataset, class_names):
    """对少数类进行过采样"""
    # 按类别分离数据
    class_datasets = {}
    for i, name in enumerate(class_names):
        class_datasets[name] = dataset.filter(
            lambda x, y: tf.argmax(y) == i
        )
    
    # 找到最大类别样本数
    max_count = max(
        sum(1 for _ in class_datasets[name]) for name in class_names
    )
    
    # 过采样使每个类别样本数一致
    balanced_datasets = []
    for name in class_names:
        ds = class_datasets[name]
        ds = ds.repeat(int(np.ceil(max_count / len(list(ds)))))
        ds = ds.take(max_count)
        balanced_datasets.append(ds)
    
    # 合并并打乱
    balanced = tf.data.Dataset.sample_from_datasets(
        balanced_datasets, weights=[1.0/len(class_names)] * len(class_names)
    )
    return balanced
```

### 方法三：数据增强补偿
对于样本量极少的类别，使用更强的数据增强策略：
```python
# 为少数类单独配置更强的增强
heavy_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])
```

---

## 5. 自定义数据集加载

### 从 CSV / DataFrame 加载
```python
import pandas as pd
import tensorflow as tf

def load_from_csv(csv_path, img_dir, img_size, batch_size):
    """从 CSV 文件加载数据"""
    df = pd.read_csv(csv_path)
    
    def parse_image(file_path, label):
        img = tf.io.read_file(img_dir + '/' + file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = preprocess_input(img)
        return img, label
    
    file_paths = df['filename'].values
    labels = df['label'].values
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
```

### 从 NumPy 数组加载（内存数据）
```python
def load_from_numpy(images, labels, batch_size):
    """从 NumPy 数组创建数据集"""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

---

## 6. 数据验证与可视化

### 检查数据集统计信息
```python
import os
import matplotlib.pyplot as plt

def inspect_dataset(data_dir, class_names):
    """检查数据集分布"""
    print(f"{'类别':<30} {'训练':<8} {'验证':<8} {'测试':<8}")
    print("-" * 54)
    for cls in class_names:
        train_count = len(os.listdir(os.path.join(data_dir, 'train', cls)))
        valid_count = len(os.listdir(os.path.join(data_dir, 'valid', cls)))
        test_count = len(os.listdir(os.path.join(data_dir, 'test', cls)))
        print(f"{cls:<30} {train_count:<8} {valid_count:<8} {test_count:<8}")
```

### 可视化增强后的样本
```python
def visualize_augmentation(datagen, sample_dir, class_name, num_samples=9):
    """展示数据增强效果"""
    generator = datagen.flow_from_directory(
        sample_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=num_samples,
        class_mode='categorical',
        classes=[class_name],
        shuffle=True
    )
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        img_batch, _ = next(generator)
        img = img_batch[0]
        # 反归一化显示
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

---

## 7. 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 训练 Loss 不下降 | 预处理器与模型不匹配 | 确认使用正确的 `preprocess_input` |
| 验证准确率远低于训练 | 数据增强导致训练分布偏移 | 降低增强强度或检查 `fill_mode` |
| GPU 利用率低 | 数据加载是瓶颈 | 添加 `.prefetch(tf.data.AUTOTUNE)` |
| 类别不平衡导致模型偏向多数类 | 未处理不平衡 | 使用 `class_weight` 或过采样 |
| `flow_from_directory` 找不到文件 | 目录结构不匹配 | 检查文件夹名与 `CLASS_NAMES` 一致 |
| OOM 内存溢出 | batch_size 过大或图像太大 | 减小 `BATCH_SIZE` 或 `IMG_SIZE` |
| 数据增强产生无效图像 | 增强参数过大 | 减小 `rotation_range` 或 `shear_range` |

---

## 8. 参考资源

- 本项目的 `data_loader.py` — ImageDataGenerator 实现参考
- 本项目的 `config.py` — 配置参数参考
- [TensorFlow Data API 文档](https://www.tensorflow.org/guide/data)
- [TensorFlow ImageDataGenerator 文档](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
