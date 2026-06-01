# Chest Cancer Classification — EfficientNetB0

基于 TensorFlow / Keras 的肺癌组织病理图像分类项目，使用 EfficientNetB0 迁移学习进行四分类：**腺癌 (Adenocarcinoma)**、**大细胞癌 (Large Cell Carcinoma)**、**鳞状细胞癌 (Squamous Cell Carcinoma)** 和 **正常组织 (Normal)**。

---

## 目录结构

```
Chest_cancer/
├── Data/                     # 数据集（已添加到 .gitignore）
│   ├── train/                # 训练集（按病例文件夹组织）
│   ├── valid/                # 验证集
│   └── test/                 # 测试集（按类别文件夹组织）
├── .github/skills/           # VS Code Copilot 技能
├── .gitignore
├── config.py                 # 配置文件
├── data_loader.py            # 数据加载与增强
├── model.py                  # 模型定义
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── log_training.py           # 训练日志工具
├── main.py                   # 主入口
├── README.md                 # 本文件
└── training_logs/            # 训练日志输出目录
```

## 环境配置

### 系统要求
- Python 3.10+
- TensorFlow 2.16+ (Keras 3)
- 推荐 GPU (项目已在 CPU 上测试通过)

### 虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 安装依赖
pip install tensorflow matplotlib seaborn scikit-learn pandas numpy
```

### VS Code 配置

1. 选择 Python 解释器为 `.venv\Scripts\python.exe`
2. Pylance 可能对 `tf.keras` 的动态导入报错，本项目已改为直接导入 `keras`：
   ```python
   from keras.applications import efficientnet
   from keras.src.legacy.preprocessing.image import ImageDataGenerator
   ```

## 数据集

### 目录结构要求

```
Data/
├── train/                    # 训练集文件夹名可含临床信息
│   ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
│   ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
│   ├── normal/
│   └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/
├── valid/                    # 验证集，结构同 train
└── test/                     # 测试集，使用简洁文件夹名
    ├── adenocarcinoma/
    ├── large.cell.carcinoma/
    ├── normal/
    └── squamous.cell.carcinoma/
```

> ⚠️ **注意**：train/valid 和 test 的文件夹名可以不同，但必须在 `config.py` 中分别配置 `TRAIN_FOLDER_NAMES` 和 `TEST_FOLDER_NAMES`，并保持语义顺序一致。

### 数据分布

| 类别 | 训练集 | 验证集 | 测试集 | 合计 |
|------|:----:|:----:|:----:|:---:|
| adenocarcinoma | 195 | 23 | 120 | 338 |
| large.cell.carcinoma | 115 | 21 | 51 | 187 |
| squamous.cell.carcinoma | 155 | 15 | 90 | 260 |
| normal | 148 | 13 | 54 | 215 |
| **总计** | **613** | **72** | **315** | **1000** |

## 配置文件 (`config.py`)

所有可调参数集中在 `config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `IMG_SIZE` | 224 | 输入图像尺寸 |
| `BATCH_SIZE` | 32 | 批次大小 |
| `CLASS_NAMES` | [...] | 类别显示名称（用于报告） |
| `TRAIN_FOLDER_NAMES` | [...] | 训练/验证集文件夹名 |
| `TEST_FOLDER_NAMES` | [...] | 测试集文件夹名 |
| `EPOCHS_INITIAL` | 50 | 第一阶段训练轮数 |
| `EPOCHS_FINETUNE` | 30 | 第二阶段训练轮数 |
| `INITIAL_LR` | 1e-3 | 第一阶段学习率 |
| `FINETUNE_LR` | 1e-5 | 第二阶段学习率 |
| `LABEL_SMOOTHING` | 0.1 | 标签平滑系数 |
| `DROPOUT_RATE` | 0.5 | Dropout 比率 |
| `L2_REG` | 1e-4 | L2 正则化系数 |
| `CLASS_WEIGHTS` | {...} | 类别权重（缓解不平衡） |

## 训练流程

### 两阶段迁移学习

1. **第一阶段 — 冻结骨干**：冻结 EfficientNetB0 预训练权重，只训练自定义分类头（50 epochs，早停 patience=8）
2. **第二阶段 — 微调**：解冻骨干后 150+ 层，以小学习率微调（30 epochs，早停 patience=8）

### 学习率调度

采用 **Warmup + Cosine Decay** 策略：
- 前 1 个 epoch 线性 warmup
- 之后按余弦曲线衰减到 `MIN_LR`

### 数据增强

```python
ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
```

## 运行

```bash
# 完整训练 + 评估
python main.py

# 仅训练
python train.py

# 仅评估（需要已有模型）
python evaluate.py

# K-Fold 交叉验证 + 训练 + 最终测试评估
python cross_validate.py
```

## 新增功能

### 🔬 CBAM 注意力机制（缓解腺癌↔大细胞癌混淆）

在 EfficientNetB0 骨干网络输出后、GlobalAveragePooling 之前，插入了 **CBAM (Convolutional Block Attention Module)**：

```
骨干特征图 → 通道注意力 → 空间注意力 → GAP → 分类头
```

- **通道注意力**：学习"哪些特征通道更重要"（如纹理通道 vs 颜色通道）
- **空间注意力**：学习"图像的哪些区域更关键"（如细胞核区域 vs 背景）

通过 `config.py` 中的 `USE_ATTENTION = True/False` 控制开关。

### 📊 K-Fold 分层交叉验证

将 train + valid 的 685 张图像合并，按类别比例分层划分为 K 折（默认 5 折），每折独立训练并报告：

- 每折验证准确率
- 平均准确率 ± 标准差
- 95% 置信区间

运行：`python cross_validate.py`，该脚本依次执行：
1. K-Fold 交叉验证
2. 全量数据完整训练
3. 测试集最终评估
4. 日志归档（含 CV 结果）

## 评估

评估输出包括：
- **测试集准确率**（Loss + Accuracy）
- **分类报告**（Precision / Recall / F1-score 按类别）
- **混淆矩阵热力图**（保存为 `confusion_matrix.png`）
- **训练曲线**（Accuracy / Loss 随 epoch 变化）

## 日志

每次训练会自动记录到 `training_logs/` 目录：

```bash
training_logs/
├── latest.json               # 最近一次训练日志
├── train_20260529_143022.json
└── ...
```

查看历史日志：
```bash
python log_training.py
```

每条日志包含：
- 时间戳
- 完整配置参数
- 每 epoch 的训练/验证指标
- 最终评估结果（准确率、分类报告、混淆矩阵）

## 训练历史

| 训练时间 | Phase1 最佳 Val Acc | Phase2 最佳 Val Acc | 测试集准确率 | 测试集 Loss | 备注 |
|:--------:|:-------------------:|:-------------------:|:-----------:|:----------:|:----:|
| 2026-05-29 18:58 | 76.39% (42 epochs) | **83.33%** (10 epochs) | 71.11% | 0.7216 | 首次完整训练，空 history 记录 |
| 2026-05-29 23:11 | 68.06% (29 epochs) | 69.44% (9 epochs) | 73.33% | 0.7205 | 含完整 epoch 记录 |
| 2026-05-31 14:27 | 73.61% (37 epochs) | 75.00% (9 epochs) | **75.24%** | **0.6839** | 🏆 **当前最佳** |

### 各次训练详细结果

#### ① 2026-05-29 18:58 — 首次训练

- **Phase1**：冻结骨干，训练 42 epochs（早停触发），最佳 Val Acc = **76.39%**
- **Phase2**：微调，训练 10 epochs，最佳 Val Acc = **83.33%**（最高验证准确率）
- **测试集**：Acc = **71.11%**，Loss = 0.7216
- **问题**：验证准确率高但测试准确率较低，存在一定过拟合；大细胞癌召回率 96% 但精确率仅 47%

#### ② 2026-05-29 23:11 — 第二次训练

- **Phase1**：冻结骨干，训练 29 epochs（早停触发），最佳 Val Acc = **68.06%**
- **Phase2**：微调，训练 9 epochs，最佳 Val Acc = **69.44%**
- **测试集**：Acc = **73.33%**，Loss = 0.7205
- **改进**：测试准确率提升 ~2%，腺癌召回率从 56% 提升至 80%
- **混淆矩阵亮点**：normal 类 54 张全部正确，仅 1 张误判

#### ③ 2026-05-31 14:27 — 当前最佳 🏆

- **Phase1**：冻结骨干，训练 37 epochs（早停触发），最佳 Val Acc = **73.61%**
- **Phase2**：微调，训练 9 epochs，最佳 Val Acc = **75.00%**
- **测试集**：Acc = **75.24%**，Loss = **0.6839** ✅
- **混淆矩阵**：
  - normal：53/54 正确（**98.1%**）
  - adenocarcinoma：85/120 正确（70.8%）
  - large.cell.carcinoma：45/51 正确（88.2%）
  - squamous.cell.carcinoma：54/90 正确（60.0%）

### 趋势分析

| 指标 | 趋势 |
|:----|:----:|
| 测试准确率 | 71.11% → 73.33% → **75.24%** 📈 |
| 测试 Loss | 0.7216 → 0.7205 → **0.6839** 📉 |
| normal 类 F1 | 保持 **0.99** 稳定 |
| 腺癌 Recall | 55.8% → 80.0% → 70.8%（波动） |
| 大细胞癌 F1 | 0.63 → 0.65 → **0.68** 📈 |

### 当前瓶颈

- **腺癌 (Adenocarcinoma) ↔ 鳞状细胞癌 (Squamous Cell Carcinoma) 混淆**：仍有 20/120 腺癌被误判为鳞癌
- **大细胞癌精确率偏低**：55.6% 的精确率说明存在较多假阳性
- **各次训练结果存在波动**：CBAM + 数据增强已缓解但仍不完全稳定

## 已知问题

1. **无 GPU 支持**：当前在 CPU 上运行，训练较慢。若使用 GPU，TensorFlow 需通过 WSL2 或 TensorFlow-DirectML 插件。
2. ~~**类别混淆**：Adenocarcinoma 和 Large Cell Carcinoma 之间仍存在一定混淆（34/120 的腺癌被误判为大细胞癌）。~~ → **已引入 CBAM 注意力机制缓解**
3. ~~**未做交叉验证**：仅依赖单次 train/valid 划分评估性能。~~ → **已实现 K-Fold 分层交叉验证**

## 许可

仅供学习研究使用。
