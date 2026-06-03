# config.py
import os

# 数据集根目录（请修改为你的实际路径）
DATA_DIR = r"D:\python\Chest_cancer\Data"

# 图像尺寸与批次大小
IMG_SIZE = 224
BATCH_SIZE = 32

# 类别显示名称（用于报告和可视化）
CLASS_NAMES = ['adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma', 'normal']
NUM_CLASSES = len(CLASS_NAMES)

# 各数据集的文件夹名（train/valid 和 test 文件夹名不同，但语义顺序保持一致）
TRAIN_FOLDER_NAMES = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',       # 0: adenocarcinoma
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',     # 1: large.cell.carcinoma
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa',  # 2: squamous.cell.carcinoma
    'normal'                                              # 3: normal
]
TEST_FOLDER_NAMES = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'squamous.cell.carcinoma',
    'normal'
]

# 训练轮数
EPOCHS_INITIAL = 50    # 第一阶段（冻结骨干）
EPOCHS_FINETUNE = 30   # 第二阶段（微调）

# 学习率
INITIAL_LR = 1e-3
FINETUNE_LR = 1e-5
MIN_LR = 1e-7

# 标签平滑（缓解过自信，对混淆严重的类有帮助）
LABEL_SMOOTHING = 0.1

# Dropout 率
DROPOUT_RATE = 0.5

# L2 正则化系数
L2_REG = 1e-4

# 类别权重（基于训练集样本数计算：总样本数/(类别数*该类样本数)）
# adenocarcinoma: 195, large.cell.carcinoma: 115, squamous: 155, normal: 148
CLASS_WEIGHTS = {
    0: 1.000,  # adenocarcinoma (195 样本) — 提高权重，缓解被误判为 large cell
    1: 1.100,  # large.cell.carcinoma (115 样本)
    2: 0.989,  # squamous.cell.carcinoma (155 样本)
    3: 1.035   # normal (148 样本)
}

# 模型保存路径
MODEL_INITIAL_PATH = "best_model_initial.keras"
MODEL_FINETUNE_PATH = "best_model_finetune.keras"

# 随机种子（保证可重复性）
RANDOM_SEED = 42

# ========================
# 骨干网络选择
# ========================
BACKBONE = 'b0'              # 可选: 'b0', 'b1', 'b2', 'b3' (越大越深，精度越高但更慢)
FREEZE_LAYER_FRACTION = 0.6  # 微调阶段冻结底层比例 (60% 底层 → 40% 顶层可训练)

# ========================
# 注意力机制 (CBAM)
# ========================
USE_ATTENTION = True       # 是否在骨干网络后加入 CBAM 注意力模块
CBAM_RATIO = 8             # 通道注意力压缩比

# ========================
# K-Fold 交叉验证
# ========================
N_FOLDS = 5                # 交叉验证折数
CV_EPOCHS_INITIAL = 30     # CV 第一阶段轮数（样本更多，可适当减少）
CV_EPOCHS_FINETUNE = 20    # CV 第二阶段轮数