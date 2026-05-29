# config.py
import os

# 数据集根目录（请修改为你的实际路径）
DATA_DIR = "./chest-ctscan-images"

# 图像尺寸与批次大小
IMG_SIZE = 224
BATCH_SIZE = 32

# 类别名称（根据文件夹名）
CLASS_NAMES = ['adenocarcinoma', 'large_cell_carcinoma', 'squamous_cell_carcinoma', 'normal']
NUM_CLASSES = len(CLASS_NAMES)

# 训练轮数
EPOCHS_INITIAL = 30    # 第一阶段（冻结骨干）
EPOCHS_FINETUNE = 15   # 第二阶段（微调）

# 学习率
INITIAL_LR = 1e-3
FINETUNE_LR = 1e-5

# 模型保存路径
MODEL_INITIAL_PATH = "best_model_initial.h5"
MODEL_FINETUNE_PATH = "best_model_finetune.h5"

# 随机种子（保证可重复性）
RANDOM_SEED = 42