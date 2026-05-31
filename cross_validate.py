"""
交叉验证 + 最终评估入口

流程:
  1. K-Fold 分层交叉验证（train+valid 合并，K 折）
  2. 使用全部 train+valid 数据在完整模型上重新训练
  3. 在 hold-out 测试集上做最终评估
  4. 保存日志和结果
"""
from train import train, train_cross_validation
from evaluate import evaluate
from log_training import save_training_log

if __name__ == "__main__":
    # ============================
    # 第一步：K-Fold 交叉验证
    # ============================
    fold_accuracies, mean_acc, std_acc = train_cross_validation()

    # ============================
    # 第二步：完整训练（全部 train+valid）
    # ============================
    print("\n" + "="*65)
    print("  🏋️  完整训练（使用全部 train+valid 数据）")
    print("="*65)
    history_initial, history_finetune = train()

    # ============================
    # 第三步：测试集最终评估
    # ============================
    print("\n" + "="*65)
    print("  📊 测试集最终评估")
    print("="*65)
    eval_results = evaluate()

    # ============================
    # 第四步：保存日志
    # ============================
    # 将交叉验证结果也写入评估记录
    if eval_results:
        eval_results["cross_validation"] = {
            "n_folds": len(fold_accuracies),
            "fold_accuracies": [round(float(a), 4) for a in fold_accuracies],
            "mean_accuracy": round(float(mean_acc), 4),
            "std_accuracy": round(float(std_acc), 4),
        }
    save_training_log(history_initial, history_finetune, eval_results)

    print("\n✅ 交叉验证 + 训练 + 评估全部完成！详情请查看 training_logs/ 目录")
