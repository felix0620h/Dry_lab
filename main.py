"""
主入口 — 训练 + 评估 + 日志记录
"""
from train import train
from evaluate import evaluate
from log_training import save_training_log, save_evaluation_results

if __name__ == "__main__":
    # 训练
    history_initial, history_finetune = train()

    # 评估
    eval_results = evaluate()

    # 保存训练日志
    save_training_log(history_initial, history_finetune, eval_results)
    print("\n✅ 训练与评估全部完成！详情请查看 training_logs/ 目录")