"""
绘制训练过程的 loss 和 WER 曲线

用法:
    python plot.py                     # 读取 ./checkpoint 下的训练日志
    python plot.py --logdir ./logs     # 从 TensorBoard 日志绘制
    python plot.py --checkpoint ./checkpoint  # 从 checkpoint trainer_state 绘制

输出:
    training_curve.png  — loss + WER 双轴曲线图
"""
import argparse
import json
import os
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_from_trainer_state(checkpoint_dir: str) -> dict:
    """从 trainer_state.json 提取训练日志。"""
    # 尝试先读主 checkpoint 目录，再尝试子目录
    candidates = [
        os.path.join(checkpoint_dir, "trainer_state.json"),
    ]
    # 也搜索 checkpoint-XXX 子目录
    for d in sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))):
        candidates.append(os.path.join(d, "trainer_state.json"))

    log_history = []
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
                log_history.extend(state.get("log_history", []))

    if not log_history:
        raise FileNotFoundError(
            f"找不到训练日志，请确认 {checkpoint_dir} 下有 trainer_state.json\n"
            f"已检查: {candidates}"
        )

    # 分离 train 和 eval 记录
    train_steps, train_loss = [], []
    eval_steps, eval_wer = [], []
    eval_loss = []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            step = entry.get("step", len(train_steps))
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_wer" in entry:
            step = entry.get("step", len(eval_steps))
            eval_steps.append(step)
            eval_wer.append(entry["eval_wer"])
            if "eval_loss" in entry:
                eval_loss.append((step, entry["eval_loss"]))

    return {
        "train_steps": np.array(train_steps),
        "train_loss": np.array(train_loss),
        "eval_steps": np.array(eval_steps),
        "eval_wer": np.array(eval_wer),
        "eval_loss": eval_loss,
    }


def smooth(data, window=10):
    """滑动窗口平滑。"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot(data: dict, output_path: str = "training_curve.png"):
    """绘制双轴曲线：loss（左轴）+ WER（右轴）。"""
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # --- 左轴：Loss ---
    color_loss = "#2196F3"
    ax1.set_xlabel("Step", fontsize=13)
    ax1.set_ylabel("Loss", color=color_loss, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    # 训练 loss（平滑后）
    if len(data["train_loss"]) > 10:
        smoothed = smooth(data["train_loss"], window=20)
        smooth_steps = data["train_steps"][len(data["train_steps"]) - len(smoothed):]
        ax1.plot(smooth_steps, smoothed, color=color_loss, linewidth=0.8, alpha=0.35)

    # 原始训练 loss（半透明）
    ax1.plot(data["train_steps"], data["train_loss"],
             color=color_loss, linewidth=0.6, alpha=0.15, label="Train Loss (raw)")

    # 验证 loss
    if data["eval_loss"]:
        e_steps, e_loss = zip(*data["eval_loss"])
        ax1.plot(e_steps, e_loss, "o-", color="#FF6F00", linewidth=2, markersize=5,
                 label="Eval Loss", zorder=5)

    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # --- 右轴：WER ---
    if len(data["eval_wer"]) > 0:
        ax2 = ax1.twinx()
        color_wer = "#E53935"
        ax2.set_ylabel("WER (字错误率)", color=color_wer, fontsize=13)
        ax2.tick_params(axis="y", labelcolor=color_wer)
        ax2.plot(data["eval_steps"], data["eval_wer"], "s-",
                 color=color_wer, linewidth=2, markersize=6,
                 label="Eval WER", zorder=5)
        ax2.set_ylim(bottom=0)

        # 标注最佳 WER
        best_idx = np.argmin(data["eval_wer"])
        best_step, best_wer = data["eval_steps"][best_idx], data["eval_wer"][best_idx]
        ax2.annotate(f"Best WER: {best_wer:.4f}\nStep {best_step}",
                     xy=(best_step, best_wer), xytext=(best_step, best_wer + 0.15),
                     arrowprops=dict(arrowstyle="->", color=color_wer),
                     fontsize=12, fontweight="bold", color=color_wer,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", alpha=0.8))

    # --- 标题 & 图例 ---
    ax1.set_title("Whisper-large-v3 LoRA 训练曲线", fontsize=16, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if len(data["eval_wer"]) > 0:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)
    else:
        ax1.legend(loc="upper right", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"📈 训练曲线已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="绘制训练曲线")
    parser.add_argument("--checkpoint", default="./checkpoint", help="checkpoint 目录")
    parser.add_argument("--output", default="./training_curve.png", help="输出图片路径")
    args = parser.parse_args()

    print("📊 读取训练日志...")
    data = load_from_trainer_state(args.checkpoint)

    print(f"  Train loss 记录: {len(data['train_loss'])} 条")
    print(f"  Eval WER 记录:   {len(data['eval_wer'])} 条")
    if len(data["eval_wer"]) > 0:
        print(f"  Best WER:        {np.min(data['eval_wer']):.4f} @ step {data['eval_steps'][np.argmin(data['eval_wer'])]}")

    plot(data, args.output)


if __name__ == "__main__":
    main()
