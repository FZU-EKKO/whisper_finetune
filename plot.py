"""
从训练日志生成 PPT 级训练曲线图。

用法:
    python plot.py                          # 默认读取 ./training_log.jsonl
    python plot.py --log ./training_log.jsonl --output ./curves/
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 中文字体 ────────────────────────────────────────────
plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# 配色
C_TRAIN = "#4A90D9"       # 训练 loss 蓝
C_TRAIN_S = "#1B5E9F"     # 平滑训练 loss 深蓝
C_EVAL_WER = "#D64545"    # WER 红
C_EVAL_LOSS = "#E8922E"   # Eval loss 橙
C_LR = "#3DA63D"          # LR 绿
C_BG = "#F7F8FA"          # 背景灰白


def load_log(log_path: str) -> dict:
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    train = [e for e in entries if e["type"] == "train"]
    evals = [e for e in entries if e["type"] == "eval"]

    return {
        "train_steps": np.array([e["step"] for e in train], dtype=int),
        "train_loss":  np.array([e["loss"] for e in train], dtype=float),
        "train_lr":    np.array([e.get("lr", 0) for e in train], dtype=float),
        "eval_steps":  np.array([e["step"] for e in evals], dtype=int),
        "eval_wer":    np.array([e["wer"] for e in evals], dtype=float),
        "eval_loss":   [(e["step"], e["epoch_loss"]) for e in evals if "epoch_loss" in e],
    }


def smooth(data, window=50):
    if len(data) < window:
        return data, np.arange(len(data))
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode="valid")
    offset = (len(data) - len(smoothed)) // 2
    return smoothed, np.arange(offset, offset + len(smoothed))


def plot_dashboard(data: dict, output_dir: str):
    """一张综合大图，适合 PPT 单页展示"""
    fig = plt.figure(figsize=(16, 10), facecolor="white")

    # ── 左栏：Loss（占 2/3 宽度，上半部分）──
    ax_loss = fig.add_axes([0.07, 0.55, 0.55, 0.38])
    ax_loss.set_facecolor(C_BG)

    # 平滑曲线
    if len(data["train_loss"]) > 50:
        s, s_steps = smooth(data["train_loss"], window=50)
        ax_loss.plot(data["train_steps"][s_steps], s,
                     color=C_TRAIN_S, linewidth=1.2, label="Train Loss (smoothed)")
    # 原始半透明
    ax_loss.plot(data["train_steps"], data["train_loss"],
                 color=C_TRAIN, linewidth=0.3, alpha=0.25)
    # Eval loss 标记
    if data["eval_loss"]:
        esteps, eloss = zip(*data["eval_loss"])
        ax_loss.plot(esteps, eloss, "D", color=C_EVAL_LOSS,
                     markersize=7, markeredgewidth=0.5, markeredgecolor="white",
                     label="Epoch Avg Loss", zorder=5)

    ax_loss.set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax_loss.set_xlabel("Step", fontsize=12)
    ax_loss.set_xlim(left=0)
    ax_loss.set_ylim(bottom=0)
    ax_loss.grid(True, alpha=0.35, linewidth=0.5)
    ax_loss.legend(loc="upper right", fontsize=10, framealpha=0.85)
    ax_loss.set_title("Training & Evaluation Loss", fontsize=15, fontweight="bold", loc="left")

    # ── 右栏：WER（占 1/3 宽度，上半部分）──
    ax_wer = fig.add_axes([0.67, 0.55, 0.29, 0.38])
    ax_wer.set_facecolor(C_BG)

    ax_wer.plot(data["eval_steps"], data["eval_wer"], "o-",
                color=C_EVAL_WER, linewidth=2, markersize=7,
                markerfacecolor="white", markeredgewidth=1.5, zorder=4)
    ax_wer.set_ylabel("WER", fontsize=14, fontweight="bold", color=C_EVAL_WER)
    ax_wer.set_xlabel("Step", fontsize=12)
    ax_wer.set_xlim(left=0)
    ax_wer.set_ylim(bottom=0)
    ax_wer.tick_params(axis="y", labelcolor=C_EVAL_WER)
    ax_wer.grid(True, alpha=0.35, linewidth=0.5)

    # Best WER 标注
    if len(data["eval_wer"]) > 0:
        best_idx = np.argmin(data["eval_wer"])
        best_step, best_wer = int(data["eval_steps"][best_idx]), data["eval_wer"][best_idx]
        ax_wer.annotate(
            f"  Best: {best_wer:.4f}\n  Step {best_step}",
            xy=(best_step, best_wer),
            xytext=(best_step * 0.7, best_wer + 0.25),
            arrowprops=dict(arrowstyle="->", color=C_EVAL_WER, lw=1.5,
                            connectionstyle="arc3,rad=0.3"),
            fontsize=11, fontweight="bold", color=C_EVAL_WER,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF0F0",
                      edgecolor=C_EVAL_WER, alpha=0.9),
        )
    ax_wer.set_title("WER (Word Error Rate)", fontsize=15, fontweight="bold", loc="left",
                     color=C_EVAL_WER)

    # ── 下半：LR Schedule ──
    ax_lr = fig.add_axes([0.07, 0.08, 0.89, 0.36])
    ax_lr.set_facecolor(C_BG)

    ax_lr.plot(data["train_steps"], data["train_lr"],
               color=C_LR, linewidth=1.2)
    ax_lr.fill_between(data["train_steps"], 0, data["train_lr"],
                       color=C_LR, alpha=0.08)
    ax_lr.set_ylabel("Learning Rate", fontsize=14, fontweight="bold", color=C_LR)
    ax_lr.set_xlabel("Step", fontsize=12)
    ax_lr.set_xlim(left=0)
    ax_lr.set_ylim(bottom=0)
    ax_lr.tick_params(axis="y", labelcolor=C_LR)
    ax_lr.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))
    ax_lr.grid(True, alpha=0.35, linewidth=0.5)

    # Warmup 标记
    warmup_steps = None
    for i, lr in enumerate(data["train_lr"]):
        if i > 0 and lr < data["train_lr"][i - 1]:
            warmup_steps = data["train_steps"][i - 1]
            break
    if warmup_steps:
        ax_lr.axvline(x=warmup_steps, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax_lr.text(warmup_steps + 50, np.max(data["train_lr"]) * 0.95,
                   f"Warmup\n{int(warmup_steps)} steps",
                   fontsize=9, color="gray", verticalalignment="top")
    ax_lr.set_title("Learning Rate Schedule (Warmup + Linear Decay)", fontsize=15,
                    fontweight="bold", loc="left", color=C_LR)

    # ── 总标题 ──
    fig.suptitle("Whisper-large-v3 LoRA Fine-tuning — Training Curves",
                 fontsize=18, fontweight="bold", y=0.99)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_dashboard.png")
    fig.savefig(path, facecolor="white")
    print(f"Dashboard saved → {path}")
    plt.close(fig)


def plot_loss_detail(data: dict, output_dir: str):
    """单独的 Loss 曲线图（适合放大展示）"""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_facecolor(C_BG)

    # 原始 loss 散点（下采样避免文件过大）
    raw = data["train_loss"]
    ds = max(1, len(raw) // 5000)
    ax.scatter(data["train_steps"][::ds], raw[::ds],
               s=1, color=C_TRAIN, alpha=0.2, label="Train Loss (per step)")

    # 平滑曲线
    if len(raw) > 50:
        s, s_steps = smooth(raw, window=100)
        ax.plot(data["train_steps"][s_steps], s,
                color=C_TRAIN_S, linewidth=1.8, label="Train Loss (smoothed, window=100)")

    # Eval loss
    if data["eval_loss"]:
        esteps, eloss = zip(*data["eval_loss"])
        ax.plot(esteps, eloss, "D-", color=C_EVAL_LOSS, linewidth=2,
                markersize=8, markeredgewidth=0.8, markeredgecolor="white",
                label="Epoch Avg Loss", zorder=5)

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=11, markerscale=1.5)
    ax.set_title("Training Loss — Whisper-large-v3 LoRA Fine-tuning",
                 fontsize=16, fontweight="bold")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path, facecolor="white")
    print(f"Loss curve saved → {path}")
    plt.close(fig)


def plot_wer_detail(data: dict, output_dir: str):
    """单独的 WER 曲线图"""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_facecolor(C_BG)

    ax.plot(data["eval_steps"], data["eval_wer"], "o-",
            color=C_EVAL_WER, linewidth=2.5, markersize=9,
            markerfacecolor="white", markeredgewidth=2, zorder=4)

    # 标注每个点的值
    for i in range(len(data["eval_steps"])):
        ax.annotate(
            f"{data['eval_wer'][i]:.4f}",
            (data["eval_steps"][i], data["eval_wer"][i]),
            textcoords="offset points", xytext=(0, 12),
            fontsize=8, color=C_EVAL_WER, ha="center",
        )

    # Best
    if len(data["eval_wer"]) > 0:
        best_idx = np.argmin(data["eval_wer"])
        best_step, best_wer = int(data["eval_steps"][best_idx]), data["eval_wer"][best_idx]
        ax.axhline(y=best_wer, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.annotate(
            f"Best WER: {best_wer:.4f} @ step {best_step}",
            xy=(best_step, best_wer),
            xytext=(data["eval_steps"][len(data["eval_steps"]) // 2], best_wer + 0.08),
            arrowprops=dict(arrowstyle="->", color=C_EVAL_WER, lw=1.8),
            fontsize=13, fontweight="bold", color=C_EVAL_WER,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=C_EVAL_WER, alpha=0.92),
        )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("WER", fontsize=13, fontweight="bold", color=C_EVAL_WER)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelcolor=C_EVAL_WER)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title("WER (Word Error Rate) — Whisper-large-v3 LoRA Fine-tuning",
                 fontsize=16, fontweight="bold")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "wer_curve.png")
    fig.savefig(path, facecolor="white")
    print(f"WER curve saved → {path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Generate training curves from log")
    p.add_argument("--log", default="./training_log.jsonl", help="训练日志文件")
    p.add_argument("--output", default="./curves", help="输出图片目录")
    args = p.parse_args()

    if not os.path.exists(args.log):
        print(f"❌ 日志文件不存在: {args.log}")
        print("   请先运行 train.py 生成训练日志")
        return

    print(f"Loading log: {args.log}")
    data = load_log(args.log)
    print(f"  Train entries: {len(data['train_steps'])}")
    print(f"  Eval  entries: {len(data['eval_steps'])}")
    if len(data["eval_wer"]) > 0:
        print(f"  Best WER: {np.min(data['eval_wer']):.4f} @ step "
              f"{data['eval_steps'][np.argmin(data['eval_wer'])]}")

    # 生成三张图
    plot_dashboard(data, args.output)    # PPT 单页综合图
    plot_loss_detail(data, args.output)  # Loss 特写
    plot_wer_detail(data, args.output)   # WER 特写

    print(f"\n✅ 全部图片已保存到 {args.output}/")


if __name__ == "__main__":
    main()
