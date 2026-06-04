"""
Plot training loss and WER curves.

Usage:
    python plot.py
    python plot.py --checkpoint ./checkpoint --output training_curve.png
"""
import argparse, json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_logs(checkpoint_dir: str) -> dict:
    candidates = [os.path.join(checkpoint_dir, "trainer_state.json")]
    for d in sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))):
        candidates.append(os.path.join(d, "trainer_state.json"))

    log_history = []
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                log_history.extend(json.load(f).get("log_history", []))

    if not log_history:
        raise FileNotFoundError(f"No trainer_state.json found in {checkpoint_dir}")

    train_steps, train_loss = [], []
    eval_steps, eval_wer, eval_loss = [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", len(train_steps)))
            train_loss.append(entry["loss"])
        if "eval_wer" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_wer.append(entry["eval_wer"])
            if "eval_loss" in entry:
                eval_loss.append((entry["step"], entry["eval_loss"]))

    return {
        "train_steps": np.array(train_steps),
        "train_loss": np.array(train_loss),
        "eval_steps": np.array(eval_steps),
        "eval_wer": np.array(eval_wer),
        "eval_loss": eval_loss,
    }


def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot(data: dict, output_path: str = "training_curve.png"):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Left axis: Loss
    c_loss = "#2196F3"
    ax1.set_xlabel("Step", fontsize=13)
    ax1.set_ylabel("Loss", color=c_loss, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=c_loss)

    if len(data["train_loss"]) > 10:
        s = smooth(data["train_loss"], window=20)
        steps = data["train_steps"][len(data["train_steps"]) - len(s):]
        ax1.plot(steps, s, color=c_loss, linewidth=0.8, alpha=0.35)

    ax1.plot(data["train_steps"], data["train_loss"],
             color=c_loss, linewidth=0.6, alpha=0.15, label="Train Loss (raw)")

    if data["eval_loss"]:
        e_steps, e_loss = zip(*data["eval_loss"])
        ax1.plot(e_steps, e_loss, "o-", color="#FF6F00", linewidth=2, markersize=5,
                 label="Eval Loss", zorder=5)

    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # Right axis: WER
    if len(data["eval_wer"]) > 0:
        ax2 = ax1.twinx()
        c_wer = "#E53935"
        ax2.set_ylabel("WER", color=c_wer, fontsize=13)
        ax2.tick_params(axis="y", labelcolor=c_wer)
        ax2.plot(data["eval_steps"], data["eval_wer"], "s-",
                 color=c_wer, linewidth=2, markersize=6, label="Eval WER", zorder=5)
        ax2.set_ylim(bottom=0)

        best_idx = np.argmin(data["eval_wer"])
        best_step, best_wer = data["eval_steps"][best_idx], data["eval_wer"][best_idx]
        ax2.annotate(f"Best WER: {best_wer:.4f} (step {best_step})",
                     xy=(best_step, best_wer), xytext=(best_step, best_wer + 0.15),
                     arrowprops=dict(arrowstyle="->", color=c_wer),
                     fontsize=12, fontweight="bold", color=c_wer,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", alpha=0.8))

    ax1.set_title("Whisper-large-v3 LoRA Training", fontsize=16, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if len(data["eval_wer"]) > 0:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)
    else:
        ax1.legend(loc="upper right", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Plot training curves")
    p.add_argument("--checkpoint", default="./checkpoint")
    p.add_argument("--output", default="./training_curve.png")
    args = p.parse_args()

    print("Loading training logs...")
    data = load_logs(args.checkpoint)

    print(f"  Train loss points: {len(data['train_loss'])}")
    print(f"  Eval WER points:   {len(data['eval_wer'])}")
    if len(data["eval_wer"]) > 0:
        print(f"  Best WER:          {np.min(data['eval_wer']):.4f} "
              f"@ step {data['eval_steps'][np.argmin(data['eval_wer'])]}")

    plot(data, args.output)


if __name__ == "__main__":
    main()
