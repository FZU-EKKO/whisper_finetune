"""
LoRA 微调 Whisper-large-v3 — 中文游戏语音识别

用法:
    python train.py --save_merged
    python train.py --epochs 10 --batch_size 8 --lr 1e-4
"""
import os, json, argparse, math, time
from dataclasses import dataclass

import torch
import torch.nn as nn
import evaluate
import soundfile as sf
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_DIR = "./whisper-large-v3"
LANGUAGE, TASK = "zh", "transcribe"
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]


# ── 自定义 Dataset ────────────────────────────────────
class WhisperDataset(TorchDataset):
    def __init__(self, metadata_path: str, max_samples: int = None):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(l) for l in f if l.strip()]
        if max_samples:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio, sr = sf.read(item["audio_filepath"])
        if audio.ndim > 1:
            audio = audio[:, 0]
        return {"audio": audio, "text": item["text"]}


# ── Data Collator ─────────────────────────────────────
@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, batch):
        audios = [b["audio"] for b in batch]
        texts = [b["text"] for b in batch]

        feats = self.processor.feature_extractor(
            audios, sampling_rate=16000, return_tensors="np")
        labels = self.processor.tokenizer(
            texts, padding=True, return_tensors="pt")

        # 将 pad token 替换为 -100
        labels["input_ids"] = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), -100)

        return {
            "input_features": torch.tensor(feats.input_features),
            "labels": labels["input_ids"],
        }


# ── 训练 ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=MODEL_DIR)
    p.add_argument("--metadata", default="./metadata.jsonl")
    p.add_argument("--output_dir", default="./whisper-lora-checkpoint")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_merged", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # Processor
    print(f"\nLoading processor from {args.model_dir}...")
    processor = WhisperProcessor.from_pretrained(args.model_dir, language=LANGUAGE, task=TASK)

    # 数据
    print("\nLoading data...")
    full_ds = WhisperDataset(args.metadata, args.max_samples)
    n = len(full_ds)
    split = int(n * 0.9)
    train_ds, eval_ds = torch.utils.data.random_split(
        full_ds, [split, n - split], generator=torch.Generator().manual_seed(42))
    print(f"  Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=Collator(processor), num_workers=4,
                              pin_memory=(device.type == "cuda"))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=Collator(processor), num_workers=2,
                             pin_memory=(device.type == "cuda"))

    # 模型 + LoRA
    print(f"\nLoading model from {args.model_dir}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_dir,
        low_cpu_mem_usage=True)

    # Encoder 梯度 hook
    model.model.encoder.conv1.register_forward_hook(
        lambda m, inp, out: out.requires_grad_(True))

    model.config.suppress_tokens = []

    for param in model.parameters():
        param.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    ))
    model.to(device)

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    to = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {tr:,} / {to:,} ({100*tr/to:.2f}%)")

    print("Setting up optimizer & metrics...")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // 1) * args.epochs  # grad_accum=1
    warmup = args.warmup_steps

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, 1.0 - (step - warmup) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # WER 指标
    wer_metric = evaluate.load("wer")

    def compute_wer(model):
        model.eval()
        preds, refs = [], []
        with torch.no_grad():
            for batch in eval_loader:
                feats = batch["input_features"].to(device, dtype=model.dtype)
                labels = batch["labels"]
                labels[labels == -100] = processor.tokenizer.pad_token_id

                mask = torch.ones(feats.shape[:2], device=device, dtype=torch.long)
                out = model.base_model.generate(
                    feats, attention_mask=mask,
                    language=LANGUAGE, task=TASK, max_length=225)
                preds.extend(processor.tokenizer.batch_decode(out, skip_special_tokens=True))
                refs.extend(processor.tokenizer.batch_decode(labels, skip_special_tokens=True))
        model.train()
        return wer_metric.compute(predictions=preds, references=refs)

    # 训练循环
    print(f"\nTraining {args.epochs} epochs, {len(train_loader)} steps/epoch")
    print("=" * 55)
    best_wer = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            feats = batch["input_features"].to(device, dtype=model.dtype)
            labels = batch["labels"].to(device)

            # 直接调 base_model，绕过 PEFT forward 的 input_ids 参数
            loss = model.base_model(input_features=feats, labels=labels).loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                print(f"  step {global_step:5d} | loss {loss.item():.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

            if global_step % args.eval_steps == 0:
                wer = compute_wer(model)
                print(f"  --- eval @ step {global_step} ---")
                print(f"  WER: {wer:.4f} (best: {best_wer:.4f})")
                if wer < best_wer:
                    best_wer = wer
                    model.save_pretrained(args.output_dir)
                    processor.save_pretrained(args.output_dir)
                    print(f"  -> saved best to {args.output_dir}")

        # Epoch 结束
        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        wer = compute_wer(model)
        print(f"\nEpoch {epoch}/{args.epochs} | loss {avg_loss:.4f} | "
              f"WER {wer:.4f} | time {elapsed:.0f}s")
        if wer < best_wer:
            best_wer = wer
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"-> saved best to {args.output_dir}")
        print("=" * 55)

    print(f"\nBest WER: {best_wer:.4f}")
    print(f"Model saved to {args.output_dir}")

    # 合并
    if args.save_merged:
        print("\nMerging LoRA...")
        base = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
        m = PeftModel.from_pretrained(base, args.output_dir).merge_and_unload()
        merged_path = "./whisper-large-v3-zh-lora"
        m.save_pretrained(merged_path)
        processor.save_pretrained(merged_path)
        print(f"Merged -> {merged_path}")

    print("Done!")


if __name__ == "__main__":
    main()
