"""
LoRA 微调 Whisper-large-v3 — 中文游戏语音识别

用法:
    python train.py --save_merged
    python train.py --epochs 10 --batch_size 8 --lr 1e-4
"""
import os, json, argparse, time, math
from dataclasses import dataclass

import torch
import evaluate
import soundfile as sf
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_DIR = "./whisper-large-v3"
LANGUAGE, TASK = "zh", "transcribe"
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]


# ── Dataset ───────────────────────────────────────────
class WhisperDataset(TorchDataset):
    def __init__(self, metadata_path, max_samples=None):
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


# ── Collator ──────────────────────────────────────────
@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, batch):
        audio = [b["audio"] for b in batch]
        text = [b["text"] for b in batch]
        feats = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np")
        labels = self.processor.tokenizer(text, padding=True, return_tensors="pt")
        labels = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        return {"input_features": torch.tensor(feats.input_features), "labels": labels}


# ── 训练 ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=MODEL_DIR)
    p.add_argument("--metadata", default="./metadata.jsonl")
    p.add_argument("--output_dir", default="./whisper-lora-checkpoint")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_merged", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # Processor
    print(f"\nLoading processor from {args.model_dir}...")
    processor = WhisperProcessor.from_pretrained(args.model_dir, language=LANGUAGE, task=TASK)

    # Data
    print("\nLoading data...")
    full = WhisperDataset(args.metadata, args.max_samples)
    n = len(full)
    split = int(n * 0.9)
    train_ds, eval_ds = torch.utils.data.random_split(
        full, [split, n - split],
        generator=torch.Generator().manual_seed(42))
    print(f"  Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=Collator(processor), num_workers=4,
                              pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=Collator(processor), num_workers=2,
                             pin_memory=True)

    # Model
    print(f"\nLoading model from {args.model_dir}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.model.encoder.conv1.register_forward_hook(
        lambda m, inp, out: out.requires_grad_(True))
    model.config.suppress_tokens = []
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    ))
    model.to(device)

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    to = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {tr:,} / {to:,} ({100*tr/to:.2f}%)")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs

    def lr_lambda(s):
        if s < args.warmup_steps:
            return s / max(1, args.warmup_steps)
        return max(0.0, 1.0 - (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # WER
    wer_metric = evaluate.load("wer")

    def eval_wer():
        model.eval()
        preds, refs = [], []
        with torch.no_grad():
            for batch in eval_loader:
                feats = batch["input_features"].to(device, dtype=dtype)
                labels = batch["labels"]
                labels[labels == -100] = processor.tokenizer.pad_token_id
                out = model.base_model.generate(
                    feats, language=LANGUAGE, task=TASK, max_length=225)
                preds.extend(processor.tokenizer.batch_decode(out, skip_special_tokens=True))
                refs.extend(processor.tokenizer.batch_decode(labels, skip_special_tokens=True))
        model.train()
        return wer_metric.compute(predictions=preds, references=refs)

    # Training
    print(f"\nTraining {args.epochs} epochs, {len(train_loader)} steps/epoch")
    best_wer = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            feats = batch["input_features"].to(device, dtype=dtype)
            labels = batch["labels"].to(device)

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
                wer = eval_wer()
                print(f"  --- eval @ step {global_step} ---")
                print(f"  WER: {wer:.4f}  (best: {best_wer:.4f})")
                if wer < best_wer:
                    best_wer = wer
                    model.save_pretrained(args.output_dir)
                    processor.save_pretrained(args.output_dir)
                    print(f"  -> saved best to {args.output_dir}")

        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        wer = eval_wer()
        print(f"\nEpoch {epoch}/{args.epochs} | loss {avg_loss:.4f} | "
              f"WER {wer:.4f} | time {elapsed:.0f}s")
        if wer < best_wer:
            best_wer = wer
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"-> saved best to {args.output_dir}")
        print(f"{'='*50}")

    print(f"\nBest WER: {best_wer:.4f}")

    # Merge
    if args.save_merged:
        print("\nMerging...")
        base = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
        merged = PeftModel.from_pretrained(base, args.output_dir).merge_and_unload()
        merged_path = "./whisper-large-v3-zh-lora"
        merged.save_pretrained(merged_path)
        processor.save_pretrained(merged_path)
        print(f"Merged -> {merged_path}")

    print("Done!")


if __name__ == "__main__":
    main()
