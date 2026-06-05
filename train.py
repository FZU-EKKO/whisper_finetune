"""
LoRA 微调 Whisper-large-v3 — 中文游戏语音识别

用法:
    python train.py --save_merged
    python train.py --epochs 10 --batch_size 8 --lr 1e-4
"""
import os, json, argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
import evaluate
import soundfile as sf
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_DIR = "./whisper-large-v3"
LANGUAGE, TASK = "zh", "transcribe"
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]


# ── 数据加载 ──────────────────────────────────────────
def load_dataset(metadata_path: str, max_samples: int = None):
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    if max_samples:
        data = data[:max_samples]
    print(f"  Loaded {len(data)} samples")

    ds = Dataset.from_list(data)
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

    def preprocess(batch):
        audio = [x["array"] for x in batch["audio_filepath"]]
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np")
        labels = processor.tokenizer(batch["text"], padding=False, return_tensors="np").input_ids
        return {"input_features": list(feats.input_features), "labels": list(labels)}

    ds = ds.map(preprocess, batched=True, batch_size=32,
                remove_columns=ds.column_names, desc="Preprocessing")
    return ds.train_test_split(test_size=0.1, seed=42)


# ── 绕过 PEFT forward 的 wrapper ──────────────────────
class WhisperPeftModel(nn.Module):
    """只覆写 forward，其余全透传给 PEFT 模型。"""
    def __init__(self, peft_model):
        super().__init__()
        self._m = peft_model

    def _cast(self, x):
        if x is not None and x.dtype != self._m.base_model.dtype:
            return x.to(self._m.base_model.dtype)
        return x

    def forward(self, input_features=None, labels=None, **kw):
        return self._m.base_model(
            input_features=self._cast(input_features), labels=labels, **kw)

    def generate(self, input_features=None, **kw):
        return self._m.base_model.generate(
            input_features=self._cast(input_features), **kw)

    def save_pretrained(self, *a, **kw):
        return self._m.save_pretrained(*a, **kw)

    def merge_and_unload(self, *a, **kw):
        return self._m.merge_and_unload(*a, **kw)

    @property
    def config(self):
        return self._m.config


# ── Data Collator ─────────────────────────────────────
@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


# ── 主函数 ────────────────────────────────────────────
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
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_merged", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # Processor
    global processor
    print(f"\nLoading processor from {args.model_dir}...")
    processor = WhisperProcessor.from_pretrained(args.model_dir, language=LANGUAGE, task=TASK)

    # 数据
    print("\nLoading data...")
    ds = load_dataset(args.metadata, args.max_samples)
    print(f"  Train: {len(ds['train'])}  Eval: {len(ds['test'])}")

    # 模型
    print(f"\nLoading model from {args.model_dir}...")
    base = WhisperForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base.model.encoder.conv1.register_forward_hook(
        lambda m, inp, out: out.requires_grad_(True))
    base.config.suppress_tokens = []

    for param in base.parameters():
        param.requires_grad = False
    peft_model = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    ))
    model = WhisperPeftModel(peft_model)
    model.print_trainable_parameters = peft_model.print_trainable_parameters
    model.print_trainable_parameters()

    # 训练参数
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        bf16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        logging_dir="./logs",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    # 指标
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        labels = pred.label_ids
        labels[labels == -100] = processor.tokenizer.pad_token_id
        refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        hyps = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=hyps, references=refs)}

    # 训练
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=Collator(processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )

    print("\nTraining...")
    trainer.train()

    # 保存
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")

    if args.save_merged:
        print("\nMerging...")
        base2 = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
        merged = PeftModel.from_pretrained(base2, args.output_dir).merge_and_unload()
        merged_path = "./whisper-large-v3-zh-lora"
        merged.save_pretrained(merged_path)
        processor.save_pretrained(merged_path)
        print(f"Merged -> {merged_path}")

    print("Done!")


if __name__ == "__main__":
    main()
