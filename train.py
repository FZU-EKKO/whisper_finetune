"""
LoRA 微调 Whisper-large-v3 for 中文游戏语音识别

用法:
    pip install -r requirements.txt
    python train.py                          # 默认参数训练
    python train.py --epochs 10 --batch_size 16 --save_merged
"""
import os, json, argparse, warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch, evaluate
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ── 常量 ──────────────────────────────────────────────
MODEL_ID = "openai/whisper-large-v3"
LANGUAGE, TASK = "zh", "transcribe"
TARGET_MODULES = ["q_proj", "v_proj", "out_proj"]


# ── Data Collator ─────────────────────────────────────
@dataclass
class DataCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[dict]) -> dict:
        inputs = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        inputs["labels"] = labels
        return inputs


# ── 数据加载 ──────────────────────────────────────────
def load_data(metadata: str, processor: WhisperProcessor, max_samples: int = None) -> Dataset:
    with open(metadata, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if max_samples:
        data = data[:max_samples]
    print(f"  加载 {len(data)} 条数据")

    ds = Dataset.from_list(data)
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
    ds = ds.rename_column("audio_filepath", "audio").rename_column("text", "sentence")

    def preprocess(batch):
        audio = [x["array"] for x in batch["audio"]]
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np")
        labels = processor.tokenizer(batch["sentence"], return_tensors="np").input_ids
        return {"input_features": list(feats.input_features), "labels": list(labels)}

    return ds.map(preprocess, batched=True, batch_size=32,
                  remove_columns=ds.column_names, desc="预处理")


# ── 模型查找 ──────────────────────────────────────────
def resolve_model(model_id: str) -> str:
    """优先本地: ./whisper-large-v3 → HF缓存 → 在线下载"""
    if os.path.isdir(model_id):
        return model_id
    for path in ["whisper-large-v3", "../whisper-large-v3"]:
        if os.path.isdir(path) and os.path.isfile(f"{path}/config.json"):
            return path
    import glob as g
    snap = os.path.expanduser("~/.cache/huggingface/hub/models--openai--whisper-large-v3/snapshots")
    if os.path.isdir(snap):
        dirs = sorted(g.glob(f"{snap}/*"))
        if dirs: return dirs[-1]
    return model_id


# ── 主函数 ────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="LoRA 微调 Whisper-large-v3")
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--metadata", default="./metadata.jsonl")
    p.add_argument("--output_dir", default="./checkpoint")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_merged", action="store_true")
    args = p.parse_args()

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  CPU 训练会很慢！")
    else:
        print(f"🖥  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)")

    # 模型 & Processor
    model_path = resolve_model(args.model_id)
    print(f"\n📦 模型: {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path, language=LANGUAGE, task=TASK)

    # 数据
    print("\n📂 加载数据...")
    ds = load_data(args.metadata, processor, args.max_samples)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    print(f"  训练: {len(ds['train'])} 条, 验证: {len(ds['test'])} 条")

    # 模型 + LoRA
    print("\n🤖 加载模型 + LoRA...")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    ))
    model.print_trainable_parameters()

    # 训练配置
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        fp16=(device == "cuda"),
        evaluation_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200,
        logging_steps=50, save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=225,
        report_to="tensorboard", logging_dir="./logs",
        dataloader_num_workers=0 if device == "cpu" else 2,
        remove_unused_columns=False, label_names=["labels"],
    )

    # 评估: WER
    wer = evaluate.load("wer")

    def compute_metrics(pred):
        labels = pred.label_ids
        labels[labels == -100] = processor.tokenizer.pad_token_id
        refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        hyps = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        return {"wer": wer.compute(predictions=hyps, references=refs)}

    # 训练
    trainer = Seq2SeqTrainer(
        model=model, args=train_args,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        data_collator=DataCollator(processor), compute_metrics=compute_metrics,
    )
    print("\n🚀 开始训练...")
    trainer.train()

    # 保存 LoRA
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\n💾 LoRA → {args.output_dir}")

    # 合并模型
    if args.save_merged:
        print("\n🔗 合并模型 → ./merged")
        base = WhisperForConditionalGeneration.from_pretrained(model_path)
        m = PeftModel.from_pretrained(base, args.output_dir).merge_and_unload()
        m.save_pretrained("./merged")
        processor.save_pretrained("./merged")

    # 最终评估
    r = trainer.evaluate()
    print(f"\n📊 验证 WER: {r.get('eval_wer', 'N/A'):.4f}")
    print("✅ 完成!")


if __name__ == "__main__":
    main()
