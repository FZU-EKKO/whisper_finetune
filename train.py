"""
LoRA 微调 Whisper-large-v3 for 中文游戏语音识别

用法:
    pip install -r requirements.txt
    python train.py                           # 训练
    python train.py --epochs 10 --batch_size 16 --save_merged
"""
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
import evaluate
import soundfile as sf
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_DIR = "./whisper-large-v3"
LANGUAGE, TASK = "zh", "transcribe"
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """数据整理器 - 正确处理 Whisper 的输入格式"""
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 分离输入特征和标签
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [f["labels"] for f in features]

        # 对 input_features 进行 padding
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 对 labels 进行 padding
        max_label_len = max(len(l) for l in label_features)
        labels_padded = torch.full((len(label_features), max_label_len), -100, dtype=torch.long)
        
        for i, label in enumerate(label_features):
            labels_padded[i, :len(label)] = torch.tensor(label, dtype=torch.long)

        batch["labels"] = labels_padded
        
        # 关键：不返回 decoder_input_ids，让模型自己处理
        return batch


def load_dataset(metadata_path: str, processor: WhisperProcessor, max_samples: int = None) -> DatasetDict:
    """加载和预处理数据集"""
    print(f"Loading data from {metadata_path}...")
    
    # 读取元数据
    data = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"  Loaded {len(data)} samples")
    
    # 转换为 HuggingFace Dataset
    dataset = Dataset.from_list(data)
    
    def prepare_dataset(batch):
        """预处理单个 batch"""
        # 读取音频
        audios = []
        for audio_path in batch["audio_filepath"]:
            audio, sr = sf.read(audio_path)
            # 转换为单声道
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audios.append(audio)
        
        # 提取 log-Mel 特征
        inputs = processor.feature_extractor(
            audios, 
            sampling_rate=16000, 
            return_tensors="np"
        )
        
        # Tokenize 文本
        labels = processor.tokenizer(
            batch["text"], 
            return_tensors="np", 
            padding=False,
            truncation=True,
            max_length=448
        ).input_ids
        
        return {
            "input_features": list(inputs.input_features),
            "labels": list(labels)
        }
    
    # 应用预处理
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=32,
        num_proc=1
    )
    
    # 分割训练集和验证集
    return dataset.train_test_split(test_size=0.1, seed=42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--metadata", type=str, default="./metadata.jsonl")
    parser.add_argument("--output_dir", type=str, default="./whisper-lora-checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_merged", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    args = parser.parse_args()
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 加载 processor
    print(f"\nLoading processor from {args.model_dir}...")
    processor = WhisperProcessor.from_pretrained(
        args.model_dir,
        language=LANGUAGE,
        task=TASK
    )
    
    # 加载数据集
    print("\nLoading dataset...")
    dataset = load_dataset(args.metadata, processor, args.max_samples)
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['test'])}")
    
    # 加载模型
    print(f"\nLoading model from {args.model_dir}...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
    
    # 配置生成参数
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # 梯度检查点需要
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 配置 LoRA
    print("\nSetting up LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params:.2f}% of {total_params:,})")

    # PEFT 内部会强传 input_ids 给 base_model，Whisper 只吃 input_features+labels
    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, input_features=None, labels=None, **kw):
            return self._m(input_features=input_features, labels=labels)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._m, name)

    model = Wrap(model)
    
    # 数据整理器
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # WER 评估
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        """计算 WER"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # 替换 -100 为 pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # 解码
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # 计算 WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    
    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        num_train_epochs=args.epochs,
        fp16=device == "cuda",
        predict_with_generate=True,
        generation_max_length=448,
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        gradient_checkpointing=True,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )
    
    # 开始训练
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()
    
    # 保存 LoRA 权重
    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # 合并并保存完整模型
    if args.save_merged:
        print("\nMerging LoRA weights with base model...")
        merged_model = model.merge_and_unload()
        merged_path = "./whisper-large-v3-zh-lora"
        merged_model.save_pretrained(merged_path)
        processor.save_pretrained(merged_path)
        print(f"Merged model saved to {merged_path}")
    
    # 最终评估
    print("\nFinal evaluation...")
    results = trainer.evaluate()
    print(f"\nFinal WER: {results['eval_wer']:.4f}")
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()