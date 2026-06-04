"""
LoRA 微调 Whisper-large-v3 for 中文游戏语音识别

用法:
    python download_model.py                  # 1. 先下载模型
    pip install -r requirements.txt
    python train.py                           # 2. 训练
    python train.py --epochs 10 --batch_size 16 --save_merged
"""
import os, json, argparse
from dataclasses import dataclass
from typing import List

import torch, evaluate, soundfile
import torch.nn as nn
from datasets import Dataset
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_DIR = "./whisper-large-v3"
LANGUAGE, TASK = "zh", "transcribe"
TARGET_MODULES = ["q_proj", "v_proj", "out_proj"]


@dataclass
class DataCollator:
    processor: WhisperProcessor
    bos_token_id: int

    def __call__(self, features: List[dict]) -> dict:
        input_features = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt")

        max_len = max(len(l) for l in labels)
        padded = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            padded[i, :len(l)] = torch.tensor(l, dtype=torch.long)
        if (padded[:, 0] == self.bos_token_id).all():
            padded = padded[:, 1:]

        return {"input_features": batch["input_features"], "labels": padded}


def load_data(metadata: str, processor: WhisperProcessor, max_samples: int = None) -> Dataset:
    with open(metadata, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if max_samples:
        data = data[:max_samples]
    print(f"  Loaded {len(data)} samples")

    ds = Dataset.from_list(data)

    def preprocess(batch):
        audios = []
        for path in batch["audio_filepath"]:
            audio, sr = soundfile.read(path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            audios.append(audio)

        feats = processor.feature_extractor(
            audios, sampling_rate=16000, return_tensors="np")
        labels = processor.tokenizer(
            batch["text"], return_tensors="np").input_ids
        return {"input_features": list(feats.input_features), "labels": list(labels)}

    return ds.map(preprocess, batched=True, batch_size=32,
                  remove_columns=ds.column_names, desc="Preprocessing")


def main():
    p = argparse.ArgumentParser(description="LoRA finetune Whisper-large-v3")
    p.add_argument("--model_dir", default=MODEL_DIR)
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

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: training on CPU will be very slow!")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # Model & Processor
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        exit(f"ERROR: model not found at {model_dir}, run download_model.py first")
    print(f"\nModel: {model_dir}")
    processor = WhisperProcessor.from_pretrained(model_dir, language=LANGUAGE, task=TASK)

    # Data
    print("\nLoading data...")
    ds = load_data(args.metadata, processor, args.max_samples)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(ds['train'])}  Validation: {len(ds['test'])}")

    # Model + LoRA
    print("\nLoading model + LoRA...")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=TARGET_MODULES, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    ))
    model.print_trainable_parameters()

    # PEFT forward 会强制传 input_ids，Whisper 只认 input_features + labels
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

    # Training args
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50, save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=225,
        report_to="tensorboard", logging_dir="./logs",
        dataloader_num_workers=0 if device == "cpu" else 2,
        remove_unused_columns=False,
    )

    # Epoch callback
    from transformers import TrainerCallback
    class EpochLogCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                best = f"{state.best_metric:.4f}" if state.best_metric else "N/A"
                print(f"\n{'='*45}")
                print(f"Epoch {int(state.epoch)} | Step {state.global_step}")
                print(f"  eval_loss: {metrics.get('eval_loss', 0):.4f}")
                print(f"  eval_wer:  {metrics.get('eval_wer', 0):.4f}")
                print(f"  best_wer:  {best}")
                print(f"{'='*45}")

    # Metrics
    wer = evaluate.load("wer")

    def compute_metrics(pred):
        labels = pred.label_ids
        labels[labels == -100] = processor.tokenizer.pad_token_id
        refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        hyps = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        return {"wer": wer.compute(predictions=hyps, references=refs)}

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model, args=train_args,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        data_collator=DataCollator(processor, processor.tokenizer.bos_token_id),
        compute_metrics=compute_metrics,
        callbacks=[EpochLogCallback()],
    )

    print("\nTraining...")
    trainer.train()

    # Save LoRA
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nLoRA saved to {args.output_dir}")

    # Merge
    if args.save_merged:
        print("Merging model to ./merged ...")
        base = WhisperForConditionalGeneration.from_pretrained(model_dir)
        m = PeftModel.from_pretrained(base, args.output_dir).merge_and_unload()
        m.save_pretrained("./merged")
        processor.save_pretrained("./merged")

    # Final
    r = trainer.evaluate()
    print(f"\nFinal WER: {r.get('eval_wer', 'N/A'):.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
