"""
Whisper LoRA 推理

用法:
    python infer.py --audio test.wav
    python infer.py --dir ../raw_data/speaker_01/
    python infer.py --lora ./checkpoint --audio test.wav
"""
import argparse, time, os, glob
import torch, librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

MODEL_DIR = "./whisper-large-v3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=MODEL_DIR, help="本地模型目录")
    p.add_argument("--lora", default="./checkpoint")
    p.add_argument("--audio", default=None)
    p.add_argument("--dir", default=None)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()

    if not args.audio and not args.dir:
        exit("❌ 请指定 --audio 或 --dir")

    device = args.device
    print(f"🖥  设备: {device}")

    # 加载模型
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        exit(f"❌ 模型不存在: {model_dir}，请先运行 python download_model.py")

    print(f"📦 模型: {model_dir}")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    if os.path.isdir(args.lora):
        print(f"🔗 LoRA: {args.lora}")
        model = PeftModel.from_pretrained(model, args.lora).merge_and_unload()
    model.to(device).eval()

    processor = WhisperProcessor.from_pretrained(
        args.lora if os.path.isdir(args.lora) else model_dir,
        language="zh", task="transcribe")

    # 推理函数
    def transcribe(path):
        audio, _ = librosa.load(path, sr=16000)
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        feats = feats.input_features.to(device)
        with torch.no_grad():
            t0 = time.time()
            ids = model.generate(feats, language="zh", task="transcribe", max_length=225)
            dt = time.time() - t0
        text = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
        return text, dt

    # 单文件 or 批量
    if args.audio:
        text, dt = transcribe(args.audio)
        print(f"\n📝 {text}\n⏱  {dt:.2f}s")
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
        print(f"\n📂 {len(files)} 个文件\n")
        total = 0
        for i, f in enumerate(files, 1):
            text, dt = transcribe(f)
            total += dt
            print(f"[{i:3d}/{len(files)}] {os.path.basename(f)}")
            print(f"         {text}  ({dt:.2f}s)")
        print(f"\n⏱  总耗时: {total:.2f}s, 平均: {total/len(files):.2f}s")


if __name__ == "__main__":
    main()
