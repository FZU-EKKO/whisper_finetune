"""
下载 Whisper-large-v3 到本地目录

用法:
    # 国内必须走镜像！
    export HF_ENDPOINT=https://hf-mirror.com
    python download_model.py                    # 下载到 ./whisper-large-v3/
    python download_model.py --dir /data/model   # 指定目录
"""
import argparse, os

MODEL_ID = "openai/whisper-large-v3"


def main():
    p = argparse.ArgumentParser(description="下载 Whisper-large-v3")
    p.add_argument("--dir", default="./whisper-large-v3", help="保存目录")
    args = p.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        print("❌ 未设置镜像！国内服务器请先执行:")
        print("   export HF_ENDPOINT=https://hf-mirror.com\n")
        if input("继续尝试直连下载? (y/n): ").strip().lower() != "y":
            return
    else:
        print(f"🌐 镜像: {os.environ['HF_ENDPOINT']}")

    print(f"⬇️  下载 {MODEL_ID} → {args.dir}")

    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    model.save_pretrained(args.dir)
    processor.save_pretrained(args.dir)

    print(f"✅ 完成! 模型已保存到 {os.path.abspath(args.dir)}")


if __name__ == "__main__":
    main()
