"""
Download Whisper-large-v3 to local directory.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com   # China mirror
    python download_model.py
    python download_model.py --dir /data/model
"""
import argparse, os

MODEL_ID = "openai/whisper-large-v3"


def main():
    p = argparse.ArgumentParser(description="Download Whisper-large-v3")
    p.add_argument("--dir", default="./whisper-large-v3")
    args = p.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        print("WARNING: HF_ENDPOINT not set. In China, set it first:")
        print("  export HF_ENDPOINT=https://hf-mirror.com\n")
        if input("Continue without mirror? (y/n): ").strip().lower() != "y":
            return
    else:
        print(f"Mirror: {os.environ['HF_ENDPOINT']}")

    print(f"Downloading {MODEL_ID} -> {args.dir}")

    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    model.save_pretrained(args.dir)
    processor.save_pretrained(args.dir)

    print(f"Done! Model saved to {os.path.abspath(args.dir)}")


if __name__ == "__main__":
    main()
