"""
将 HuggingFace transformers 格式的 Whisper 模型转换为 CTranslate2 格式，
供 faster_whisper / ekko_asr_backend 使用。

用法:
    python convert_to_ct2.py
    python convert_to_ct2.py --model ./whisper-large-v3-zh-lora --output ./whisper-large-v3-zh-lora-ct2
"""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="Convert HF Whisper → CTranslate2")
    p.add_argument("--model", default="./whisper-large-v3-zh-lora",
                   help="合并后的 HuggingFace 模型目录（save_merged 的输出）")
    p.add_argument("--output", default="./whisper-large-v3-zh-lora-ct2",
                   help="输出的 CTranslate2 模型目录")
    p.add_argument("--quantization", default="float16",
                   choices=["float16", "int8_float16", "int8"],
                   help="量化方式（默认 float16，和 faster_whisper 默认一致）")
    args = p.parse_args()

    cmd = [
        sys.executable, "-m", "ct2_transformers.converter",
        "--model", args.model,
        "--output_dir", args.output,
        "--quantization", args.quantization,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]

    print(f"转换中...")
    print(f"  源模型: {args.model}")
    print(f"  输出:   {args.output}")
    print(f"  量化:   {args.quantization}")
    print(f"  {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n❌ 转换失败！请确保已安装 ctranslate2: pip install ctranslate2")
        sys.exit(1)

    print(f"\n✅ 转换完成！")
    print(f"在 .env 中设置: EKKO_ASR_MODEL_PATH={args.output}")


if __name__ == "__main__":
    main()
