"""
将 HuggingFace transformers 格式的 Whisper 模型转换为 CTranslate2 格式，
供 faster_whisper / ekko_asr_backend 使用。

用法:
    python convert_to_ct2.py
    python convert_to_ct2.py --model ./whisper-large-v3-zh-lora --output ./whisper-large-v3-zh-lora-ct2
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def fix_config(model_dir: Path):
    """移除不兼容字段，避免旧版 transformers / ctranslate2 加载报错。"""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    changed = False
    for key in ("dtype", "torch_dtype"):
        if key in cfg:
            cfg.pop(key)
            changed = True
    if changed:
        bak = config_path.with_suffix(".json.bak")
        if not bak.exists():
            shutil.copy(config_path, bak)
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   🔧 已修复 config.json（备份: {bak.name}）")


def main():
    p = argparse.ArgumentParser(description="Convert HF Whisper → CTranslate2")
    p.add_argument("--model", default="./whisper-large-v3-zh-lora",
                   help="合并后的 HuggingFace 模型目录（save_merged 的输出）")
    p.add_argument("--output", default="./whisper-large-v3-zh-lora-ct2",
                   help="输出的 CTranslate2 模型目录")
    p.add_argument("--quantization", default="float16",
                   choices=["float16", "int8_float16", "int8",
                            "int8_float32", "int8_bfloat16", "int16", "bfloat16", "float32"],
                   help="量化方式（默认 float16）")
    args = p.parse_args()

    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        sys.exit(1)

    # 修复 config.json 上的兼容问题
    fix_config(model_dir)

    # 尝试新版 CLI，不行再回退旧版 Python 模块方式
    cmd = [
        "ct2-transformers-converter",
        "--model", str(model_dir),
        "--output_dir", args.output,
        "--quantization", args.quantization,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]

    print(f"转换中...")
    print(f"  源模型: {model_dir}")
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
