"""阶段4-a：把 LoRA adapter 合并回 base，导出完整模型供 value_server 常驻加载。

用法：
    python merge_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/adapter --out merged/value-1p5b
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="runs/adapter")
    ap.add_argument("--out", default="merged/value-1p5b")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=1, torch_dtype=torch.float16, trust_remote_code=True,
    )
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()   # 含 modules_to_save 的 score 头

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"合并完成 → {out}")


if __name__ == "__main__":
    main()
