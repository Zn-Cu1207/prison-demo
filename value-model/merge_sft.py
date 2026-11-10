"""把对话 SFT 的 LoRA 适配器合并回基座，产出可转 GGUF 的 fp16 HF 权重。

默认在 CPU 上合并（--device cpu）：7B fp16 合并吃内存不吃显存，
这样不跟 T4 上常驻的价值服务抢显存。合并后用 llama.cpp 转 GGUF 喂 Ollama。

用法：
    .venv/bin/python merge_sft.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --adapter runs/convo-7b-v1 \
        --out merged/convo-7b-v1
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    print(f"[merge] 载入基座 {args.base}（{args.device}, fp16）…")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, trust_remote_code=True,
        device_map={"": args.device} if args.device == "cuda" else None,
    )
    if args.device == "cpu":
        base = base.to("cpu")
    print(f"[merge] 挂上适配器 {args.adapter} 并合并…")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tok.save_pretrained(args.out)
    print(f"[merge] 合并完成 → {args.out}")


if __name__ == "__main__":
    main()
