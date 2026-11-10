"""对话大脑 SFT：Qwen2.5-7B-Instruct + QLoRA(4bit) 因果 LM 微调。

把 opus 产的"金标群像对白"蒸馏进本地 7B——只在 assistant 段算 loss，
system/user 段 label=-100（不学 prompt，只学"该怎么说话"）。

- base 可配：将来 1.5B 价值模型也想换更大的，改 --base 即可，脚本不动。
- 精度按卡自动：Ampere+ 用 bf16 计算；T4(Turing, sm75) 没有高效 bf16，用 fp16。
- 4bit 量化基座 + 梯度检查点 + paged_adamw_8bit，把 7B 训练压进 T4 的 15GB。

用法（T4）：
    .venv/bin/python train_sft.py \
        --data data/convo_sft.jsonl \
        --base Qwen/Qwen2.5-7B-Instruct \
        --out runs/convo-7b-v1 \
        --epochs 3 --bs 1 --accum 16 --max-len 1280
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


class MemFrugalTrainer(Trainer):
    """T4(15GB) 显存友好的 CausalLM SFT。

    痛点：Qwen2ForCausalLM 传 labels 时会把整段 logits 转 fp32(logits.float())，
    再做 cross_entropy——在 max_len≈1280、词表 15 万的情况下，仅 logits 一项就 ~0.78GB，
    加上 softmax 中间量又一份，峰值直接把 15GB T4 顶穿(观测到 720MiB 分配失败)。

    做法：① 调 model 时不传 labels → 模型只吐 fp16 logits、跳过内部 .float()+CE；
          ② 自己按 token 分块算 CE，任一时刻只有 CHUNK 个 token 落到 fp32，峰值可控。
    数值上等价于对有效(非 -100)token 求平均。
    """

    LOSS_CHUNK = 512

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits                       # [B, T, V] fp16，未 float 化
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)

        total = shift_logits.new_zeros(())
        n_tok = (shift_labels != -100).sum().clamp(min=1)
        for i in range(0, shift_logits.size(0), self.LOSS_CHUNK):
            lg = shift_logits[i:i + self.LOSS_CHUNK].float()   # 仅本块转 fp32
            lb = shift_labels[i:i + self.LOSS_CHUNK]
            total = total + F.cross_entropy(lg, lb, ignore_index=-100, reduction="sum")
        loss = total / n_tok
        return (loss, outputs) if return_outputs else loss


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("messages"):
                rows.append(obj)
    return rows


def build_example(tok, messages: list[dict], max_len: int) -> dict | None:
    """apply_chat_template 拼全文；prompt 段(system+user)label 置 -100，只在 assistant 段算 loss。

    超长时从 **左侧** 截断（丢最早的 prompt token），因为要学的 assistant 回复在尾部——
    右截会把要学的内容切掉。左截保证 assistant 段永远完整，只损失最早的一点 persona 上下文。
    这样 T4(15GB) 能把序列压到显存内，且几乎不丢样本。
    """
    prompt_msgs = messages[:-1]
    prompt_ids = tok.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(full_ids) <= len(prompt_ids):
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    input_ids = full_ids[-max_len:]        # 左截：保尾部(assistant)完整
    labels = labels[-max_len:]
    if all(x == -100 for x in labels):   # assistant 段为空，学不到东西
        return None
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help=">0 时按步数封顶(覆盖 epochs)。无人值守过夜训练的时限保护阀：")
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cap = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    use_bf16 = cap >= 8                      # Ampere+ 才有高效 bf16；T4(7.5) 走 fp16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[sft] device_cap={cap} bf16={use_bf16} compute_dtype={compute_dtype}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_rows(args.data)
    random.shuffle(rows)
    exs = []
    for r in rows:
        e = build_example(tok, r["messages"], args.max_len)
        if e is not None:
            exs.append(e)
    n_val = max(1, int(len(exs) * args.val_frac))
    val, train = exs[:n_val], exs[n_val:]
    lens = sorted(len(e["input_ids"]) for e in exs)
    print(f"[sft] 样本 {len(exs)}（train {len(train)} / val {len(val)}）"
          f" token 长度 p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True, torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    print(f"[sft] gradient_checkpointing={getattr(model, 'is_gradient_checkpointing', None)}",
          flush=True)

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,          # >0 覆盖 epochs：过夜训练的时限保护，跑到上限就存
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",              # 关 eval：省掉 val 前向的 logits 聚集峰值，T4 显存更稳
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=[],
        save_total_limit=1,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = MemFrugalTrainer(
        model=model, args=targs, train_dataset=train,
        data_collator=collator,
    )
    print(f"[sft] 训练前显存 {round(torch.cuda.memory_allocated()/1e6)}MB "
          f"reserved {round(torch.cuda.memory_reserved()/1e6)}MB", flush=True)
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("[sft] CUDA OOM，换更短 max_len 再试", flush=True)
        sys.exit(75)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[sft] 适配器已保存 → {args.out}")


if __name__ == "__main__":
    main()
