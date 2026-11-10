"""阶段2：微调价值模型（Motif 第②步）。

Qwen2.5-1.5B-Instruct + LoRA + 标量价值头(num_labels=1) + Bradley-Terry 损失：
    P(chosen ≻ rejected) = σ(v_chosen − v_rejected)
    L = − confidence · log σ(v_chosen − v_rejected)     （按标注置信度加权）
推理时对**单个候选**输出标量 v（pointwise），与 value_server 完全一致、免疫位置偏见。

- 按 situation_id 整组切 train/val，防同情境样本泄漏。
- 输入渲染只走 common.build_input（train-serve 命根子）。
- 精度按卡自动选：Ampere+ 走纯 bf16；T4(Turing) 没有 bf16，走 fp16 + GradScaler
  且可训练参数留 fp32。同一份脚本两台机器都能跑，不需要改参数。

用法：
    python train_value.py --pref data/preferences.jsonl --out runs/adapter --epochs 3
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import build_input


class PairDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # 两侧上下文默认相同（同一情境比两个候选）。
        # 强度对是例外：同一个候选、只有某个需求的数值不同，用来教"多急"的绝对刻度——
        # BT 损失只约束同情境内的相对顺序，跨情境的强度标度它学不到，
        # 所以必须显式给一批两侧 context 不同的样本。
        cc = r.get("context_chosen") or r["context"]
        cr = r.get("context_rejected") or r["context"]
        chosen = build_input(cc, r["chosen"])
        rejected = build_input(cr, r["rejected"])
        return chosen, rejected, float(r.get("confidence", 1.0))


def make_collate(tok, max_len):
    """chosen 与 rejected 拼成**一个** batch 一次前向（2N 条），再对半切开算 BT 损失。

    比分两次 forward 快：GPU 上一次大 batch 的利用率明显高于两次小 batch。
    """
    def collate(batch):
        texts = [b[0] for b in batch] + [b[1] for b in batch]
        conf = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc, conf, len(batch)
    return collate


def split_by_situation(rows, val_frac, seed):
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["situation_id"], []).append(r)
    sids = list(groups.keys())
    random.Random(seed).shuffle(sids)
    n_val = max(1, int(len(sids) * val_frac))
    val_sids = set(sids[:n_val])
    train, val = [], []
    for sid, rs in groups.items():
        (val if sid in val_sids else train).extend(rs)
    return train, val


def value_pair(model, enc, n, device):
    """一次前向拿到 (v_chosen, v_rejected)。"""
    enc = {k: v.to(device) for k, v in enc.items()}
    v = model(**enc).logits.squeeze(-1)  # [2N]
    return v[:n], v[n:]


@torch.no_grad()
def eval_acc(model, loader, device, amp: bool = False):
    model.eval()
    correct = tot = 0
    for enc, _conf, n in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            vc, vr = value_pair(model, enc, n, device)
        correct += (vc.float() > vr.float()).sum().item()
        tot += vc.numel()
    return correct / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref", nargs="+", default=["data/preferences.jsonl"],
                    help="可给多份，例如 data/preferences_v2.jsonl data/intensity_prefs.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="runs/adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条样本（0=全部），快速试跑用")
    ap.add_argument("--grad-ckpt", action="store_true", default=True,
                    help="梯度检查点(默认开)：省显存，避免 12G 卡爆显存后掉进共享内存变慢")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--resume", default="",
                    help="已有 LoRA 目录，接着训（例如 runs/adapter-v4），不从零初始化")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 精度按卡自动选，同一份脚本要能同时跑 3060(Ampere) 和 T4(Turing)：
    #   · Ampere+ 支持 bf16 → 直接纯 bf16 训练，指数范围宽，不需要损失缩放。
    #   · T4 是 Turing(sm75)，**没有 bf16**。fp16 不能照搬 bf16 的做法：动态范围窄，
    #     不做损失缩放梯度会下溢成 0，轻则训不动、重则 NaN。所以走标准混合精度：
    #     冻结的底座留 fp16，可训练参数(LoRA A/B + score 头)提到 fp32，
    #     前向包 autocast、反向配 GradScaler。
    # 判断依据必须是计算能力，**不能用 torch.cuda.is_bf16_supported()**：
    # 它在 T4(sm75) 上也返回 True（CUDA 11+ 提供 bf16 存储/转换即算"支持"），
    # 但 Turing 没有 bf16 张量核心，实测 4096³ 矩阵乘：
    #     T4    fp16 24.52 TFLOPS / bf16 2.38 TFLOPS / fp32 4.31 TFLOPS  ← bf16 比 fp32 还慢
    #     3060  fp16  8.86 TFLOPS / bf16 8.91 TFLOPS                     ← Ampere 上两者等价
    # 信了那个 API 就会在 T4 上白慢十倍。sm80(Ampere) 起才有原生 bf16。
    bf16_ok = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    if device != "cuda":
        dtype, amp = torch.float32, False
    elif bf16_ok:
        dtype, amp = torch.bfloat16, False
    else:
        dtype, amp = torch.float16, True
    if device == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)} "
              f"sm{''.join(map(str, torch.cuda.get_device_capability(0)))} bf16={bf16_ok}")
    print(f"device={device} dtype={dtype} 混合精度(GradScaler)={amp}")

    # 允许多份偏好文件：主偏好(大模型排序派生) + 强度对(规则标注) 分开落盘更好检视，
    # 但训练要合在一起。按 situation_id 整组切分，所以两份混合不会串到 val 里。
    rows: list[dict] = []
    for p in args.pref:
        part = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  载入 {p}: {len(part)} 条")
        rows.extend(part)
    if args.limit:
        rows = rows[: args.limit]
    train_rows, val_rows = split_by_situation(rows, args.val, args.seed)
    print(f"样本 {len(rows)} → train {len(train_rows)} / val {len(val_rows)}")
    src = Counter(r.get("label_by", "?") for r in rows)
    print(f"  标注来源分布：{dict(src)}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=1, torch_dtype=dtype, trust_remote_code=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    # 训练只做单次前向打分，不需要 KV cache。开着会白占显存并拖慢速度。
    model.config.use_cache = False

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    if args.resume:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print(f"从 adapter 接着训：{args.resume}", flush=True)
    else:
        model = get_peft_model(model, lora)
    # 梯度检查点：3060 只有 12G，不开的话激活值会把显存顶满，Windows 驱动会把超出的
    # 部分悄悄挪到系统内存(WDDM 共享显存)，训练慢十几倍。开了之后激活值降一个量级，
    # 全程留在显存里，虽然多一次重算，实测反而快得多。
    if args.grad_ckpt:
        model.enable_input_require_grads()   # PEFT + checkpointing 必须，否则没有梯度入口
        model.gradient_checkpointing_enable()
    model.to(device)
    if amp:
        # fp16 路径：可训练参数必须留 fp32（master weights）。
        # 它们只有一两千万个（LoRA r=16 的 q/k/v/o + score 头），提 fp32 的显存代价可忽略，
        # 但能挡住 fp16 更新量被舍入成 0 的问题。
        n_fp32 = 0
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
                n_fp32 += p.numel()
        print(f"fp16 混合精度：{n_fp32/1e6:.1f}M 个可训练参数已提到 fp32(master weights)")
    model.print_trainable_parameters()

    collate = make_collate(tok, args.maxlen)
    train_loader = DataLoader(PairDataset(train_rows), batch_size=args.bs, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PairDataset(val_rows), batch_size=args.bs, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    print(f"起始 val 准确率：{eval_acc(model, val_loader, device, amp):.3f}（随机基线 ~0.5）", flush=True)
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        running = 0.0
        for step, (enc, conf, n) in enumerate(train_loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                vc, vr = value_pair(model, enc, n, device)
                conf = conf.to(device)
                # BT 损失在 fp32 里算：logsigmoid 在 fp16 下容易失精度
                loss = -(conf * F.logsigmoid(vc.float() - vr.float())).mean() / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)          # 先还原真实梯度再裁剪，否则裁的是被放大过的值
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            if (step + 1) % 50 == 0:
                print(f"  ep{ep} step{step + 1}/{len(train_loader)} loss={running / (step + 1):.4f}",
                      flush=True)
        acc = eval_acc(model, val_loader, device, amp)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        print(f"[epoch {ep}] 平均loss={running / max(1, len(train_loader)):.4f}  "
              f"val准确率={acc:.3f}  显存峰值={peak:.1f}GB", flush=True)
        # 每轮都存一次，训练中断也不至于全丢
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"LoRA adapter + tokenizer 已存 → {out}")


if __name__ == "__main__":
    main()
