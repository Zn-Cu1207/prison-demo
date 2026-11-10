"""阶段0-c：生成"强度对" → intensity_prefs.jsonl（规则标注，不烧一次 API）。

为什么必须单独造这批数据
------------------------
Bradley-Terry 损失只约束**同一情境内**两个候选的相对顺序：
    L = −log σ(v_chosen − v_rejected)
同情境的样本再多，也只能告诉模型"这里该吃饭而不是解闷"，
**说不出"饥饿95 比饥饿60 更急"**——因为这两个数值从来没在同一个损失项里比较过。
结果就是 v1 的毛病：需求强度是平的，饥饿 60 和 95 的分差不足，"多急"表达不出来。

做法：造**配对情境**——除了目标需求的数值，其余字段逐字相同，
同一个候选行动在两侧出现，规则裁定"数值高的那侧更急"。
因为只有一个变量在动，标签无歧义，不需要大模型判，置信度给满。

用法：
    python gen_intensity.py --n 900 --out data/intensity_prefs.jsonl --seed 11
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.context import (  # noqa: E402
    GRUDGE_NONE,
    PERFORM_BAD,
    PERFORM_GOOD,
    PERFORM_MID,
    PHASE_CLASS,
    PHASE_FREE,
    PHASE_SLEEP,
    PHASE_TRAIN,
    PHASE_YARD,
    SUPPLY_OK,
    Candidate,
    ContextState,
    NeedState,
    build_context_caption,
)

# 需求维度 → 对应的候选行动。只取有明确"越高越急"语义的六个。
DIM_GOAL = {
    "hunger": "Eat",
    "thirst": "Drink",
    "fatigue": "Sleep",
    "injury": "Cure",
    "boredom": "DeclineBoredom",
    "social": "Socialize",
}
# 各维度的行动能在哪些时段成立（照 world._gen_candidates 的时段约束，别造出线上不存在的情境）
DIM_PHASES = {
    "hunger": [PHASE_YARD, PHASE_FREE, PHASE_TRAIN],
    "thirst": [PHASE_YARD, PHASE_FREE, PHASE_TRAIN],
    "fatigue": [PHASE_SLEEP, PHASE_FREE],
    "injury": [PHASE_YARD, PHASE_FREE, PHASE_TRAIN],
    "boredom": [PHASE_YARD, PHASE_FREE, PHASE_CLASS],
    "social": [PHASE_YARD, PHASE_FREE],
}
# 强度档：跨档才配对。相邻档差距小、跨档差距大，让模型学出单调且有坡度的刻度。
BANDS = [(30, 45), (50, 65), (70, 82), (86, 100)]

PERSONALITIES = ["暴躁", "书呆", "老油条", "新人", "普通"]
MOTIVES = ["想早点平安假释", "惦记着还清外面欠下的债", "就想熬到刑满平安出去", "无"]


def base_ctx(rng: random.Random, dim: str) -> ContextState:
    """造一个"干净底座"：与强度无关的字段随机但两侧共用，避免混入其它变量。"""
    return ContextState(
        needs=NeedState(),
        threat="无",
        event="无",
        phase=rng.choice(DIM_PHASES[dim]),
        personality=rng.choice(PERSONALITIES),
        surveillance=rng.choice(["中", "低"]),
        backer_present=rng.random() < 0.3,
        enemy_present=False,
        yard_crowd=rng.choice(["适中", "空旷"]),
        parole_near=False,
        has_corpse=False,
        has_labor=rng.random() < 0.3,
        motive=rng.choice(MOTIVES),
        player_command="无",
        # 四个新轴固定在"中性"值：这批数据只教强度，别把新轴的信号混进来
        food_supply=SUPPLY_OK,
        grudge=GRUDGE_NONE,
        perform=rng.choice([PERFORM_GOOD, PERFORM_MID, PERFORM_BAD]),
        busted_recently=False,
    )


def fill_other_needs(rng: random.Random, ns: NeedState, dim: str) -> None:
    """其余需求压在低位：只让目标维度成为"最急的那件事"，标签才干净。"""
    for d in ("hunger", "thirst", "fatigue", "injury", "boredom", "social"):
        if d != dim:
            setattr(ns, d, rng.randint(0, 25))
    ns.stress = rng.randint(10, 45)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900, help="生成多少对")
    ap.add_argument("--out", default="data/intensity_prefs.jsonl")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dims = list(DIM_GOAL)
    n_by_dim: dict[str, int] = {}
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            dim = dims[i % len(dims)]          # 六个维度轮着来，别偏科
            goal = DIM_GOAL[dim]
            # 取两个不同档位，低档做 rejected、高档做 chosen
            lo_band, hi_band = rng.sample(range(len(BANDS)), 2)
            if lo_band > hi_band:
                lo_band, hi_band = hi_band, lo_band
            lo_v = rng.randint(*BANDS[lo_band])
            hi_v = rng.randint(*BANDS[hi_band])

            ctx = base_ctx(rng, dim)
            fill_other_needs(rng, ctx.needs, dim)

            # 关键：两侧除了目标维度的数值，其它字段逐字相同
            setattr(ctx.needs, dim, hi_v)
            cap_hi = build_context_caption(ctx)
            setattr(ctx.needs, dim, lo_v)
            cap_lo = build_context_caption(ctx)

            cand = Candidate(goal, "need").to_payload()
            # 档差越大越有把握：相邻档 0.75，跨两三档给到 0.95
            conf = {1: 0.75, 2: 0.9, 3: 0.95}[hi_band - lo_band]
            rec = {
                "situation_id": f"int_{dim}_{i:06d}",
                "pair_id": f"int_{dim}_{i:06d}_p0",
                # context 字段留着给不认识新键的旧代码兜底
                "context": cap_hi,
                "context_chosen": cap_hi,
                "context_rejected": cap_lo,
                "chosen": cand,
                "rejected": cand,
                "confidence": conf,
                "label_by": "intensity",
                "reason": f"同一情境下{dim}={hi_v} 比 {dim}={lo_v} 更紧急（其余字段完全相同）",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            n_by_dim[dim] = n_by_dim.get(dim, 0) + 1

    print(f"写出 {written} 条强度对 → {out}")
    print("各维度分布：", dict(sorted(n_by_dim.items())))
    print("这批全是规则标注（label_by=intensity），不消耗任何 API 调用。")


if __name__ == "__main__":
    main()
