"""生成"新四轴对" → axis_prefs.jsonl（规则标注，零 API 成本）。

为什么需要这批数据
------------------
v2 给 caption 追加了四轴（食物/仇怨/表现/近期被抓），但只靠 gen_situations + 大模型排序
教不动它们，原因有两个，都是这次审计才暴露出来的：

1) **共线**：生成器里 `仇怨` 只在 `threat=="仇人"` 时才非默认，于是"仇怨:有"和
   "仇人:在场"永远同时出现、同时消失。两个字段完全共线时，模型没法把仇怨当独立信号——
   它只要看"仇人"那一位就够了，多出来的这一轴等于没有。
2) **样本太薄**：审计实测仇怨非默认只占 5%（107/2200），学不出刻度。

解法和 gen_intensity 一样：造**只差一个轴**的配对情境，同一个候选行动放两侧，
规则裁定哪侧该做。因为只有一个变量在动，标签无歧义，也不必烧 API。

一条重要的现实约束（差点踩坑）
------------------------------
引擎里 `_hostile_near` 要求 grudge ≥ GRUDGE_TO_FIGHT(35)，而 `_grudge_axis` 恰好也是
≥35 记"有"。所以 **"仇人:在场 + 仇怨:无" 在游戏里根本不可能出现**。
造这种样本就是在训练现实中不存在的输入，反而破坏 train-serve 一致。
因此仇怨轴只教合法的那个对比：仇人在场时，**仇怨深 vs 仇怨有**（该动手还是只吵两句）。

用法：
    python gen_axis_pairs.py --n 800 --out data/axis_prefs.jsonl --seed 13
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.context import (  # noqa: E402
    GRUDGE_DEEP,
    GRUDGE_NONE,
    GRUDGE_SOME,
    PERFORM_BAD,
    PERFORM_GOOD,
    PERFORM_MID,
    PHASE_FREE,
    PHASE_SLEEP,
    PHASE_TRAIN,
    PHASE_YARD,
    SUPPLY_OK,
    SUPPLY_OUT,
    SUPPLY_TIGHT,
    Candidate,
    ContextState,
    NeedState,
    build_context_caption,
)

PERSONALITIES = ["暴躁", "书呆", "老油条", "新人", "普通"]
MOTIVES = ["想早点平安假释", "惦记着还清外面欠下的债", "就想熬到刑满平安出去", "无"]


def base_ctx(rng: random.Random, phases: list[str]) -> ContextState:
    """中性底座。四个新轴一律先压到默认值，之后只由每种对去动**一个**。"""
    return ContextState(
        needs=NeedState(
            hunger=rng.randint(0, 25), thirst=rng.randint(0, 25),
            fatigue=rng.randint(0, 25), injury=rng.randint(0, 15),
            boredom=rng.randint(20, 60), social=rng.randint(0, 40),
            stress=rng.randint(10, 45),
        ),
        threat="无",
        event="无",
        phase=rng.choice(phases),
        personality=rng.choice(PERSONALITIES),
        surveillance=rng.choice(["中", "低"]),
        backer_present=rng.random() < 0.3,
        enemy_present=False,
        yard_crowd=rng.choice(["适中", "空旷"]),
        parole_near=False,
        has_corpse=False,
        has_labor=False,
        motive=rng.choice(MOTIVES),
        player_command="无",
        food_supply=SUPPLY_OK,
        grudge=GRUDGE_NONE,
        perform=PERFORM_MID,
        busted_recently=False,
    )


# ---------------------------------------------------------------------------
# 每种对的定义：怎么造底座、把哪一轴设成什么、哪侧该做。
# 只列**判断上没有歧义**的规则；模棱两可的（比如"仇怨深时该不该改成吵架"）一律不造，
# 宁可少而干净——错标比缺标更伤模型。
# ---------------------------------------------------------------------------
def make_food_eat(rng: random.Random) -> tuple[ContextState, str, str, str, float, str]:
    """食物轴 · Eat：有饭吃时才该去食堂；断供了去了是白跑一趟。"""
    ctx = base_ctx(rng, [PHASE_YARD, PHASE_FREE, PHASE_TRAIN])
    ctx.needs.hunger = rng.randint(55, 92)      # 饿着，但没到濒死（濒死时白跑也得跑）
    hi, lo = SUPPLY_OK, rng.choice([SUPPLY_OUT, SUPPLY_TIGHT])
    return ctx, "food_supply", hi, lo, 0.9, "有饭(充足)时去吃饭才有意义；断供/紧张时多半白跑"


def make_food_work(rng: random.Random) -> tuple[ContextState, str, str, str, float, str]:
    """食物轴 · 帮厨/换货：牢里没饭了，该去弄吃的而不是干等。"""
    ctx = base_ctx(rng, [PHASE_TRAIN, PHASE_FREE])
    ctx.needs.hunger = rng.randint(50, 88)
    ctx.has_labor = True                         # KitchenDuty 需要被派苦役
    hi, lo = SUPPLY_OUT, SUPPLY_OK
    return ctx, "food_supply", hi, lo, 0.7, "断供时更该去后厨/黑市弄吃的；有饭时没这个必要"


def make_grudge_fight(rng: random.Random) -> tuple[ContextState, str, str, str, float, str]:
    """仇怨轴 · Fight：仇人就在眼前，仇怨越深越说得通动手。

    两侧都必须 仇人:在场——因为引擎里 grudge≥35 才算"在场"，
    "在场+仇怨:无"是不可能状态，不能造。
    """
    ctx = base_ctx(rng, [PHASE_YARD, PHASE_FREE, PHASE_TRAIN])
    ctx.threat = "仇人"
    ctx.enemy_present = True
    ctx.needs.stress = rng.randint(40, 80)
    hi, lo = GRUDGE_DEEP, GRUDGE_SOME
    return ctx, "grudge", hi, lo, 0.85, "仇人在场时，仇怨深(≥70)比仅仅有仇(35~70)更该动手"


def make_busted_illicit(rng: random.Random) -> tuple[ContextState, str, str, str, float, str]:
    """近期被抓轴 · 违禁行为：刚从禁闭出来正被盯着，这一阵该老实。"""
    ctx = base_ctx(rng, [PHASE_YARD, PHASE_FREE])
    return ctx, "busted_recently", False, True, 0.85, \
        "刚被抓过(近期被抓:是)时违禁行为风险陡增，该收着；没被抓过时才敢干"


def make_perform_illicit(rng: random.Random) -> tuple[ContextState, str, str, str, float, str]:
    """表现轴 · 违禁行为：表现好的珍惜记录，表现差的没什么可失去。"""
    ctx = base_ctx(rng, [PHASE_YARD, PHASE_FREE])
    hi, lo = PERFORM_BAD, PERFORM_GOOD
    return ctx, "perform", hi, lo, 0.8, \
        "表现差的人破罐破摔、更可能违禁；表现好的人格外珍惜记录，违禁要排后"


def make_weapon_kill(rng: random.Random) -> tuple[ContextState, str, bool, bool, float, str]:
    """凶器轴 · Assassinate：没刀空挥，有刀才说得通动手。"""
    ctx = base_ctx(rng, [PHASE_YARD, PHASE_FREE, PHASE_TRAIN])
    ctx.threat = "仇人"
    ctx.enemy_present = True
    ctx.grudge = GRUDGE_DEEP
    ctx.surveillance = "低"
    return ctx, "has_weapon", True, False, 0.9, \
        "手里有刀才说得通下杀手；没刀去动手是空挥"


def make_weapon_get(rng: random.Random) -> tuple[ContextState, str, bool, bool, float, str]:
    """凶器轴 · GetWeapon：没刀该去弄，已经有刀再跑一趟没必要。"""
    ctx = base_ctx(rng, [PHASE_YARD, PHASE_FREE])
    ctx.threat = "仇人"
    ctx.enemy_present = True
    ctx.grudge = GRUDGE_DEEP
    return ctx, "has_weapon", False, True, 0.85, \
        "没刀时该去弄凶器；已经有刀再跑一趟没必要"


# 违禁候选（挑当前时段真的会上桌的那几个，别造线上不存在的组合）
ILLICIT_PICK = [("Gamble", "daily"), ("Barter", "daily"), ("Tattoo", "daily")]

# (权重, 造法, 候选选取函数)
SPECS = [
    (0.16, make_food_eat, lambda rng: ("Eat", "need")),
    (0.10, make_food_work, lambda rng: rng.choice([("KitchenDuty", "worktask"),
                                                   ("Barter", "daily")])),
    (0.12, make_grudge_fight, lambda rng: ("Fight", "threat")),
    (0.10, make_busted_illicit, lambda rng: rng.choice(ILLICIT_PICK)),
    (0.08, make_perform_illicit, lambda rng: rng.choice(ILLICIT_PICK)),
    (0.22, make_weapon_kill, lambda rng: ("Assassinate", "threat")),
    (0.22, make_weapon_get, lambda rng: ("GetWeapon", "daily")),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default="data/axis_prefs.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    plan: list[int] = []
    for idx, (w, _f, _c) in enumerate(SPECS):
        plan += [idx] * int(args.n * w)
    while len(plan) < args.n:
        plan.append(0)
    rng.shuffle(plan)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    tattoo_note = 0
    with out.open("w", encoding="utf-8") as f:
        for i, si in enumerate(plan):
            _w, build, pick_cand = SPECS[si]
            ctx, axis, hi_val, lo_val, conf, why = build(rng)
            goal, source = pick_cand(rng)
            # Tattoo 只在 自由活动/就寝 菜单里；时段不符就换一个，别造线上不存在的组合
            if goal == "Tattoo" and ctx.phase not in (PHASE_FREE, PHASE_SLEEP):
                goal, source = "Gamble", "daily"
                tattoo_note += 1

            setattr(ctx, axis, hi_val)
            cap_hi = build_context_caption(ctx)
            setattr(ctx, axis, lo_val)
            cap_lo = build_context_caption(ctx)
            if cap_hi == cap_lo:                 # 万一轴值没真的变，跳过（否则是零信息样本）
                continue

            cand = Candidate(goal, source).to_payload()
            sid = f"ax_{axis}_{goal}_{i:06d}"
            rec = {
                "situation_id": sid,
                "pair_id": f"{sid}_p0",
                "context": cap_hi,               # 兜底给不认识新键的旧代码
                "context_chosen": cap_hi,
                "context_rejected": cap_lo,
                "chosen": cand,
                "rejected": cand,
                "confidence": conf,
                "label_by": "axis",
                "reason": f"{why}（只有『{axis}』一个字段不同：{hi_val} vs {lo_val}）",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[f"{axis}/{goal}"] += 1

    total = sum(stats.values())
    print(f"写出 {total} 条新轴对 → {out}")
    print("分布：")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28} {v:>4}")
    if tattoo_note:
        print(f"（{tattoo_note} 条 Tattoo 因时段不符换成 Gamble）")
    print("全部规则标注（label_by=axis），零 API 消耗。")


if __name__ == "__main__":
    main()
