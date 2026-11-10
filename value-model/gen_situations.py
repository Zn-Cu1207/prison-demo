"""阶段0-a：合成监狱情境 + 候选行动 → situations.jsonl

- 直接复用 backend/app 的 ContextState / build_context_caption / Candidate / ACTIONS，
  保证 caption 和候选 desc 与线上游戏**逐字一致**（train-serve 命根子）。
- 候选生成是 world._gen_candidates 的独立移植（同阈值/同触发条件）。
- **分层采样**：强行按配额生成 command / threat / worktask / 濒死need / event / 日常
  六类情境，避免稀有情境被"闲逛"淹没（RA §5 的关键教训）。

用法：
    python gen_situations.py --n 2000 --out data/situations.jsonl --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# 复用后端真实模块（app 是空 __init__ 的纯包，只拉 stdlib，不会拽起 fastapi）
BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.actions import BY_GOAL, GOALS_NEW, ILLICIT_GOALS  # noqa: E402
from app.context import (  # noqa: E402
    GRUDGE_DEEP,
    GRUDGE_NONE,
    GRUDGE_SOME,
    INTENT_ALLY,
    INTENT_DEAL,
    INTENT_FRIENDLY,
    INTENT_HOSTILE,
    INTENT_NONE,
    INTENT_UNKNOWN,
    INTENT_WEAK,
    NEAR_ALLY,
    NEAR_CREDITOR,
    NEAR_DEBTOR,
    NEAR_FOE,
    NEAR_NONE,
    NEAR_STRANGER,
    OBLIG_NONE,
    OBLIG_OWE,
    OBLIG_OWED,
    PERFORM_BAD,
    PERFORM_GOOD,
    PERFORM_MID,
    PHASE_CLASS,
    PHASE_FREE,
    PHASE_ROLLCALL,
    PHASE_SLEEP,
    PHASE_TRAIN,
    PHASE_YARD,
    SOC_ALLIED,
    SOC_DEAL,
    SOC_NONE,
    SOC_PROTECTED,
    SOC_RECONCILED,
    SOC_THREATENED,
    SUPPLY_OK,
    SUPPLY_OUT,
    SUPPLY_TIGHT,
    Candidate,
    ContextState,
    NeedState,
    build_context_caption,
)

THRESHOLDS = {"hunger": 30, "thirst": 30, "fatigue": 50, "boredom": 12, "social": 30, "injury": 30}

THREATS = ["无", "仇人", "狱警", "狱犬"]
EVENTS = ["无", "斗殴", "吵架声", "狱警搜牢", "警报"]
PHASES = [PHASE_ROLLCALL, PHASE_TRAIN, PHASE_CLASS, PHASE_YARD, PHASE_SLEEP, PHASE_FREE]
PERSONALITIES = ["暴躁", "书呆", "老油条", "新人", "普通"]
SURVEIL = ["高", "中", "低"]
CROWD = ["拥挤", "适中", "空旷"]
MOTIVES = [
    "想早点平安假释", "想在里头找个靠山别被欺负", "惦记着还清外面欠下的债",
    "一心想翻案证明自己清白", "想抢块地盘自己说了算", "就想熬到刑满平安出去", "无",
]
SURVIVAL_GOALS = {"Eat", "Drink", "Sleep", "Cure"}
# 违禁/高风险：指令 vs 被盯 时优先抽这些，让「当着管教面还下令动手」进训练。
WATCH_COMMANDS = [
    "Gamble", "Barter", "GetWeapon", "Assassinate", "PrisonBreak",
    "Fight", "Tunnel", "Hide", "Threaten", "Betray", "Extort",
]


def _command_text(goal: str) -> str:
    """与 world._build_context 逐字一致：命令去{label}({desc})。"""
    act = BY_GOAL.get(goal)
    if not act:
        return f"命令去{goal}"
    return f"命令去{act.label}({act.desc})"


def _pick_command(rng: random.Random, avoid: set[str] | None = None) -> str:
    """70% 抽 seen=new，让典狱长新动词真正进训练；其余从全集抽。"""
    avoid = avoid or set()
    new = [g for g in GOALS_NEW if g not in avoid]
    allg = [g for g in BY_GOAL if g not in avoid]
    pool = new if new and rng.random() < 0.7 else (allg or list(BY_GOAL))
    return rng.choice(pool)

# 时段消遣/门路菜单（近似 world._activity_menu 的分时段可选集）
MENU_BY_PHASE = {
    PHASE_YARD: ["Workout", "Read", "Gamble", "Smoke", "Chess", "Music", "Barter",
                 "Extort", "SellInfo", "SeekProtector", "CurryFavor", "Watch", "Investigate"],
    PHASE_FREE: ["Workout", "Shower", "Read", "Gamble", "Smoke", "Chess", "Music", "Pray",
                 "Barter", "Tattoo", "Visit", "SellInfo", "ReviewFiles", "CurryFavor", "Investigate"],
    PHASE_TRAIN: ["Laundry", "Clean", "Workout"],
    PHASE_CLASS: ["Read"],
    PHASE_SLEEP: ["Pray", "Tattoo"],
    PHASE_ROLLCALL: [],
}


def _cand_dict(goal: str, source: str, force: bool = False) -> dict:
    c = Candidate(goal, source, b_force_plan=force)
    d = c.to_payload()
    d["illicit"] = goal in ILLICIT_GOALS
    return d


def gen_candidates(ctx: ContextState, rng: random.Random) -> list[dict]:
    """world._gen_candidates 的独立移植（同阈值/触发）。"""
    n = ctx.needs
    phase = ctx.phase
    c: list[dict] = []
    if n.hunger >= THRESHOLDS["hunger"]:
        c.append(_cand_dict("Eat", "need"))
    if n.thirst >= THRESHOLDS["thirst"]:
        c.append(_cand_dict("Drink", "need"))
    if n.fatigue >= THRESHOLDS["fatigue"]:
        c.append(_cand_dict("Sleep", "need"))
    if n.boredom >= THRESHOLDS["boredom"]:
        c.append(_cand_dict("DeclineBoredom", "need"))
    if n.social >= THRESHOLDS["social"]:
        c.append(_cand_dict("Socialize", "need"))
    if n.injury >= THRESHOLDS["injury"]:
        c.append(_cand_dict("Cure", "need"))
    # 被狱警/狱犬盯上 → 收敛
    if ctx.threat in ("狱警", "狱犬"):
        c.append(_cand_dict("LayLow", "threat"))
    if ctx.enemy_present and phase in (PHASE_YARD, PHASE_FREE, PHASE_TRAIN):
        c.append(_cand_dict("Fight", "threat"))
        c.append(_cand_dict("Argue", "threat"))
    if phase == PHASE_CLASS:
        c.append(_cand_dict("Study", "daily"))
    if ctx.has_labor and phase in (PHASE_TRAIN, PHASE_FREE):
        c.append(_cand_dict("Labor", "worktask"))
        c.append(_cand_dict("KitchenDuty", "worktask"))
    if phase == PHASE_SLEEP and not ctx.parole_near:
        has_tool = rng.random() < 0.4
        breached = has_tool and rng.random() < 0.5
        if has_tool and not breached:
            c.append(_cand_dict("Tunnel", "daily"))
        if has_tool and breached:
            c.append(_cand_dict("PrisonBreak", "daily"))
    if ctx.event == "狱警搜牢":
        c.append(_cand_dict("Hide", "event"))
    if ctx.surveillance != "低" and (ctx.event == "斗殴" or ctx.has_corpse or ctx.enemy_present):
        c.append(_cand_dict("Snitch", "event"))
    if ctx.surveillance == "低" and phase in (PHASE_YARD, PHASE_FREE):
        c.append(_cand_dict("Watch", "daily"))
    if n.social >= 25 and phase in (PHASE_YARD, PHASE_FREE):
        c.append(_cand_dict("Phone", "need"))
    if ctx.has_corpse:
        c.append(_cand_dict("Bury", "event"))
        c.append(_cand_dict("Mourn", "event"))
    # v3/P2 定向社交动作：按"近旁那个人是谁 + 对方想干嘛 + 我手里的人情"上桌。
    # 与 world 的社交候选生成同规则，train-serve 一致。
    if phase in (PHASE_YARD, PHASE_FREE) and ctx.near_rel != NEAR_NONE:
        if ctx.near_rel in (NEAR_STRANGER, NEAR_DEBTOR, NEAR_CREDITOR, NEAR_ALLY):
            c.append(_cand_dict("Propose", "daily"))
        if ctx.near_rel in (NEAR_STRANGER, NEAR_ALLY) or ctx.other_intent == INTENT_ALLY:
            c.append(_cand_dict("Ally", "daily"))
        if ctx.near_rel == NEAR_FOE or ctx.other_intent == INTENT_HOSTILE:
            c.append(_cand_dict("Threaten", "daily"))
        if ctx.near_rel in (NEAR_ALLY, NEAR_DEBTOR):
            c.append(_cand_dict("Betray", "daily"))
        if ctx.other_intent == INTENT_WEAK or ctx.near_rel == NEAR_DEBTOR:
            c.append(_cand_dict("Protect", "daily"))
    if ctx.obligation == OBLIG_OWE and ctx.near_rel == NEAR_CREDITOR:
        c.append(_cand_dict("Repay", "daily"))
    # 弄凶器 / 下杀手：对齐 world._gen_candidates 的仇杀链（有仇但没刀 → 先弄；有刀且没被盯 → 动手）
    if ctx.enemy_present or ctx.grudge == GRUDGE_DEEP:
        if not ctx.has_weapon:
            if not any(x["goal"] == "GetWeapon" for x in c):
                c.append(_cand_dict("GetWeapon", "daily"))
        elif ctx.threat not in ("狱警", "狱犬"):
            if not any(x["goal"] == "Assassinate" for x in c):
                c.append(_cand_dict("Assassinate", "threat"))
    # 时段消遣菜单，取样 2-4 个
    menu = [g for g in MENU_BY_PHASE.get(phase, []) if g in BY_GOAL]
    rng.shuffle(menu)
    for g in menu[: rng.randint(2, 4)]:
        if not any(x["goal"] == g for x in c):
            c.append(_cand_dict(g, "daily"))
    # 兜底
    for g in ("Patrol", "Rest"):
        if not any(x["goal"] == g for x in c):
            c.append(_cand_dict(g, "daily"))
    # 玩家指令
    if ctx.player_command and ctx.player_command != "无":
        cmd_goal = ctx._cmd_goal  # type: ignore[attr-defined]
        c = [x for x in c if x["goal"] != cmd_goal]
        c.append(_cand_dict(cmd_goal, "command", force=True))
    return c


SURVIVAL_DIMS = ["hunger", "thirst", "fatigue", "injury"]


def _sample_needs(rng: random.Random, kind: str) -> NeedState:
    def r(lo, hi):
        return rng.randint(lo, hi)
    if kind in ("crisis", "cmd_crisis"):
        ns = NeedState(r(0, 40), r(0, 40), r(0, 40), r(0, 25), r(0, 40), r(0, 40), r(0, 40))
        setattr(ns, rng.choice(SURVIVAL_DIMS), r(85, 100))     # 抬一个维度到濒死
        return ns
    if kind == "multicrisis":
        # v2 关键补数据：**同时**有 2~3 个濒死维度，逼标注者把生存需求互相排序。
        # v1 缺的正是这个——crisis 只抬一维、其余压在阈值下（受伤 0~20 连候选都进不去），
        # 所以"饥饿99 + 受伤95 该先治伤还是先吃饭"这类样本一条都没有，
        # 模型自然学不会，上线后就表现为重伤的人先去吃饭。
        ns = NeedState(r(0, 45), r(0, 45), r(0, 45), r(0, 45), r(0, 45), r(0, 45), r(20, 70))
        for d in rng.sample(SURVIVAL_DIMS, rng.randint(2, 3)):
            setattr(ns, d, r(80, 100))
        return ns
    if kind == "scarcity":
        # 饿着但没饭：教模型看"食物:断供/紧张"这一轴，别再往食堂跑
        ns = NeedState(r(60, 100), r(0, 50), r(0, 50), r(0, 40), r(0, 50), r(0, 50), r(20, 70))
        return ns
    lo, hi = (0, 55) if rng.random() < 0.5 else (0, 80)
    # 受伤上限从 35 抬到 60：v1 里受伤几乎永远够不到 Cure 的阈值(30)，
    # 治伤极少作为候选出现，训练信号严重不足。
    return NeedState(r(lo, hi), r(lo, hi), r(0, hi), r(0, 60), r(lo, hi), r(lo, hi), r(0, hi))


def make_situation(rng: random.Random, kind: str) -> ContextState:
    """按类别造情境。kind ∈ command/cmd_crisis/cmd_watch/threat/worktask/crisis/multicrisis/scarcity/event/daily/social/kill/shift"""
    phase = rng.choice(PHASES)
    parole_near = rng.random() < 0.2
    # v2 新增四轴的采样：与既有轴保持相关性，别造出自相矛盾的情境
    #（比如"仇人在场"却"仇怨:无"，或"假释临近"却"表现:差"）
    food = rng.choices([SUPPLY_OK, SUPPLY_TIGHT, SUPPLY_OUT], weights=[0.7, 0.2, 0.1])[0]
    perform = (PERFORM_GOOD if parole_near
               else rng.choices([PERFORM_GOOD, PERFORM_MID, PERFORM_BAD],
                                weights=[0.2, 0.55, 0.25])[0])
    ctx = ContextState(
        needs=_sample_needs(rng, kind),
        threat="无",
        event="无",
        phase=phase,
        personality=rng.choice(PERSONALITIES),
        surveillance=rng.choice(SURVEIL),
        backer_present=rng.random() < 0.3,
        enemy_present=False,
        yard_crowd=rng.choice(CROWD),
        parole_near=parole_near,
        has_corpse=rng.random() < 0.06,
        has_labor=rng.random() < 0.3,
        motive=rng.choice(MOTIVES),
        player_command="无",
        food_supply=food,
        grudge=GRUDGE_NONE,
        perform=perform,
        busted_recently=rng.random() < 0.12,
        near_rel=NEAR_NONE,
        other_intent=INTENT_NONE,
        social_recent=SOC_NONE,
        obligation=rng.choices([OBLIG_NONE, OBLIG_OWE, OBLIG_OWED],
                               weights=[0.8, 0.12, 0.08])[0],
        has_weapon=rng.random() < 0.12,
    )
    ctx._cmd_goal = None  # type: ignore[attr-defined]

    if kind == "scarcity":
        # 断供/紧张二选一，饿着却没饭吃才是这类情境的要点
        ctx.food_supply = rng.choices([SUPPLY_OUT, SUPPLY_TIGHT], weights=[0.7, 0.3])[0]
    if kind == "threat":
        ctx.threat = rng.choice(["仇人", "狱犬", "狱警"])
        if ctx.threat == "仇人":
            ctx.enemy_present = True
            # 仇人在场必然有仇怨，深浅决定该吵架还是该动手
            ctx.grudge = rng.choices([GRUDGE_SOME, GRUDGE_DEEP], weights=[0.5, 0.5])[0]
        ctx.phase = rng.choice([PHASE_YARD, PHASE_FREE, PHASE_TRAIN])
    elif kind == "worktask":
        ctx.has_labor = True
        ctx.phase = rng.choice([PHASE_TRAIN, PHASE_FREE])
    elif kind == "event":
        ctx.event = rng.choice(["斗殴", "狱警搜牢", "警报", "吵架声"])
        if ctx.event in ("斗殴", "警报"):
            ctx.surveillance = rng.choice(["高", "中"])
    elif kind in ("command", "cmd_crisis", "cmd_watch"):
        avoid = SURVIVAL_GOALS if kind == "cmd_crisis" else set()
        if kind == "cmd_watch" and rng.random() < 0.55:
            cmd = rng.choice(WATCH_COMMANDS)
        else:
            cmd = _pick_command(rng, avoid)
        ctx.player_command = _command_text(cmd)
        ctx._cmd_goal = cmd  # type: ignore[attr-defined]
        if kind == "cmd_watch":
            ctx.threat = rng.choice(["狱警", "狱犬"])
            ctx.surveillance = "高"
            ctx.phase = rng.choice([PHASE_YARD, PHASE_FREE, PHASE_TRAIN])
    elif kind == "kill":
        ctx.phase = rng.choice([PHASE_YARD, PHASE_FREE, PHASE_TRAIN])
        ctx.enemy_present = True
        ctx.grudge = rng.choices([GRUDGE_SOME, GRUDGE_DEEP], weights=[0.35, 0.65])[0]
        ctx.threat = "仇人"
        ctx.has_weapon = rng.random() < 0.5
        if ctx.has_weapon:
            ctx.surveillance = rng.choice(["低", "中"])
        else:
            ctx.surveillance = rng.choice(["低", "中", "高"])
    elif kind == "shift":
        # 典狱长夜班：玩家走近谁，谁的 caption 就是 威胁:狱警 + 监视度:高。
        # 再叠缴械余波 / 藏器 / 警报，让模型看见处置之后的世界。
        ctx.threat = "狱警"
        ctx.surveillance = "高"
        ctx.phase = rng.choice([PHASE_YARD, PHASE_FREE, PHASE_SLEEP, PHASE_TRAIN])
        roll = rng.random()
        if roll < 0.25:
            # 藏器被撞见 / 刚被栽赃：刀还在手里，典狱长站在眼前。
            ctx.has_weapon = True
            ctx.enemy_present = True
            ctx.grudge = GRUDGE_DEEP
        elif roll < 0.45:
            ctx.event = rng.choice(["警报", "狱警搜牢", "斗殴"])
            ctx.has_weapon = rng.random() < 0.3
        elif roll < 0.65:
            # 搜身收缴之后：刀没了，刚被抓过。
            ctx.has_weapon = False
            ctx.enemy_present = True
            ctx.grudge = GRUDGE_DEEP
            ctx.busted_recently = True
        elif roll < 0.82:
            ctx.needs.injury = rng.randint(50, 90)
        else:
            # 遣回牢房：指令通道，引擎会强制，模型也要看见。
            ctx.player_command = _command_text("Rest")
            ctx._cmd_goal = "Rest"  # type: ignore[attr-defined]
    elif kind == "social":
        # v3/P2 的核心新情境：近旁有个具体的人，我该冲他走哪步社交棋。
        # 造一组**自洽**的社交轴（关系↔对方意图↔人情不互相打架）。
        ctx.phase = rng.choice([PHASE_YARD, PHASE_FREE])
        rel = rng.choice([NEAR_FOE, NEAR_ALLY, NEAR_CREDITOR, NEAR_DEBTOR, NEAR_STRANGER])
        ctx.near_rel = rel
        if rel == NEAR_FOE:
            ctx.other_intent = rng.choice([INTENT_HOSTILE, INTENT_UNKNOWN])
            ctx.enemy_present = True
            ctx.grudge = rng.choice([GRUDGE_SOME, GRUDGE_DEEP])
            ctx.social_recent = rng.choice([SOC_THREATENED, SOC_NONE])
        elif rel == NEAR_ALLY:
            ctx.other_intent = rng.choice([INTENT_FRIENDLY, INTENT_ALLY])
            ctx.social_recent = rng.choice([SOC_ALLIED, SOC_NONE])
        elif rel == NEAR_CREDITOR:              # 债主在近旁 → 我欠着人情
            ctx.other_intent = rng.choice([INTENT_FRIENDLY, INTENT_DEAL, INTENT_UNKNOWN])
            ctx.obligation = OBLIG_OWE
            ctx.social_recent = rng.choice([SOC_DEAL, SOC_NONE])
        elif rel == NEAR_DEBTOR:                # 欠我的人在近旁 → 有人欠我
            ctx.other_intent = rng.choice([INTENT_WEAK, INTENT_FRIENDLY, INTENT_UNKNOWN])
            ctx.obligation = OBLIG_OWED
            ctx.social_recent = rng.choice([SOC_PROTECTED, SOC_DEAL, SOC_NONE])
        else:                                   # 生人
            ctx.other_intent = rng.choice([INTENT_DEAL, INTENT_FRIENDLY, INTENT_WEAK,
                                           INTENT_UNKNOWN])
            ctx.social_recent = rng.choice([SOC_RECONCILED, SOC_NONE])
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/situations.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 配额保稀有。v4 加了 cmd_crisis / cmd_watch / kill：
    # 典狱长命令 vs 濒死、命令 vs 被盯、弄凶器/下杀手，都是线上已经有、模型没见过的。
    quota = {"command": 0.08, "cmd_crisis": 0.07, "cmd_watch": 0.08,
             "threat": 0.08, "worktask": 0.06,
             "crisis": 0.09, "multicrisis": 0.09, "scarcity": 0.08,
             "event": 0.10, "social": 0.08, "kill": 0.08,
             "shift": 0.07, "daily": 0.04}
    plan: list[str] = []
    for k, frac in quota.items():
        plan += [k] * int(args.n * frac)
    while len(plan) < args.n:
        plan.append("daily")
    rng.shuffle(plan)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    cat_count: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as f:
        for i, kind in enumerate(plan):
            ctx = make_situation(rng, kind)
            cands = gen_candidates(ctx, rng)
            if len(cands) < 2:
                continue
            rec = {
                "situation_id": f"sit_{i:06d}",
                "kind": kind,
                "context": build_context_caption(ctx),
                "ctx": {
                    "needs": vars(ctx.needs),
                    "threat": ctx.threat, "event": ctx.event, "phase": ctx.phase,
                    "personality": ctx.personality, "surveillance": ctx.surveillance,
                    "backer_present": ctx.backer_present, "enemy_present": ctx.enemy_present,
                    "yard_crowd": ctx.yard_crowd, "parole_near": ctx.parole_near,
                    "has_corpse": ctx.has_corpse, "has_labor": ctx.has_labor,
                    "motive": ctx.motive, "player_command": ctx.player_command,
                    "food_supply": ctx.food_supply, "grudge": ctx.grudge,
                    "perform": ctx.perform, "busted_recently": ctx.busted_recently,
                    "near_rel": ctx.near_rel, "other_intent": ctx.other_intent,
                    "social_recent": ctx.social_recent, "obligation": ctx.obligation,
                    "has_weapon": ctx.has_weapon,
                },
                "candidates": cands,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            for c in cands:
                cat_count[c["source"]] = cat_count.get(c["source"], 0) + 1
    print(f"写出 {kept} 条情境 → {out}")
    print("候选来源分布：", dict(sorted(cat_count.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
