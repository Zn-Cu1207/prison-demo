"""v5 硬规则对：v4 没学会的点 + 典狱长夜班的因果。

和 intensity/axis 一样零 API：同一情境两个候选，规则裁定。
BT 损失只比较同情境相对顺序——这正是 v4 行为探针在测的东西。

覆盖：
  1) 生存：重伤 Cure > 中等口渴 Drink / 饥饿 Eat
  2) 凶器：没刀 GetWeapon > Assassinate；有刀反过来；狱警面前 LayLow > 下手
  3) 被盯：狱警/典狱长走近 → LayLow > Eat/Fight/GetWeapon（夜班主循环）
  4) 夜班事件：警报/搜牢 → Hide/LayLow > 越狱/赌钱
  5) 断供：没饭时 Barter/KitchenDuty > Eat
  6) 缴械余波：凶器:无 时不要空挥

用法：
    python gen_hard_pairs.py --n 3000 --out data/hard_prefs_v5.jsonl
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

from app.actions import BY_GOAL, ILLICIT_GOALS  # noqa: E402
from app.context import (  # noqa: E402
    GRUDGE_DEEP,
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
    Candidate,
    ContextState,
    NeedState,
    build_context_caption,
)

PHASES = [PHASE_YARD, PHASE_FREE, PHASE_TRAIN]


def _needs(inj=10, hun=25, thi=25, fat=15, **kw) -> NeedState:
    return NeedState(hunger=hun, thirst=thi, fatigue=fat, injury=inj,
                     boredom=kw.get("bor", 20), social=kw.get("soc", 15),
                     stress=kw.get("st", 30))


def _ctx(**kw) -> ContextState:
    d = dict(
        needs=_needs(), threat="无", event="无", phase=PHASE_YARD,
        personality="普通", surveillance="中", enemy_present=False,
        food_supply=SUPPLY_OK, grudge="无", perform=PERFORM_MID,
        has_weapon=False,
    )
    d.update(kw)
    return ContextState(**d)


def _cand(goal: str, source: str) -> dict:
    d = Candidate(goal, source, b_force_plan=(source == "command")).to_payload()
    d["illicit"] = goal in ILLICIT_GOALS
    return d


def _pair(sid: str, ctx: ContextState, ch: tuple[str, str], rj: tuple[str, str],
          why: str, conf: float = 0.95) -> dict:
    cap = build_context_caption(ctx)
    return {
        "situation_id": sid,
        "pair_id": f"{sid}_p0",
        "context": cap,
        "chosen": _cand(*ch),
        "rejected": _cand(*rj),
        "confidence": conf,
        "label_by": "hard",
        "reason": why,
    }


def _cmd_text(goal: str) -> str:
    act = BY_GOAL.get(goal)
    if act is None:
        return f"命令去{goal}"
    return f"命令去{act.label}({act.desc})"


def make_survival(rng: random.Random, i: int) -> dict:
    """重伤必须压过尚未濒死的饥渴——v4 探针 Drink 仍居首。"""
    if rng.random() < 0.28:
        # 对齐 eval 探针：饥饿99 + 受伤95 + 口渴40 → Cure 必须压过 Drink/Eat。
        ctx = _ctx(needs=_needs(inj=95, hun=99, thi=40),
                   phase=rng.choice(PHASES), personality=rng.choice(
                       ["暴躁", "书呆", "老油条", "新人", "普通"]))
        rej = rng.choice([("Drink", "need"), ("Eat", "need")])
        return _pair(f"hd_surv_eval_{i:05d}", ctx, ("Cure", "need"), rej,
                     f"探针伤95饥99渴40：Cure > {rej[0]}")
    kind = rng.choice(["cure_drink", "cure_eat", "cure_both"])
    thi = rng.randint(30, 55) if kind != "cure_eat" else rng.randint(10, 40)
    hun = rng.randint(30, 70) if kind != "cure_drink" else rng.randint(10, 40)
    inj = rng.randint(85, 100)
    ctx = _ctx(needs=_needs(inj=inj, hun=hun, thi=thi),
               phase=rng.choice(PHASES), personality=rng.choice(
                   ["暴躁", "书呆", "老油条", "新人", "普通"]))
    if kind == "cure_drink":
        return _pair(f"hd_surv_cd_{i:05d}", ctx, ("Cure", "need"), ("Drink", "need"),
                     f"受伤{inj}濒死，口渴{thi}未濒死，Cure > Drink")
    if kind == "cure_eat":
        return _pair(f"hd_surv_ce_{i:05d}", ctx, ("Cure", "need"), ("Eat", "need"),
                     f"受伤{inj}濒死，饥饿{hun}还能撑，Cure > Eat")
    return _pair(f"hd_surv_cb_{i:05d}", ctx, ("Cure", "need"),
                 rng.choice([("Drink", "need"), ("Eat", "need")]),
                 f"受伤{inj}，饥{hun}渴{thi}，先止血")


def make_weapon(rng: random.Random, i: int) -> dict:
    """凶器轴：同情境两个动作，比 axis 的「同一动作换 caption」更贴探针。"""
    armed = rng.random() < 0.5
    ctx = _ctx(
        needs=_needs(inj=rng.randint(0, 20), st=50),
        threat="仇人", enemy_present=True, grudge=GRUDGE_DEEP,
        surveillance="低", phase=rng.choice(PHASES), has_weapon=armed,
        personality=rng.choice(["暴躁", "老油条", "普通"]),
    )
    if armed:
        return _pair(f"hd_wp_on_{i:05d}", ctx, ("Assassinate", "threat"),
                     ("GetWeapon", "daily"), "有刀：下手 > 再去弄刀")
    return _pair(f"hd_wp_off_{i:05d}", ctx, ("GetWeapon", "daily"),
                 ("Assassinate", "threat"), "没刀：先弄凶器，空挥没意义")


def make_watch(rng: random.Random, i: int) -> dict:
    """夜班主循环：典狱长走近 = 威胁:狱警。收敛压过吃饭/动手/弄刀。"""
    hun = rng.randint(35, 62)   # 中等饿，不是濒死
    armed = rng.random() < 0.45
    other = rng.choice([
        ("Eat", "need"), ("Fight", "threat"), ("Assassinate", "threat"),
        ("GetWeapon", "daily"), ("Gamble", "daily"), ("PrisonBreak", "daily"),
    ])
    if other[0] in ("Fight", "Assassinate"):
        enemy, grudge, threat_extra = True, GRUDGE_DEEP, "狱警"
    else:
        enemy, grudge, threat_extra = False, "无", "狱警"
    ctx = _ctx(
        needs=_needs(hun=hun, inj=rng.randint(0, 20)),
        threat=threat_extra, surveillance="高", phase=rng.choice(PHASES),
        enemy_present=enemy, grudge=grudge, has_weapon=armed,
        personality=rng.choice(["暴躁", "书呆", "老油条", "新人", "普通"]),
    )
    return _pair(f"hd_watch_{other[0]}_{i:05d}", ctx, ("LayLow", "threat"), other,
                 f"典狱长/狱警在眼前，收敛 > {other[0]}（饥饿{hun}未濒死）")


def make_shift_event(rng: random.Random, i: int) -> dict:
    """夜班警报/搜牢/骚乱。"""
    ev = rng.choice(["警报", "狱警搜牢", "斗殴"])
    ctx = _ctx(
        needs=_needs(hun=rng.randint(10, 45), fat=rng.randint(20, 50)),
        threat=rng.choice(["狱警", "无"]), event=ev, surveillance="高",
        phase=rng.choice([PHASE_YARD, PHASE_FREE, PHASE_SLEEP]),
        has_weapon=rng.random() < 0.3,
    )
    if ev == "狱警搜牢":
        return _pair(f"hd_ev_hide_{i:05d}", ctx, ("Hide", "event"),
                     rng.choice([("PrisonBreak", "daily"), ("Gamble", "daily"),
                                 ("GetWeapon", "daily")]),
                     "搜牢：藏东西 > 越狱/赌/弄刀")
    if ev == "警报":
        return _pair(f"hd_ev_alarm_{i:05d}", ctx, ("LayLow", "threat"),
                     rng.choice([("PrisonBreak", "daily"), ("Gamble", "daily"),
                                 ("Fight", "threat"), ("Assassinate", "threat")]),
                     "警报：收敛 > 趁乱动手/越狱")
    return _pair(f"hd_ev_riot_{i:05d}", ctx, ("Snitch", "event"),
                 rng.choice([("Gamble", "daily"), ("Patrol", "daily")]),
                 "斗殴+高监视：告密/收敛比闲逛更说得通")


def make_food(rng: random.Random, i: int) -> dict:
    """断供时去食堂是白跑——v4 把方向学反了。同情境 Eat vs 弄吃的。"""
    hun = rng.randint(50, 82)  # 饿但没到 85 濒死
    ctx = _ctx(
        needs=_needs(hun=hun), food_supply=SUPPLY_OUT,
        has_labor=True, phase=rng.choice([PHASE_FREE, PHASE_TRAIN]),
    )
    alt = rng.choice([("KitchenDuty", "worktask"), ("Barter", "daily")])
    return _pair(f"hd_food_{i:05d}", ctx, alt, ("Eat", "need"),
                 f"食物断供且饥饿{hun}未濒死：{alt[0]} > Eat")


def make_confiscate(rng: random.Random, i: int) -> dict:
    """夜班搜缴之后：刀没了，别空挥；若典狱长还站在旁边，先收敛。"""
    still_watched = rng.random() < 0.5
    ctx = _ctx(
        needs=_needs(st=45), threat="狱警" if still_watched else "仇人",
        surveillance="高" if still_watched else "低",
        enemy_present=True, grudge=GRUDGE_DEEP, has_weapon=False,
        phase=rng.choice(PHASES), busted_recently=True,
    )
    if still_watched:
        return _pair(f"hd_cf_w_{i:05d}", ctx, ("LayLow", "threat"),
                     rng.choice([("Assassinate", "threat"), ("GetWeapon", "daily")]),
                     "刚被缴械且典狱长还在眼前：收敛 > 下手/再去弄刀")
    return _pair(f"hd_cf_n_{i:05d}", ctx, ("GetWeapon", "daily"),
                 ("Assassinate", "threat"),
                 "刚被缴械、管教已走：先弄凶器，空挥没意义")


def make_grudge(rng: random.Random, i: int) -> dict:
    """夜班调解后仇怨变浅：该吵不该打。"""
    deep = rng.random() < 0.5
    ctx = _ctx(
        needs=_needs(st=rng.randint(40, 80)), threat="仇人",
        enemy_present=True, grudge=GRUDGE_DEEP if deep else GRUDGE_SOME,
        surveillance="低", phase=rng.choice(PHASES),
    )
    if deep:
        return _pair(f"hd_gd_{i:05d}", ctx, ("Fight", "threat"), ("Argue", "threat"),
                     "仇怨深：动手 > 吵两句")
    return _pair(f"hd_gs_{i:05d}", ctx, ("Argue", "threat"), ("Fight", "threat"),
                 "仇怨只有「有」（调解后）：吵架 > 动手")


def make_plant(rng: random.Random, i: int) -> dict:
    """夜班栽赃：caption 变成 凶器:有。典狱长还在就收敛；走开才能动手。"""
    still_watched = rng.random() < 0.55
    ctx = _ctx(
        needs=_needs(st=55, hun=rng.randint(20, 50)),
        threat="狱警" if still_watched else "仇人",
        surveillance="高" if still_watched else "低",
        enemy_present=True, grudge=GRUDGE_DEEP, has_weapon=True,
        phase=rng.choice(PHASES),
    )
    if still_watched:
        return _pair(f"hd_pl_w_{i:05d}", ctx, ("LayLow", "threat"),
                     rng.choice([("Assassinate", "threat"), ("Fight", "threat"),
                                 ("GetWeapon", "daily")]),
                     "被栽赃后刀在手里，但典狱长还在眼前：收敛 > 动手")
    return _pair(f"hd_pl_n_{i:05d}", ctx, ("Assassinate", "threat"),
                 ("GetWeapon", "daily"),
                 "被栽赃且管教已走：下手 > 再去弄刀")


def make_recall(rng: random.Random, i: int) -> dict:
    """夜班遣回牢房：玩家指令回铺，非濒死时压过吃饭/闲逛。"""
    hun = rng.randint(30, 70)
    ctx = _ctx(
        needs=_needs(hun=hun, inj=rng.randint(0, 25)),
        phase=rng.choice(PHASES), player_command=_cmd_text("Rest"),
        threat=rng.choice(["无", "狱警"]),
        surveillance=rng.choice(["中", "高"]),
    )
    return _pair(f"hd_rc_{i:05d}", ctx, ("Rest", "command"),
                 rng.choice([("Eat", "need"), ("Patrol", "daily"), ("Gamble", "daily")]),
                 f"典狱长遣回牢房，饥饿{hun}未濒死：指令 > 吃饭/闲逛")


def make_perform(rng: random.Random, i: int) -> dict:
    ctx = _ctx(
        needs=_needs(), phase=PHASE_FREE, surveillance="低",
        perform=PERFORM_GOOD, parole_near=True,
    )
    return _pair(f"hd_pf_{i:05d}", ctx,
                 rng.choice([("Patrol", "daily"), ("Read", "daily")]),
                 rng.choice([("Gamble", "daily"), ("Assassinate", "threat"),
                             ("GetWeapon", "daily")]),
                 "表现好/快假释：守规矩 > 违禁")


SPECS = [
    (0.18, make_survival),
    (0.14, make_weapon),
    (0.18, make_watch),
    (0.12, make_shift_event),
    (0.10, make_food),
    (0.08, make_confiscate),
    (0.06, make_plant),
    (0.06, make_grudge),
    (0.04, make_perform),
    (0.04, make_recall),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/hard_prefs_v5.jsonl")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    plan: list[int] = []
    for idx, (w, _fn) in enumerate(SPECS):
        plan += [idx] * int(args.n * w)
    while len(plan) < args.n:
        plan.append(0)
    rng.shuffle(plan)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, si in enumerate(plan):
            rec = SPECS[si][1](rng, i)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"写出 {args.n} 条硬规则对 → {out}")
    tags = Counter()
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tags[json.loads(line)["situation_id"].split("_")[1]] += 1
    print("种类：", dict(sorted(tags.items())))
    print("全部 label_by=hard，零 API。")


if __name__ == "__main__":
    main()
