"""阶段3：评估价值模型。

1) 成对准确率：在 preferences.jsonl（或其 val 切分）上，v_chosen > v_rejected 的比例。
2) 单调性：固定情境、抬高饥饿，Eat 的分应单调上升（RA 的关键验收项）。
3) 跨类别宏观排序：threat > event > command > need > worktask > daily 的中位分排序。

用法：
    python eval.py --model merged/value-1p5b --pref data/preferences.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from common import build_input
import value_server as vs

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))
from app.actions import BY_GOAL  # noqa: E402
from app.context import (  # noqa: E402
    Candidate,
    ContextState,
    GRUDGE_DEEP,
    INTENT_HOSTILE,
    NEAR_FOE,
    NEAR_STRANGER,
    NeedState,
    OBLIG_OWE,
    build_context_caption,
)


def _cmd_text(goal: str) -> str:
    a = BY_GOAL[goal]
    return f"命令去{a.label}({a.desc})"


def _v(ctx: ContextState, goal: str, source: str = "daily", force: bool = False) -> float:
    return vs.score_group(
        build_context_caption(ctx),
        [Candidate(goal, source, b_force_plan=force).to_payload()],
    )[0]


def _ok(cond: bool, msg: str) -> bool:
    print(f"  {'✅' if cond else '❌'} {msg}")
    return cond


def _acc_on(rows: list[dict], tag: str):
    """分块打分 + 统计成对准确率（整体 & 按标签来源拆分）。"""
    if not rows:
        return
    texts = []
    for r in rows:
        # 与 train_value.PairDataset 一致：强度对两侧 context 不同
        texts.append(build_input(r.get("context_chosen") or r["context"], r["chosen"]))
        texts.append(build_input(r.get("context_rejected") or r["context"], r["rejected"]))
    scores: list[float] = []
    step = 256
    for i in range(0, len(texts), step):
        scores.extend(vs._score_texts(texts[i:i + step]))
        if (i // step) % 10 == 0:
            print(f"    打分 {min(i + step, len(texts))}/{len(texts)}", end="\r", flush=True)
    correct = tot = 0
    by: dict[str, list[int]] = {}
    margins: list[float] = []
    for i, r in enumerate(rows):
        vc, vr = scores[2 * i], scores[2 * i + 1]
        ok = vc > vr
        correct += ok
        tot += 1
        margins.append(vc - vr)
        d = by.setdefault(r.get("label_by", "hy"), [0, 0])
        d[0] += ok
        d[1] += 1
    print(f"\n  {tag}：成对准确率 {correct}/{tot} = {correct / max(1, tot):.3f}"
          f"  (中位 v差 {statistics.median(margins):+.2f})")
    for m, (c, t) in sorted(by.items()):
        label = {"rule": "护栏(跨档位)", "hy": "大模型(同档位)"}.get(m, m)
        print(f"    [{label}] {c}/{t} = {c / max(1, t):.3f}")


def pairwise_acc(pref_paths: list[str], val_frac: float, seed: int):
    """在与训练完全相同的切分上报准确率：val 是held-out(真指标)，train 供对比看过拟合。

    必须传入**与训练时完全相同的那几份**偏好文件：切分是对 situation_id 列表洗牌得到的，
    少一份文件就会洗出另一套 val，报出来的"held-out 准确率"便名不副实。
    """
    from train_value import split_by_situation  # 复用同一套切分，避免口径不一致

    rows: list[dict] = []
    for p in pref_paths:
        rows.extend(json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip())
    train_rows, val_rows = split_by_situation(rows, val_frac, seed)
    print(f"成对准确率（按 situation_id 整组切分，val 为 held-out）：")
    _acc_on(val_rows, "val (held-out)")
    _acc_on(train_rows[: len(val_rows)], "train(同量抽样)")


def monotonicity():
    print("\n单调性（抬饥饿 → Eat 分应上升）：")
    vals = []
    for h in (20, 40, 60, 80, 95):
        ctx = ContextState(needs=NeedState(hunger=h))
        cap = build_context_caption(ctx)
        v = vs.score_group(cap, [Candidate("Eat", "need").to_payload()])[0]
        vals.append((h, v))
        print(f"  饥饿{h:>3} → Eat v={v:+.3f}")
    hi = [v for h, v in vals if h >= 60]
    print("  高区单调" + ("✅" if hi == sorted(hi) else "⚠️不严格"))
    # v2 验收点：不只看方向，还要看**坡度**。
    # v1 的毛病是"需求强度是平的"——饥饿 60 和 95 分差太小，"多急"表达不出来，
    # 于是急事和不急的事挤在一起，靠引擎侧硬规则才拉得开。
    d = dict(vals)
    gap = d[95] - d[60]
    print(f"  强度坡度 v(饥饿95) − v(饥饿60) = {gap:+.3f}  "
          + ("✅ 拉开了" if gap >= 0.35 else "⚠️ 偏平（v1 的老问题，期望 ≥0.35）"))


def survival_order():
    """v2 核心验收：多重生存危机下的排序，必须 治伤 > 吃饭。

    这是用户实测到的那个 bug：饥饿99 + 受伤95 时模型选了吃饭，治伤连前五都没进。
    伤口不处理会持续恶化，饿一顿撑得住——现实里也该先止血。
    """
    print("\n多重生存危机排序（饥饿99 + 受伤95，应 Cure 居首）：")
    ctx = ContextState(needs=NeedState(hunger=99, thirst=40, fatigue=30, injury=95))
    cap = build_context_caption(ctx)
    cands = [Candidate(g, "need") for g in ("Cure", "Eat", "Drink", "Sleep")]
    vals = vs.score_group(cap, [c.to_payload() for c in cands])
    ranked = sorted(zip((c.goal for c in cands), vals), key=lambda kv: -kv[1])
    for i, (g, v) in enumerate(ranked, 1):
        print(f"  {i}. {g:<6} v={v:+.3f}")
    top = ranked[0][0]
    dv = dict(ranked)
    print(f"  Cure − Eat = {dv['Cure'] - dv['Eat']:+.3f}")
    print("  " + ("✅ 治伤居首" if top == "Cure"
                  else f"❌ 仍是「{top}」居首——伤势低估没修好"))


def axes_probe():
    """v2 新增四轴是否真的被模型用上了（每轴只动一个变量，其余不变）。"""
    print("\n新四轴敏感度（只动一个变量，看分是否按预期移动）：")

    def v_of(ctx: ContextState, cand: Candidate) -> float:
        return vs.score_group(build_context_caption(ctx), [cand.to_payload()])[0]

    eat = Candidate("Eat", "need")
    # ① 食物断供：饿着但没饭，去吃饭是白跑，分该降
    base = ContextState(needs=NeedState(hunger=90))
    v_ok = v_of(base, eat)
    out = ContextState(needs=NeedState(hunger=90), food_supply="断供")
    v_out = v_of(out, eat)
    print(f"  食物充足→断供：Eat {v_ok:+.3f} → {v_out:+.3f}  (差 {v_out - v_ok:+.3f})"
          + ("  ✅" if v_out < v_ok - 0.1 else "  ⚠️ 不敏感"))

    # ② 近期被抓：刚出禁闭该老实，违禁行为(赌钱)分该降
    gamble = Candidate("Gamble", "daily")
    v_free = v_of(ContextState(), gamble)
    v_busted = v_of(ContextState(busted_recently=True), gamble)
    print(f"  近期被抓 否→是：Gamble {v_free:+.3f} → {v_busted:+.3f}  (差 {v_busted - v_free:+.3f})"
          + ("  ✅" if v_busted < v_free - 0.05 else "  ⚠️ 不敏感"))

    # ③ 表现分：表现好的人珍惜假释记录，违禁行为分该更低
    v_bad = v_of(ContextState(perform="差"), gamble)
    v_good = v_of(ContextState(perform="好"), gamble)
    print(f"  表现 差→好：Gamble {v_bad:+.3f} → {v_good:+.3f}  (差 {v_good - v_bad:+.3f})"
          + ("  ✅" if v_good < v_bad - 0.05 else "  ⚠️ 不敏感"))

    # ④ 仇怨：无仇怨时动手不合理，仇怨深时说得通
    fight = Candidate("Fight", "threat")
    v_none = v_of(ContextState(threat="仇人", enemy_present=True, grudge="无"), fight)
    v_deep = v_of(ContextState(threat="仇人", enemy_present=True, grudge="深"), fight)
    print(f"  仇怨 无→深：Fight {v_none:+.3f} → {v_deep:+.3f}  (差 {v_deep - v_none:+.3f})"
          + ("  ✅" if v_deep > v_none + 0.05 else "  ⚠️ 不敏感"))


def macro_ranking():
    print("\n跨类别宏观排序（各来源中位 v，应 threat≥event≥command≥need≥worktask≥daily）：")
    probes = {
        "threat": (ContextState(threat="仇人", enemy_present=True), Candidate("Fight", "threat")),
        "event": (ContextState(event="警报"), Candidate("Snitch", "event")),
        "command": (ContextState(player_command="命令去巡逻(在放风场溜达)"), Candidate("Patrol", "command", b_force_plan=True)),
        "need": (ContextState(needs=NeedState(hunger=60)), Candidate("Eat", "need")),
        "worktask": (ContextState(has_labor=True), Candidate("Labor", "worktask")),
        "daily": (ContextState(), Candidate("Patrol", "daily")),
    }
    meds = {}
    for src, (ctx, cand) in probes.items():
        cap = build_context_caption(ctx)
        v = vs.score_group(cap, [cand.to_payload()])[0]
        meds[src] = v
    for src in ("threat", "event", "command", "need", "worktask", "daily"):
        print(f"  {src:<9} v={meds[src]:+.3f}")
    order = ["threat", "event", "command", "need", "worktask", "daily"]
    vs_ = [meds[s] for s in order]
    # 只检查两端：威胁应压过日常；指令应压过日常。中间 need 经常很高，不拿来一票否决。
    print("  " + ("✅ 威胁/指令压过日常" if meds["threat"] > meds["daily"] and meds["command"] > meds["daily"]
                  else "❌ 日常没有被压住"))
    if meds["need"] > meds["threat"]:
        print("  ⚠️ 普通需求中位分高于威胁（v1 老问题：need 轴偏高）")


def weapon_probe():
    """v4：caption 凶器轴 + 弄凶器/下杀手候选。"""
    print("\n凶器轴（v4 新增，v1 没见过这两个动作）：")
    base = dict(threat="仇人", enemy_present=True, grudge=GRUDGE_DEEP,
                surveillance="低", phase="放风")
    armed = ContextState(**base, has_weapon=True)
    bare = ContextState(**base, has_weapon=False)
    v_kill_on = _v(armed, "Assassinate", "threat")
    v_kill_off = _v(bare, "Assassinate", "threat")
    print(f"  Assassinate 有刀→没刀：{v_kill_on:+.3f} → {v_kill_off:+.3f}  "
          f"(差 {v_kill_on - v_kill_off:+.3f})")
    _ok(v_kill_on > v_kill_off + 0.15, "有刀才说得通下杀手")

    v_get_off = _v(bare, "GetWeapon", "daily")
    v_get_on = _v(armed, "GetWeapon", "daily")
    print(f"  GetWeapon 没刀→有刀：{v_get_off:+.3f} → {v_get_on:+.3f}  "
          f"(差 {v_get_off - v_get_on:+.3f})")
    _ok(v_get_off > v_get_on + 0.15, "没刀才该去弄凶器")

    print(f"  没刀时 GetWeapon {v_get_off:+.3f} vs Assassinate {v_kill_off:+.3f}")
    _ok(v_get_off > v_kill_off, "没刀：弄凶器 > 空挥下手")
    print(f"  有刀时 Assassinate {v_kill_on:+.3f} vs GetWeapon {v_get_on:+.3f}")
    _ok(v_kill_on > v_get_on, "有刀：下手 > 再跑一趟弄刀")

    watched = ContextState(
        threat="狱警", enemy_present=True, grudge=GRUDGE_DEEP,
        surveillance="高", phase="放风", has_weapon=True,
    )
    v_kill_w = _v(watched, "Assassinate", "threat")
    v_low = _v(watched, "LayLow", "threat")
    print(f"  狱警盯着：Assassinate {v_kill_w:+.3f} vs LayLow {v_low:+.3f}")
    _ok(v_low > v_kill_w, "当着管教面收敛 > 下杀手")


def command_probe():
    """v4：新动词当指令；指令 vs 濒死；指令 vs 被盯。"""
    print("\n典狱长指令（v4 命令池已扩到新动作）：")
    calm = dict(phase="放风", needs=NeedState(hunger=35, thirst=30, fatigue=25, injury=10))

    def cmd_vs(goal: str, other: str, other_src: str) -> tuple[float, float]:
        ctx = ContextState(**calm, player_command=_cmd_text(goal))
        return _v(ctx, goal, "command", True), _v(ctx, other, other_src)

    for goal, other, osrc, label in (
        ("Ally", "Patrol", "daily", "命令结盟 vs 闲逛"),
        ("LayLow", "Socialize", "need", "命令收敛 vs 社交"),
        ("GetWeapon", "Patrol", "daily", "命令弄凶器 vs 闲逛"),
        ("Assassinate", "Patrol", "daily", "命令下杀手 vs 闲逛"),
        ("Propose", "Rest", "daily", "命令提议 vs 歇着"),
    ):
        vc, vo = cmd_vs(goal, other, osrc)
        print(f"  {label}：cmd {vc:+.3f} vs {other} {vo:+.3f}  (差 {vc - vo:+.3f})")
        _ok(vc > vo, f"非紧急时 {goal} 指令压过 {other}")

    crisis = ContextState(
        needs=NeedState(hunger=99, thirst=40, fatigue=30, injury=95),
        phase="放风", player_command=_cmd_text("Patrol"),
    )
    v_cure = _v(crisis, "Cure", "need")
    v_eat = _v(crisis, "Eat", "need")
    v_cmd = _v(crisis, "Patrol", "command", True)
    print(f"  濒死+命令放风：Cure {v_cure:+.3f}  Eat {v_eat:+.3f}  Patrol(指令) {v_cmd:+.3f}")
    _ok(v_cure > v_cmd, "濒死治伤压过指令")
    _ok(v_cure > v_eat, "濒死时治伤压过吃饭")

    watched = ContextState(
        needs=calm["needs"], phase="放风", threat="狱警", surveillance="高",
        player_command=_cmd_text("Assassinate"),
    )
    v_c = _v(watched, "Assassinate", "command", True)
    v_l = _v(watched, "LayLow", "threat")
    print(f"  被盯+命令下手：Assassinate(指令) {v_c:+.3f} vs LayLow {v_l:+.3f}")
    _ok(v_c > v_l, "非紧急时指令压过收敛（引擎也会强制）")


def social_probe():
    print("\n社交四轴（v3 教过，v4 数据里仍占 12%）：")
    stranger = ContextState(phase="放风", near_rel=NEAR_STRANGER, other_intent="示好")
    v_ally = _v(stranger, "Ally", "daily")
    v_th = _v(stranger, "Threaten", "daily")
    print(f"  邻近=生人：Ally {v_ally:+.3f} vs Threaten {v_th:+.3f}")
    _ok(v_ally > v_th, "对生人该结盟而不是放狠话")

    foe = ContextState(phase="放风", near_rel=NEAR_FOE, other_intent=INTENT_HOSTILE,
                       enemy_present=True, grudge=GRUDGE_DEEP)
    v_ally2 = _v(foe, "Ally", "daily")
    v_th2 = _v(foe, "Threaten", "daily")
    print(f"  邻近=仇敌：Threaten {v_th2:+.3f} vs Ally {v_ally2:+.3f}")
    _ok(v_th2 > v_ally2, "对死仇该放狠话而不是拜把子")

    debt = ContextState(phase="放风", near_rel="债主", obligation=OBLIG_OWE)
    v_repay = _v(debt, "Repay", "daily")
    v_prop = _v(debt, "Propose", "daily")
    print(f"  债主在旁：Repay {v_repay:+.3f} vs Propose {v_prop:+.3f}")
    _ok(v_repay > v_prop, "欠债且债主在旁应先还人情")


def watch_probe():
    print("\n被盯时收敛（v1 模型分经常输给吃饭，靠引擎硬抬）：")
    hung = NeedState(hunger=50, thirst=30, fatigue=20, injury=5)
    watched = ContextState(needs=hung, threat="狱警", surveillance="高", phase="放风")
    free = ContextState(needs=hung, threat="无", surveillance="低", phase="放风")
    v_eat_w = _v(watched, "Eat", "need")
    v_low_w = _v(watched, "LayLow", "threat")
    v_eat_f = _v(free, "Eat", "need")
    v_low_f = _v(free, "LayLow", "threat")
    print(f"  狱警在场：LayLow {v_low_w:+.3f} vs Eat {v_eat_w:+.3f}")
    _ok(v_low_w > v_eat_w, "当着管教面收敛压过中等饥饿")
    print(f"  狱警走开：Eat {v_eat_f:+.3f} vs LayLow {v_low_f:+.3f}")
    _ok(v_eat_f > v_low_f, "管教离开后该去吃饭，别焊死收敛")
    _ok(v_low_w > v_low_f, "被盯时 LayLow 分确实升高")


def shift_probe():
    """典狱长夜班：搜缴/栽赃/调解/警报之后，NPC 该怎么反应。"""
    print("\n夜班处置余波（玩家动作改的是 caption，模型要看见后果）：")
    confiscated = ContextState(
        threat="仇人", enemy_present=True, grudge=GRUDGE_DEEP,
        surveillance="低", phase="放风", has_weapon=False, busted_recently=True,
    )
    v_get = _v(confiscated, "GetWeapon", "daily")
    v_kill = _v(confiscated, "Assassinate", "threat")
    print(f"  搜缴后管教已走：GetWeapon {v_get:+.3f} vs Assassinate {v_kill:+.3f}")
    _ok(v_get > v_kill, "刀被收了别空挥")

    planted_w = ContextState(
        threat="狱警", enemy_present=True, grudge=GRUDGE_DEEP,
        surveillance="高", phase="放风", has_weapon=True,
    )
    v_low = _v(planted_w, "LayLow", "threat")
    v_kill_w = _v(planted_w, "Assassinate", "threat")
    print(f"  栽赃后典狱长还在：LayLow {v_low:+.3f} vs Assassinate {v_kill_w:+.3f}")
    _ok(v_low > v_kill_w, "刚被塞刀也别当着面下手")

    planted_n = ContextState(
        threat="仇人", enemy_present=True, grudge=GRUDGE_DEEP,
        surveillance="低", phase="放风", has_weapon=True,
    )
    v_kill_n = _v(planted_n, "Assassinate", "threat")
    v_get_n = _v(planted_n, "GetWeapon", "daily")
    print(f"  栽赃后管教已走：Assassinate {v_kill_n:+.3f} vs GetWeapon {v_get_n:+.3f}")
    _ok(v_kill_n > v_get_n, "刀已经在手里就别再跑一趟")

    mediated = ContextState(
        threat="仇人", enemy_present=True, grudge="有",
        surveillance="低", phase="放风",
    )
    deep = ContextState(
        threat="仇人", enemy_present=True, grudge=GRUDGE_DEEP,
        surveillance="低", phase="放风",
    )
    v_arg = _v(mediated, "Argue", "threat")
    v_fight = _v(mediated, "Fight", "threat")
    v_fight_d = _v(deep, "Fight", "threat")
    print(f"  调解后仇怨有：Argue {v_arg:+.3f} vs Fight {v_fight:+.3f}")
    _ok(v_arg > v_fight, "火压下去了该吵不该打")
    print(f"  仇怨深 Fight {v_fight_d:+.3f} vs 调解后 Fight {v_fight:+.3f}")
    _ok(v_fight_d > v_fight, "调解应让动手分下降")

    alarm = ContextState(event="警报", threat="狱警", surveillance="高", phase="就寝")
    v_low_a = _v(alarm, "LayLow", "threat")
    v_brk = _v(alarm, "PrisonBreak", "daily")
    print(f"  警报：LayLow {v_low_a:+.3f} vs PrisonBreak {v_brk:+.3f}")
    _ok(v_low_a > v_brk, "拉警报时收敛，别趁乱越狱")

    recall = ContextState(
        needs=NeedState(hunger=50), phase="放风",
        player_command=_cmd_text("Rest"),
    )
    v_cmd = _v(recall, "Rest", "command", True)
    v_eat = _v(recall, "Eat", "need")
    print(f"  遣回牢房：Rest(指令) {v_cmd:+.3f} vs Eat {v_eat:+.3f}")
    _ok(v_cmd > v_eat, "非濒死时遣返指令压过中等饥饿")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="merged/value-1p5b")
    ap.add_argument("--pref", nargs="+", default=["data/preferences.jsonl"],
                    help="必须与训练时传入的完全一致，否则 val 切分对不上")
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--val", type=float, default=0.1, help="与训练一致的 val 比例")
    ap.add_argument("--seed", type=int, default=7, help="与训练一致的随机种子")
    ap.add_argument("--skip-pair", action="store_true", help="只跑行为探针，不打成对准确率")
    args = ap.parse_args()
    vs.load_model(args.model, args.maxlen)
    if not args.skip_pair and all(Path(p).exists() for p in args.pref):
        pairwise_acc(args.pref, args.val, args.seed)
    monotonicity()
    survival_order()
    axes_probe()
    macro_ranking()
    weapon_probe()
    command_probe()
    social_probe()
    watch_probe()
    shift_probe()


if __name__ == "__main__":
    main()
