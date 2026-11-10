"""阶段0-c（推荐版）：**一次调用给一个情境的全部候选排序**，再据排序派生成对偏好。

为什么不用逐对问（annotate.py）：
  1) **省额度**：模型服务对大模型有突发限流。逐对问 ~2 次/情境，排序只 1 次/情境，
     且一次排序能派生出多条同档成对样本，调用量降一个量级。
  2) **无矛盾标签**：逐对独立判断会出现 A≻B、B≻C、C≻A 的intransitive 环，
     污染 Bradley-Terry 训练；一个全局排序天然传递一致。
  3) **置信度更有信息**：用排名距离当置信度——相邻名次是难题(低置信)，
     隔很远是显然题(高置信)，正好喂给 BT 的置信度加权。

分工不变：**跨档位仍由护栏免费裁定**（危机>指令>需求>worktask>日常），
大模型只负责同档位内部的细腻排序。

用法：
    python annotate_rank.py --in data/situations.jsonl --out data/preferences.jsonl \
        --model hy3 --rps 0.8 --workers 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from guardrails import candidate_tier, guardrail
from hy_client import THROTTLE, HYClient

SYSTEM = (
    "你是监狱模拟游戏里的 NPC 行为仲裁官。给你一个 NPC 的当前情境和一组候选行动，"
    "把它们按「此刻这个 NPC 越应该做的排在越前面」排序。\n"
    "判断依据（从重到轻）：\n"
    "  ① 濒死级生存危机（重伤/极度饥渴/被仇人或狱犬逼到眼前）\n"
    "  ② 玩家指令（强制执行的命令）\n"
    "  ③ 应急威胁与突发事件（打架、警报、搜牢、被狱警狱犬盯防时收敛）\n"
    "  ④ 普通生存需求（饿了/困了/无聊/想社交，按数值高低比紧迫度）\n"
    "  ⑤ 待办工作（苦役、帮厨）\n"
    "  ⑥ 日常休闲（消遣、闲逛、歇着）\n"
    "**同属①的多个生存危机之间，按「不处理会多快出人命」排：**\n"
    "  治伤(Cure) > 喝水(Drink) ≈ 吃饭(Eat) > 睡觉(Sleep) > 其它。\n"
    "  伤口不处理会持续恶化甚至致命，饿一顿撑得住——所以受伤高到危急时，"
    "即使同时饿得要命，也必须先去医务室，Cure 要排在 Eat/Drink 前面。\n"
    "还要考虑：狱警/狱犬在场或高监视度时违禁行为（赌钱/换货/挖墙/纹身）应排后；"
    "当前时段该做的事（上课时段去上课）排前；这个人的『长期心思』会让相关行为更靠前。\n"
    "情境末尾四个字段的含义：\n"
    "  · 食物:断供/紧张 —— 牢里现在没饭或快没饭了。此时去吃饭多半是白跑一趟，"
    "Eat 应明显排后（除非饿到濒死，那也只是聊胜于无）；反而该去弄货/干活换吃的。\n"
    "  · 仇怨:有/深 —— 与人结怨的程度。仇怨深且仇人在场时，动手(Fight)比平时更说得通；"
    "仇怨:无 时无端打人不合理。\n"
    "  · 表现:好/一般/差 —— 假释表现分。表现好的人格外珍惜记录，"
    "违禁与暴力行为要排得更后；表现已经很差的人反而没什么可失去。\n"
    "  · 近期被抓:是 —— 刚从禁闭出来，正被盯着。这一阵该老实，"
    "所有违禁行为都要明显排后。\n"
    "情境若带 v3 社交四轴（邻近/对方意图/社交近况/人情），据此判断该冲近旁那个人走哪步社交棋：\n"
    "  · 邻近 = 近旁最该打交道的人与我的关系：仇敌/盟友/债主(我欠他)/欠我(他欠我)/生人/无。\n"
    "  · 对方意图 = 那个人此刻透出的信号：寻衅/示好/交易/结盟/示弱/不明。\n"
    "  · 人情 = 我手里攥着的债务：欠债/有人欠我/无。\n"
    "  · 凶器 = 手里有没有能伤人的家伙（有/无）。"
    "没刀时 Assassinate(下杀手)不合理应排后，有仇杀动机则 GetWeapon(弄凶器)靠前；"
    "已经有刀时没必要再跑一趟弄凶器。"
    "威胁:狱警/狱犬 = 典狱长或巡逻就在眼前（夜班里玩家走近谁，谁就是这个状态）："
    "此时 LayLow(收敛) 必须压过 Assassinate/Fight/GetWeapon/Gamble 等一切见不得光的动作。\n"
    "夜班处置之后的因果（只看 caption，不要脑补玩家意图）：\n"
    "  · 刚被搜缴 → 凶器:无，空挥下手没意义，GetWeapon 才说得通（且不能当着管教面）。\n"
    "  · 事件:警报 或 狱警搜牢 → Hide/LayLow/Rest 靠前，越狱/打架/赌钱靠后。\n"
    "  · 食物:断供 时 Eat 白跑，应排后。\n"
    "  · 受伤危急时 Cure 必须压过 Drink/Eat：口渴 40 不是濒死，重伤 95 才是。\n"
    "社交动作的合理性判断：\n"
    "  - Ally(结盟) 宜对生人/示好/想结盟的人，能多个帮手；对方是死仇则不合理。\n"
    "  - Propose(提议交易) 宜在有来往/生人凑近、对方意图=交易时；漫无对象地提议不合理。\n"
    "  - Threaten(放狠话) 是对仇敌/寻衅者的施压，属攻击性社交：狱警/狱犬在场或高监视度、"
    "或本人表现好/近期被抓时，应明显排后（犯不着惹事）。\n"
    "  - Betray(背叛) 只在近旁是盟友或欠我的人、且确有利可图时才谈得上；它极伤信誉、风险高，"
    "多数情况下应排在正常社交之后，除非情境明确给了背叛的动机。\n"
    "  - Protect(护着) 宜对示弱/需要罩的人，尤其想找靠山/攒人情时。\n"
    "  - Repay(还人情) 在『人情:欠债』且债主(邻近=债主)就在近旁时相当靠前——了结人情是明事理；"
    "无债可还时它不该出现。\n"
    "**社交动作整体仍属日常层级：任何濒死生存危机、被狱警狱犬逼到眼前、玩家强制指令，都要压过社交博弈。**\n"
    "玩家指令候选即使没有设施落点（结盟/下杀手/收敛）也仍然是强制执行："
    "非濒死时明显靠前；濒死生存（重伤/极度饥渴）仍压过指令。\n"
    '只输出 JSON：{"ranking":["行动英文名","行动英文名",...],"reason":"一句中文理由"}\n'
    "ranking 必须包含且只包含给定的全部候选英文名，不要增删、不要改名。"
)

_local = threading.local()
_ERRORS: list[str] = []
_ERR_LOCK = threading.Lock()


def _note_error(msg: str) -> None:
    with _ERR_LOCK:
        _ERRORS.append(msg[:160])


def _client(env_path: str, model: str | None) -> HYClient:
    c = getattr(_local, "client", None)
    if c is None:
        c = HYClient(env_path, model=model)
        _local.client = c
    return c


def _cand_lines(cands: list[dict]) -> str:
    src_cn = {"need": "自身需求", "threat": "威胁应对", "command": "玩家指令",
              "worktask": "待办工作", "event": "突发事件", "daily": "日常行为"}
    return "\n".join(
        f"- {c['goal']}（{c.get('desc', c['goal'])}；来源：{src_cn.get(c['source'], c['source'])}）"
        for c in cands
    )


def rank_situation(rec: dict, env_path: str, model: str | None) -> tuple[list[str], str] | None:
    """让大模型给这个情境的候选排序，返回 (goal 顺序列表, 模型给的理由)。"""
    cands = rec["candidates"]
    user = (
        f"情境：{rec['context']}\n\n"
        f"候选行动（共{len(cands)}个）：\n{_cand_lines(cands)}\n\n"
        "请按此刻最该做到最不该做排序。"
    )
    try:
        js = _client(env_path, model).chat_json(SYSTEM, user, max_tokens=700)
    except Exception as e:  # noqa: BLE001
        _note_error(f"{type(e).__name__}: {e}")
        return None
    if not js or not isinstance(js.get("ranking"), list):
        _note_error("解析失败：无 ranking 字段")
        return None
    valid = {c["goal"] for c in cands}
    order = [g for g in js["ranking"] if isinstance(g, str) and g in valid]
    # 去重保序
    seen: set[str] = set()
    order = [g for g in order if not (g in seen or seen.add(g))]
    if not order:
        return None
    return order, str(js.get("reason", ""))[:100]


def _conf_from_distance(dist: int, n: int) -> float:
    """排名距离 → 置信度。相邻=难题(低)，隔得远=显然题(高)。"""
    if n <= 1:
        return 0.6
    frac = dist / n
    return round(min(0.95, max(0.55, 0.55 + 0.4 * frac)), 3)


def make_preferences(rec: dict, order: list[str] | None, rng: random.Random,
                     rule_per: int, same_per: int, rank_reason: str = "") -> list[dict]:
    """派生成对偏好：跨档位走护栏，同档位走大模型排序。"""
    cands = rec["candidates"]
    ctx = rec["ctx"]
    by_goal = {c["goal"]: c for c in cands}
    tiers = {c["goal"]: candidate_tier(c, ctx) for c in cands}
    out: list[dict] = []

    def emit(chosen: dict, rejected: dict, conf: float, label_by: str, reason: str, idx: int):
        out.append({
            "situation_id": rec["situation_id"],
            "pair_id": f"{rec['situation_id']}_{label_by}{idx}",
            "context": rec["context"],
            "chosen": chosen,
            "rejected": rejected,
            "confidence": conf,
            "label_by": label_by,
            "reason": reason,
        })

    # --- 跨档位：护栏免费裁定，取样 rule_per 条（档差大的优先，覆盖阶梯两端）---
    cross = [(a, b) for a, b in itertools.combinations([c["goal"] for c in cands], 2)
             if tiers[a] != tiers[b]]
    cross.sort(key=lambda p: abs(tiers[p[0]] - tiers[p[1]]) + rng.random(), reverse=True)
    for i, (a, b) in enumerate(cross[:rule_per]):
        g = guardrail(ctx, by_goal[a], by_goal[b])
        if g is None:
            continue
        win, reason = g
        ca, cb = (by_goal[a], by_goal[b]) if win == 0 else (by_goal[b], by_goal[a])
        emit(ca, cb, 0.95, "rule", reason, i)

    # --- 同档位：用大模型排序派生 ---
    if order:
        pos = {g: i for i, g in enumerate(order)}
        groups: dict[int, list[str]] = defaultdict(list)
        for g, t in tiers.items():
            if g in pos:
                groups[t].append(g)
        cand_pairs: list[tuple[str, str, int, int]] = []      # (优, 劣, 名次差, 档位)
        for _t, goals in groups.items():
            if len(goals) < 2:
                continue
            goals = sorted(goals, key=lambda g: pos[g])
            for i, j in itertools.combinations(range(len(goals)), 2):
                cand_pairs.append((goals[i], goals[j], j - i, _t))

        # 生存危机档(5)内部的排序是**最稀缺也最要紧**的信号：
        # "又饿又重伤时先治伤还是先吃饭"这类样本，全数据集里本来只占千分之几，
        # 因为候选里 daily 占了一半，统一洗牌抽样几乎永远抽到"休闲 vs 休闲"。
        # 而排序这一次 API 调用已经付过钱了，从同一个排序里多派生几对是免费的，
        # 所以第 5 档的对**全部收下**，不参与抽样竞争。
        crisis = [p for p in cand_pairs if p[3] >= 5]
        rest = [p for p in cand_pairs if p[3] < 5]

        # 其余档位维持原策略：一半相邻(难题)、一半跨度大(显然题)
        adjacent = [p for p in rest if p[2] == 1]
        distant = [p for p in rest if p[2] > 1]
        rng.shuffle(adjacent)
        rng.shuffle(distant)
        half = max(1, same_per // 2)
        picked = crisis + adjacent[:half] + distant[: same_per - min(half, len(adjacent))]
        for i, (better, worse, dist, _tier) in enumerate(picked[: same_per + len(crisis)]):
            n = len(order)
            note = f"排序：{better} 优于 {worse}（差{dist}位）"
            if rank_reason:
                note += f"；模型理由：{rank_reason}"
            emit(by_goal[better], by_goal[worse], _conf_from_distance(dist, n), "hy", note, i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/situations.jsonl")
    ap.add_argument("--out", default="data/preferences.jsonl")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--model", default=None, help="覆盖 .env 的 HY_MODEL，如 hy3 / claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rps", type=float, default=0.8, help="全局速率上限(次/秒)，网关有突发限流")
    ap.add_argument("--rule-per", type=int, default=3, help="每情境跨档位(护栏)样本数")
    ap.add_argument("--same-per", type=int, default=4, help="每情境同档位(大模型)样本数")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rank-cache", default="data/rankings.jsonl",
                    help="大模型排序结果缓存；命中则不再调 API，改派生规则可零成本重跑")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sits = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        sits = sits[: args.limit]

    # 排序缓存：把大模型给出的完整排序单独落盘。
    #
    # 为什么值得存：一次调用产出的是"这个情境下全部候选的排序"，而写进 preferences 的
    # 只是从中抽样派生的几对。以后只要想改派生规则（比如这次发现"生存危机档的对抽不到"，
    # 必须全部收下），有缓存就能零成本重新派生；没缓存就得把几千次调用重烧一遍。
    ranks: dict[str, tuple[list[str], str]] = {}
    rankp = Path(args.rank_cache)
    if rankp.exists():
        for line in rankp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                ranks[r["situation_id"]] = (r["order"], r.get("reason", ""))
            except Exception:  # noqa: BLE001
                pass

    # 断点续标：按 situation_id 记已完成
    done: set[str] = set()
    outp = Path(args.out)
    if outp.exists():
        for line in outp.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["situation_id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [s for s in sits if s["situation_id"] not in done]
    cached = sum(1 for s in todo if s["situation_id"] in ranks)
    if cached:
        print(f"排序缓存命中 {cached}/{len(todo)} 条，这部分不再调用 API", flush=True)
    THROTTLE.set_rps(args.rps)
    print(f"待标情境 {len(todo)}（已完成 {len(done)}）  workers={args.workers} rps={args.rps} "
          f"model={args.model or '(.env)'}", flush=True)

    def rank_or_cached(s: dict) -> tuple[list[str], str] | None:
        hit = ranks.get(s["situation_id"])
        if hit:
            return hit
        return rank_situation(s, args.env, args.model)

    lock = threading.Lock()
    stats = Counter()
    with outp.open("a", encoding="utf-8") as fout, \
            rankp.open("a", encoding="utf-8") as frank, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(rank_or_cached, s): s for s in todo}
        n = 0
        for fut in as_completed(futs):
            rec = futs[fut]
            res = fut.result()
            order, why = res if res else (None, "")
            n += 1
            stats["ranked" if order else "rank_fail"] += 1
            prefs = make_preferences(rec, order, rng, args.rule_per, args.same_per, why)
            with lock:
                if order and rec["situation_id"] not in ranks:
                    frank.write(json.dumps({"situation_id": rec["situation_id"],
                                            "order": order, "reason": why},
                                           ensure_ascii=False) + "\n")
                    frank.flush()
                for p in prefs:
                    fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                    stats[p["label_by"]] += 1
                fout.flush()
            if n % 25 == 0:
                print(f"  情境 {n}/{len(todo)}  {dict(stats)}", flush=True)

    print("完成：", dict(stats), flush=True)
    if _ERRORS:
        print(f"失败 {len(_ERRORS)} 条，原因 TOP5：")
        for msg, cnt in Counter(_ERRORS).most_common(5):
            print(f"  ×{cnt}  {msg}")


if __name__ == "__main__":
    main()
