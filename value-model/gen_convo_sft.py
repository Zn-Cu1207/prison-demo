"""对话大脑 SFT 数据管线：NPC↔NPC 群像对白（蒸馏 opus → 本地 7B）。

两段式（各自可续跑 / 增量），和价值模型 gen→annotate 同构：

  capture   —— 用【真实后端 World】离线跑仿真（随机打分器，不吃 GPU），
               每检测到一对该攀谈的 NPC，就用**线上同一套** npc_chat_context + convo_director
               拼出导演稿 system prompt，去重后落 data/convo_prompts.jsonl。
               这样训练 prompt 与线上逐字一致（train-serve 命根子）。

  annotate  —— 拿每条 system prompt 去问 opus（claude-opus-4-8）产"金标"6 句对白，
               落成 data/convo_sft.jsonl（{messages:[system,user,assistant]}）。
               按 prompt 指纹断点续跑，随时能 top-up 加数据重训。

用法：
    python gen_convo_sft.py capture  --n 3000 --seed 7
    python gen_convo_sft.py annotate --limit 3000            # 可多次跑，自动跳过已标
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import random
import re
import statistics
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.world import World, CONVO_SOCIAL_GOALS  # noqa: E402
from app.deepseek_client import convo_director, monologue_director  # noqa: E402
from app.model_client import ScoredCandidate  # noqa: E402
from app.quests import quest_of  # noqa: E402

PROMPTS = HERE / "data" / "convo_prompts.jsonl"
SFT = HERE / "data" / "convo_sft.jsonl"
USER_TURN = "开始，直接输出含 6 句的 JSON 数组。"

# 话题钩子的分层配额（上限占比）。按 _convo_plan 的措辞反查钩子类型：不能按它返回的
# score 判别 —— 生死网固定 120，而旧仇是 90+g 且 g>=CONVO_GRUDGE_MIN(35) → 至少 125，
# 跨对取最高分时旧仇反而压过生死网（world._plan_convo 里就是这么比的，运行时同样受影响）。
HOOK_PATTERNS = (
    ("生死网", "杀心藏在闲话底下"),
    ("军火债", "催那笔"),
    ("旧仇", "被坑的旧账翻脸"),
    ("结盟", "拴在一条绳上"),
)
HOOK_QUOTA = {"生死网": 0.22, "军火债": 0.18, "旧仇": 0.30, "结盟": 0.18, "fallback": 0.16}

# —— 质量准入阈值（check_lines）。都对着 convo_director 里那份写作要求来定，
#    并用旧数据校准过：整体拒收率应落在两三成，太高说明阈值定苛了、白烧 API。
MAX_MEAN_LEN = 52        # 平均句长上限（要求 12~40 字，留出"情到浓处可更长"的余量）
MIN_MEAN_LEN = 12        # 平均句长下限：太短是敷衍
# 句长变异系数下限：句句一样长＝排比化流水账。定 0.10 时实测 9/9 条被拒的样本变异系数
# 都恰好压在 0.100 —— 阈值正落在模型自然分布的中位，等于为一个观感问题反复烧 API，
# 而重试也过不去。降到 0.07 只拦真正整齐的。
MIN_LEN_CV = 0.07
MIN_DETAIL_LINES = 2     # 至少几句要影射给定细节（prompt 的硬要求就是 2）
# 信号集的套话过滤阈值：细节原文的 n-gram 里，出现在超过这个比例的 prompt 中的一律不算
# 影射（"恨之入骨"占 89%、"很不对付"占 84%，全是模板字样）。取值依据见 _detail_signals。
DETAIL_DF_MAX = 0.20
# 细节池丰度门槛：注入的细节稀薄时，"≥2 句影射"是模型再重试也达不到的要求，
# 硬卡只会把 API 烧在不可能通过的 prompt 上。所以要求随注入量伸缩。
# 切点随信号集换代重标定过：新信号池中位 93、10 分位 62（旧切点 40 已永不触发），
# 池 40~69 的通过率 70.3%、≥70 是 91.5%，故切在 60 —— 只放宽最吃紧的那 10%。
DETAIL_POOL_RICH = 60
# 单句与细节原文的最长连续重合字数上限：奖励字面重叠就必然招来照抄，得有反向约束。
# 旧门（只认 3-gram 精确重合）实测把照抄率抬了一倍——它选出的样本 45.1% 含≥5 字照抄、
# 11.4% 含≥7 字，而同批被它拒掉的产出只有 22.8% / 4.9%。切在 8 字：只拒掉 2.1% 的
# 自然产出（7 字要拒 4.9%，说明七字重合还会自然发生），但能砍掉最露骨的复读。
MAX_VERBATIM = 8
QUALITY_RETRIES = 3      # 同一条 prompt 最多重生成几次（拒收即重试）
REJECTS = HERE / "data" / "convo_rejects.jsonl"   # 每次拒收的原始输出与原因，留作回溯
_NON_CJK = re.compile(r"[^\u4e00-\u9fffa-zA-Z0-9]")


def _hook_of(topic: str) -> str:
    for name, pat in HOOK_PATTERNS:
        if pat in topic:
            return name
    return "fallback"

# —— 独白 SFT（替掉 narrate.py 的模板自言自语）——————————————————————————————
MONO_PROMPTS = HERE / "data" / "mono_prompts.jsonl"
MONO_SFT = HERE / "data" / "mono_sft.jsonl"
MONO_USER = "此刻你一个人待着，心里在想什么？说一句不超过28字的内心独白（自言自语，不是对别人说），只输出这一句。"
# 临时心境：叠在真实处境之上，逼独白覆盖不同情绪（尤其"按捺不住主线"这类承诺压力场景）
MONO_MOODS = [
    ("心焦", "主线大事被眼前的吃喝拉撒一再耽搁，你按捺不住、心里正拧着劲，惦记着该去办的那件事"),
    ("平静", "此刻还算安生，可心思一刻没停，盘算着往后的路"),
    ("警惕", "四下晃着狱警或仇家的影子，你压着火、掂量着分寸"),
    ("算计", "你在心里盘算下一步怎么走、怎么占上风、拿谁开刀"),
    ("疲惫", "熬得心力交瘁，可有件事怎么都撂不下"),
    ("起狠", "被逼到墙角，心里起了狠念头"),
    ("后怕", "刚出的岔子还惊魂未定，越想越怕露馅"),
    ("惦记", "心里搁着一个人／一笔账，翻来覆去放不下"),
]

# 注入 recent_events 的物理层事件池（和引擎里真实会发生的事同调，别凭空造）
PHYS_EVENTS = [
    "刚在放风场跟人推搡了两下", "窝头被人抢了半个", "手背上叫铁栏杆刮了道口子",
    "赌局上输了半条烟", "被狱警点名训了一顿", "夜里没睡好，眼皮直打架",
    "医务室排队没排上", "牢头暗示要'孝敬'点东西", "听见隔壁号子夜里有动静",
    "水房的龙头又被人占着", "刚躲过一次搜牢", "跟老相识对上了眼、没敢吭声",
]
WATCHERS = ["", "", "铁面", "哮天"]
CARRY = ["", "", "", "food", "medkit"]


class RandomScorer:
    """离线打分器：给每个候选一个随机分，让 NPC 的 goal 自然发散（不吃 GPU/不联网）。"""

    label = "random-offline"

    async def evaluate(self, items):
        scored, says = [], []
        for _cap, cands, _p in items:
            row = []
            for c in cands:
                v = random.uniform(0.3, 6.0)
                row.append(ScoredCandidate(candidate=c, v=v, m_value=v, score=v))
            scored.append(row)
            says.append(None)
        return scored, says


def _scramble(world: World) -> None:
    """打散全场状态，逼出多样的处境/任务进度/恩怨/近况 —— 在 tick 之前调用，
    好让这一拍的 goal/drivers/caption 都从打散后的状态重新算出来（自洽）。"""
    npcs = [n for n in world.npcs if not getattr(n, "dead", False)]
    for n in npcs:
        nd = n.needs
        for k in ("hunger", "thirst", "fatigue", "boredom", "social", "stress"):
            if hasattr(nd, k):
                setattr(nd, k, float(random.randint(5, 100)))
        # 伤势：多数轻、偶尔重（让"带伤对话"也进分布）
        if hasattr(nd, "injury"):
            nd.injury = float(random.choice([0, 0, 0, 15, 35, 60, 85]))
        q = quest_of(n.id)
        if q and q.steps:
            n.quest_step = random.randint(0, len(q.steps) - 1)
            n.step_progress = float(random.randint(0, max(1, q.steps[n.quest_step].need)))
        n.drive_progress = float(random.randint(0, 100))
        n.watched_by = random.choice(WATCHERS)
        n.carrying = random.choice(CARRY)
        n.solitary_ticks = random.choice([0, 0, 0, 0, random.randint(1, 6)])
        n.parole_lost = random.random() < 0.12
        # 近况：0~2 条，混合"当前分目标的 beat"与物理层小事
        try:
            n.recent_events.clear()
        except Exception:
            pass
        pool = list(PHYS_EVENTS)
        if q and q.steps and n.quest_step < len(q.steps):
            beat = q.steps[n.quest_step].beat
            if beat:
                pool.append(beat)
        for _ in range(random.randint(0, 2)):
            world._remember(n, random.choice(pool))
    # 恩怨：偶尔给一对播下仇怨（让"口角"有据可依，且强弱不一）。
    # 注意别播太狠——否则全场都在结仇、chat 样本几乎绝迹（口角/闲聊要均衡）。
    if len(npcs) >= 2 and random.random() < 0.45:
        a, b = random.sample(npcs, 2)
        a.grudge[b.id] = float(random.randint(20, 95))
        if random.random() < 0.5:
            b.grudge[a.id] = float(random.randint(20, 95))
    # 位置：把随机几个人拉到一处，制造"狭路相逢"（否则大多离得远）
    if len(npcs) >= 2:
        cx, cz = random.choice(npcs).x, random.choice(npcs).z
        for n in random.sample(npcs, min(3, len(npcs))):
            n.x = cx + random.uniform(-2.0, 2.0)
            n.z = cz + random.uniform(-2.0, 2.0)


def _fingerprint(a_ctx: dict, b_ctx: dict, kind: str, topic: str) -> str:
    """按"有意义的内容"去重：性格/任务这一步/发声归因/近况/恩怨/话题。"""
    def sig(c):
        dr = c.get("drivers") or {}
        return "|".join(str(x) for x in (
            c.get("name"), c.get("goal_label"), c.get("substep"),
            dr.get("why"), dr.get("tier"), c.get("grudge_with"), c.get("body"),
            tuple((c.get("recent_events") or [])[-2:]),
        ))
    raw = f"{kind}::{topic}::{sig(a_ctx)}::{sig(b_ctx)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _harvest(world: World, seen: set, out: list, target: int) -> None:
    """本拍从场上挑几对能攀谈的，用线上同一套函数拼 system prompt。

    话题来源必须与线上一致：走 `world._convo_plan`（生死网/军火债/旧仇/结盟四级钩子），
    没有钩子才退回 `_fallback_topic`。此前这里调的是 `world._convo_topic(a,b,kind)` ——
    world.py 重构后该方法已不存在，capture 会直接 AttributeError 挂掉；8/5 之后没人跑过，
    所以一直没暴露。旧数据就是那次重构之前采的：48% 的样本话题是"发发牢骚、交换点风声"
    这句万能套话，于是 prompt 里"整段围绕给定话题"这条约束在一半数据上等于空。

    按钩子类型做分层配额（同 gen_situations.py 对 command/threat/social 的做法）：
    生死网只涉及 quests.py 里那两对人，若不给配额就会被"旧仇"淹没 —— 而它恰恰是最有戏、
    最该让模型学会怎么说的一类。
    """
    actives = [n for n in world.npcs
               if not getattr(n, "dead", False) and not getattr(n, "escaped", False)
               and getattr(n, "solitary_ticks", 0) <= 0]
    random.shuffle(actives)
    # 先把本拍所有能攀谈的对收齐，再按"谁最欠配额"挑 —— 不能把配额写成硬上限逐个否决：
    # fallback 在当前世界几乎不触发，四类钩子必须占满 100%，而它们的上限之和不足 1，
    # 逐个否决会让四类同时超额、一条也加不进去（实测采到 86 条就彻底卡死）。
    cands: list[tuple[float, str, str, object, object]] = []
    counts = collections.Counter(r.get("hook", "") for r in out)
    n_out = max(1, len(out))
    for i, a in enumerate(actives):
        for b in actives[i + 1:]:
            plan = world._convo_plan(a, b)
            if plan is None:
                if a.goal in CONVO_SOCIAL_GOALS and b.goal in CONVO_SOCIAL_GOALS:
                    kind, topic = "chat", world._fallback_topic()
                else:
                    continue
            else:
                kind, topic, _score, _tier = plan
            hook = _hook_of(topic)
            # 欠额 = 目标占比 - 当前占比；越大越该补。加一点随机抖动打破平手时的固定偏好。
            deficit = (HOOK_QUOTA.get(hook, 0.25) - counts[hook] / n_out
                       + random.uniform(-0.02, 0.02))
            cands.append((-deficit, kind, topic, a, b))
            break
    cands.sort(key=lambda x: x[0])
    made = 0
    for _rank, kind, topic, a, b in cands:
        if made >= 3 or len(out) >= target:
            break
        a_ctx = world.npc_chat_context(a.id, other_id=b.id)
        b_ctx = world.npc_chat_context(b.id, other_id=a.id)
        if not a_ctx or not b_ctx:
            continue
        fp = _fingerprint(a_ctx, b_ctx, kind, topic)
        if fp in seen:
            continue
        seen.add(fp)
        out.append({
            "fp": fp, "kind": kind, "topic": topic, "hook": _hook_of(topic),
            "a": a.id, "b": b.id, "a_name": a_ctx["name"], "b_name": b_ctx["name"],
            "system": convo_director(a_ctx, b_ctx, topic, kind),
        })
        made += 1


async def capture(n: int, seed: int, append: bool = False) -> None:
    """采 n 条导演稿 prompt。`append=True` 时保留已有的，只往上补到 n 条。

    追加模式是必须的，不是方便功能：`world.tick()` 是异步的、含依赖时钟的分支，
    所以**同一 seed 重跑并不能复现同一批 prompt**（实测重跑 6000 条，前 3000 条里
    只有 71 条 fp 与上一次一致）。而 annotate 的断点续跑是靠 fp 匹配的 ——
    覆盖式重跑会让已经烧掉的标注全部作废。要扩容就只能在原文件上追加。
    """
    random.seed(seed)
    world = World(RandomScorer())
    world.reset()
    seen: set = set()
    out: list = []
    if append and PROMPTS.exists():
        for line in PROMPTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:                                # noqa: BLE001
                continue
            out.append(r)
            seen.add(r["fp"])
        print(f"[capture] 追加模式：已有 {len(out)} 条，补到 {n} 条")
        if len(out) >= n:
            print("[capture] 已经够了，不动")
            await world.aclose()
            return
    step = 0
    max_steps = n * 30
    while len(out) < n and step < max_steps:
        step += 1
        if step % 4 == 0:
            _scramble(world)
        try:
            await world.tick(0.5)
        except Exception as e:  # noqa: BLE001
            # 随机打分下偶发的边界情况：跳过这拍，别让整轮挂掉
            print(f"[capture] tick 跳过：{type(e).__name__}: {e}", file=sys.stderr)
        _harvest(world, seen, out, n)
        if step % 200 == 0:
            print(f"[capture] step={step} 已采 {len(out)}/{n}")
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    # 原子替换：annotate 的断点续跑要读这个文件，不能留下半截的中间态
    tmp = PROMPTS.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, PROMPTS)
    kinds = {}
    for r in out:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[capture] 完成：{len(out)} 条 → {PROMPTS}（{kinds}），用了 {step} 拍")
    await world.aclose()


def _mono_fp(ctx: dict, mood_tag: str) -> str:
    """独白去重指纹：同一人+同一步+同身体+同恩怨+同近况+同心境 只留一条。"""
    dr = ctx.get("drivers") or {}
    raw = "|".join(str(x) for x in (
        "mono", ctx.get("name"), ctx.get("goal_label"), ctx.get("substep"),
        dr.get("why"), dr.get("tier"), ctx.get("body"), ctx.get("grudge_with"),
        tuple((ctx.get("recent_events") or [])[-1:]), mood_tag,
    ))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _harvest_mono(world: World, seen: set, out: list, target: int) -> None:
    """本拍从场上每个能自处的 NPC 采一条独白 prompt（单角色，不需配对，采得快）。
    复用线上 npc_chat_context + monologue_director，train-serve 逐字一致。"""
    actives = [n for n in world.npcs
               if not getattr(n, "dead", False) and not getattr(n, "escaped", False)
               and getattr(n, "solitary_ticks", 0) <= 0]
    random.shuffle(actives)
    for n in actives:
        if len(out) >= target:
            break
        ctx = world.npc_chat_context(n.id)
        if not ctx:
            continue
        mood_tag, mood_hint = random.choice(MONO_MOODS)
        fp = _mono_fp(ctx, mood_tag)
        if fp in seen:
            continue
        seen.add(fp)
        out.append({
            "fp": fp, "npc": n.id, "name": ctx["name"], "mood": mood_tag,
            "system": monologue_director(ctx, mood_hint),
        })


async def mono_capture(n: int, seed: int) -> None:
    random.seed(seed)
    world = World(RandomScorer())
    world.reset()
    seen: set = set()
    out: list = []
    step = 0
    max_steps = n * 12
    while len(out) < n and step < max_steps:
        step += 1
        if step % 3 == 0:
            _scramble(world)
        try:
            await world.tick(0.5)
        except Exception as e:  # noqa: BLE001
            print(f"[mono-capture] tick 跳过：{type(e).__name__}: {e}", file=sys.stderr)
        _harvest_mono(world, seen, out, n)
        if step % 200 == 0:
            print(f"[mono-capture] step={step} 已采 {len(out)}/{n}")
    MONO_PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with MONO_PROMPTS.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    moods: dict = {}
    for r in out:
        moods[r["mood"]] = moods.get(r["mood"], 0) + 1
    print(f"[mono-capture] 完成：{len(out)} 条 → {MONO_PROMPTS}（{moods}），用了 {step} 拍")
    await world.aclose()


def _extract_array(text: str) -> list | None:
    t = (text or "").strip()
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        arr = json.loads(t[i:j + 1])
    except Exception:
        return None
    return arr if isinstance(arr, list) else None


DETAIL_KEEP = ("他眼下一门心思要做的这一步：", "刚刚发生在他身上的事：",
               "他还记着的旧事：", "恩怨：", "他的案底/身份底色：")
_DF_LOCK = threading.Lock()
_DF: tuple[collections.Counter, int] | None = None


def _detail_raw(system: str) -> str:
    """导演稿里"具体细节"那几行拼成的原文（去标点）。身体状况是生理值，不算叙事细节。"""
    buf: list[str] = []
    for ln in (system or "").split("\n"):
        b = ln.strip()
        for p in DETAIL_KEEP:
            if b.startswith(p):
                buf.append(_NON_CJK.sub("", b[len(p):]))
                break
    return "".join(buf)


def _grams(s: str) -> set[str]:
    return ({s[i:i + 2] for i in range(len(s) - 1)}
            | {s[i:i + 3] for i in range(len(s) - 2)})


def _detail_df() -> tuple[collections.Counter, int]:
    """全语料的 n-gram 文档频率，用来剔掉"每份细节里都有"的套话。惰性算一次（约 2 秒）。"""
    global _DF
    with _DF_LOCK:
        if _DF is None:
            df: collections.Counter = collections.Counter()
            n = 0
            if PROMPTS.exists():
                for line in PROMPTS.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        sysmsg = json.loads(line)["system"]
                    except Exception:                        # noqa: BLE001
                        continue
                    n += 1
                    for x in _grams(_detail_raw(sysmsg)):
                        df[x] += 1
            _DF = (df, max(n, 1))
    return _DF


def _detail_signals(system: str) -> set[str]:
    """判"台词有没有真咬住给定细节"的信号集：细节原文的 2/3-gram，剔掉全语料套话。

    上一版只认 3-gram 精确重合，理由是"三字组够具体，不必靠语料词频筛显著性"。
    拿 3421 份落盘产出校准后发现这个理由站不住，两头都错：

    · **漏**：两字诨名永远进不了 3-gram 池，而"提到恩怨里那个人"恰恰是最要紧的一类影射；
      换个说法也漏（"留条后路"→"留后路"差一字就不算）。实测被它拒掉的产出里 76% 本该通过，
      标注因此要烧 3.83 次调用才换一条样本。
    · **错收**：不筛词频，"恨之入骨"(89% 的 prompt 都有)、"很不对付"(84%) 这些模板字样
      本身就在池子里 —— 一句"我恨之入骨"就算影射到细节了。

    所以改成 2/3-gram + DF 过滤。阈值是扫出来的：判据不是通过率（放宽就能刷），而是
    真配对与**错配**（把产出配到别人 prompt 上）的通过率之差。DF≤20% 时 85.9% vs 26.9%，
    gap 59 为各方案最优；不滤套话是 98.5% vs 86.7%（gap 12，等于橡皮章）；
    另外试过加"第三方诨名"匹配，反而把 gap 压到 31 —— 全场只有 5 个角色、恩怨行天天列名，
    提名字没有区分度。
    """
    df, n = _detail_df()
    return {x for x in _grams(_detail_raw(system)) if df[x] / n <= DETAIL_DF_MAX}


def _detail_texts(system: str) -> list[str]:
    """把导演稿里那几行"具体细节"的原文抽出来，用于重试时明确告诉模型该咬住什么。"""
    keep = ("他眼下一门心思要做的这一步：", "刚刚发生在他身上的事：",
            "他还记着的旧事：", "恩怨：")
    out: list[str] = []
    for ln in (system or "").split("\n"):
        b = ln.strip()
        for p in keep:
            if b.startswith(p):
                out.append(b[len(p):].rstrip("。"))
                break
    return [x for x in out if x]


def preflight_hint(system: str) -> str:
    """第一次调用就把最常踩的三个坑说明白（只加在**请求**里，不进训练样本）。

    可以这么做的前提：落盘的样本永远是 (system, USER_TURN, 台词)，提示词只影响这一次
    请求怎么问 opus，不写进 messages。所以"用提示把目标行为引出来、再让学生模型学着
    不用提示也这么写"是免费的 —— 现在数据集里本就有一半是靠 retry_hint 救回来的样本。

    值得前置是因为吞吐卡在别处：网关把总调用量锁在约 32 次/分钟（12 个 worker 与 28 个
    worker 实测都是这个数，加 worker 只会把单次延迟从 22 秒推到 53 秒，还撞 429 突发限额）。
    速率因此完全由"每条样本耗几次调用"决定，把首次通过率提上去是唯一的提速手段。
    """
    bits = []
    det = _detail_texts(system)
    if det:
        bits.append("【必须做到】至少两句要明确提到下面这些具体的事（提那件事、那个人、那笔账本身，"
                    "不要泛泛而谈）：" + "；".join(f"「{d}」" for d in det[:4]))
        bits.append("【必须做到】提这些事要换成人物嘴里会说的词，不许成段照搬上面的原话。")
    bits.append("【必须做到】多数句子 12~35 字，长短错落——有短句也有长句，平均别超过 45 字。")
    return "\n" + "\n".join(bits)


def retry_hint(system: str, bad: list[str]) -> str:
    """带反馈的重试：把上一稿的具体毛病和必须咬住的细节原文摊给模型。

    盲重试（同 prompt 原样重采）浪费很大：实测 22.5% 的 prompt 三次全废、
    每条白烧 3 次调用，占总花费的 35%，而且失败原因几乎全是"没引用给定细节"——
    模型并不知道自己漏了什么。把要求点明之后，同样的重试次数能救回明显更多。
    """
    bits = ["", "【上一稿被退回，原因】" + "；".join(bad[:3])]
    if any("影射" in b for b in bad):
        det = _detail_texts(system)
        if det:
            bits.append("【必须做到】至少两句要明确提到下面这些具体的事（提那件事、那个人、"
                        "那笔账本身，不要泛泛而谈）：" + "；".join(f"「{d}」" for d in det[:4]))
    if any("照抄" in b for b in bad):
        bits.append("【必须做到】提这些事要用人物自己的说法带出来，不许成段照搬上面的原话——"
                    "换成他嘴里会说的词，别整句复读。")
    if any("齐整" in b for b in bad):
        bits.append("【必须做到】句子长短错落：有短句也有长句，别都写成差不多的长度。")
    if any("过长" in b for b in bad):
        bits.append("【必须做到】压缩篇幅，多数句子控制在 12~40 字。")
    if any("顺序" in b for b in bad):
        bits.append("【必须做到】严格按 A、B、A、B、A、B 轮流，who 字段不许错位。")
    bits.append("请重写完整的 6 句 JSON 数组。")
    return "\n".join(bits)


def check_lines(arr: list, system: str) -> tuple[list | None, list[str]]:
    """质量准入（拒收制）：不合格就退回让模型重生成，**绝不就地修正**。

    为什么从"格式校验"改成"质量准入"：旧版只查"是个 list、长度≥4、text 非空"，
    3000 条 prompt 收下 2999 条（通过率 99.97%）——等于没有校验。于是 convo_director
    里那份很严的写作要求全都只是期望，一条也没进准入标准，不合格的产出照样进训练集。
    SFT 只学收下来的样本分布，"格式完美、内容空洞"就是这么教出来的。

    尤其注意旧版这一行：`who = "A" if (who == "A" or k % 2 == 0) else "B"`
    —— 它不是校验而是**改写**：模型把顺序说错了（连着两句 A、或漏一句），它按位置奇偶
    把标签强行改成交替，于是"本该 A 说的话被贴上 B 的标签"批量进了训练集。prompt 里花了
    三行强调"绝不许张冠李戴"，校验器却在亲手制造张冠李戴。现在改成顺序不对直接拒收。

    返回 (规整后的 lines 或 None, 拒收原因列表)。
    """
    bad: list[str] = []
    if not isinstance(arr, list):
        return None, ["不是数组"]
    if len(arr) != 6:
        bad.append(f"句数={len(arr)}（须 6）")
    out: list[dict] = []
    for k, item in enumerate(arr):
        if not isinstance(item, dict):
            bad.append(f"第{k + 1}项不是对象")
            continue
        who = str(item.get("who", "")).strip().upper()
        text = str(item.get("text", "")).strip()
        if not text:
            bad.append(f"第{k + 1}句为空")
            continue
        if who not in ("A", "B"):
            bad.append(f"第{k + 1}句 who={who!r} 非法")
            continue
        if who != ("A" if k % 2 == 0 else "B"):
            bad.append(f"第{k + 1}句说话人顺序错（应 {'A' if k % 2 == 0 else 'B'}）")
        out.append({"who": who, "text": text})
    if len(out) != 6 or bad:
        return None, bad or ["规整后不足 6 句"]

    lens = [len(x["text"]) for x in out]
    mean = sum(lens) / len(lens)
    if mean > MAX_MEAN_LEN:
        bad.append(f"平均句长 {mean:.0f} 字，过长（>{MAX_MEAN_LEN}）")
    if mean < MIN_MEAN_LEN:
        bad.append(f"平均句长 {mean:.0f} 字，过短（<{MIN_MEAN_LEN}）")
    cv = (statistics.pstdev(lens) / mean) if mean else 0.0
    if cv < MIN_LEN_CV:
        bad.append(f"句长过于齐整（变异系数 {cv:.2f}<{MIN_LEN_CV}），像排比流水账")

    sig = _detail_signals(system)
    if sig:
        need = MIN_DETAIL_LINES if len(sig) >= DETAIL_POOL_RICH else 1
        hits = sum(1 for x in out if _detail_hit(x["text"], sig))
        if hits < need:
            bad.append(f"只有 {hits} 句影射到给定细节（须≥{need}）")
    ref = _detail_raw(system)
    if ref:
        for k, x in enumerate(out):
            span = _verbatim_span(x["text"], ref)
            if span >= MAX_VERBATIM:
                bad.append(f"第{k + 1}句照抄了细节原文 {span} 字（须<{MAX_VERBATIM}）")

    return (None, bad) if bad else (out, [])


def _detail_hit(text: str, sig: set[str]) -> bool:
    return bool(_grams(_NON_CJK.sub("", text or "")) & sig)


def _verbatim_span(text: str, ref: str) -> int:
    """台词与细节原文的最长连续重合字数——用来抓"照抄 caption"。"""
    s = _NON_CJK.sub("", text or "")
    best = 0
    for i in range(len(s)):
        for j in range(i + best + 1, len(s) + 1):
            if s[i:j] in ref:
                best = j - i
            else:
                break
    return best


def _valid_lines(arr: list) -> list | None:
    """旧接口：保留给 mono 支线等老调用点，内部走宽松规则（不做质量准入）。"""
    if not isinstance(arr, list) or len(arr) < 4:
        return None
    out = []
    for k, item in enumerate(arr[:6]):
        if not isinstance(item, dict):
            return None
        text = str(item.get("text", "")).strip()
        if not text:
            return None
        out.append({"who": "A" if k % 2 == 0 else "B", "text": text})
    return out if len(out) >= 4 else None


def revalidate() -> None:
    """用当前的 check_lines 重过一遍存量 SFT，不合格的挪进 dropped 文件。

    门槛改过之后必须回灌一次，否则训练集里混着两套标准 —— 前一批是按"只认 3-gram 精确
    重合"收的，那道门对照抄毫无约束（实测它把≥5 字照抄的比例抬到 45.1%，是自然产出的两倍）。
    被挪走的条目 fp 会重新回到 annotate 的待标队列，重标一遍即可。
    """
    if not (SFT.exists() and PROMPTS.exists()):
        sys.exit("缺 convo_sft.jsonl 或 convo_prompts.jsonl")
    sysmsg = {}
    for line in PROMPTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            sysmsg[r["fp"]] = r["system"]
    _detail_df()
    keep: list[str] = []
    drop: list[dict] = []
    reasons: collections.Counter = collections.Counter()
    total = 0
    for line in SFT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        rec = json.loads(line)
        fp = rec["meta"]["fp"]
        if fp not in sysmsg:
            keep.append(line)      # prompt 已不在池子里，无从复核，留着
            continue
        try:
            arr = json.loads(rec["messages"][-1]["content"])
        except Exception:                                    # noqa: BLE001
            arr = None
        lines, bad = check_lines(arr, sysmsg[fp]) if arr is not None else (None, ["无法解析"])
        if lines:
            keep.append(line)
        else:
            drop.append({"fp": fp, "bad": bad, "rec": rec})
            for b in bad:
                reasons["照抄细节原文" if "照抄" in b
                        else "影射不足" if "影射" in b
                        else b.split("（")[0]] += 1
    tmp = SFT.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for line in keep:
            f.write(line + "\n")
    os.replace(tmp, SFT)
    dropped = SFT.parent / "convo_sft_dropped.jsonl"
    with dropped.open("a", encoding="utf-8") as f:
        for d in drop:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[revalidate] 复核 {total} 条：留 {len(keep)}，挪走 {len(drop)} → {dropped}")
    for k, v in reasons.most_common():
        print(f"    {k:14s} {v}")


def annotate(limit: int, model: str, rps: float, workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from hy_client import HYClient, THROTTLE
    THROTTLE.set_rps(rps)          # 全局节流：所有线程共享同一节奏，避开网关突发限流
    if not PROMPTS.exists():
        sys.exit(f"没有 {PROMPTS}，先跑 capture")
    prompts = [json.loads(l) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    done: set = set()
    if SFT.exists():
        for l in SFT.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            try:
                done.add(json.loads(l)["meta"]["fp"])
            except Exception:
                pass
    todo = [p for p in prompts if p["fp"] not in done][:limit]
    df, dn = _detail_df()          # 开线程前先把语料词频算好，避免多线程重复初始化
    print(f"[annotate] 待标 {len(todo)}（已标 {len(done)}，共 {len(prompts)}）"
          f"model={model} rps={rps} workers={workers} 词频语料={dn} 条/{len(df)} gram")
    client = HYClient(model=model, timeout_s=90.0)   # httpx.Client 线程安全，多线程共享
    SFT.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0, "retried": 0, "calls": 0}
    f = SFT.open("a", encoding="utf-8")
    rej = REJECTS.open("a", encoding="utf-8")

    def work(p):
        """拒收即重生成（rejection sampling）：不合格不修、不收，退回让模型重写。

        比"事后筛掉"强：筛掉只会缩小数据集，重生成能把同一条 prompt 救回来。
        原始输出与每次拒收的原因都落盘 —— 上一版只存了规整后的结果，
        导致"A/B 顺序被改写过多少次"事后完全无法回溯。
        """
        attempts: list[dict] = []
        for _ in range(QUALITY_RETRIES):
            if attempts and attempts[-1]["bad"]:
                user = USER_TURN + retry_hint(p["system"], attempts[-1]["bad"])
            else:
                user = USER_TURN + preflight_hint(p["system"])
            try:
                raw = client.chat(p["system"], user, max_tokens=900, retries=5)
            except Exception as e:  # noqa: BLE001
                attempts.append({"raw": "", "bad": [f"调用失败：{e}"]})
                continue
            with lock:
                counters["calls"] += 1
            lines, bad = check_lines(_extract_array(raw) or [], p["system"])
            if lines:
                rec = {
                    "messages": [
                        {"role": "system", "content": p["system"]},
                        {"role": "user", "content": USER_TURN},
                        {"role": "assistant", "content": json.dumps(lines, ensure_ascii=False)},
                    ],
                    "meta": {"fp": p["fp"], "kind": p["kind"], "hook": p.get("hook", ""),
                             "topic": p.get("topic", ""), "a": p["a"], "b": p["b"],
                             "attempts": len(attempts) + 1, "raw": raw},
                }
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    if attempts:
                        counters["retried"] += 1
                        for at in attempts:
                            rej.write(json.dumps(
                                {"fp": p["fp"], "hook": p.get("hook", ""),
                                 "final": "ok", **at}, ensure_ascii=False) + "\n")
                        rej.flush()
                return ("ok", None)
            attempts.append({"raw": raw, "bad": bad})
        with lock:
            for at in attempts:
                rej.write(json.dumps({"fp": p["fp"], "hook": p.get("hook", ""),
                                      "final": "give_up", **at},
                                     ensure_ascii=False) + "\n")
            rej.flush()
        last = attempts[-1]["bad"] if attempts else ["无产出"]
        return ("fail", f"{QUALITY_RETRIES} 次仍不合格：{'；'.join(last[:2])}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, p) for p in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            status, err = fut.result()
            counters[status] += 1
            if err and status == "fail":
                print(f"[annotate] 失败：{err}", file=sys.stderr)
            if i % 25 == 0:
                print(f"[annotate] {i}/{len(todo)}  ok={counters['ok']} "
                      f"fail={counters['fail']} 重试救回={counters['retried']} "
                      f"调用={counters['calls']}")
    f.close()
    rej.close()
    client.close()
    per = counters["calls"] / max(1, counters["ok"])
    print(f"[annotate] 完成：新增 {counters['ok']} 条（放弃 {counters['fail']}，"
          f"其中 {counters['retried']} 条靠重试救回）→ {SFT}")
    print(f"[annotate] 共 {counters['calls']} 次调用，平均 {per:.2f} 次/条；"
          f"拒收明细 → {REJECTS}")


def _valid_mono(arr: list | None) -> list | None:
    """校验独白：字符串数组，每句 6~40 字，去空去重，最少 4 句。"""
    if not isinstance(arr, list):
        return None
    out, seen = [], set()
    for s in arr[:8]:
        if isinstance(s, dict):
            s = s.get("text") or s.get("say") or s.get("line") or ""
        s = str(s).strip().strip("“”\"'")
        if 6 <= len(s) <= 40 and s not in seen:
            seen.add(s)
            out.append(s)
    return out if len(out) >= 4 else None


def mono_annotate(limit: int, model: str, rps: float, workers: int) -> None:
    """拿独白 prompt 问 opus 产 8 句心里话，拆成 8 条单句训练样本落 mono_sft.jsonl。
    一次调用出多条样本 → 比对话便宜得多。按 prompt fp 断点续跑。"""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from hy_client import HYClient, THROTTLE
    THROTTLE.set_rps(rps)
    if not MONO_PROMPTS.exists():
        sys.exit(f"没有 {MONO_PROMPTS}，先跑 mono-capture")
    prompts = [json.loads(l) for l in MONO_PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    done: set = set()
    if MONO_SFT.exists():
        for l in MONO_SFT.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            try:
                done.add(json.loads(l)["meta"]["fp"])
            except Exception:
                pass
    todo = [p for p in prompts if p["fp"] not in done][:limit]
    print(f"[mono-annotate] 待标 prompt {len(todo)}（已标 {len(done)}，共 {len(prompts)}）"
          f" model={model} rps={rps} workers={workers}")
    client = HYClient(model=model, timeout_s=90.0)
    MONO_SFT.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0, "lines": 0}

    def work(p):
        try:
            raw = client.chat(p["system"], MONO_USER, max_tokens=520, retries=5)
        except Exception as e:  # noqa: BLE001
            return ("fail", 0, f"调用失败：{e}")
        lines = _valid_mono(_extract_array(raw))
        if not lines:
            return ("fail", 0, "产出不合格")
        recs = [{
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": MONO_USER},
                {"role": "assistant", "content": s},
            ],
            "meta": {"fp": p["fp"], "i": i, "kind": "mono", "npc": p["npc"], "mood": p.get("mood")},
        } for i, s in enumerate(lines)]
        with lock:
            with MONO_SFT.open("a", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return ("ok", len(recs), None)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, p) for p in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            status, nlines, err = fut.result()
            counters[status] += 1
            counters["lines"] += nlines
            if err and status == "fail":
                print(f"[mono-annotate] 失败：{err}", file=sys.stderr)
            if i % 25 == 0:
                print(f"[mono-annotate] {i}/{len(todo)}  ok={counters['ok']} "
                      f"fail={counters['fail']} 样本={counters['lines']}")
    client.close()
    print(f"[mono-annotate] 完成：{counters['ok']} 个 prompt → {counters['lines']} 条样本"
          f"（失败 {counters['fail']}）→ {MONO_SFT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--n", type=int, default=3000)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--append", action="store_true",
                   help="保留已有 prompt、只补到 --n 条（扩容必须用这个，见 capture 的说明）")
    sub.add_parser("revalidate")
    a = sub.add_parser("annotate")
    a.add_argument("--limit", type=int, default=100000)
    a.add_argument("--model", default="claude-opus-4-8")
    a.add_argument("--rps", type=float, default=6.0)
    a.add_argument("--workers", type=int, default=8)
    mc = sub.add_parser("mono-capture")
    mc.add_argument("--n", type=int, default=400)
    mc.add_argument("--seed", type=int, default=11)
    ma = sub.add_parser("mono-annotate")
    ma.add_argument("--limit", type=int, default=100000)
    ma.add_argument("--model", default="claude-opus-4-8")
    ma.add_argument("--rps", type=float, default=6.0)
    ma.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "capture":
        asyncio.run(capture(args.n, args.seed, args.append))
    elif args.cmd == "revalidate":
        revalidate()
    elif args.cmd == "mono-capture":
        asyncio.run(mono_capture(args.n, args.seed))
    elif args.cmd == "mono-annotate":
        mono_annotate(args.limit, args.model, args.rps, args.workers)
    else:
        annotate(args.limit, args.model, args.rps, args.workers)


if __name__ == "__main__":
    main()
