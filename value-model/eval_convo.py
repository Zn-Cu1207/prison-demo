"""语言脑（NPC↔NPC 对白）质量评测：把"感觉说得不好"变成可复现的数。

为什么需要这个（和决策脑的 eval.py 对称）：
    决策脑有 eval.py —— 成对准确率/单调性/survival_order/axes_probe，所以能诊断、能改数据、
    能证明 v3 比 v1 好。语言脑此前**一个质量指标都没有**：gen_convo_sft._valid_lines 只校验
    "是个 list、长度≥4、text 非空"，3000 条 prompt 收下 2999 条金标（通过率 99.97%）——
    等于没有校验。于是 convo_director 里那份很严的写作要求（围绕话题、至少 2 句影射细节、
    后一句接住前一句、不复读、人设不许串味）全都只是"期望"，没有一条进了准入标准。
    SFT 只学收下来的样本分布，所以"格式完美、内容空洞"是数据教出来的，不是模型笨。

    本脚本给出五个可自动计算、零 API 成本的指标（可反复跑、可用于回归），
    外加一个 --judge 模式用 opus 在小样本上打分，用来校准这些自动指标是否和人的判断同向。
    校准过的自动指标才敢当验收线用。

指标（全部只依赖字符 n-gram，不需要分词库）：
    1. 格式        句数是否恰好 6、句长分布、句长齐整度（都一样长=排比化流水账）
    2. 话题保持率  每句是否命中 topic 的显著 bigram；样本级看"命中≥2 句"的比例
    3. 细节引用率  是否影射注入的 substep/刚发生的事/旧事/恩怨/案底（prompt 要求≥2 句）
    4. 接话率      相邻两句是否有显著词重叠（零重叠=各说各话）
    5. 说话人可辨识度  只看台词、用字符 bigram 最近质心分类器认说话人（5 类，随机基线 20%）
                   —— 这正是 prompt 里"读者只看台词就该认出谁是谁"的可测版本
    6. 第三方漂移  台词提到了不在场的人。再拆一刀：那个人是否出现在 topic 或注入细节里。
                   若大量"来自注入记忆但不在 topic"，就证明是记忆检索不看对话对象把人硬塞进来的。

用法：
    python eval_convo.py                                  # 全量自动指标 → convo_eval_report.txt
    python eval_convo.py --detail data/convo_quality.jsonl # 另存每样本明细，便于挑最差的重标
    python eval_convo.py --judge 80 --model claude-opus-4-8  # 抽 80 条让 opus 打分并算相关性
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SFT = HERE / "data" / "convo_sft.jsonl"

# 一个 bigram 在全部台词里出现的文档频率低于此值，才算"显著"（有区分度）。
# 太高的（"的话""什么""知道"）是通用词，命中它们不能说明贴题。
SALIENT_DF = 0.02
# 相邻两句共享几个显著 bigram 才算"接住了话头"
LINK_MIN = 1
# 同一样本内两句的 bigram Jaccard 超过此值 → 判为复读
REPEAT_J = 0.5

# convo_director / _convo_who_line 里的固定前缀（改那边的措辞，这里要同步）
FIELD_PREFIX = {
    "crime": "他的案底/身份底色：",
    "substep": "他眼下一门心思要做的这一步：",
    "body": "身体状况：",
    "grudge": "恩怨：",
    "recent": "刚刚发生在他身上的事：",
    "rel": "他与旁人的关系：",
    "memory": "他还记着的旧事：",
}
# 进"细节池"的字段：prompt 明确要求台词咬住这些（body 是生理状态，不算叙事细节）
DETAIL_FIELDS = ("substep", "recent", "memory", "grudge", "crime")

_WHO_RE = re.compile(r"^([AB])=「(.+?)」（(.*?)，(.*?)）：此刻(.*)$")
_SPEECH_RE = re.compile(r"^【(.+?)的说话腔调，务必照着写】(.*)$")
_CJK = re.compile(r"[^\u4e00-\u9fffa-zA-Z0-9]")


def bigrams(text: str) -> set[str]:
    """字符 bigram：中文无需分词也能有相当好的主题/风格区分度。"""
    s = _CJK.sub("", text or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def parse_system(sys_text: str) -> dict:
    """把导演稿 system prompt 反解成结构化字段。

    capture 阶段只把 a_ctx/b_ctx 渲染进了 system 字符串、没有单独存结构化副本，
    所以这里按固定前缀反解。前缀一旦在 deepseek_client 改了，这里会解不出来 ——
    parse_ok 计数就是这个护栏（大面积解不出会在报告里显性暴露）。
    """
    out: dict = {"A": {}, "B": {}, "topic": ""}
    cur: str | None = None
    for ln in sys_text.split("\n"):
        m = _WHO_RE.match(ln)
        if m:
            cur = m.group(1)
            out[cur].update(name=m.group(2), personality=m.group(3),
                            role=m.group(4), doing=m.group(5).rstrip("。"))
            continue
        if ln.startswith("话题："):
            cur = None
            t = ln[len("话题："):]
            for sep in ("。这是一场", "。这是狱友"):
                if sep in t:
                    t = t.split(sep)[0]
                    break
            out["topic"] = t
            continue
        if cur and ln.startswith("  "):
            body = ln.strip()
            ms = _SPEECH_RE.match(body)
            if ms:
                out[cur]["speech"] = ms.group(2).rstrip("。")
                continue
            for key, prefix in FIELD_PREFIX.items():
                if body.startswith(prefix):
                    out[cur][key] = body[len(prefix):].rstrip("。")
                    break
            continue
        if cur and not ln.startswith("  "):
            cur = None
    return out


def load(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
            msgs = {m["role"]: m["content"] for m in rec["messages"]}
            lines = json.loads(msgs["assistant"])
        except Exception:                                    # noqa: BLE001
            continue
        ctx = parse_system(msgs.get("system", ""))
        rows.append({"meta": rec.get("meta", {}), "ctx": ctx, "lines": lines,
                     "system": msgs.get("system", "")})
    return rows


def build_df(rows: list[dict]) -> dict[str, float]:
    """全语料 bigram 文档频率（文档=一句台词），用来判显著性。"""
    df: Counter = Counter()
    total = 0
    for r in rows:
        for ln in r["lines"]:
            total += 1
            df.update(bigrams(ln.get("text", "")))
    return {g: c / max(1, total) for g, c in df.items()}


def salient(text: str, df: dict[str, float]) -> set[str]:
    return {g for g in bigrams(text) if df.get(g, 0.0) < SALIENT_DF}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def idf_of(df: dict[str, float]) -> dict[str, float]:
    return {g: math.log(1.0 / max(v, 1e-6)) for g, v in df.items()}


def wov(dialog: set[str], ref: set[str], idf: dict[str, float]) -> float:
    """idf 加权重叠，按参考文本长度归一（免得长话题天然占优）。

    刻意**不做**低频阈值筛选：话题词会因为"大家都在贴题讨论它"而变高频，
    用低频当显著性判据会把真正的贴题证据全过滤掉（第一版就栽在这儿：
    话题命中 0.098/6，91.9% 判为完全跑题，明显是指标坏了而不是数据坏了）。
    """
    if not dialog or not ref:
        return 0.0
    return sum(idf.get(g, 0.0) for g in (dialog & ref)) / (len(ref) ** 0.5)


def discriminative(rows: list[dict], get_ref, idf: dict[str, float],
                   draws: int = 20, seed: int = 7) -> tuple[float, float]:
    """判别式贴合度：这段对话和**它自己的**参考文本，是否比和随机抽来的参考文本更像。

    不依赖任何阈值，因此对"什么词算显著"完全免疫；随机基线天然是 50%。
    返回 (AUC, 严格胜出率)：AUC=所有 own-vs-random 对比中 own 更高的比例；
    严格胜出率=own 打败全部 draws 个随机参考的样本比例。
    """
    import random as _r
    rnd = _r.Random(seed)
    refs = [bigrams(get_ref(r)) for r in rows]
    dials = [bigrams(" ".join(ln.get("text", "") for ln in r["lines"])) for r in rows]
    n = len(rows)
    wins = 0.0
    total = 0
    strict = 0
    for i in range(n):
        own = wov(dials[i], refs[i], idf)
        beat_all = True
        for _ in range(draws):
            j = rnd.randrange(n)
            if j == i:
                continue
            other = wov(dials[i], refs[j], idf)
            total += 1
            # 并列记 0.5 分（AUC 的标准处理）。这里必须这么做：话题字符串重复率极高
            # （2999 条样本只有 21 种话题、其中一种占 48%），随机抽到的常常就是同一个话题，
            # 把并列当失败会把 AUC 压到基线以下，看着像"对话跟自己的话题反而更不像"。
            if own > other:
                wins += 1
            elif own == other:
                wins += 0.5
            else:
                beat_all = False
        strict += int(beat_all)
    return (wins / max(1, total), strict / max(1, n))


def speaker_classifier(rows: list[dict]) -> tuple[float, int, dict]:
    """说话人可辨识度：只看台词，用字符 bigram 最近质心分类器认人。

    每个 NPC 的质心 = 他所有台词的 bigram 频率向量（按语料 df 做 idf 式压制，
    免得"的话/知道"这类通用词主导相似度）。留一法：算某句的归属时，把这句自己
    从质心里扣掉，避免自证。随机基线 = 1/角色数。
    """
    by_name: dict[str, Counter] = defaultdict(Counter)
    samples: list[tuple[str, set[str]]] = []
    for r in rows:
        for ln in r["lines"]:
            who = ln.get("who")
            name = r["ctx"].get(who, {}).get("name")
            text = ln.get("text", "")
            if not name or not text:
                continue
            g = bigrams(text)
            by_name[name].update(g)
            samples.append((name, g))
    names = sorted(by_name)
    if len(names) < 2 or not samples:
        return 0.0, 0, {}
    hit = 0
    per_name: dict[str, list[int]] = defaultdict(list)
    for name, g in samples:
        best, best_s = None, -1.0
        for cand in names:
            c = by_name[cand]
            # 留一：这句本身对自己质心的贡献扣掉
            score = sum((c[x] - 1) if cand == name else c[x] for x in g)
            score /= max(1.0, sum(c.values()) ** 0.5)
            if score > best_s:
                best, best_s = cand, score
        ok = int(best == name)
        hit += ok
        per_name[name].append(ok)
    acc = hit / len(samples)
    per = {n: round(sum(v) / len(v), 3) for n, v in per_name.items()}
    return acc, len(names), per


def evaluate(rows: list[dict]) -> tuple[dict, list[dict]]:
    df = build_df(rows)
    all_names = {r["ctx"][t].get("name") for r in rows for t in ("A", "B")
                 if r["ctx"].get(t, {}).get("name")}

    detail: list[dict] = []
    for r in rows:
        ctx, lines = r["ctx"], r["lines"]
        texts = [ln.get("text", "") for ln in lines]
        lens = [len(t) for t in texts]
        topic_sal = salient(ctx.get("topic", ""), df)

        pool_text = " ".join(
            ctx.get(t, {}).get(f, "") for t in ("A", "B") for f in DETAIL_FIELDS)
        pool_sal = salient(pool_text, df)

        topic_hits = sum(1 for t in texts if salient(t, df) & topic_sal)
        detail_hits = sum(1 for t in texts if salient(t, df) & pool_sal)

        # 接话率：相邻两句共享显著 bigram
        links, link_ok = 0, 0
        for i in range(len(texts) - 1):
            links += 1
            if len(salient(texts[i], df) & salient(texts[i + 1], df)) >= LINK_MIN:
                link_ok += 1

        # 复读：样本内句对最大 Jaccard
        max_j = 0.0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                max_j = max(max_j, jaccard(bigrams(texts[i]), bigrams(texts[j])))

        # 第三方漂移：提到不在场的人；再看那人是否有"出处"（topic / 注入细节）
        here = {ctx.get(t, {}).get("name") for t in ("A", "B")}
        joined = " ".join(texts)
        third = sorted(n for n in all_names - here if n and n in joined)
        from_topic = [n for n in third if n in ctx.get("topic", "")]
        from_inject = [n for n in third if n not in ctx.get("topic", "") and n in pool_text]
        unanchored = [n for n in third
                      if n not in ctx.get("topic", "") and n not in pool_text]

        detail.append({
            "fp": r["meta"].get("fp", ""),
            "kind": r["meta"].get("kind", ""),
            "n_lines": len(texts),
            "len_mean": round(statistics.fmean(lens), 1) if lens else 0,
            "len_cv": round(statistics.pstdev(lens) / statistics.fmean(lens), 3)
                      if len(lens) > 1 and statistics.fmean(lens) else 0.0,
            "in_band": round(sum(1 for x in lens if 12 <= x <= 40) / max(1, len(lens)), 3),
            "topic_hits": topic_hits,
            "detail_hits": detail_hits,
            "link_rate": round(link_ok / links, 3) if links else 0.0,
            "max_repeat_j": round(max_j, 3),
            "third_party": third,
            "third_from_topic": from_topic,
            "third_from_inject": from_inject,
            "third_unanchored": unanchored,
        })

    n = len(detail)
    def frac(pred) -> float:
        return round(sum(1 for d in detail if pred(d)) / max(1, n), 4)
    def avg(key) -> float:
        return round(statistics.fmean([d[key] for d in detail]) if detail else 0.0, 3)

    acc, n_names, per_name = speaker_classifier(rows)
    idf = idf_of(df)
    topic_auc, topic_strict = discriminative(rows, lambda r: r["ctx"].get("topic", ""), idf)
    pool_auc, pool_strict = discriminative(
        rows, lambda r: " ".join(r["ctx"].get(t, {}).get(f, "")
                                 for t in ("A", "B") for f in DETAIL_FIELDS), idf)
    summary = {
        "n_samples": n,
        "fmt_exactly_6": frac(lambda d: d["n_lines"] == 6),
        "fmt_len_mean": avg("len_mean"),
        "fmt_len_cv_median": round(statistics.median([d["len_cv"] for d in detail]), 3) if detail else 0,
        "fmt_in_band_12_40": avg("in_band"),
        "topic_auc": round(topic_auc, 4),
        "topic_strict": round(topic_strict, 4),
        "pool_auc": round(pool_auc, 4),
        "pool_strict": round(pool_strict, 4),
        "detail_hit_lines_avg": avg("detail_hits"),
        "detail_ge2_rate": frac(lambda d: d["detail_hits"] >= 2),
        "detail_zero_rate": frac(lambda d: d["detail_hits"] == 0),
        "link_rate_avg": avg("link_rate"),
        "link_broken_rate": frac(lambda d: d["link_rate"] < 0.34),
        "repeat_rate": frac(lambda d: d["max_repeat_j"] >= REPEAT_J),
        "speaker_id_acc": round(acc, 4),
        "speaker_id_baseline": round(1.0 / max(1, n_names), 4),
        "speaker_id_per_name": per_name,
        "third_party_rate": frac(lambda d: bool(d["third_party"])),
        "third_from_topic_rate": frac(lambda d: bool(d["third_from_topic"])),
        "third_from_inject_rate": frac(lambda d: bool(d["third_from_inject"])),
        "third_unanchored_rate": frac(lambda d: bool(d["third_unanchored"])),
    }
    return summary, detail


def render(summary: dict) -> str:
    s = summary
    L = ["=" * 68, "语言脑对白质量基线（自动指标，零 API）", "=" * 68,
         f"样本数：{s['n_samples']}", ""]
    L += ["【1 格式】  prompt 要求：正好 6 句、每句 12~40 字、长短错落",
          f"  恰好 6 句              {s['fmt_exactly_6']:.1%}",
          f"  平均句长              {s['fmt_len_mean']} 字",
          f"  落在 12~40 字区间      {s['fmt_in_band_12_40']:.1%}",
          f"  句长变异系数(中位)     {s['fmt_len_cv_median']}   ← 越小越『句句一样长』",
          "",
          "【2 话题保持】  判别式：这段对话是否比随机话题更贴自己的话题（基线 50%）",
          f"  话题 AUC              {s['topic_auc']:.1%}",
          f"  打败全部 20 个随机话题 {s['topic_strict']:.1%}",
          f"  注入细节 AUC          {s['pool_auc']:.1%}"
          "   ← 同法验证细节确实被用上了",
          f"  打败全部随机细节池     {s['pool_strict']:.1%}",
          "",
          "【3 细节引用】  prompt 要求：至少 2 句具体影射 眼下这一步/刚发生的事/恩怨",
          f"  平均引用句数          {s['detail_hit_lines_avg']} / 6",
          f"  达标(≥2 句)的样本      {s['detail_ge2_rate']:.1%}   ← 这是 prompt 的硬要求",
          f"  完全不引用的样本       {s['detail_zero_rate']:.1%}",
          "",
          "【4 接话】  prompt 要求：后一句必须接住前一句，绝不各说各话",
          f"  相邻句有效衔接率       {s['link_rate_avg']:.1%}",
          f"  基本不接话的样本       {s['link_broken_rate']:.1%}",
          f"  含复读句对的样本       {s['repeat_rate']:.1%}",
          "",
          "【5 说话人可辨识度】  prompt 要求：只看台词就该认出谁是谁",
          f"  最近质心分类准确率     {s['speaker_id_acc']:.1%}"
          f"（随机基线 {s['speaker_id_baseline']:.1%}）",
          f"  分角色                {s['speaker_id_per_name']}",
          "",
          "【6 第三方漂移】  台词提到了不在场的人",
          f"  提到不在场者的样本     {s['third_party_rate']:.1%}",
          f"    └ 出自 topic（合理）  {s['third_from_topic_rate']:.1%}",
          f"    └ 出自注入的记忆/关系 {s['third_from_inject_rate']:.1%}"
          "   ← 检索不看对话对象的代价",
          f"    └ 无出处（凭空扯）    {s['third_unanchored_rate']:.1%}",
          "=" * 68]
    return "\n".join(L)


# --------------------------------------------------------------------------
# opus 裁判：只在小样本上跑，目的是校准上面的自动指标是否和人的判断同向。
# 没校准过的自动指标不能当验收线——这是这一步存在的唯一理由。
# --------------------------------------------------------------------------
JUDGE_RUBRIC = """你是一位严格的剧本评审。下面给出一段监狱题材对手戏的【创作要求】和【实际产出】。
请只依据要求逐项打分，不要因为文笔漂亮而放宽。每项 0/1/2 分：0=完全没做到，1=部分做到，2=充分做到。

评分项：
- topic：整段是否围绕给定话题，没有跑题。
- detail：是否有至少 2 句**具体**影射到给定的「眼下这一步/刚发生的事/恩怨/案底」（提到那件事、那个人、那笔账，而不是泛泛而谈）。
- turn：是否真正一来一回、后一句接住前一句（追问/反驳/还价/揭底），而不是各说各话或空客套。
- voice：A 与 B 的腔调是否明显不同、各自贴合标注的说话腔调，没有串味或张冠李戴。

严格只输出一个 JSON 对象，不要任何多余字符：
{"topic":0,"detail":0,"turn":0,"voice":0,"why":"一句中文理由"}"""


def judge(rows: list[dict], detail: list[dict], k: int, model: str,
          rps: float, workers: int) -> None:
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sys.path.insert(0, str(HERE))
    from hy_client import HYClient, THROTTLE                 # noqa: PLC0415

    THROTTLE.set_rps(rps)
    idx = list(range(len(rows)))
    random.seed(7)
    random.shuffle(idx)
    idx = idx[:k]
    client = HYClient(model=model, timeout_s=90.0)
    lock = threading.Lock()
    scored: list[tuple[int, dict]] = []

    def work(i: int):
        r = rows[i]
        # 只把创作要求那一段 + 双方设定喂给裁判，避免把 6 句示例也塞进去带偏
        req = r["system"]
        payload = (f"【创作要求】\n{req}\n\n【实际产出】\n"
                   + json.dumps(r["lines"], ensure_ascii=False))
        try:
            raw = client.chat(JUDGE_RUBRIC, payload, max_tokens=300, retries=4)
        except Exception as e:                               # noqa: BLE001
            return None, f"调用失败：{e}"
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None, "裁判没给出 JSON"
        try:
            d = json.loads(m.group(0))
        except Exception:                                    # noqa: BLE001
            return None, "裁判 JSON 解析失败"
        return (i, d), None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, i) for i in idx]
        for done, fut in enumerate(as_completed(futs), 1):
            got, err = fut.result()
            if err:
                print(f"[judge] {err}", file=sys.stderr)
                continue
            with lock:
                scored.append(got)
            if done % 20 == 0:
                print(f"[judge] {done}/{len(idx)}")
    client.close()
    if not scored:
        print("[judge] 一条都没打成，跳过校准")
        return

    dims = ("topic", "detail", "turn", "voice")
    print("\n" + "=" * 68)
    print(f"opus 裁判（n={len(scored)}，满分 2）")
    print("=" * 68)
    for d in dims:
        vals = [s[1].get(d, 0) for s in scored]
        print(f"  {d:<8} 均分 {statistics.fmean(vals):.2f}   "
              f"0分占比 {sum(1 for v in vals if v == 0) / len(vals):.1%}")

    # 校准：自动指标 vs 裁判分的 Spearman 秩相关
    pairs = {
        "topic":  ("topic_hits", "topic"),
        "detail": ("detail_hits", "detail"),
        "turn":   ("link_rate", "turn"),
    }
    print("\n  自动指标与裁判分的秩相关（Spearman，越接近 1 越可信）：")
    for label, (auto_key, jd) in pairs.items():
        xs = [detail[i][auto_key] for i, _ in scored]
        ys = [s.get(jd, 0) for _, s in scored]
        print(f"    {label:<8} {spearman(xs, ys):+.3f}")
    print("=" * 68)


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return 0.0
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def selftest(rows: list[dict]) -> str:
    """指标的验收：人为破坏数据，看对应指标是否真的掉下去。

    为什么需要这一步：没有人工标注、而 opus 当裁判又给自己的产出打满分（实测 topic/detail/
    turn 全 2.00、零方差 → Spearman 恒为 0，校准不出任何东西），所以改用**合成对照**：
    针对每个指标做一种定向破坏，若指标不动，说明它测的不是它声称测的东西。
    这和 eval.py 里 axes_probe 用"只改一个轴"的探针判模型是否吃这个轴，是同一个思路。

    随机基线参照：话题/细节 AUC 应回到 50%，说话人可辨识度应回到 20%。
    """
    import copy
    import random as _r
    rnd = _r.Random(11)
    n = len(rows)

    def other(i: int) -> int:
        j = rnd.randrange(n)
        return (i + 1) % n if j == i else j

    variants: dict[str, list[dict]] = {}
    # 打乱句序：只该打到"接话率"，贴合类指标（用词袋）应基本不动
    variants["打乱句序"] = [
        dict(r, lines=rnd.sample(r["lines"], len(r["lines"]))) for r in rows]
    # 换成随机话题：只该打到话题 AUC
    variants["换随机话题"] = [
        dict(r, ctx={**r["ctx"], "topic": rows[other(i)]["ctx"].get("topic", "")})
        for i, r in enumerate(rows)]
    # 换成随机注入细节：只该打到细节 AUC / 引用率
    def swap_pool(i: int, r: dict) -> dict:
        j = other(i)
        ctx = copy.deepcopy(r["ctx"])
        for t in ("A", "B"):
            for f in DETAIL_FIELDS:
                ctx[t][f] = rows[j]["ctx"].get(t, {}).get(f, "")
        return dict(r, ctx=ctx)
    variants["换随机细节"] = [swap_pool(i, r) for i, r in enumerate(rows)]
    # A/B 标签互换：只该打到说话人可辨识度（应掉到随机附近）
    variants["互换说话人"] = [
        dict(r, lines=[{**ln, "who": ("B" if ln.get("who") == "A" else "A")}
                       for ln in r["lines"]]) for r in rows]
    # 整段换成别人的对白：所有贴合类指标都该掉到基线
    variants["整段换对白"] = [
        dict(r, lines=rows[other(i)]["lines"]) for i, r in enumerate(rows)]

    keys = [("topic_auc", "话题AUC"), ("pool_auc", "细节AUC"),
            ("detail_ge2_rate", "细节达标"), ("link_rate_avg", "接话率"),
            ("speaker_id_acc", "可辨识度")]
    base, _ = evaluate(rows)
    rowsout = [("原始数据", base)]
    for name, v in variants.items():
        s, _ = evaluate(v)
        rowsout.append((name, s))

    w = 10
    head = "破坏方式".ljust(12) + "".join(lbl.rjust(w) for _k, lbl in keys)
    out = ["", "=" * 68, "指标自检：定向破坏后各指标的变化（掉了才说明指标有效）",
           "=" * 68, head, "-" * len(head)]
    for name, s in rowsout:
        out.append(name.ljust(12) + "".join(f"{s[k]:>{w}.1%}" for k, _l in keys))
    out += ["-" * len(head),
            "随机基线".ljust(12) + f"{0.5:>{w}.1%}{0.5:>{w}.1%}"
            + " " * w + " " * w + f"{base['speaker_id_baseline']:>{w}.1%}",
            "=" * 68]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(SFT))
    ap.add_argument("--detail", default="", help="每样本明细落盘路径（挑最差的重标用）")
    ap.add_argument("--report", default=str(HERE / "convo_eval_report.txt"))
    ap.add_argument("--judge", type=int, default=0, help="抽多少条让 opus 打分（0=不跑）")
    ap.add_argument("--selftest", action="store_true",
                    help="跑指标自检：定向破坏数据，看对应指标是否真的掉")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--rps", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    path = Path(args.sft)
    if not path.exists():
        sys.exit(f"没有 {path}")
    rows = load(path)
    if not rows:
        sys.exit("一条都没读出来（格式变了？）")
    bad = sum(1 for r in rows if not r["ctx"].get("topic"))
    if bad:
        print(f"[warn] {bad} 条没解析出 topic —— deepseek_client 的措辞可能改过，"
              f"检查 FIELD_PREFIX / 话题前缀", file=sys.stderr)

    summary, detail = evaluate(rows)
    text = render(summary)
    if args.selftest:
        text += "\n" + selftest(rows)
    print(text)
    Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(f"\n[report] → {args.report}")

    if args.detail:
        with Path(args.detail).open("w", encoding="utf-8") as f:
            for d in detail:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[detail] → {args.detail}（{len(detail)} 条）")

    if args.judge:
        judge(rows, detail, args.judge, args.model, args.rps, args.workers)


if __name__ == "__main__":
    main()
