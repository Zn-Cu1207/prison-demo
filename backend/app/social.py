"""P1 · 语言↔世界双向桥的「纯逻辑」层：关系轴定义 + 对白社交意图解析。

背景（见 emergent gameplay 分析的 P1）：
    在此之前，NPC↔NPC 对话是**只读的镜子**——LLM 读世界、说台词，但台词不写回世界。
    于是"谈成一笔军火 / 放狠话 / 求人罩着"这些互动**不产生任何后果**，对话永远只是
    动作的配音，长不出新剧情。关系也只有一个标量 grudge（只有"恨"这一维）。

    这个模块补两块**纯逻辑**（不依赖 world，可单测）：
      1) 关系轴 REL_AXES：把单一 grudge 扩成 typed edge——信任/恐惧/欠债/结盟/敬畏。
         （grudge 仍单独保留为"敌意"维，老的打架逻辑一个字不动。）
      2) parse_conversation：把**已经生成好的对白**解析成结构化 SocialEvent
         （威胁 / 军火交易 / 求护结盟 / 讲和 / 通风报信）。

    真正改世界（转移凶器、加减 grudge、记记忆、记义务、生成靠山）由 world 侧
    _apply_social_event 落地——把"解析(纯)"和"改状态(领域)"分开，前者可单测、后者好审计。

设计取舍：先做**确定性关键词解析**，理由——① 零延迟、可测、可控，不给本地小模型
再加一次调用；② 作为"护栏兜底"，将来对话脑直接吐结构化意图(JSON)时，这套解析退化成
校验层即可，接口不变。宁可漏（没命中就当闲聊，无副作用），不可错（乱改世界更糟）。
"""
from __future__ import annotations

from dataclasses import dataclass

# 关系图的五个轴（对某个具体对象的态度，0~100）。grudge=敌意 仍在 NPC 上单列。
REL_AXES = ("trust", "fear", "debt", "ally", "respect")
REL_CN = {"trust": "信任", "fear": "怕", "debt": "欠", "ally": "结盟", "respect": "敬"}


def blank_rel() -> dict[str, float]:
    return {a: 0.0 for a in REL_AXES}


# 关键词表（中文；保守匹配：命中才触发）。刻意用"黑话/口语"里高频的表达。
_KW = {
    "threat": ("弄死", "做了你", "做掉", "收拾你", "别怪我", "有你好看", "要你命",
               "送你上路", "见红", "捅了", "捅死", "白刀子", "敢动", "找死", "活腻"),
    "deal": ("成交", "换", "多少钱", "这价", "卖你", "买", "一手交", "货", "生意"),
    "weapon": ("刀", "电棍", "电击", "家伙", "凶器", "见血", "尖的", "铁"),
    "protect_ask": ("带上我", "求你带", "罩着我", "护着我", "跟着你", "收下我", "求你了"),
    "protect_offer": ("我罩着你", "我挺你", "有我在", "跟着我", "护着你", "算我兄弟", "别怕"),
    "reconcile": ("对不住", "对不起", "是我不对", "算了", "揭过", "一笔勾销", "讲和",
                  "别计较", "服个软"),
    "info": ("跟你说", "听说", "名单", "风声", "透个底", "递个话", "告诉你个事"),
    "snitch": ("告密", "举报", "报告狱警", "通风报信", "打小报告", "去说一声"),
}


def _hit(text: str, keys: tuple[str, ...]) -> bool:
    return any(k in text for k in keys)


@dataclass
class SocialEvent:
    """一条从对白里解出来的社交意图。src/dst 用「效果发起方/承受方」的语义约定：

    - threat   : src 威胁 dst
    - deal_weapon: src、dst 谈成军火（谁买谁卖由 world 按持械/任务判定，与文本无关）
    - protect  : src=保护者，dst=被保护者
    - reconcile: src、dst 讲和（对称）
    - info     : src 向 dst 透风
    - snitch   : src 放话要去告密（震慑 dst）
    """

    kind: str
    src: str
    dst: str
    text: str = ""


def parse_conversation(a_id: str, b_id: str, kind: str,
                       lines: list[dict]) -> list[SocialEvent]:
    """把整段对白解析成社交事件。每类最多产一条，避免一场对话把关系改爆。

    lines: [{"who": npc_id, "name": str, "text": str}, ...]
    """
    seen: set[str] = set()
    out: list[SocialEvent] = []

    def other(who: str) -> str:
        return b_id if who == a_id else a_id

    for ln in lines:
        who = ln.get("who")
        txt = ln.get("text", "") or ""
        if who not in (a_id, b_id) or not txt:
            continue
        dst = other(who)

        if "threat" not in seen and _hit(txt, _KW["threat"]):
            out.append(SocialEvent("threat", who, dst, txt))
            seen.add("threat")

        # 军火交易：同一句里既有"交易词"又有"凶器词"，才算谈军火（避免把普通换烟误判）
        if "deal_weapon" not in seen and _hit(txt, _KW["deal"]) and _hit(txt, _KW["weapon"]):
            out.append(SocialEvent("deal_weapon", who, dst, txt))
            seen.add("deal_weapon")

        if "protect" not in seen and _hit(txt, _KW["protect_ask"]):
            # who 在求罩 → who 是被保护者，dst 是保护者
            out.append(SocialEvent("protect", src=dst, dst=who, text=txt))
            seen.add("protect")
        elif "protect" not in seen and _hit(txt, _KW["protect_offer"]):
            # who 在给罩 → who 是保护者
            out.append(SocialEvent("protect", src=who, dst=dst, text=txt))
            seen.add("protect")

        if "reconcile" not in seen and _hit(txt, _KW["reconcile"]):
            out.append(SocialEvent("reconcile", who, dst, txt))
            seen.add("reconcile")

        if "snitch" not in seen and _hit(txt, _KW["snitch"]):
            out.append(SocialEvent("snitch", who, dst, txt))
            seen.add("snitch")
        elif "info" not in seen and _hit(txt, _KW["info"]):
            out.append(SocialEvent("info", who, dst, txt))
            seen.add("info")

    return out
