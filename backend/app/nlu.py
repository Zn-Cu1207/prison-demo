"""玩家意图解析（两轨架构·第一轨：NLU 意图与状态解析）。

用户设计的两轨里，这是"传统代码把玩家自然语言死死卡进结构化意图"的那一轨：
  玩家自由文本 → PlayerIntent(意图/诚意/涉及道具/涉及证据/情绪) → world 据此改黑板。

刻意做成**确定性关键词解析**，不依赖任何模型：
  - 离线可跑、可写探针断言、延迟为零（用户明确说延迟后期再优化）；
  - 它就是"状态机数值结算"的安全护栏，LLM 的 JSON-mode 解析日后可叠在上面增强，
    但绝不能让不稳定的模型输出直接决定world 状态（会剧情暴走）。

第二轨（动态台词生成）仍由 deepseek_client.roleplay_system + LLM 负责，读同一块黑板。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .plot import ALCOHOL, CIGS, KEYCARD, PENSION_LIST, SEDATIVE

# 意图类别（对齐用户 PlayerIntentAnalysis 的 Literal 意图集，做了本地化裁剪）
GIVE = "give"                 # 给/交出某样东西（触发交换）
SHOW = "show_evidence"        # 出示/摆出证据（触发收买/施压）
ACCUSE = "accuse"             # 当面指控/揭穿
BRIBE = "bribe"               # 收买/买通/走后门
HELP = "help_seeking"         # 求助/打听门路
THREATEN = "threaten"         # 威胁/硬来
CHIT = "chit_chat"            # 闲聊/套近乎

# —— 关键词表（够用即可，命中优先级从强到弱）——
_KW_GIVE = ("给你", "拿去", "交给你", "换", "这个给", "送你", "递给", "喏", "拿着这")
_KW_SHOW = ("你看这", "看这个", "我这有", "证据", "我手里有", "瞧瞧这", "这是", "摆在你面前")
_KW_ACCUSE = ("是你", "就是你干的", "揭穿", "别装了", "你骗", "你陷害", "凶手是你",
              "露馅", "别演了", "我知道你", "承认吧", "幕后")
_KW_BRIBE = ("买通", "收买", "行个方便", "通融", "打点", "给你好处", "睁只眼", "放我")
_KW_HELP = ("怎么办", "帮帮", "帮我", "求你", "带我", "怎么出", "门路", "教教", "指条路",
            "有没有办法", "怎么才能")
_KW_THREATEN = ("信不信", "小心点", "别怪我", "揍你", "弄死", "威胁", "试试看")

# 道具 / 证据的自然语言指代 → 内部 id
_ITEM_KW: dict[str, str] = {
    ALCOHOL: ("酒精", "医用酒精"),
    SEDATIVE: ("镇静剂", "安定", "迷药", "药"),
    KEYCARD: ("磁卡", "门卡", "卡"),
    PENSION_LIST: ("抚恤金", "名单", "克扣"),
    CIGS: ("烟", "香烟", "华子"),
}
# 证据（线索 id）的自然语言指代
_EVID_KW: dict[str, tuple[str, ...]] = {
    "softshoe": ("软底", "布鞋", "没声音", "没脚步", "脚步声"),
    "laundering": ("洗钱", "黑账", "分赃", "账本"),
    "ledger": ("欠条", "赌债", "借条"),
    "pension_list": ("抚恤金", "克扣", "遗属"),
    "medlog": ("用药", "病历", "出入库", "药品记录", "篡改"),
    "prescription": ("配药单", "药方", "处方"),
    "photo": ("照片", "合影", "旧照", "现场"),
}
# 诚意加分词：有推理/条件/证据味道的表达，玩家"讲得在理"
_REASON_KW = ("因为", "所以", "证据", "记录", "对不上", "说明", "证明", "如果", "只要",
              "名单", "亲眼", "当晚", "时间", "对上了", "逻辑", "线索")


@dataclass
class PlayerIntent:
    intent: str = CHIT
    sincerity: int = 5              # 1~10：话说得有多在理/有多有诚意
    item: str | None = None        # 涉及的道具 id
    evidence: str | None = None    # 涉及的证据/线索 id
    emotion: str = "neutral"       # aggressive / friendly / neutral / fearful
    raw: str = ""
    matched: list[str] = field(default_factory=list)  # 命中的关键词（调试/探针）


def _first_hit(text: str, table: dict[str, tuple[str, ...]]) -> str | None:
    for key, kws in table.items():
        if any(k in text for k in kws):
            return key
    return None


def _sincerity(text: str) -> int:
    """诚意/逻辑严密度 1~10：长度 + 讲理关键词。短促空话低分，摆事实讲逻辑高分。"""
    n = len(text)
    base = 3 + min(4, n // 10)                       # 越长越像认真说事（封顶 +4）
    reason = min(4, sum(1 for k in _REASON_KW if k in text) * 2)
    return max(1, min(10, base + reason))


def _emotion(text: str) -> str:
    if any(k in text for k in _KW_THREATEN) or any(k in text for k in ("滚", "混蛋", "去死")):
        return "aggressive"
    if any(k in text for k in ("谢谢", "拜托", "求", "兄弟", "帮帮", "多谢", "大哥")):
        return "friendly"
    if any(k in text for k in ("怕", "害怕", "紧张", "别抓我", "饶")):
        return "fearful"
    return "neutral"


def parse_intent(text: str) -> PlayerIntent:
    """把玩家这句话解析成结构化意图。确定性、可离线、可断言。"""
    t = (text or "").strip()
    pi = PlayerIntent(raw=t)
    if not t:
        return pi
    pi.item = _first_hit(t, _ITEM_KW)
    pi.evidence = _first_hit(t, _EVID_KW)
    pi.sincerity = _sincerity(t)
    # 摆出了具体证据/线索本身就是最硬的"讲理"，据此拔高诚意（有凭有据≠空口指控）
    if pi.evidence:
        pi.sincerity = max(pi.sincerity, 7)
    pi.emotion = _emotion(t)

    def hit(kws: tuple[str, ...]) -> bool:
        m = [k for k in kws if k in t]
        pi.matched.extend(m)
        return bool(m)

    # 优先级：给东西 > 指控 > 出示证据/收买 > 威胁 > 求助 > 闲聊
    if hit(_KW_ACCUSE):
        pi.intent = ACCUSE
    elif hit(_KW_GIVE) or (pi.item and any(k in t for k in ("换", "给", "拿"))):
        pi.intent = GIVE
    elif hit(_KW_BRIBE):
        pi.intent = BRIBE
    elif hit(_KW_SHOW) or (pi.evidence and any(k in t for k in ("看", "这", "有"))):
        pi.intent = SHOW
    elif hit(_KW_THREATEN):
        pi.intent = THREATEN
    elif hit(_KW_HELP):
        pi.intent = HELP
    else:
        pi.intent = CHIT
    return pi
