"""context 拼装（监狱原生词表版）。

设计决策（2026-07-24）：**游戏该是什么属性就用什么属性**，不向现有模型的旧词表做映射。
监狱原生的威胁/事件/时段/性格/新轴 直接进 caption，现在先喂现有模型观察其泛化能力，
后续按这套格式重训一版原生听懂监狱话的模型（格式不变、无缝替换）。

格式（唯一真相源，重训目标格式，见 监狱Demo-属性与IO蓝图 §0'）：

    饥饿{h}/100,口渴{t}/100,疲劳{f}/100,受伤{inj}/100,无聊{b}/100,社交欲{s}/100,压力{st}/100; \
    威胁:{仇人|狱警|狱犬|无}; 事件:{斗殴|吵架声|狱警搜牢|警报|无}; 时段:{点名|训练|上课|放风|就寝|自由活动}; \
    性格:{暴躁|书呆|老油条|新人|普通}; 监视度:{高|中|低}; 靠山:{在场|无}; 仇人:{在场|无}; \
    放风拥挤:{拥挤|适中|空旷}; 假释:{临近|尚早}; 有尸体:{是|否}; 苦役:{有|无}; \
    长期心思:{一句泛泛的长期动机，如"想早点平安假释"|无}; 玩家指令:{命令去标签(desc)|无}; \
    食物:{充足|紧张|断供}; 仇怨:{无|有|深}; 表现:{好|一般|差}; 近期被抓:{是|否}

v2（2026-07-31）在末尾追加了后四轴。它们对应物理层落地后真实影响行为的四件事
（物资短缺、仇怨强度、假释表现分、刚被关过禁闭），此前模型完全看不见、只能靠引擎侧
硬规则兜，于是出现"没饭还往食堂跑""重伤了还先去吃饭"。追加在末尾而不插进中间，
是为了让既有字段顺序一个字不动，v1/v2 的数据能逐字对比、加轴本身可回滚。

『长期心思』(motive)是宏观动机（drives.py）投进 caption 的唯一通道，**刻意只放一句泛泛
的倾向**，不写具体动作——越具体模型/LLM 越容易照着复读，行为就重复不自然。具体的偏置交给
引擎侧 M_drive 乘数（world._m_drive），模型看不见、不会说出来，行为才像人。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# 监狱原生枚举取值
# ---------------------------------------------------------------------------
# 威胁：仇人 / 狱警 / 狱犬 / 无
THREAT_NONE = "无"

# 事件：斗殴 / 吵架声 / 狱警搜牢 / 警报 / 无（监狱里没枪，用拳头/口角/警报替代）
EVENT_NONE = "无"

# 时段（监狱作息）：点名 / 训练 / 上课 / 放风 / 就寝 / 自由活动
PHASE_ROLLCALL = "点名"
PHASE_TRAIN = "训练"
PHASE_CLASS = "上课"
PHASE_YARD = "放风"
PHASE_SLEEP = "就寝"
PHASE_FREE = "自由活动"

# 性格（监狱原生，不再映射到旧 6 词）
PERSONALITY_NORMAL = "普通"

# 监视度 / 拥挤 / 假释
SURVEIL_HIGH, SURVEIL_MID, SURVEIL_LOW = "高", "中", "低"
CROWD_PACKED, CROWD_MID, CROWD_EMPTY = "拥挤", "适中", "空旷"

# v2 新增四轴的取值
SUPPLY_OK, SUPPLY_TIGHT, SUPPLY_OUT = "充足", "紧张", "断供"
GRUDGE_NONE, GRUDGE_SOME, GRUDGE_DEEP = "无", "有", "深"
PERFORM_GOOD, PERFORM_MID, PERFORM_BAD = "好", "一般", "差"

# ---------------------------------------------------------------------------
# v3 社交四轴取值（2026-08-06，P2：让决策脑从"独我"走向"看得见彼此"）
# 追加在 caption 末尾（同 v2 的加轴纪律：既有字段顺序一字不动，可逐字对比、可回滚）。
# 只在 PRISON_CAPTION=v3 时渲染——线上默认仍是 v2，避免把新词表喂给还没重训的 v2 模型。
# ---------------------------------------------------------------------------
# 邻近：此刻近旁最该打交道的那个人，与我的关系（取最强的一条关系边）
NEAR_FOE, NEAR_ALLY = "仇敌", "盟友"
NEAR_CREDITOR, NEAR_DEBTOR = "债主", "欠我"      # 债主=我欠他；欠我=他欠我
NEAR_STRANGER, NEAR_NONE = "生人", "无"
# 对方可见意图（由对方当前 goal 归纳出的社交信号）
INTENT_HOSTILE, INTENT_FRIENDLY = "寻衅", "示好"
INTENT_DEAL, INTENT_ALLY = "交易", "结盟"
INTENT_WEAK, INTENT_UNKNOWN, INTENT_NONE = "示弱", "不明", "无"
# 我身上刚发生的社交事件（P1 social_events / memory 投进来的最近一条）
SOC_THREATENED, SOC_DEAL = "被威胁", "谈成买卖"
SOC_ALLIED, SOC_PROTECTED = "新结盟", "被求护"
SOC_SNITCHED, SOC_RECONCILED, SOC_NONE = "被告发", "刚讲和", "无"
# 我持有的人情/债务（P1 义务账本投进来）
OBLIG_OWE, OBLIG_OWED, OBLIG_NONE = "欠债", "有人欠我", "无"


def caption_version() -> str:
    """线上 caption 版本。默认 v3（2026-08-06 起 v3 社交价值模型已上线 T4）。

    历史：v3 上线前默认 v2 以保护老模型。现在线上服务的就是 v3（带社交四轴），
    默认置 v3 防手滑——否则不带 PRISON_CAPTION 重启后端会用 v2 caption 喂 v3 模型，
    格式对不上、NPC 悄悄变傻还不报错。要回滚 v2 显式设 PRISON_CAPTION=v2。
    """
    return os.environ.get("PRISON_CAPTION", "v3").strip().lower()

# ---------------------------------------------------------------------------
# source 中文映射（渲染候选文本用）。训练侧六类：need/threat/command/worktask/event/daily
# ---------------------------------------------------------------------------
SOURCE_CN = {
    "need": "自身需求",
    "threat": "威胁应对",
    "command": "玩家指令",
    "worktask": "待办工作",
    "event": "突发事件",
    "daily": "日常行为",
}

Source = Literal["need", "threat", "command", "worktask", "event", "daily"]

# ---------------------------------------------------------------------------
# goal → 描述/标签。全部从行动全集 actions.py 派生（单一真相源），此处仅再导出，
# 让老调用方 `from .context import DESC_MAP, GOAL_LABEL` 继续可用。
# ---------------------------------------------------------------------------
from .actions import DESC_MAP, GOAL_LABEL  # noqa: E402,F401


@dataclass
class NeedState:
    """7 个需求维度，全部 0~100 整数（越高越强烈）。"""

    hunger: int = 0      # 饥饿
    thirst: int = 0      # 口渴
    fatigue: int = 0     # 疲劳
    injury: int = 0      # 受伤
    boredom: int = 0     # 无聊
    social: int = 0      # 社交欲
    stress: int = 0      # 压力


@dataclass
class ContextState:
    """一个 NPC 当前的完整监狱情境。"""

    needs: NeedState = field(default_factory=NeedState)
    threat: str = THREAT_NONE               # 仇人/狱警/狱犬/无
    event: str = EVENT_NONE                 # 斗殴/吵架声/狱警搜牢/警报/无
    phase: str = PHASE_FREE                 # 作息时段
    personality: str = PERSONALITY_NORMAL   # 暴躁/书呆/老油条/新人/普通
    surveillance: str = SURVEIL_MID         # 监视度 高/中/低
    backer_present: bool = False            # 靠山在场
    enemy_present: bool = False             # 仇人在场
    yard_crowd: str = CROWD_MID             # 放风拥挤 拥挤/适中/空旷
    parole_near: bool = False               # 假释临近
    has_corpse: bool = False                # 有尸体
    has_labor: bool = False                 # 苦役
    motive: str = "无"                      # 长期心思（宏观动机的泛泛倾向，见 drives.py）
    player_command: str = "无"              # "命令去X(desc)" 或 "无"
    # -- v2 新增四轴（2026-07-31）------------------------------------------
    # 这四件事在物理层落地后已经真实影响行为，但模型此前完全看不见（全靠引擎侧硬规则兜），
    # 于是出现"没饭还往食堂跑""重伤了还先去吃饭"这类不合理。加轴 + 重标重训一起修。
    food_supply: str = SUPPLY_OK            # 食物：充足/紧张/断供
    grudge: str = GRUDGE_NONE               # 仇怨：无/有/深（原来只有"仇人:在场"二值）
    perform: str = PERFORM_MID              # 表现：好/一般/差（假释表现分分档）
    busted_recently: bool = False           # 近期被抓（刚从禁闭出来，该老实一阵）
    # -- v3 社交四轴（2026-08-06，P2）：只在 PRISON_CAPTION=v3 时渲染进 caption ------
    # 让价值模型第一次"看得见彼此"——身边是谁/关系、对方想干嘛、我刚经历的社交事件、
    # 我手里攥着的人情。有了它，动作才可能从交互里长出来（结盟/背叛/讨债/护人）。
    near_rel: str = NEAR_NONE               # 邻近：仇敌/盟友/债主/欠我/生人/无
    other_intent: str = INTENT_NONE         # 对方意图：寻衅/示好/交易/结盟/示弱/不明/无
    social_recent: str = SOC_NONE           # 社交近况：被威胁/谈成买卖/新结盟/被求护/被告发/刚讲和/无
    obligation: str = OBLIG_NONE            # 人情：欠债/有人欠我/无
    # -- v4 凶器轴（2026-08-18）：行政手段栽赃/没收改的是 npc.weapon，模型此前完全看不见。
    # 追加在 v3 社交四轴之后（既有字段顺序一字不动）。有刀才说得通下杀手，没刀该先去弄凶器。
    has_weapon: bool = False                # 凶器：有/无


def build_context_caption(s: ContextState) -> str:
    """把监狱情境拼成 caption（字段顺序/标点固定）。"""
    n = s.needs
    needs_part = (
        f"饥饿{n.hunger}/100,"
        f"口渴{n.thirst}/100,"
        f"疲劳{n.fatigue}/100,"
        f"受伤{n.injury}/100,"
        f"无聊{n.boredom}/100,"
        f"社交欲{n.social}/100,"
        f"压力{n.stress}/100"
    )
    return (
        f"{needs_part}; "
        f"威胁:{s.threat}; "
        f"事件:{s.event}; "
        f"时段:{s.phase}; "
        f"性格:{s.personality}; "
        f"监视度:{s.surveillance}; "
        f"靠山:{'在场' if s.backer_present else '无'}; "
        f"仇人:{'在场' if s.enemy_present else '无'}; "
        f"放风拥挤:{s.yard_crowd}; "
        f"假释:{'临近' if s.parole_near else '尚早'}; "
        f"有尸体:{'是' if s.has_corpse else '否'}; "
        f"苦役:{'有' if s.has_labor else '无'}; "
        f"长期心思:{s.motive}; "
        f"玩家指令:{s.player_command}; "
        # v2 新增四轴一律追加在末尾：既有字段顺序一个字都不动，
        # 方便和 v1 的数据/日志逐字对比，也让"加轴"这件事本身是可回滚的。
        f"食物:{s.food_supply}; "
        f"仇怨:{s.grudge}; "
        f"表现:{s.perform}; "
        f"近期被抓:{'是' if s.busted_recently else '否'}"
        # v3 社交四轴：仅在 PRISON_CAPTION=v3 时追加（同末尾追加纪律，v2 数据/日志逐字可比）。
        + (
            f"; 邻近:{s.near_rel}; "
            f"对方意图:{s.other_intent}; "
            f"社交近况:{s.social_recent}; "
            f"人情:{s.obligation}; "
            f"凶器:{'有' if s.has_weapon else '无'}"
            if caption_version() == "v3" else ""
        )
    )


# source → priority 默认值。value_server 只在 command 时把它拼进输入文本
# （"，优先级{priority}、强制执行"），所以这个字段直接影响模型看到的原文，
# 必须和 RA 训练数据一致，不能一律填 n/a。
PRIORITY_BY_SOURCE = {"command": "Precise", "worktask": "Normal"}


@dataclass
class Candidate:
    """候选目标。字段对齐 value_server 请求体的 candidate 结构。"""

    goal: str
    source: Source = "need"
    b_force_plan: bool = False
    priority: str | None = None      # None = 按 source 取默认

    @property
    def desc(self) -> str:
        return DESC_MAP.get(self.goal, self.goal)

    @property
    def prio(self) -> str:
        if self.priority is not None:
            return self.priority
        return PRIORITY_BY_SOURCE.get(self.source, "n/a")

    @property
    def label(self) -> str:
        return GOAL_LABEL.get(self.goal, self.goal)

    def to_payload(self) -> dict:
        return {
            "goal": self.goal,
            "desc": self.desc,
            "source": self.source,
            "priority": self.prio,
            "bForcePlan": self.b_force_plan,
        }
