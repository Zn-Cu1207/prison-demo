"""打架：多人 bout + 真实伤害。

改这版之前的状态：
    `Fight` 是**单人**行为——候选只在静态布尔 `enemy_present` 为真时生成（全场只有
    疤脸是 True），选中后走到放风场，拿到的收益和"溜达"一模一样，不产生任何伤害。
    于是 `injury` 全场只有"纹身 +4" 一个来源，`Cure`/医务室基本没有意义。

现在打架是一场**有参与者名单的事件**（bout）：

    甲选了打架 且 视线内有具体的仇人 → 开一场 bout，把对方拖进来（被打不是可选项）
    旁边的暴躁分子 / 对场内某人有仇的人 → 可能加入 → 三人以上群殴
    每帧按双方"战力"互相造成 injury → 有人被打到 KO_INJURY → 输，逃去医务室
    伤者去医务室要消耗医疗包 → 医疗包会见底 → 伤势真的留在身上

战力构成刻意做得可解释（暴躁能打、有靠山敢下手、人多欺负人少、自己有伤会掉战力），
这样前端把拆解打出来，玩家能看懂"为什么他被打成这样"。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# -- 手感常量（调这里就是调打架的激烈程度）--------------------------------
FIGHT_RANGE = 6.0          # 多近才能"看见仇人"进而开打
JOIN_RANGE = 7.0           # 旁人多近可能被卷进来
GRUDGE_TO_FIGHT = 35.0     # 仇怨到多少才愿意动手
FIGHT_MAX_TICKS = 10       # 一场架最多打多久（到点各自散开）
KO_INJURY = 68.0           # 伤到这个程度就算被打输
DMG_PER_TICK = 6.0         # 基准每帧伤害
STRESS_PER_TICK = 3.0
JOIN_CHANCE = 0.22         # 每帧旁人加入的概率（满足条件时）

# 打输/被卷入之后仇怨怎么长
GRUDGE_ON_LOSS = 30.0
GRUDGE_ON_HIT = 6.0
# 抢物资结仇：最后一份被别人拿走，而自己正饿着
GRUDGE_ON_STOLEN_SUPPLY = 18.0

# ---------------------------------------------------------------------------
# 初始仇怨。刻意跟《囚笼悖论》的剧情对齐，而不是随机撒仇 ——
# 这样打起来的时候，前端能说清"他们为什么有仇"，也和 NPC 台词自洽。
#   (甲, 乙): 甲对乙的仇怨值
# ---------------------------------------------------------------------------
SEED_GRUDGE: dict[tuple[str, str], float] = {
    # 帮派头目 vs 走私犯：黑市地盘之争，互相看不顺眼
    ("shitou", "laok"): 55.0,
    ("laok", "shitou"): 42.0,
    # 疤脸想拿管制药，却被黑医捏着货源
    ("shitou", "yanjing"): 40.0,
    # 黑医要让"目击者"永远说不清话（疯子是命案目击者）
    ("yanjing", "xiaodou"): 48.0,
    # —— 五角仇杀闭环新增的三条边（与 quests.py 的 kill_target 对齐）——
    # 疤脸恨坑他顶罪、害他多蹲的老牙（他想亲手了结的人）
    ("shitou", "laochen"): 58.0,
    # 老牙要灭口攥着他把柄的瘸子：表面搭伙、暗地起了杀心（背叛的引信）
    ("laochen", "laok"): 46.0,
    # 瘸子恨常年拿医疗档案拿捏他、还害死过弟兄的黑医
    ("laok", "yanjing"): 50.0,
    # 疯子偏执地认定是疤脸把他打进禁闭室、害他成了今天这样
    ("xiaodou", "shitou"): 44.0,
}

# ---------------------------------------------------------------------------
# 初始关系图（结盟/欠债/怕/敬），同样跟《囚笼悖论》剧情对齐。
# 作用：给"结盟/背叛/护着/讨债"这些定向社交动作一开局就有燃料——否则开局全是生人
# (NEAR_STRANGER)，社交动作几拍才冒一个，上帝视角看半天没戏。
#   (甲, 乙): {轴: 值}  —— 甲对乙的态度（trust/fear/debt/ally/respect，0~100）
# ---------------------------------------------------------------------------
SEED_REL: dict[tuple[str, str], dict[str, float]] = {
    # 瘸子(走私犯,drive=债) 欠疤脸一笔黑市旧账 → 疤脸是他的债主，怕、想还也想赖
    ("laok", "shitou"): {"debt": 42.0, "fear": 20.0},
    # 疯子(命案目击者)求老牙罩着，怕被灭口 → 结盟 + 信任靠山
    ("xiaodou", "laochen"): {"ally": 34.0, "trust": 28.0, "fear": 14.0},
    ("laochen", "xiaodou"): {"ally": 30.0},
    # 两个老油条(老牙&瘸子)搭伙：给一条"看着牢靠、随时能反水"的盟约（背叛的引信）
    ("laochen", "laok"): {"ally": 24.0, "trust": 18.0},
    ("laok", "laochen"): {"ally": 24.0, "trust": 18.0},
    # 黑医敬畏帮派头目（黑市地盘），还欠他一笔药钱/保护费 → 非敌非盟的"债主"关系，
    # 这样 Repay(还人情) 才有机会真的摆上桌（盟友/仇敌会盖过"债主"归档，还不成）。
    ("yanjing", "shitou"): {"fear": 22.0, "respect": 16.0, "debt": 24.0},
}

# 开局义务账本：与 SEED_REL 的欠债对齐，让 Repay/讨债 一开局就摆得上桌。
#   {debtor 欠, creditor 债主, kind, text}
SEED_OBLIGATION: list[dict] = [
    {"debtor": "laok", "creditor": "shitou", "kind": "debt",
     "text": "瘸子欠疤脸一笔黑市旧账"},
    {"debtor": "yanjing", "creditor": "shitou", "kind": "debt",
     "text": "柳叶刀欠疤脸一笔药钱／保护费"},
]


@dataclass
class FightBout:
    """一场正在进行的打架。"""

    id: str
    members: list[str] = field(default_factory=list)   # 参与者 npc id
    ticks: int = 0
    x: float = 0.0
    z: float = 0.0
    log: list[str] = field(default_factory=list)
    loser: str = ""
    busted: bool = False        # 是否被狱警当场抓到
    over: bool = False

    def to_message(self) -> dict:
        return {
            "id": self.id, "members": list(self.members), "ticks": self.ticks,
            "pos": [round(self.x, 2), round(self.z, 2)],
            "loser": self.loser, "busted": self.busted, "over": self.over,
            "log": self.log[-4:],
        }


def strength(npc, allies: int, enemies: int) -> float:
    """战力。刻意每一项都能用一句话解释，方便前端展示"为什么打输"。"""
    s = 10.0
    if npc.personality == "暴躁":
        s += 6.0            # 下手最狠
    elif npc.personality == "老油条":
        s += 2.5            # 会打，但更爱见风使舵
    elif npc.personality == "书呆":
        s -= 3.0
    elif npc.personality == "新人":
        s -= 2.0
    if npc.backer_present:
        s += 3.0            # 有靠山罩着，敢下死手
    s -= npc.needs.injury * 0.06      # 自己带伤，越打越弱
    s -= npc.needs.fatigue * 0.02
    # 人多欺负人少
    s += 2.2 * max(0, allies - enemies)
    return max(2.0, s)


def roll_damage(attacker_strength: float, defender_strength: float) -> float:
    """按战力比出伤害。比值封顶，避免一帧把人打死。"""
    ratio = attacker_strength / max(2.0, defender_strength)
    ratio = max(0.45, min(2.2, ratio))
    return DMG_PER_TICK * ratio * random.uniform(0.75, 1.25)
