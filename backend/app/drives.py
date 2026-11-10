"""宏观动机（macro motive）——给每个犯人一条长期心思，让"中间那条宽带"有方向。

为什么要有这层：
    原来无紧急需求时，行为来自"时段菜单轮换取样 + 模型在随机菜单里挑"，
    等于当前 tick 数的余数决定石头去赌钱还是去洗澡——五个人做的事看不出区别。
    宏观动机占领这条宽带：老陈刷表现保假释、石头立威抢地盘、老K卖消息还债……
    行为、背景、对话由同一份 Drive 派生，一致性不靠"努力对齐"而是天然成立。

三条注入通道（刻意分层，避免"照着 prompt 复读动作"的不自然）：
    1) **caption/prompt** 只放 `motive` 这一句**泛泛的长期心思**（不写"去打扫"这种具体指令，
       越具体越容易复读、越重复）。它是一种倾向，不是任务清单。
    2) **引擎侧 M_drive 乘数**（world._m_drive）做真正的具体偏置：推进动机的行为抬一档、
       损耗的压一档、禁忌几乎归零。这一步模型看不见，所以不会被"说出来"，行为自然。
    3) **签名动作**（signature）——只有这个动机才会长出的专属动作，带冷却、低频出现，
       让"五个人不一样"看得见，又不会每拍复读。

破罐破摔（DESPERATE）：老陈的假释被狱警搞黄后，动机整体转向——禁忌解除、
    反而倾向违禁/越狱。这把"背景设定"变成了玩家可操作、可涌现的机制。

重训备注：M_drive 与 M_crowd/M_command 同类，都是 RA 公式外的引擎侧乘数，
    等监狱专属轴（含"动机"）重训进模型后，这一项应收回模型内部。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# 无所事事的兜底行为：谁都能干、不推进任何长期心思。有动机的人会略微避开它们，
# 转去做真正在乎的事——除非某人的动机本就把它们收进了 boost。
AIMLESS = frozenset({"Rest", "Patrol", "DeclineBoredom"})


@dataclass(frozen=True)
class Drive:
    key: str
    motive: str                 # 泛泛的长期心思（进 caption/prompt，不写具体动作）
    stake: str                  # 怕失去什么（供对话）
    signature: str              # 签名动作 goal（只有这个动机会长出，低频+冷却）
    advance: dict[str, int] = field(default_factory=dict)   # 完成后推进动机进度的行为
    boost: frozenset[str] = frozenset()   # M_drive 抬一档的行为
    damp: frozenset[str] = frozenset()    # M_drive 压一档的行为
    taboo: frozenset[str] = frozenset()   # 几乎不做的行为（m_drive≈0）

    def m_drive(self, goal: str) -> float:
        """动机对某行为的乘数：签名 > 推进/加成 > 中性 > 无所事事 > 压制 > 禁忌。
        生存刚需（Eat/Drink/Sleep/Cure）从不进任何集合，始终 1.0——动机不跟命抢。
        幅度按用户要求"偏前者"（动机主导中间宽带）标定，但仍压不过告急的生存分。"""
        if goal in self.taboo:
            return 0.03
        if goal == self.signature:
            return 3.0
        if goal in self.advance or goal in self.boost:
            return 2.2
        if goal in self.damp:
            return 0.5
        # 有长期心思的人不会一直无所事事地闲逛/歇着 —— 除非闲逛正是它动机的一部分。
        # 这一档也直接治"五个人都在解闷/歇着/放风"的重复感（用户要"减少重复"）。
        if goal in AIMLESS:
            return 0.7
        return 1.0

    def drive_goals(self) -> set[str]:
        """该动机希望"上桌"的专属候选：推进项 + 签名动作（供 world 每拍注入）。"""
        goals = set(self.advance)
        if self.signature:
            goals.add(self.signature)
        return goals


# ---------------------------------------------------------------------------
# 五条动机。key 与 personas.PersonaSeed.drive 对齐。
# ---------------------------------------------------------------------------
DRIVES: dict[str, Drive] = {
    # 老陈：快假释，处处避险，最见不得沾事——一旦被抓翻车最惨（→DESPERATE）
    "parole": Drive(
        key="parole",
        motive="还有俩月假释，只想平平安安熬到出去",
        stake="快到手的假释资格",
        signature="CurryFavor",
        advance={"CurryFavor": 9, "Labor": 5, "KitchenDuty": 4, "Clean": 4,
                 "Laundry": 4, "Study": 3},
        boost=frozenset({"CurryFavor", "Labor", "KitchenDuty", "Clean",
                         "Laundry", "Study", "LayLow"}),
        damp=frozenset({"Gamble", "Smoke", "Barter", "Argue"}),
        taboo=frozenset({"Fight", "Tattoo", "Tunnel", "PrisonBreak"}),
    ),
    # 石头：想吞黑市、压住对家，靠立威/换货/动手扩张，最鄙视告密
    "market": Drive(
        key="market",
        motive="想在牢里立住脚，把地盘和门路攥在自己手里",
        stake="在牢里的地盘和面子",
        signature="Extort",
        advance={"Extort": 9, "Barter": 5, "Fight": 4, "Gamble": 3},
        boost=frozenset({"Extort", "Barter", "Fight", "Gamble", "Workout", "Smoke"}),
        damp=frozenset({"Study", "Read", "Pray", "LayLow"}),
        taboo=frozenset({"Snitch"}),
    ),
    # 老K：外面欠着赌债，得攒钱——卖消息/赌/换货来钱，什么都能谈
    "debt": Drive(
        key="debt",
        motive="外面欠着一屁股赌债，得想法子攒钱还上",
        stake="外面逼得紧、还不上就要命的赌债",
        signature="SellInfo",
        advance={"SellInfo": 8, "Gamble": 5, "Barter": 5},
        boost=frozenset({"SellInfo", "Gamble", "Barter", "Socialize", "Phone"}),
        damp=frozenset({"Study", "Workout"}),
        taboo=frozenset(),
    ),
    # 眼镜：一心翻案证清白，泡在书/卷宗/探视里查证据，讨厌动粗
    "clear": Drive(
        key="clear",
        motive="一心想翻自己那桩案子，证明自己是清白的",
        stake="洗清冤屈、还自己一个清白的机会",
        signature="ReviewFiles",
        advance={"ReviewFiles": 9, "Read": 5, "Study": 4, "Visit": 3},
        boost=frozenset({"ReviewFiles", "Read", "Study", "Visit", "Phone"}),
        damp=frozenset({"Gamble", "Smoke", "Barter"}),
        taboo=frozenset({"Fight", "Tattoo"}),
    ),
    # 黑医柳叶刀：手里攥着药和病历，谁的把柄都留一手，稳坐钓鱼台——靠信息/交换拿捏人
    "control": Drive(
        key="control",
        motive="攥着药品和病历，谁的把柄都留一手，稳坐钓鱼台",
        stake="手里那些能拿捏人的把柄和门路",
        signature="SellInfo",
        advance={"SellInfo": 8, "ReviewFiles": 5, "Barter": 4},
        boost=frozenset({"SellInfo", "ReviewFiles", "Barter", "Read"}),
        damp=frozenset({"Fight", "Workout", "Gamble"}),
        taboo=frozenset({"Snitch"}),
    ),
    # 小豆：人生地不熟，想找靠山活下去——贴近强者/望风/社交，怕事不敢招惹
    "safety": Drive(
        key="safety",
        motive="人生地不熟，就想找个靠山、别挨欺负地活下去",
        stake="有没有人罩着、能不能别挨欺负",
        signature="SeekProtector",
        advance={"SeekProtector": 8, "Socialize": 5, "Watch": 4, "Pray": 2},
        boost=frozenset({"SeekProtector", "Socialize", "Watch", "Pray"}),
        damp=frozenset({"Fight", "Argue", "Gamble"}),
        taboo=frozenset({"Tunnel", "PrisonBreak"}),
    ),
}

# 破罐破摔：假释被狱警搞黄后套在原动机之上。禁忌全解，反而倾向违禁/越狱。
DESPERATE = Drive(
    key="desperate",
    motive="假释黄了，破罐子破摔——横竖这样了，还怕个啥",
    stake="已经没什么可失去的了",
    signature="",
    advance={"Barter": 5, "Fight": 4, "Tunnel": 6, "Gamble": 4},
    boost=frozenset({"Fight", "Barter", "Tunnel", "PrisonBreak", "Gamble", "Smoke", "Argue"}),
    damp=frozenset({"CurryFavor", "LayLow", "Labor"}),
    taboo=frozenset(),
)


def get_drive(key: str, parole_lost: bool = False) -> Drive:
    """取生效动机：假释崩塌后一律走破罐破摔。"""
    if parole_lost:
        return DESPERATE
    return DRIVES.get(key, DRIVES["safety"])
