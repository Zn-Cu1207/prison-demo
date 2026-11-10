"""行为全集（NPC 能做的每一件事）——全项目唯一真相源。

为什么单独立这个文件：
    "一个 NPC 能做哪些事" 原来散落在三处——context.py 的 DESC_MAP/GOAL_LABEL、
    world.py 的 GOAL_TO_KIND/ACT_RELIEF、以及 _gen_candidates 里的触发条件。
    改一个漏一个就会出现"标签有、设施没有"之类的静默错位。
    这里把每个行动的全部元信息收成一张表，其余模块一律从这里派生。

同时它就是**微调数据采集的行动清单**：
    - `goal` 是喂给模型的英文行动名（训练标签，改名等于换了个模型不认识的动作，勿轻改）。
    - `desc` 是候选文本里的监狱语义描述，会随训练数据一起固化。
    - `seen` 标注现有 RA 模型是否给这个动作打过分：
        "yes"  = 训练见过，分可信；
        "near" = 借用近义旧动作，将信将疑；
        "new"  = 全新监狱动作，现有模型只能瞎猜 —— 这些正是重训最该覆盖的。
    采数据时按 `seen` 分层，能把标注预算优先砸在 new/near 上。

字段：
    goal      英文行动名（训练标签）
    label     中文短标签（前端显示）
    desc      监狱语义描述（喂模型的候选文本）
    kind      落地设施 kind（对应 world.Facility.kind；模型不看，引擎用来寻路）
    source    主要来源类别 need/threat/command/worktask/event/daily
    relieves  到达后的需求增量（负=缓解）。仅"丰富活动"走这条；生存/放风类见 effect
    effect    人类可读的效果说明（含 kind 分支处理的生存/放风类）
    trigger   候选何时生成（对 _gen_candidates 条件的自然语言归纳）
    seen      yes/near/new —— 现有模型的熟悉度，决定重训/采数优先级
    category  归类，仅用于文档分组
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    goal: str
    label: str
    desc: str
    kind: str
    source: str
    effect: str
    trigger: str
    seen: str
    category: str
    relieves: dict[str, int] = field(default_factory=dict)
    illicit: bool = False   # 见不得光：狱警/狱犬在视线内时会被压制（m_watch≈0）
    prep: str = ""          # 越狱链的准备步：完成后往 escape_prep 记这个 key


# ---------------------------------------------------------------------------
# 行动全集。新增一个 NPC 能做的事 = 在这里加一行，其余模块自动生效。
# ---------------------------------------------------------------------------
ACTIONS: list[Action] = [
    # —— 生存刚需 ——
    # 这四个走 world_objects 的分阶段执行：前往→取用真实物资→执行 N 帧→结算。
    # 取不到物资（饭吃光/龙头占满/铺位满/药柜空）就是**真的做不成**，需求不缓解。
    # relieves 留空是因为结算表在 world_objects.PERFORM_EFFECT，不走 ACT_RELIEF。
    Action("Eat", "吃饭", "去食堂吃饭", "eat", "need",
           "取一份窝头吃掉(6帧)：饥饿-70；食堂没饭则失败", "饥饿≥30", "yes", "生存"),
    Action("Drink", "喝水", "去打水喝", "drink", "need",
           "占一个龙头喝(3帧)：口渴-70；龙头占满则失败", "口渴≥30", "yes", "生存"),
    Action("Sleep", "睡觉", "回牢房睡觉", "sleep", "need",
           "占一张铺位睡(24帧)：疲劳-80、压力-10；铺位满则失败", "疲劳≥50", "yes", "生存"),
    Action("Cure", "治伤", "处理伤口", "cure", "need",
           "用一份医疗包(8帧)：受伤-70、压力-6；药柜空则失败", "受伤≥30", "yes", "生存"),

    # —— 情绪 / 社交 ——
    Action("DeclineBoredom", "解闷", "找点乐子解闷", "yard", "need",
           "无聊-40、社交-30（放风场）", "无聊≥12", "new", "情绪社交"),
    Action("Socialize", "社交", "找狱友唠嗑", "yard", "need",
           "无聊-40、社交-30（放风场）", "社交欲≥30", "yes", "情绪社交"),
    Action("Phone", "打电话", "去电话亭给家里打电话", "phone", "need",
           "社交-26、压力-8", "社交欲≥25 且 放风/自由活动", "new", "情绪社交",
           {"social": -26, "stress": -8}),
    Action("Visit", "探视", "去探视室会见家属", "visit", "daily",
           "社交-28、压力-10", "自由活动菜单", "new", "情绪社交",
           {"social": -28, "stress": -10}),

    # —— 威胁应对 ——
    # 打架不再是单人动作：开一场多人 bout（见 combat.py），逐帧互相造成 injury，
    # 旁人可能加入，输家被推去医务室，被狱警撞见则全体禁闭+加刑。
    Action("Fight", "打架", "用拳头教训对方", "yard", "threat",
           "开一场群殴：双方逐帧受伤、输家进医务室、被狱警看到则禁闭加刑",
           "视线内有具体仇人(仇怨≥35) 且 放风/自由/训练", "yes", "威胁应对",
           illicit=True),
    Action("Argue", "吵架", "跟对方吵两句", "yard", "threat",
           "压力-10、社交-6", "仇人在场（与打架并列，升级/降级二选一）", "new", "威胁应对",
           {"stress": -10, "social": -6}),
    Action("LayLow", "收敛", "狱警/狱犬盯着，原地老实待着别出格", "", "threat",
           "原地规矩、暂停手头的事、压力+4", "视线内有狱警或狱犬", "new", "威胁应对",
           {"stress": 4}),

    # —— 日常作息 ——
    Action("Patrol", "巡逻", "在放风场溜达", "yard", "daily",
           "无聊-40、社交-30（放风场）", "兜底：去放风场（El Farol 的『去』）", "yes", "日常"),
    Action("Rest", "歇着", "回牢房待着躲清静", "sleep", "daily",
           "无聊-12、压力-8（不睡）", "兜底：回牢房（El Farol 的『不去』）", "near", "日常"),
    Action("Study", "学习", "去上课学习", "class", "daily",
           "无聊-30（教室）", "上课时段", "new", "日常"),
    Action("Labor", "苦役", "去干苦役", "labor", "worktask",
           "劳动（无直接需求缓解）", "被派苦役 且 训练/自由活动", "yes", "劳动"),
    Action("KitchenDuty", "帮厨", "去后厨帮厨打下手", "eat", "worktask",
           "饥饿-22、无聊-6、疲劳+4", "被派苦役 且 训练/自由活动", "new", "劳动",
           {"hunger": -22, "boredom": -6, "fatigue": 4}),
    Action("Laundry", "洗衣", "去洗衣房搓洗囚服", "laundry", "daily",
           "无聊-6、疲劳+4", "训练时段菜单", "new", "劳动",
           {"boredom": -6, "fatigue": 4}),
    Action("Clean", "打扫", "打扫牢区卫生", "labor", "daily",
           "无聊-8", "训练时段菜单", "new", "劳动",
           {"boredom": -8}),

    # —— 消遣（放风/自由活动菜单，轮换采样）——
    Action("Workout", "健身", "去健身区举铁锻炼", "gym", "daily",
           "压力-15、无聊-12", "训练/放风/自由活动菜单", "new", "消遣",
           {"stress": -15, "boredom": -12}),
    Action("Shower", "洗澡", "去澡堂洗澡", "shower", "daily",
           "压力-12、疲劳-8", "自由活动菜单", "new", "消遣",
           {"stress": -12, "fatigue": -8}),
    Action("Read", "看书", "去图书室看书", "library", "daily",
           "无聊-16", "上课/自由活动菜单", "new", "消遣",
           {"boredom": -16}),
    Action("Gamble", "赌钱", "蹲角落聚众赌两把", "yard", "daily",
           "无聊-18、社交-10", "放风/自由活动菜单", "new", "消遣",
           {"boredom": -18, "social": -10}, illicit=True),
    Action("Smoke", "抽烟", "靠墙抽根烟", "yard", "daily",
           "压力-16", "放风/自由活动菜单", "new", "消遣",
           {"stress": -16}),
    Action("Chess", "下棋", "去娱乐室下盘棋", "commons", "daily",
           "无聊-20、社交-12", "上课/自由活动菜单", "new", "消遣",
           {"boredom": -20, "social": -12}),
    Action("Music", "唱歌", "在娱乐室弹唱两句", "commons", "daily",
           "无聊-14、压力-12、社交-8", "放风/自由活动菜单", "new", "消遣",
           {"boredom": -14, "stress": -12, "social": -8}),
    Action("Pray", "祷告", "在铺上祷告静心", "sleep", "daily",
           "压力-16", "自由活动/就寝菜单", "new", "消遣",
           {"stress": -16}),

    # —— 门路 / 人情 / 风险 ——
    Action("Barter", "换货", "去黑市角落换点东西", "market", "daily",
           "无聊-14、压力-6；顺带换到越狱工具", "放风/自由活动菜单", "new", "门路风险",
           {"boredom": -14, "stress": -6}, illicit=True, prep="tool"),
    Action("Tattoo", "纹身", "找人纹个身", "commons", "daily",
           "无聊-12、压力-8、受伤+4", "自由活动/就寝菜单", "new", "门路风险",
           {"boredom": -12, "stress": -8, "injury": 4}, illicit=True),
    Action("Watch", "望风", "替人望风放哨", "yard", "daily",
           "无聊-6、压力+6（紧张）", "监视度低 且 放风/自由活动", "new", "门路风险",
           {"boredom": -6, "stress": 6}),
    # 翻找线索：给 NPC 与玩家同等的"查案/搜证"能力——四处翻查、搜寻可用的线索或把柄。
    Action("Investigate", "翻找线索", "四处翻查、搜寻可用的线索或把柄", "library", "daily",
           "无聊-12、压力+4（翻找费神）", "查案/攥把柄的人会做的事", "new", "门路风险",
           {"boredom": -12, "stress": 4}),
    # 弄凶器：去黑市找门路弄一件能伤人的家伙（刀/电击枪），为动手做准备。
    Action("GetWeapon", "弄凶器", "去黑市弄一件能伤人的家伙", "market", "daily",
           "弄到凶器（刀/电击枪）", "有仇杀目标却手无寸铁时", "new", "门路风险",
           {"stress": 4}, illicit=True),
    # 下杀手：持凶器、仇杀目标就在近旁、且没被狱警/狱犬盯着时——对目标下杀手（可能致命）。
    Action("Assassinate", "下杀手", "趁人不备对仇杀目标下杀手", "", "threat",
           "对目标下杀手（需持凶器+目标在侧+未被盯）", "持凶器且仇杀目标在近旁且未被盯", "new", "威胁应对",
           illicit=True),
    Action("Snitch", "告密", "找管教打小报告", "guardpost", "event",
           "压力-8", "有情报(斗殴/尸体/仇人) 且 监视度非低", "new", "门路风险",
           {"stress": -8}),
    Action("Hide", "藏东西", "把违禁品藏起来", "sleep", "event",
           "压力-10", "触发『狱警搜牢』事件", "new", "门路风险",
           {"stress": -10}, illicit=True),
    Action("Tunnel", "撬门", "在大门撬锁挖出路", "gate", "daily",
           "推进越狱准备（需无人盯防）", "夜晚+在大门+已有工具+未被盯上", "new", "越狱",
           illicit=True, prep="breach"),
    Action("PrisonBreak", "越狱", "翻出州立监狱大门", "gate", "daily",
           "逃出监狱（准备齐 + 夜晚 + 未被盯）", "工具+挖通齐备 且 就寝 且 未被盯上", "new", "越狱",
           illicit=True),

    # —— 宏观动机专属动作（各人长期心思长出的签名动作，带冷却、低频）——
    Action("CurryFavor", "刷表现", "找管教表现、抢着干活攒表现分", "guardpost", "daily",
           "推进假释、压力+4、无聊-6", "快假释者的签名动作（避开生存紧急时）", "new", "动机",
           {"stress": 4, "boredom": -6}),
    Action("Extort", "立威", "在黑市角落立威、收保护费", "market", "daily",
           "推进地盘、压力-10、无聊-8、社交-8", "想抢地盘者的签名动作", "new", "动机",
           {"stress": -10, "boredom": -8, "social": -8}, illicit=True),
    Action("SellInfo", "卖消息", "把打听来的消息兜售出去换钱", "market", "daily",
           "推进还债、社交-10、无聊-8", "缺钱者的签名动作", "new", "动机",
           {"social": -10, "boredom": -8}),
    Action("ReviewFiles", "核卷宗", "去图书室核对卷宗、找破绽", "library", "daily",
           "推进翻案、无聊-14、社交+4", "想翻案者的签名动作", "new", "动机",
           {"boredom": -14, "social": 4}),
    Action("SeekProtector", "贴靠山", "凑到强者身边、求人罩着", "commons", "daily",
           "推进求安全、社交-20、压力-8", "怕事新人的签名动作", "new", "动机",
           {"social": -20, "stress": -8}),

    # —— 尸体（M3 关系/命案后开放）——
    Action("Bury", "埋葬", "埋葬尸体", "yard", "event",
           "处理尸体", "有尸体", "yes", "命案"),
    Action("Mourn", "哀悼", "哀悼死者", "yard", "event",
           "哀悼", "有尸体", "yes", "命案"),

    # —— 定向社交博弈（v3/P2：冲着"近旁那个具体的人"发起的一步棋）——
    # 这些动作的对象由引擎在近旁按关系挑定（同 Fight 挑 fight_target），caption 的
    # 邻近/对方意图/社交近况/人情 四轴给模型判断"此刻该冲谁走哪步"的依据。
    # 它们与 P1 的对白写回同源：选中即对目标施加相应关系效果（见 world 社交动作执行）。
    Action("Propose", "提议", "凑近某人提一笔交易或条件", "", "daily",
           "向近旁的人提议交易/条件（谈成则按内容转移物资、记人情）",
           "近旁有可打交道的人(生人/有往来)", "new", "社交博弈"),
    Action("Ally", "结盟", "跟近旁的人拜把子、结成攻守同盟", "", "daily",
           "与近旁的人结盟：双方 ally↑、从此互不动手、危难相帮",
           "近旁有可结盟的人(非死仇)", "new", "社交博弈"),
    Action("Protect", "护着", "把近旁的弱者收到自己羽翼下", "", "daily",
           "罩着近旁的人：对方靠山在场、ally↑、你多一份人情债权",
           "近旁有示弱/求护的人", "new", "社交博弈"),
    Action("Repay", "还人情", "找债主把欠下的人情或债了结", "", "daily",
           "偿还一笔欠债：debt 清零、对方 trust↑",
           "身上欠着人情且债主在近旁", "new", "社交博弈"),
    # 威胁/背叛是攻击性社交，狱警/狱犬在场或高监视度下会被压制（illicit）。
    Action("Threaten", "放狠话", "当面威胁、拿捏近旁的人", "", "daily",
           "威胁近旁的人：对方 fear↑、grudge↑（可能逼其就范）",
           "近旁有想拿捏的人且未被狱警盯", "new", "社交博弈", illicit=True),
    Action("Betray", "背叛", "出卖近旁的盟友或欠自己的人", "", "daily",
           "背叛：ally 清零、grudge↑、可能趁机夺物；风险高",
           "近旁有盟友/欠我的人且有利可图", "new", "社交博弈", illicit=True),
]

# 别名/校验用索引
BY_GOAL: dict[str, Action] = {a.goal: a for a in ACTIONS}
assert len(BY_GOAL) == len(ACTIONS), "actions.py 有重复 goal"

# ---------------------------------------------------------------------------
# 其余模块从这里派生（不要再手抄一份）
# ---------------------------------------------------------------------------
DESC_MAP: dict[str, str] = {a.goal: a.desc for a in ACTIONS}
GOAL_LABEL: dict[str, str] = {a.goal: a.label for a in ACTIONS}
GOAL_TO_KIND: dict[str, str] = {a.goal: a.kind for a in ACTIONS}
# ACT_RELIEF 只收"丰富活动"这条路（生存/放风类走 world._drift_needs 的 kind 分支）
ACT_RELIEF: dict[str, dict[str, int]] = {a.goal: dict(a.relieves) for a in ACTIONS if a.relieves}

# 采数分层：现有模型没见过的动作，重训/标注优先覆盖
GOALS_NEW: list[str] = [a.goal for a in ACTIONS if a.seen == "new"]
GOALS_SEEN: list[str] = [a.goal for a in ACTIONS if a.seen == "yes"]

# 见不得光的动作：狱警/狱犬在视线内会被压制
ILLICIT_GOALS: set[str] = {a.goal for a in ACTIONS if a.illicit}
# 越狱链准备步：goal -> escape_prep 的 key（完成后记账）
PREP_BY_GOAL: dict[str, str] = {a.goal: a.prep for a in ACTIONS if a.prep}


def catalog_rows() -> list[dict]:
    """导出为纯 dict 行，供 dump_actions.py 出 JSON / 文档。"""
    return [
        {
            "goal": a.goal, "label": a.label, "desc": a.desc, "kind": a.kind,
            "source": a.source, "effect": a.effect, "trigger": a.trigger,
            "seen": a.seen, "category": a.category, "relieves": dict(a.relieves),
            "illicit": a.illicit, "prep": a.prep,
        }
        for a in ACTIONS
    ]
