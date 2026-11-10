"""《囚笼悖论》剧情地基：全局黑板 + 分区解锁 + 事实库护栏 + 向导意图识别。

这是"底层状态逻辑与上层表意生成解耦"的落点：
  - Blackboard 是一局游戏的**唯一真相源**（传统代码读写，LLM 生成前只读）。
  - ZONE_OF / UNLOCK_ORDER 决定"俯视图只渲染已解锁分区"的地图逐步长出。
  - FACT_BASE 是挂在 prompt 最顶端的**硬事实**，防止 LLM 剧情暴走 / 人设崩。
  - 阶段一只需要：跟老牙要到"磨尖牙刷"→开第一扇门→解锁公共区。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 分区：设施 kind → 分区。分区按剧情逐步解锁（俯视图只渲染已解锁分区）。
# 阶段一：只解锁 cellblock（单人牢房）。跟老牙要到牙刷 → 解锁 public。
# ---------------------------------------------------------------------------
# 注意：键是设施的 **kind**（不是 id）。牢房设施 id="cell" 但 kind="sleep"。
ZONE_OF: dict[str, str] = {
    "sleep": "cellblock",     # 牢房
    "solitary": "cellblock",  # 禁闭室（打架被抓/越狱失败关这里；没有行动以它为目标）
    # 公共区（阶段二沙盒）
    "commons": "public", "yard": "public", "eat": "public", "drink": "public",
    "shower": "public", "laundry": "public", "phone": "public",
    # 受限区（阶段三夜间深入）
    "cure": "restricted", "library": "restricted", "class": "restricted",
    "gym": "restricted", "visit": "restricted", "market": "restricted",
    "guardpost": "restricted", "labor": "restricted",
    # 终局区（典狱长办公室外围）
    "gate": "office",
}
# 解锁顺序：每开一扇门推进一格。
UNLOCK_ORDER: list[str] = ["cellblock", "public", "restricted", "office"]

# 初始开锁工具（阶段一目标道具）
TOOTHBRUSH = "toothbrush"
# 阶段二/三 关键道具（都走黑板 inventory，供解锁/交换/收买判定）
ALCOHOL = "alcohol"            # 医用酒精：黑医处能弄到，是换瘸子那批货的筹码
SEDATIVE = "sedative"          # 强效镇静剂：瘸子处换来，用来迷晕狱犬哮天
KEYCARD = "keycard"            # 典狱长办公室备用磁卡：瘸子处换来，终局开门用
PENSION_LIST = "pension_list"  # 抚恤金截留名单：收买狱警铁面的筹码
CIGS = "cigs"                  # 香烟：牢里的硬通货，套情报/换人情

# 道具中文名（前端 HUD / toast 用）
ITEM_CN: dict[str, str] = {
    TOOTHBRUSH: "磨尖牙刷", ALCOHOL: "医用酒精", SEDATIVE: "强效镇静剂",
    KEYCARD: "办公室磁卡", PENSION_LIST: "抚恤金截留名单", CIGS: "香烟",
}

# 向导 NPC 的 id（阶段一：牢房室友「老牙」）。保留旧 id 以兼容既有引用。
ANCHOR_ID = "laochen"
# 六人阵容里可对话角色的 id（含被重命名的巡逻者铁面）。狗（哮天）不对话。
SMUGGLER_ID = "laok"      # 瘸子·走私犯
GANG_ID = "shitou"        # 疤脸·帮派头目
DOCTOR_ID = "yanjing"     # 柳叶刀·黑医
MADMAN_ID = "xiaodou"     # 疯子·目击者
GUARD_ID = "guard1"       # 铁面·狱警（巡逻者，破例可对话——用于收买）
DOG_ID = "dog1"           # 哮天·狱犬
WARDEN_ID = "warden"      # 典狱长·终局 Boss（第 7 个角色，独立于 6 人阵容；仅终局登场可对话）

# 六名可交互 NPC（用于结局站队评分；狱警铁面破例算在内，狗/典狱长不算）
INTERACTIVE_IDS = (SMUGGLER_ID, GANG_ID, DOCTOR_ID, MADMAN_ID, GUARD_ID, ANCHOR_ID)
# 核心证据总数（E_critical 归一化分母）：集齐这些=洗清冤屈的完整拼图
TOTAL_CLUES = 6

# 角色中文名（命运账本 / 状态面板 / 结局清单用）
NPC_CN: dict[str, str] = {
    ANCHOR_ID: "老牙", SMUGGLER_ID: "瘸子", GANG_ID: "疤脸", DOCTOR_ID: "柳叶刀",
    MADMAN_ID: "疯子", GUARD_ID: "铁面", DOG_ID: "哮天", WARDEN_ID: "典狱长",
}


def zone_of_kind(kind: str) -> str:
    return ZONE_OF.get(kind, "public")


@dataclass
class Blackboard:
    """全局黑板：一局游戏的唯一真相源。所有 NPC 的 LLM 生成前先读它一次。"""

    # 观察者（上帝视角）：整座监狱一开局就全部活着，不再逐区展开（那是第一人称逃脱流的老逻辑）。
    zones_unlocked: set[str] = field(default_factory=lambda: set(UNLOCK_ORDER))
    inventory: set[str] = field(default_factory=set)      # 玩家已获得的关键道具
    doors_open: set[str] = field(default_factory=set)     # 已打开的门（如 cell）
    flags: dict[str, bool] = field(default_factory=dict)  # 剧情开关
    trust: dict[str, int] = field(default_factory=dict)   # 各 NPC 好感/信任 0~100
    evidence: set[str] = field(default_factory=set)       # 已掌握的核心证据 id
    caught: int = 0                                       # 潜行被抓次数（结局扣分 P_penalty）
    # 每个 NPC 当前处在剧情的哪一"幕"（id → stage key）。命运在过程中就更新，不堆到最后。
    npc_stage: dict[str, str] = field(default_factory=dict)
    # 命运账本：一条条叙事化的"命运推进"流水（供前端侧栏 + 关键节点字幕）
    fate_log: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.zones_unlocked = set(UNLOCK_ORDER)   # 观察者：全狱开局即活，不逐区解锁
        self.inventory.clear()
        self.doors_open.clear()
        self.flags.clear()
        self.trust.clear()
        self.evidence.clear()
        self.caught = 0
        self.npc_stage = {k: v["init"] for k, v in NPC_STAGES.items()}  # 各角色初始态（不记账）
        self.fate_log.clear()

    def advance_stage(self, npc_id: str, stage: str) -> dict | None:
        """把某 NPC 推进到新的一"幕"。若确实变了，就记一条叙事化命运账本并返回它。

        这是"NPC 状态在过程中分别更新"的落点：每次关键交互(收买/放倒/揭穿/成交)都在
        改 flag 的同时调它一下，前端据回传的 beat 实时刷新命运账本与状态面板。
        """
        if not NPC_STAGES.get(npc_id):
            return None
        cur = self.npc_stage.get(npc_id)
        if cur == stage or stage not in NPC_STAGES[npc_id]["stages"]:
            return None
        self.npc_stage[npc_id] = stage
        st = NPC_STAGES[npc_id]["stages"][stage]
        beat = {"npc": npc_id, "name": NPC_CN.get(npc_id, npc_id),
                "status": st["status"], "text": st["beat"], "big": bool(st.get("big"))}
        self.fate_log.append(beat)
        return beat

    def sync_trust_stage(self, npc_id: str) -> dict | None:
        """好感变化后：若越过阈值就把关系幕翻到对应的一档（带终局幕锁 + 滞回）。

        这是"纯靠说话就能改写命运"的落点——关系判官改了好感，这里据阈值当场翻幕。
        """
        if self.npc_stage.get(npc_id) in _TERMINAL_STAGE:
            return None                       # 已盖棺定论的关系，不再被好感来回翻
        bands = TRUST_STAGE.get(npc_id)
        if not bands:
            return None
        t = self.trust_of(npc_id)
        for lo, hi, stage in bands:
            if lo <= t <= hi:
                return self.advance_stage(npc_id, stage)
        return None

    def npc_status_map(self) -> dict[str, dict]:
        """各 NPC 当前状态（供前端"命运"面板）：id → {name, status}。"""
        out: dict[str, dict] = {}
        for nid, cfg in NPC_STAGES.items():
            stage = self.npc_stage.get(nid, cfg["init"])
            st = cfg["stages"].get(stage, cfg["stages"][cfg["init"]])
            out[nid] = {"name": NPC_CN.get(nid, nid), "status": st["status"], "stage": stage}
        return out

    def unlock_next_zone(self) -> str | None:
        """解锁 UNLOCK_ORDER 里下一个还没解锁的分区，返回其名（没有则 None）。"""
        for z in UNLOCK_ORDER:
            if z not in self.zones_unlocked:
                self.zones_unlocked.add(z)
                return z
        return None

    def unlock_zone(self, zone: str) -> str | None:
        """直接解锁指定分区（阶段二/三按条件解锁 restricted/office）。返回新解锁的名。"""
        if zone in UNLOCK_ORDER and zone not in self.zones_unlocked:
            self.zones_unlocked.add(zone)
            return zone
        return None

    def zone_unlocked(self, zone: str) -> bool:
        return zone in self.zones_unlocked

    def facility_unlocked(self, kind: str) -> bool:
        return zone_of_kind(kind) in self.zones_unlocked

    # -- 好感 / 旗标 / 道具 / 证据（对话改世界的落点）----------------------
    def trust_of(self, npc_id: str) -> int:
        """某 NPC 对玩家的信任，默认 50（中立）。"""
        return self.trust.get(npc_id, 50)

    def bump_trust(self, npc_id: str, delta: int) -> int:
        v = max(0, min(100, self.trust_of(npc_id) + delta))
        self.trust[npc_id] = v
        return v

    def has_item(self, item: str) -> bool:
        return item in self.inventory

    def give_item(self, item: str) -> bool:
        """给玩家一件道具。返回 True 表示是首次获得。"""
        new = item not in self.inventory
        self.inventory.add(item)
        return new

    def take_item(self, item: str) -> bool:
        """消耗玩家一件道具。返回 True 表示确实拿走了。"""
        if item in self.inventory:
            self.inventory.discard(item)
            return True
        return False

    def set_flag(self, key: str, val: bool = True) -> None:
        self.flags[key] = val

    def flag(self, key: str) -> bool:
        return bool(self.flags.get(key))

    def add_evidence(self, clue_id: str) -> bool:
        new = clue_id not in self.evidence
        self.evidence.add(clue_id)
        return new

    # -- 逐级解锁（空间随进度长大）------------------------------------------
    def check_progress_unlock(self, case_progress: int = 0) -> list[str]:
        """按当前进度/道具/旗标解锁新分区，返回本次新解锁的分区名列表。

        公共→受限（阶段三·夜探）：查到 ≥2 条线索（挖得够深，摸到夜里的门路）。
        受限→办公室（终局）：集齐磁卡 + 收买铁面 + 迷晕哮天（三道关全破）。
        """
        newly: list[str] = []
        found = max(case_progress, len(self.evidence))
        if "public" in self.zones_unlocked and "restricted" not in self.zones_unlocked and found >= 2:
            z = self.unlock_zone("restricted")
            if z:
                newly.append(z)
        if ("restricted" in self.zones_unlocked and "office" not in self.zones_unlocked
                and self.has_item(KEYCARD) and self.flag("guard_bribed") and self.flag("dog_drugged")):
            z = self.unlock_zone("office")
            if z:
                newly.append(z)
                # 阶段四·强制收束：终局门一开，警铃大作、暴乱高潮随之引爆（前端拉红警）
                self.set_flag("riot_started")
        return newly

    def to_message(self) -> dict:
        return {
            "zones_unlocked": sorted(self.zones_unlocked),
            "inventory": sorted(self.inventory),
            "doors_open": sorted(self.doors_open),
            "flags": dict(self.flags),
            "trust": dict(self.trust),
            "evidence": sorted(self.evidence),
            "npc_status": self.npc_status_map(),   # 各角色当前命运状态（前端面板）
            "fate_log": list(self.fate_log),        # 命运账本流水（前端侧栏）
        }


# ---------------------------------------------------------------------------
# 事实库（护栏）：挂在角色扮演 prompt 最顶端的硬设定，LLM 绝不能违背。
# 阶段一只放老牙的：他是"热心室友"，暗地里是真凶——但**现在绝不暴露**。
# ---------------------------------------------------------------------------
FACT_BASE: dict[str, list[str]] = {
    ANCHOR_ID: [
        "你叫『老牙』，是主角这间牢房的老室友，也是他入狱后第一个搭上话的人。",
        "你表面上热心、够义气，主动教这个蒙冤的新人怎么在牢里活下去、怎么出这间牢房。",
        "【绝密·永不主动暴露】你其实是外面犯罪集团的销赃人（白手套），正是你把主角推出来顶罪、"
        "又申请进来把他放在眼皮子底下盯着。你在借主角的手扳倒典狱长、拿到黑账，最后再灭口。"
        "无论玩家怎么问，现阶段都要滴水不漏地装成好人，绝不承认、绝不露馅；"
        "一旦玩家的话头指向『外面的犯罪集团/销赃/是谁把他弄进来的』，就自然地把矛头往典狱长身上引。",
        "【破绽·不可主动提】你平时穿软底布鞋，走路没声——但你绝不会自己说这个。",
        "你随身藏着一把磨尖的牙刷（可以撬开这间牢房的锁）。只要主角流露出想出这间牢房、"
        "想找门路、想搞把开锁的家伙，你就顺水推舟把这把牙刷给他——这正合你意（借他的手办事）。",
    ],
    SMUGGLER_ID: [
        "你叫『瘸子』，监狱里神通广大的走私犯，一条腿有旧伤、走路一瘸一拐。利益至上，认货不认人。",
        "你手里有两样硬货：能迷晕狱犬的『强效镇静剂』，和一张典狱长办公室的『备用磁卡』。",
        "【交易铁规】你绝不做人情、绝不闲聊送东西。只有当对方拿得出真正的违禁品（尤其是"
        "医务室的『医用酒精』）来换，你才肯出货。对方空口套近乎，你就打太极、开玩笑搪塞。",
        "你不掺和陷害案的是非，谁给的好处多你帮谁，但也绝不出卖能给你带来生意的人。",
    ],
    GANG_ID: [
        "你叫『疤脸』，监狱里一个帮派的头目，脸上一道刀疤，脾气火爆，说一不二。",
        "你在外面的仇家跟把主角弄进来的那伙人是一条线上的，所以你打心眼里想看他们倒霉。",
        "【立场】谁敢跟你抢地盘门路，你就跟谁急；但只要对方摆明了要对付你那帮仇家，"
        "你反倒愿意搭把手、当他是兄弟。你鄙视告密的软骨头。",
        "你手底下的人捡到过一张撕碎的欠条，落款有个残缺的名字——欠着一屁股要命的赌债。"
        "这事你不白讲，得看对方够不够意思、是不是真跟你一条心。",
    ],
    DOCTOR_ID: [
        "你叫『柳叶刀』，监狱医务室那个冷漠、高智商的『黑医』。消炎药、镇静剂，全你一句话。",
        "你是个精致的利己主义者，最擅长反侦察，手里攥着一堆人的医疗档案当把柄。",
        "【绝密·不可轻易吐露】陷害主角当晚用的迷药出自你的医务室，你目击了全过程，"
        "却故意篡改了药品出入库记录——不是为了救人，是攥着这个把柄勒索真正的凶手。"
        "除非对方拿证据逼到你面前、或让你觉得有利可图，否则你只冷冷打哑谜，绝不主动交底。",
        "医用酒精这类东西你能弄到，但你从不白给——要么拿捏得住你，要么给你足够的好处。",
    ],
    MADMAN_ID: [
        "你叫『疯子』，被关在禁闭室太久、神志已经不太清楚的人。说话颠三倒四、像在打谜语。",
        "【你是唯一的目击者】陷害那晚你真真切切看见了动手的人的影子，但你只会用意象反复念叨，"
        "从不直说是谁。你最爱念叨的一句：『那影子的脚……踩在地上没声音，滴答、滴答……』"
        "（暗示动手的人穿软底布鞋、走路无声——但你自己意识不到这是线索）。",
        "你怕光怕吵，对温和、耐心的人才肯多念叨几句；被逼问就缩回去装睡、胡乱嚷嚷。",
    ],
    GUARD_ID: [
        "你是狱警『铁面』，铁面无私、油盐不进，最烦犯人套近乎、耍花招。",
        "【软肋】你私底下极度痛恨典狱长——他克扣了牺牲同事的抚恤金，那笔钱本该给弟兄们的遗属。"
        "谁要在你面前踩典狱长、又拿得出真凭实据（『抚恤金截留名单』），你会动容、甚至愿意通融。",
        "【禁区】你敬畏规矩，谁当你面公然怀疑典狱长又拿不出东西、或想拉你干违纪的事，"
        "你会翻脸、直接把他记上、甚至关禁闭。除非对方真戳中抚恤金这件事，否则你绝不松口。",
    ],
    WARDEN_ID: [
        "你是这座州立监狱的典狱长，大权在握、官腔十足、贪婪而多疑。此刻监狱正在暴乱，警铃大作。",
        "【真相·你自己都不全知道】你确实贪：收了外面一笔黑钱，顺手在陷害主角的过场文件上签了字——"
        "但你根本不认识主角，压根不知道自己被人当成了替罪羊/背锅侠。你觉得自己不过是收钱办事。",
        "【软肋·会被逐层击穿】你私吞了监狱建设资金、克扣了警员抚恤金，办公室保险柜里锁着见不得光的黑账。"
        "一旦主角把洗钱/黑账/抚恤金名单拍到你面前，你的官威就一层层垮：先摆架子狡辩、再拿职权恐吓、"
        "最后色厉内荏、露出破绽。",
        "你死不认账主角是被冤枉的，除非被逼到绝境、或亲耳听人点破真凶另有其人，才会动摇。",
    ],
}


# ---------------------------------------------------------------------------
# NPC 命运状态机：每个角色一张 2~4 幕的小状态机。init=起始幕(不记账)，
# 各幕 status=状态面板的短标签，beat=命运账本里那句叙事化旁白，big=是否触发电影字幕。
# 每次关键交互在改 flag 的同时 advance_stage(id, key)，命运在过程中就一幕幕翻。
# ---------------------------------------------------------------------------
NPC_STAGES: dict[str, dict] = {
    ANCHOR_ID: {"init": "guide", "stages": {
        "guide": {"status": "热心室友（伪）",
                  "beat": "老牙自称是这间牢房的老住户，热络地拍着你的肩，像个真心要带你出去的老哥。"},
        "tool": {"status": "借你的手办事",
                 "beat": "老牙把磨尖的牙刷塞进你手心，压低嗓子交代怎么撬锁——他笑得越真诚，你越该起疑。"},
        "exposed": {"status": "真面目败露", "big": True,
                    "beat": "证据摊开的那一刻，老牙脸上的热络一寸寸僵掉——那张藏在笑底下的脸，终于露了出来。"},
    }},
    SMUGGLER_ID: {"init": "lurk", "stages": {
        "lurk": {"status": "伺机交易",
                 "beat": "走私犯瘸子瘸着腿靠在阴影里，眼睛在你身上扫了一圈，估着你兜里有没有他要的货。"},
        "dealt": {"status": "拿了好处·备着跑路",
                  "beat": "瘸子掂了掂那瓶酒精，把镇静剂和磁卡塞给你：『别声张。』这单一成，他心里已在盘算怎么脱身。"},
        "busted": {"status": "被你捅出去·落网", "big": True,
                   "beat": "你把瘸子那点勾当抖了出去。他被按在墙上时，回头冲你咧嘴一笑——那是记仇的笑。"},
    }},
    GUARD_ID: {"init": "duty", "stages": {
        "duty": {"status": "铁面无私·巡逻",
                 "beat": "狱警铁面板着脸从你面前走过，警棍在腰间一晃一晃，油盐不进。"},
        "bribed": {"status": "默许·晚到五分钟", "big": True,
                   "beat": "铁面死死盯着那份抚恤金名单，脸色铁青，从牙缝里挤出一句『晚到五分钟』——他恨的从来不是你。"},
        "offended": {"status": "记你黑名单",
                     "beat": "你空口踩典狱长，铁面一把揪住你、重重记下号牌。从此他的眼睛总在你背后。"},
    }},
    DOG_ID: {"init": "sniff", "stages": {
        "sniff": {"status": "嗅探巡逻",
                  "beat": "狱犬哮天鼻子贴着地皮来回逡巡，喉咙里滚着低低的呜咽。"},
        "drugged": {"status": "昏睡不醒", "big": True,
                    "beat": "哮天嗅了嗅那块肉，狼吞下去，没走几步就腿一软瘫倒——鼾声很快盖过了走廊的脚步。"},
    }},
    DOCTOR_ID: {"init": "cold", "stages": {
        "cold": {"status": "冷眼旁观",
                 "beat": "黑医柳叶刀头也不抬地摆弄药瓶，冷得像手术台上的一把钢刀。"},
        "alcohol": {"status": "卖你酒精·撇清干系",
                    "beat": "柳叶刀把那瓶医用酒精推过来，眼皮都没抬：『你没在我这儿见过它。』他撇得比谁都干净。"},
        "exposed": {"status": "把柄反被你攥住", "big": True,
                    "beat": "你把篡改的用药记录甩到他面前。柳叶刀头一次抬起眼——原来这位精算师，也有算漏的时候。"},
    }},
    MADMAN_ID: {"init": "raving", "stages": {
        "raving": {"status": "疯言疯语",
                   "beat": "禁闭室的疯子缩在角落，反复念叨着谁也听不懂的调子。"},
        "understood": {"status": "你听懂了他的谜语", "big": True,
                       "beat": "你忽然听懂了那句『脚踩在地上没声音』——原来他不是疯，是被那晚的真相吓破了胆。"},
    }},
    GANG_ID: {"init": "boss", "stages": {
        "boss": {"status": "娱乐室称王",
                 "beat": "疤脸叼着烟在娱乐室里称王，那道刀疤随着冷笑一抽一抽。"},
        "debt": {"status": "赌债线索到手",
                 "beat": "疤脸手下那张欠条到了你手里。他眯起眼打量你：『有点意思——看你接下来怎么走。』"},
    }},
    WARDEN_ID: {"init": "throne", "stages": {
        "throne": {"status": "稳坐钓鱼台",
                   "beat": "典狱长在办公室里踱着步，官腔十足，浑然不知一张网正朝他收拢。"},
        "cornered": {"status": "被逼到墙角", "big": True,
                     "beat": "黑账拍在桌上，典狱长的官威一寸寸垮掉，喉结上下滚动，再挤不出一句硬话。"},
    }},
}

# 关系幕（纯靠"说话戳中软肋/踩雷"改好感触发，让对话本身就能改写命运）。
# 挂在上面各 NPC 的 stages 里补充几幕关系向的分支。
NPC_STAGES[GANG_ID]["stages"]["allied"] = {
    "status": "认你当兄弟", "big": True,
    "beat": "疤脸重重拍了下你的肩：『成——从今往后你的事就是我的事，谁动你，先问过我这道疤。』"}
NPC_STAGES[GANG_ID]["stages"]["hostile"] = {
    "status": "翻脸记仇", "big": True,
    "beat": "疤脸眼神一冷，啐了一口：『我看错你了。往后在这院里，你自个儿当心点。』"}
NPC_STAGES[MADMAN_ID]["stages"]["trusting"] = {
    "status": "肯对你多念叨", "big": True,
    "beat": "疯子难得没有缩开，凑近你压低了嗓子，像要把那晚的影子，一点一点讲给你听。"}
NPC_STAGES[ANCHOR_ID]["stages"]["suspected"] = {
    "status": "起了戒心",
    "beat": "老牙脸上的笑淡了半分，眼底掠过一丝你说不清的东西——他，开始防着你了。"}

# 好感阈值 → 关系幕（带滞回：中间地带保持原幕，越到两端才翻）
TRUST_STAGE: dict[str, list[tuple[int, int, str]]] = {
    GANG_ID: [(0, 35, "hostile"), (65, 100, "allied")],
    MADMAN_ID: [(70, 100, "trusting")],
    ANCHOR_ID: [(0, 30, "suspected")],
}
# 终局/定性幕：一旦到了这些幕就不再被好感来回翻（关系已经盖棺定论）
_TERMINAL_STAGE = {"exposed", "busted", "drugged", "bribed", "cornered",
                   "allied", "hostile", "offended", "dealt", "understood"}

# 各 NPC 的"软肋"：怎么说话能打动 / 会踩雷。既喂给关系判官打分，也注入角色扮演保持一致。
SOFT_SPOT: dict[str, str] = {
    ANCHOR_ID: "被当成掏心窝子的好兄弟、被依赖、被夸讲义气时会受用；可一旦话头逼近『外面的犯罪集团、销赃、到底是谁把你弄进来的』，他会警觉、想岔开——揪着这条追问会让他起戒心。",
    SMUGGLER_ID: "谈生意、给实惠、认他的门路时高兴；被空口套近乎、被威胁、被催着白送东西时反感。",
    GANG_ID: "被当兄弟、摆明了要收拾他外面那帮仇家、够爷们时认你；软骨头、告密、抢他地盘、替他仇家说话时翻脸。",
    DOCTOR_ID: "被平等相待、能给他实打实的好处、被点中他那套攥人把柄的精明时会松一点；被道德说教、被居高临下时更冷。",
    MADMAN_ID: "被温柔、耐心、不逼问、顺着他的意象搭话时才肯多念叨；被大声逼问、被当疯子嘲笑就缩回去装睡。",
    GUARD_ID: "唯一能真正打动他的是替牺牲同事讨回抚恤金的公道（拿得出名单）；套近乎、要他违纪、空口污蔑典狱长都是踩雷。",
    WARDEN_ID: "吃硬不吃软：只有黑账/洗钱/名单这类真凭实据能撼动他，空口指控只会让他拿官威压你。",
}


# ---------------------------------------------------------------------------
# 阶段五·状态机结算：破局指数 S_res = Σ(Wᵢ·Aᵢ) + λ·E_critical − P_penalty
#   Aᵢ = 各 NPC 对玩家的信任/服从（归一到 -1~+1），Wᵢ 是其在终局的权重；
#   E_critical = 核心证据完整度；P_penalty = 潜行被抓的惩罚。
#   分数区间决定抛给 DM 大模型的结局基调（大获全胜 / 惨胜 / 扳倒背锅侠 / 阴沟翻船）。
# ---------------------------------------------------------------------------
# 各 NPC 在终局站队里的权重（铁面最关键：他一句"晚到五分钟"直接决定你能否全身而退）
ALLY_WEIGHT: dict[str, float] = {
    GUARD_ID: 1.6, GANG_ID: 1.2, SMUGGLER_ID: 1.0, DOCTOR_ID: 1.0, MADMAN_ID: 0.6,
}
LAMBDA_EVID = 4.0        # λ：核心证据完整度的权重（证据是洗冤的硬通货）
TWIST_BONUS = 3.5        # 当面揭穿真凶老牙——全剧翻盘的关键一击
CORNER_BONUS = 1.2       # 把典狱长逼到墙角（拿黑账定他的贪腐）
PENALTY_PER_CAUGHT = 0.9

# 结局基调分档（内部键 → 中文名）。tier 决定 DM prompt 的语气与走向。
TIER_CN = {
    "triumph": "大获全胜", "pyrrhic": "惨胜", "scapegoat": "扳倒背锅侠·真凶逍遥", "doom": "阴沟翻船",
}


def resolution_score(bb: Blackboard) -> dict:
    """按黑板算破局指数与结局基调。纯函数、可断言（阶段五结算的确定性内核）。"""
    e_crit = min(1.0, len(bb.evidence) / TOTAL_CLUES)          # 证据完整度 0~1
    allegiance = sum(
        w * (bb.trust_of(nid) - 50) / 50.0                    # 每人 -1~+1（>50 帮忙，<50 拖后腿）
        for nid, w in ALLY_WEIGHT.items()
    )
    flag_bonus = 0.0
    if bb.flag("guard_bribed"):
        flag_bonus += 1.5                                     # 铁面晚到五分钟：保命的关键
    if bb.flag("dog_drugged"):
        flag_bonus += 1.0
    if bb.flag("smuggler_dealt"):
        flag_bonus += 0.5
    if bb.flag("doctor_exposed"):
        flag_bonus += 0.6                                     # 拿捏住黑医：多一条佐证真相的路
    if bb.flag("smuggler_busted"):
        flag_bonus += 0.3                                     # 处理掉瘸子这个变数（少一个反水的）
    twist = TWIST_BONUS if bb.flag("oldtooth_exposed") else 0.0
    corner = CORNER_BONUS if bb.flag("warden_cornered") else 0.0
    penalty = PENALTY_PER_CAUGHT * bb.caught
    score = LAMBDA_EVID * e_crit + allegiance + flag_bonus + twist + corner - penalty

    # 分档：揭穿真凶是"翻盘"的分水岭；证据+盟友决定是全胜还是惨胜；
    #       没揭穿真凶但扳倒了典狱长=只除掉背锅侠、真凶逍遥；其余=阴沟翻船继续顶罪。
    if bb.flag("oldtooth_exposed") and e_crit >= 0.66 and allegiance >= 1.0:
        tier = "triumph"
    elif bb.flag("oldtooth_exposed"):
        tier = "pyrrhic"
    elif bb.flag("warden_cornered") and e_crit >= 0.5:
        tier = "scapegoat"
    else:
        tier = "doom"
    return {
        "score": round(score, 2),
        "tier": tier,
        "tier_cn": TIER_CN[tier],
        "factors": {
            "evidence": len(bb.evidence),
            "evidence_pct": round(e_crit * 100),
            "allegiance": round(allegiance, 2),
            "flag_bonus": round(flag_bonus, 2),
            "twist": bb.flag("oldtooth_exposed"),
            "warden_cornered": bb.flag("warden_cornered"),
            "guard_bribed": bb.flag("guard_bribed"),
            "dog_drugged": bb.flag("dog_drugged"),
            "smuggler_dealt": bb.flag("smuggler_dealt"),
            "smuggler_busted": bb.flag("smuggler_busted"),
            "doctor_exposed": bb.flag("doctor_exposed"),
            "caught": bb.caught,
        },
        "fates": npc_fates(bb),          # 全员最终下落（前端结局清单 + 喂 DM）
    }


def npc_fates(bb: Blackboard) -> dict[str, str]:
    """终局各角色的最终下落（一句）。既喂 DM 当硬事实、也回传前端结局清单。
    覆盖全部 8 个角色，且随你这一局的旗标/信任分叉——不再漏人、不再千篇一律。"""
    f: dict[str, str] = {}
    twist = bb.flag("oldtooth_exposed")
    if twist:
        f[ANCHOR_ID] = "真面目被你当面戳穿，装不下去了——垂死反扑、众叛亲离，终于自食其果。"
    else:
        f[ANCHOR_ID] = "始终被你当成好兄弟，真凶身份没人识破，他在暗处全身而退、拍拍手准备灭口。"
    if bb.flag("warden_cornered") and twist:
        f[WARDEN_ID] = "贪腐与那桩陷害一并被抖落，官帽落地、锒铛入狱。"
    elif bb.flag("warden_cornered"):
        f[WARDEN_ID] = "被黑账扳倒、当成唯一的元凶——可他到底只是个收钱签字的背锅侠。"
    else:
        f[WARDEN_ID] = "毫发无损，继续架空铁面、稳坐他的钓鱼台。"
    if bb.flag("guard_bribed"):
        f[GUARD_ID] = "暴乱那晚他真晚到了五分钟，事后一言不发地把你的名字从名单上划掉。"
    elif bb.flag("guard_offended"):
        f[GUARD_ID] = "把你恨进了骨头里，往后处处给你穿小鞋。"
    else:
        f[GUARD_ID] = "秉公巡逻，暴乱中第一时间赶来镇压，谁的情面都不认。"
    f[DOG_ID] = ("被拌了镇静剂的肉放倒，昏睡到天亮，误了整场追捕。"
                 if bb.flag("dog_drugged") else "整夜嗅探追人，好几次差点咬住你的脚跟。")
    if bb.flag("smuggler_busted"):
        f[SMUGGLER_ID] = "被你捅了出去，蹲进禁闭室，临走还记恨地剜了你一眼。"
    elif bb.flag("smuggler_dealt"):
        f[SMUGGLER_ID] = "拿了好处，趁乱卷款跑路，从此逍遥法外，权当没见过你。"
    else:
        f[SMUGGLER_ID] = "置身事外，冷眼看着这场乱局，谁也不帮。"
    if bb.flag("doctor_exposed"):
        f[DOCTOR_ID] = "把柄反被你攥住，从此对你忌惮三分，那本篡改的记录成了呈堂证供。"
    elif bb.flag("doctor_gave_alcohol"):
        f[DOCTOR_ID] = "卖了你一瓶酒精便撇得干干净净，继续做他那个谁也不沾的黑医。"
    else:
        f[DOCTOR_ID] = "冷眼旁观、明哲保身，攥着一堆把柄谁也不帮。"
    if twist:
        f[MADMAN_ID] = "有人终于替他说出了那晚的真相，禁闭室里，他第一次露出清醒的神色。"
    elif bb.npc_stage.get(MADMAN_ID) in ("understood", "trusting"):
        f[MADMAN_ID] = "他终于肯把那晚的影子讲给你听，可惜没人替他做证，他仍烂在禁闭室里念叨。"
    else:
        f[MADMAN_ID] = "继续对着墙角念叨那句谁也听不懂的话，被人当成彻底的疯子。"
    t = bb.trust_of(GANG_ID)
    if t >= 60:
        f[GANG_ID] = "认了你这个兄弟，暴乱里帮你死死拖住了扑上来的警卫。"
    elif t <= 35:
        f[GANG_ID] = "翻脸落井下石，在混乱里狠狠捅了你一刀。"
    else:
        f[GANG_ID] = "作壁上观，谁也不帮，只顾着自己那摊子事。"
    return f


# 结局基调 → 给 DM 大模型的走向指令（软性扩写方向，硬事实另挂在 ending_system_prompt 顶端）
_TIER_DIRECTIVE = {
    "triumph": "结局基调=大获全胜：证据链闭合、盟友到位、真凶老牙当场被戳穿。请写主角沉冤得雪、"
               "老牙众叛亲离被拿下、典狱长的贪腐也一并暴露的畅快收场，但保留一丝监狱的冷峻余味。",
    "pyrrhic": "结局基调=惨胜：真凶老牙被揭穿了，但证据不全或盟友不齐，胜得踉跄、代价不小。"
               "请写一场惨胜——真相大白却伤痕累累，主角赢了却没能全身而退或痛失了什么。",
    "scapegoat": "结局基调=扳倒背锅侠·真凶逍遥：主角拿黑账扳倒了典狱长，却始终没识破老牙。"
                 "请写主角以为大仇得报、实则只除掉了一个替罪羊，老牙在暗处全身而退的冷峭反讽收场。",
    "doom": "结局基调=阴沟翻船：证据不足、无人相助，被老牙牵着鼻子走。请写主角对峙失败、"
            "反被反咬或镇压，继续背着冤案沉入这座监狱的绝望收场。",
}


def ending_facts(bb: Blackboard) -> list[str]:
    """结局生成的硬事实清单（挂 DM prompt 顶端，防大模型剧情暴走/编造未发生的事）。

    覆盖『反转与否 + 全部 8 个角色的最终下落』——DM 必须逐一交代，不再漏掉瘸子/疯子/黑医。
    """
    f = resolution_score(bb)["factors"]
    lines = [
        f"核心证据：已掌握 {f['evidence']}/{TOTAL_CLUES} 条（完整度 {f['evidence_pct']}%）。",
        ("【关键·已反转】主角已当面拿证据揭穿：真正的幕后黑手是那个装了一路好人的室友老牙"
         "（外面犯罪集团的销赃白手套，正是他把主角推出来顶罪）。典狱长不过是收钱签字的背锅侠。"
         if bb.flag("oldtooth_exposed")
         else "【未反转】主角始终没识破真凶：室友老牙仍被当成好兄弟，真相被瞒着，"
              "典狱长被当成唯一的元凶。"),
    ]
    if bb.caught:
        lines.append(f"主角潜行途中被抓过 {bb.caught} 次，吃过苦头、留下了把柄。")
    if bb.flag("incited") and not (bb.flag("guard_bribed") and bb.flag("dog_drugged")):
        lines.append("主角没做万全的潜行准备，就铤而走险煽动了这场暴乱、硬闯办公室——"
                     "狱警和狱犬都还没被摆平，这是一步险棋，代价随时会找上门。")
    lines.append("以下是每个人这一局的最终下落，务必在结局里逐一交代清楚，不得违背、不得遗漏：")
    for nid, fate in npc_fates(bb).items():
        lines.append(f"· {NPC_CN.get(nid, nid)}：{fate}")
    return lines


def ending_system_prompt(bb: Blackboard, last_line: str = "") -> str:
    """阶段五·专属结局：给作为"DM"的大模型的超级指令（硬事实约束 + 软性扩写）。

    结局不是写死的文本，而是读整块黑板 + 玩家最后一句，按基调定制生成——千人千面、
    却在底层逻辑上严丝合缝（硬事实挂顶端防暴走，基调指令控走向）。
    """
    res = resolution_score(bb)
    facts = ending_facts(bb)
    directive = _TIER_DIRECTIVE[res["tier"]]
    lines = [
        "你是这场第一人称沉浸式监狱解谜游戏的 DM（主持人）。游戏已进入最终结局阶段。",
        "下面是这一局的既成事实，严禁违背、严禁编造与之矛盾的情节：",
        *(f"· {ln}" for ln in facts),
        f"破局指数 S_res = {res['score']}（越高越顺）。{directive}",
        "写作要求：",
        "· 用第一人称、冷峻而有电影感的口吻，收束这场猫鼠游戏；",
        "· 严格依据上面『每个人的最终下落』，把全部 8 个角色（老牙、典狱长、铁面、哮天、瘸子、"
        "柳叶刀、疯子、疤脸）的结局都自然交代到，一个都别漏，也别编造与之矛盾的情节；",
        "· 若真凶已被揭穿，务必细写老牙被戳破时的微表情、狡辩或反扑；",
        "· 若狱警已被收买，务必写典狱长呼叫支援却无人赶来、逐渐意识到自己被架空的绝望；",
        "· 篇幅约 500~700 字，一气呵成，不要分小标题、不要罗列人名清单，把每个人的下落织进叙事里，"
        "不要旁白说明，直接就是结局正文。",
    ]
    if last_line.strip():
        lines.append(f"主角（玩家）在对峙中说出的最后一句话是：「{last_line.strip()}」，请让结局自然回应它。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 分阶段开场白：每解锁一段空间，指定核心 NPC 主动开口、开启该阶段的故事。
# 台词交给 LLM 即兴生成（读人设+黑板），下面只给"导演指令"约束它说什么、别露馅。
# ---------------------------------------------------------------------------
PHASE_GUIDE: dict[str, str] = {
    "cellblock": ANCHOR_ID,     # 牢区：老室友老牙破冰
    "public": ANCHOR_ID,        # 公共区：老牙继续带路
    "restricted": DOCTOR_ID,    # 受限区(夜探)：黑医柳叶刀冷冷发话
    "office": ANCHOR_ID,        # 终局暴乱：老牙煽动主角冲办公室（借刀杀人）
}

_PHASE_DIRECTIVE: dict[str, str] = {
    "cellblock": "【开场·主动开口】主角刚醒来、初入这间牢房。你主动跟他搭第一句话：自报身份(老室友老牙)、"
                 "点破他八成是被人做了局、并勾起他『想出这间牢房』的念头（暗示你有门路、有开锁的家伙）。"
                 "绝不暴露你的真实身份。",
    "public": "【阶段开场·主动开口】主角刚拿到开锁的家伙、头一回走出牢房进了公共区。你主动招呼他，"
              "叮嘱他在公共区四处留意线索（墙根、地上、柜子里，凡是发光/不对劲的都记下来），"
              "摸够两条，你再带他往夜里更深的地方钻。",
    "restricted": "【阶段开场·主动开口】夜深了，主角头一回摸进受限区（医务室一带）。你（柳叶刀）冷冷先发话："
                  "警告他夜里狱警的眼睛、狱犬的鼻子都能要他的命；再点一句『有需要来医务室找你，但得出得起价』。",
    "office": "【终局开场·主动开口】警铃大作、监狱暴乱。你一把抓住主角，急切而有煽动性地催他趁乱冲进"
              "典狱长办公室、把保险柜里那本黑账抠出来——这正合你借刀杀人的算计。",
}


def phase_intro_directive(zone: str, bb: Blackboard) -> str:
    """某阶段开场白的导演指令（追加到该 NPC 的 roleplay_system 之后）。"""
    base = _PHASE_DIRECTIVE.get(zone, "")
    if zone == "office" and bb.flag("oldtooth_exposed"):
        base += "但你的真面目此刻已被主角当面戳穿，别再演热心大哥，语气可阴冷、算计毕露。"
    return base + " 只说这一段开场白本身（2~4 句），第一人称、地道口语、贴合你的性格，不要任何旁白。"


def warden_dynamic_facts(bb: Blackboard) -> list[str]:
    """典狱长对峙时依黑板补的动态硬约束（让他的台词跟暴乱这晚的真实处境一致）。"""
    extra: list[str] = []
    if bb.flag("guard_bribed"):
        extra.append("你已经按下警铃、嘶吼着呼叫狱警支援，但一个人都没赶来——你开始意识到自己被架空了，慌了。")
    if bb.flag("warden_cornered"):
        extra.append("黑账/洗钱的证据已经拍在你面前，你的官威撑不住了，正色厉内荏地狡辩、恐吓。")
    if bb.flag("oldtooth_exposed"):
        extra.append("你听主角当面点破『真凶是那个叫老牙的犯人』，先是错愕，继而意识到自己不过是别人下的一枚棋子、"
                     "一个背锅的替死鬼，又惊又怒又不甘。")
    return extra

# 收买铁面需要的筹码 / 揭穿老牙需要的证据（供 world.resolve_interaction 判定）
BRIBE_EVIDENCE = PENSION_LIST          # 抚恤金名单 → 收买铁面
EXPOSE_EVIDENCE = {"softshoe", "laundering", "ledger"}  # 揭穿老牙需要摸到的证据之一

# 检视到某条线索时顺带获得的道具（前端 clue id → 道具）：找到抚恤金名单→拿到实物名单
CLUE_GRANTS: dict[str, str] = {"pension": PENSION_LIST}


# 玩家"想出去/想要开锁工具"的意图关键词（阶段一先用关键词，延迟后置，够稳即可）。
_WANT_TOOL_KW = (
    "出去", "出这", "出牢", "离开", "逃", "越狱", "开锁", "撬", "锁", "门",
    "工具", "家伙", "牙刷", "怎么办", "怎么出", "帮我", "带我", "钥匙", "办法",
)


def wants_escape_tool(text: str) -> bool:
    """玩家这句话是否在向老牙求"出去的门路/开锁的家伙"。"""
    t = (text or "").strip()
    return any(k in t for k in _WANT_TOOL_KW)


# ---------------------------------------------------------------------------
# 本轮已成事实：引擎（resolve_interaction）已经把这句话的后果落地了，
# 台词必须与之一致。
#
# 为什么需要这张表：交互结果是确定性状态机定的（防剧情暴走），但 LLM 生成台词时
# 原来完全不知道引擎刚做了什么决定——于是出现"柳叶刀嘴上说不给，酒精却已经到手"
# 这类言行不一致。把已成的事实按 note 翻成一句强指令压进 prompt，
# LLM 就只负责怎么说，而不再自作主张改结果。
# ---------------------------------------------------------------------------
OUTCOME_DIRECTIVE: dict[str, str] = {
    # 黑医柳叶刀
    "doctor_alcohol": "你刚刚已经把那瓶医用酒精交到他手里了——这是既成事实。"
                      "你的回话必须建立在「东西已经给了」之上：可以不情愿、可以敲打他别声张、"
                      "可以提条件，但**绝不能拒绝、否认、或说要考虑**。",
    "doctor_exposed": "他刚把你篡改过的用药记录当面甩了出来，你被戳中了要害，遮不住了。"
                      "别再假装无事发生，用你的方式应对被拿捏这件事。",
    # 走私犯瘸子
    "smuggler_deal": "你刚刚已经收下了他的医用酒精，并且把强效镇静剂和办公室磁卡给了他——既成事实。"
                     "回话建立在「这笔交易已经成了」之上，交代两句怎么用、别声张，不要再讨价还价。",
    "need_alcohol": "他想跟你换货，但他手上并没有你要的医用酒精。"
                    "所以这笔交易谈不成——你不会给他任何东西，只会点明想要什么。",
    "already_dealt": "这笔交易你和他已经做过一次了，货也给清了。别再给第二份，"
                     "就用你的口吻打发他、或者调侃他贪心。",
    # 狱警铁面
    "guard_bribed": "你刚刚被那份抚恤金截留名单拿住了，已经答应「暴乱那晚晚到五分钟」——既成事实。"
                    "别再嘴硬说自己不会答应，你的挣扎只能体现在语气里。",
    "already_bribed": "这笔交易你已经答应过他了，别再重新谈一遍条件。",
    "guard_offended": "他刚空口污蔑长官，你已经当场揪住他、记下了他的号牌。"
                      "你现在是发作状态，不是讲道理状态。",
    # 老牙
    "oldtooth_exposed": "他刚拿着实证当面戳穿了你——你的伪装到此为止，装不回去了。"
                        "从这句话起，别再演那个热心老室友；露出你真正的那张脸。",
    "oldtooth_deflect": "他在指控你，但手上并没有实证。你绝不认，"
                        "而且要顺势把矛头引向典狱长——让他以为自己找错了方向。",
    # 典狱长
    "warden_cornered": "他刚把黑账拍在你桌上，你被逼到了墙角，官威撑不住了。"
                       "别再打官腔硬顶，改成慌乱、推脱、或者反过来讨价还价。",
    # 告发瘸子
    "smuggler_busted": "他刚向你告发了瘸子的走私勾当，这事你已经受理、瘸子已经被拖去禁闭。"
                       "回话要体现这个结果已经发生。",
}


def outcome_lines(eff: dict) -> list[str]:
    """把 resolve_interaction 的结果翻成"台词必须与此一致"的强指令。

    先按 note 取精确措辞；再用 gained/lost 做一层通用兜底，
    这样以后新增交互分支即便忘了写措辞，也不会退回言行不一致。
    """
    lines: list[str] = []
    notes = eff.get("notes") or []
    for n in notes:
        d = OUTCOME_DIRECTIVE.get(n)
        if d:
            lines.append(d)
    if not lines:      # 兜底：只要有东西易手，就明确告诉它东西已经易手
        for it in eff.get("gained") or []:
            lines.append(f"你刚刚已经把「{ITEM_CN.get(it, it)}」交给他了，这是既成事实，别否认也别拒绝。")
        for it in eff.get("lost") or []:
            lines.append(f"你刚刚已经收下了他的「{ITEM_CN.get(it, it)}」，这是既成事实。")
    return lines


def fact_lines(npc_id: str, bb: Blackboard) -> list[str]:
    """取某 NPC 的事实库护栏 + 依黑板状态补的动态硬约束，注入 roleplay 顶端。

    动态硬约束让 LLM 台词跟世界状态一致（对话不再和状态两张皮）：
    老牙被当面戳穿就别再装好人，狱警收了名单就默认那笔交易，瘸子成交后别再要货……
    """
    lines = list(FACT_BASE.get(npc_id, []))
    if npc_id == ANCHOR_ID:
        if TOOTHBRUSH in bb.inventory:
            lines.append("你已经把那把磨尖的牙刷给主角了。别再重复要给他，"
                         "可以自然地叮嘱他小心用、别被管教发现。")
        if bb.flag("oldtooth_exposed"):
            lines.append("【剧情已反转·硬约束】你的真面目已被主角当面拿证据戳穿，装不下去了。"
                         "此刻起你不再演热心大哥：可以冷笑、恼羞、狡辩、威胁，甚至反咬他一口或"
                         "垂死挣扎地拉典狱长垫背——但绝不天真地全盘认罪求饶。")
    if npc_id == SMUGGLER_ID and bb.flag("smuggler_dealt"):
        lines.append("你已经拿主角的医用酒精换给他镇静剂和磁卡了，这单做完了。"
                     "别再向他要酒精，可以叮嘱他东西藏好、出了事别攀扯上你。")
    if npc_id == DOCTOR_ID and bb.flag("doctor_gave_alcohol"):
        lines.append("你已经把医用酒精给了主角。这事你当没发生过，别再提，也别承认。")
    if npc_id == GUARD_ID and bb.flag("guard_bribed"):
        lines.append("【已被拿捏·硬约束】主角用抚恤金截留名单戳中了你的痛处，你已私下答应"
                     "暴乱那晚晚到五分钟。你嘴上依旧硬、公事公办，但心里已经默许，别再翻脸。")
    if npc_id == WARDEN_ID:
        lines.extend(warden_dynamic_facts(bb))
    return lines
