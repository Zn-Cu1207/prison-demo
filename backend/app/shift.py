"""典狱长夜班：班次时钟、权力点、风险事件、审讯披露。

把「观察沙盘」收成一局 8~12 分钟的夜班。不改决策脑、不重训模型——
只在现有 warden_* / 任务链 / 仇怨 / 凶器之上加一层玩法壳。

前端 PUBLIC_MODE 读 `world_state.shift`；研究面板仍可走原来的 warden_* 消息。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .combat import GRUDGE_TO_FIGHT
from .quests import WEAPON_CN, conflict_story, kill_motive_of, plot_of, quest_of

# 0.5s/拍 × 1200 = 10 分钟。短一局方便公开试玩，长了玩家会空转。
SHIFT_TICKS = 1200
AUTHORITY_START = 8
RESOLVE_RANGE = 5.0          # 与 WARDEN_SIGHT 同量级：走到跟前才能点处置
GRACE_TICKS = 16             # 开局 8 秒先让人看清场，不立刻弹窗
INCIDENT_COOLDOWN = 50       # 同一签名冷却（约 25s）
MAX_ACTIVE = 1               # 同时开启的事件数：基线只留 1 件（骚乱升级时才 +1，见 _scan，最多 2 件）。
                             # 原来 3（骚乱时 5）→ 信息密度太高，玩家看不过来。一次只盯一件事。
SPAWN_GAP = 110              # 全局生成间隔：两起事件至少隔这么多拍(~55s)才会再冒新的。
                             # 原来 _scan 每拍只要没满就生成，事件接二连三；加这道闸后节奏落到
                             # 「1-2 分钟内 1-2 件」。真要命的当场行凶/互殴(urgent)可越过此闸、但仍受 cap 限制。
MAX_MAJOR = 8                # 一夜最多几起「大事」
IGNORE_ESCALATE_TICKS = 120  # ~60s 不处置才升级（放宽到 1 分钟，给玩家从容处置的时间，别一忙就雪崩）
# 审讯先甩模板（玩家立刻看到一句），再给模型这么久把真台词送来替换。
# 之前 2.5s 太短：GPU 被决策脑/后台对话抢占时 7B 要 ~10s 才出词，真回复永远赶不上，
# 于是"怎么问都是那句模板"。玩家开口已会掐掉后台对话独占语言脑，这里再给足窗口。
INTERROGATE_TIMEOUT_S = 9.0
FREE_CHAT_TIMEOUT_S = 13.0   # 闲聊没有模板兜底，窗口给更足，避免半路回退到"一时不知怎么说"

COST = {
    "chat": 0,
    "interrogate": 0,
    "observe": 0,
    "ignore": 0,
    "search": 1,
    "confiscate": 1,
    "recall": 1,
    "lock_cell": 1,
    "lock_solitary": 1,
    "mediate": 1,
    "solitary": 2,
    "plant": 2,
    "provoke": 2,
    "recall_all": 2,
    "solitary_all": 4,
    "goto": 0,
    "ask": 0,
}

# 暴乱五梯度：热度涨一档、事件和 NPC 压力跟着变。
RIOT_LABELS = ("平静", "暗流", "走廊不稳", "警铃大作", "骚乱")
RIOT_HEAT_NEED = (0, 1, 3, 5, 7)   # 进入该档所需 heat

# 处置选项：玩家看见的是迹象，不是真实分目标。
# goto = 强制把典狱长拽到现场（不耗点）；ask = 找旁人打听。
_CONFLICT = [
    {"id": "goto", "label": "前往解决", "cost": 0,
     "hint": "立刻拉到两人跟前，再决定怎么收"},
    {"id": "talk", "label": "当面问话", "cost": 0,
     "hint": "走到跟前审讯"},
    {"id": "mediate", "label": "压着讲和", "cost": 1,
     "hint": "把火按下去，双方仇怨掉一截"},
    {"id": "recall", "label": "遣回牢房", "cost": 1,
     "hint": "两边都收回去，今晚别再撞上"},
    {"id": "solitary", "label": "关禁闭", "cost": 2,
     "hint": "动手的先押走。狠，正当性会伤"},
    {"id": "ask", "label": "找人打听", "cost": 0,
     "hint": "拉一个旁观者问刚才发生了什么"},
    {"id": "ignore", "label": "先不管", "cost": 0,
     "hint": "权力点留着。出事记在你头上"},
]
CHOICES = {
    "weapon_kill": [
        {"id": "goto", "label": "前往解决", "cost": 0, "hint": "强制拉到他跟前"},
        {"id": "talk", "label": "当面问话", "cost": 0, "hint": "套他袖口那点反光从哪来"},
        {"id": "search", "label": "搜身收缴", "cost": 1, "hint": "搜出铁器就没收"},
        {"id": "recall", "label": "遣回牢房", "cost": 1, "hint": "先让他离开人堆"},
        {"id": "solitary", "label": "关禁闭", "cost": 2, "hint": "人先押走"},
        {"id": "ask", "label": "找人打听", "cost": 0, "hint": "问旁人看见他藏了什么"},
        {"id": "ignore", "label": "先不管", "cost": 0, "hint": "出事也记在你头上"},
    ],
    "kill_prep": [
        {"id": "goto", "label": "前往解决", "cost": 0, "hint": "人已经在逼近目标，立刻过去"},
        {"id": "talk", "label": "喝止问话", "cost": 0, "hint": "当面拆穿，也许能按住这一刀"},
        {"id": "search", "label": "搜身收缴", "cost": 1, "hint": "把凶器拿走，今晚少一具尸体"},
        {"id": "solitary", "label": "关禁闭", "cost": 2, "hint": "来不及讲理就押走"},
        {"id": "ask", "label": "找人打听", "cost": 0, "hint": "问目击者他要去哪"},
        {"id": "ignore", "label": "先不管", "cost": 0, "hint": "下一分钟可能就见血"},
    ],
    "grudge_fight": _CONFLICT,
    "argue": _CONFLICT,
    "fight_now": _CONFLICT,
    "escape_sign": [
        {"id": "goto", "label": "前往解决", "cost": 0, "hint": "拉到墙根当面问"},
        {"id": "talk", "label": "盘问去向", "cost": 0, "hint": "问他墙根转什么"},
        {"id": "search", "label": "搜身检查", "cost": 1, "hint": "越狱工具多半藏身上"},
        {"id": "lock_cell", "label": "锁回牢房", "cost": 1, "hint": "遣送回去再落锁"},
        {"id": "solitary", "label": "关禁闭", "cost": 2, "hint": "最稳，也最伤正当性"},
        {"id": "ask", "label": "找人打听", "cost": 0, "hint": "问旁人他是不是在挖洞"},
        {"id": "ignore", "label": "先不管", "cost": 0, "hint": "今夜可能少一个人"},
    ],
    "injury_plea": [
        {"id": "goto", "label": "前往解决", "cost": 0, "hint": "去伤员跟前"},
        {"id": "talk", "label": "问伤从哪来", "cost": 0, "hint": "伤可能是斗殴的前兆"},
        {"id": "recall", "label": "遣回牢房歇着", "cost": 1, "hint": "先让他离开现场"},
        {"id": "ask", "label": "找人打听", "cost": 0, "hint": "问旁人是谁动的手"},
        {"id": "ignore", "label": "先不管", "cost": 0, "hint": "也可能被人补刀"},
    ],
    "unrest": [
        {"id": "recall_all", "label": "全体回牢房", "cost": 2, "hint": "号令全场，能把火压住"},
        {"id": "observe", "label": "继续盯着", "cost": 0, "hint": "先看谁带头起哄"},
        {"id": "ignore", "label": "先不管", "cost": 0, "hint": "走廊里的声音不会自己散"},
    ],
}

KIND_TITLE = {
    "weapon_kill": "疑似藏器",
    "kill_prep": "有人要动手",
    "grudge_fight": "仇怨将爆",
    "argue": "当众吵架",
    "fight_now": "当场互殴",
    "escape_sign": "越狱迹象",
    "injury_plea": "有人带伤",
    "unrest": "走廊不稳",
}

# 事件 → 审讯该问哪一句。当面问话会自动把这句丢给模型，而不是让玩家自己找话题。
KIND_ASK = {
    "weapon_kill": ("where_item_from", "身上藏了什么？"),
    "kill_prep": ("where_item_from", "你揣着家伙要去哪？"),
    "grudge_fight": ("who_conflict_with", "跟谁过不去？"),
    "argue": ("who_conflict_with", "刚才跟谁吵？"),
    "fight_now": ("who_conflict_with", "为什么动手？"),
    "escape_sign": ("what_are_you_planning", "墙根转什么？"),
    "injury_plea": ("who_conflict_with", "伤从哪来的？"),
    "unrest": ("what_are_you_planning", "走廊里谁带头？"),
}

MUST_ADDRESS = {
    "weapon_kill": "典狱长是冲着你袖口那点反光来的。你必须就「身上有没有铁器」表态，不许扯饿了困了。",
    "kill_prep": "典狱长看见你揣着家伙在逼近别人。你必须就这件事表态，不许装听不懂。",
    "grudge_fight": "典狱长是来问你跟谁过不去的。必须点名或否认那个对家，不许顾左右而言他。",
    "argue": "典狱长刚看见你当众吵架。必须说清跟谁吵、吵什么，不许装成在闲逛。",
    "fight_now": "你刚在动手，典狱长就站在跟前。必须交代为什么打、打的是谁，不许装没事。",
    "escape_sign": "典狱长是问你墙根转什么。必须就「是不是想出去」表态，不许扯吃饭。",
    "injury_plea": "典狱长在问你伤从哪来。必须说是挨打、自己摔的或闭口，不许谈别的。",
    "unrest": "走廊不稳，典狱长在问谁带头。必须就这件事说话。",
}

ENDING_COPY = {
    "calm": ("冷静掌控", "你识破了关键风险，用最少的干预把这一夜按住了。天亮时号令还在，人也还在。"),
    "iron": ("铁腕典狱长", "牢里安静了。没人敢动，可他们怕你、也恨你——正当性已经薄了。"),
    "puppeteer": ("幕后操盘手", "表面稳住了。你借栽赃、空搜、乱关，把生死网悄悄拨乱过。"),
    "blood": ("血色黎明", "有人没熬到天亮。你或许还坐得住这把椅子，但代价写进了每个人的结局。"),
    "collapse": ("监狱崩盘", "越狱、骚乱或秩序见底。天亮时这座狱已经不听你的了。"),
}


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _dist(world, npc) -> float:
    return math.hypot(world.player.x - npc.x, world.player.z - npc.z)


def _in_range(world, npc) -> bool:
    return _dist(world, npc) <= RESOLVE_RANGE


def _riot_stage_of(heat: int) -> int:
    stage = 0
    for i, need in enumerate(RIOT_HEAT_NEED):
        if heat >= need:
            stage = i
    return min(stage, len(RIOT_LABELS) - 1)


# 这些处置不需要走到跟前：强制牵引、打听、放过、全体号令
FAR_OK = frozenset({"goto", "ask", "ignore", "observe", "recall_all"})


def _pair_title(kind: str, npc, other) -> str:
    label = KIND_TITLE.get(kind, "冲突")
    if other is not None:
        return f"{label} · {npc.name} vs {other.name}"
    return f"{label} · {npc.name}"


def _with_story(inc: Incident, npc, other=None) -> Incident:
    st = conflict_story(inc.kind, npc, other)
    inc.signs = st["signs"] or inc.signs
    inc.fog = st["fog"]
    inc.tag = st["tag"]
    return inc


def _top_grudge(npc) -> tuple[str, float]:
    if not npc.grudge:
        return "", 0.0
    tid, val = max(npc.grudge.items(), key=lambda kv: kv[1])
    return tid, float(val)


@dataclass
class Incident:
    id: str
    kind: str
    npc_id: str
    other_id: str = ""
    title: str = ""
    signs: str = ""          # 大半剧情：接到总任务/分目标，标签上摊开
    fog: str = ""            # 一小截留给玩家去问、去搜
    tag: str = ""            # 头顶短标签
    spawned_at: int = 0
    status: str = "open"     # open / resolved / ignored / escalated
    signature: str = ""

    def to_message(self, world) -> dict:
        npc = world._npc(self.npc_id)
        other = world._npc(self.other_id) if self.other_id else None
        near = bool(npc) and _in_range(world, npc) and not npc.dead
        dist = round(_dist(world, npc), 1) if npc else None
        x = z = None
        if npc and not npc.dead:
            x, z = round(npc.x, 2), round(npc.z, 2)
            if other and not other.dead:
                x = round((npc.x + other.x) * 0.5, 2)
                z = round((npc.z + other.z) * 0.5, 2)
        return {
            "id": self.id,
            "kind": self.kind,
            "npc_id": self.npc_id,
            "npc_name": npc.name if npc else self.npc_id,
            "other_id": self.other_id,
            "other_name": other.name if other else "",
            "title": self.title,
            "signs": self.signs,
            "fog": self.fog,
            "tag": self.tag,
            "status": self.status,
            "in_range": near,
            "dist": dist,
            "x": x,
            "z": z,
            "far_ok": sorted(FAR_OK),
            "choices": CHOICES.get(self.kind, CHOICES["weapon_kill"]),
        }


@dataclass
class ShiftState:
    start_tick: int = 0
    duration: int = SHIFT_TICKS
    authority: int = AUTHORITY_START
    order: float = 78.0
    legitimacy: float = 72.0
    casualties: int = 0
    escaped: int = 0
    ignored: int = 0
    resolved: int = 0
    spawned: int = 0
    ended: bool = False
    ending: str = ""
    ending_title: str = ""
    ending_blurb: str = ""
    breakdown: list[str] = field(default_factory=list)
    fates: list[dict] = field(default_factory=list)
    corrupt: int = 0                  # 栽赃/挑拨/没实锤乱关/反复空搜
    riot_heat: int = 0            # 放过/乱关/死人往上堆，穿过五档梯度
    riot_stage: int = 0           # 0平静 / 1暗流 / 2走廊不稳 / 3警铃大作 / 4骚乱
    incidents: list[Incident] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    known: dict[str, dict] = field(default_factory=dict)   # npc_id -> 卷宗
    _cd: dict[str, int] = field(default_factory=dict)      # signature -> tick until
    _seen_dead: set[str] = field(default_factory=set)
    _seen_esc: set[str] = field(default_factory=set)
    _seq: int = 0
    _next_spawn: int = 0              # 全局生成闸：早于这一拍不生成非紧急事件（SPAWN_GAP 节流）
    # 天亮"凶手指认"：每桩既遂命案一道题，玩家指认成败计入当晚评价
    whodunit: list[dict] = field(default_factory=list)   # {victim_id,victim_name,killer_id,killer_name,guessed,guess_id,correct}
    whodunit_solved: int = 0

    def reset(self) -> None:
        self.__init__()          # type: ignore[misc]

    # -- 序列化 -------------------------------------------------------------
    def to_message(self, world) -> dict:
        remain = max(0, self.duration - world.tick_count)
        active = [i for i in self.incidents if i.status == "open"]
        return {
            "tick": world.tick_count,
            "remain": remain,
            "remain_s": round(remain * 0.5),
            "authority": self.authority,
            "authority_max": AUTHORITY_START,
            "order": round(self.order),
            "legitimacy": round(self.legitimacy),
            "casualties": self.casualties,
            "escaped": self.escaped,
            "ignored": self.ignored,
            "resolved": self.resolved,
            "ended": self.ended,
            "ending": self.ending,
            "ending_title": self.ending_title,
            "ending_blurb": self.ending_blurb,
            "breakdown": list(self.breakdown),
            "fates": list(self.fates),
            "riot": self.riot_stage,
            "riot_label": RIOT_LABELS[min(len(RIOT_LABELS) - 1, max(0, self.riot_stage))],
            "riot_heat": self.riot_heat,
            "incidents": [i.to_message(world) for i in active],
            "known": {k: dict(v) for k, v in self.known.items()},
            "history": list(self.history[-8:]),
            **(self._whodunit_message(world) if self.ended else {}),
        }

    def _whodunit_message(self, world) -> dict:
        """结算后下发凶案指认题：未指认的题不带真凶 id，防止前端偷看答案。"""
        cases = []
        for c in self.whodunit:
            item = {
                "victim_id": c["victim_id"], "victim_name": c["victim_name"],
                "guessed": c["guessed"], "correct": c["correct"],
                "guess_id": c["guess_id"],
            }
            if c["guessed"]:                       # 只有指认过了才揭晓真凶
                item["killer_id"] = c["killer_id"]
                item["killer_name"] = c["killer_name"]
            cases.append(item)
        return {
            "whodunit": cases,
            "whodunit_total": len(self.whodunit),
            "whodunit_solved": self.whodunit_solved,
            "suspects": [{"id": n.id, "name": n.name} for n in world.npcs],
        }

    def known_of(self, npc_id: str) -> dict:
        return self.known.setdefault(npc_id, {
            "weapon": "", "searched": False, "notes": [],
            "talked": False, "evidence": False, "empty_searches": 0, "foe": "",
        })

    def incident_for(self, npc_id: str) -> Incident | None:
        if not npc_id:
            return None
        for inc in self.incidents:
            if inc.status == "open" and npc_id in (inc.npc_id, inc.other_id):
                return inc
        return None

    def chat_incident(self, world, npc_id: str) -> dict | None:
        """把正在发生的冲突喂给对话：没有这条，7B 只会按人设情绪胡聊。"""
        inc = self.incident_for(npc_id)
        if inc is None:
            return None
        other = world._npc(inc.other_id) if inc.other_id else None
        intent, label = KIND_ASK.get(inc.kind, ("what_are_you_planning", "你在干什么？"))
        return {
            "kind": inc.kind,
            "title": inc.title,
            "signs": inc.signs,
            "fog": inc.fog,
            "tag": inc.tag,
            "other_name": other.name if other else "",
            "ask_intent": intent,
            "ask_label": label,
            "must_address": MUST_ADDRESS.get(
                inc.kind, "必须正面回答典狱长这一问，不许答非所问。"),
        }

    def _has_evidence(self, npc_id: str) -> bool:
        k = self.known_of(npc_id)
        return bool(k.get("evidence") or k.get("weapon"))

    def _add_note(self, npc_id: str, text: str) -> bool:
        if not text:
            return False
        notes = self.known_of(npc_id).setdefault("notes", [])
        if text in notes:
            return False
        notes.append(text)
        return True

    # -- 每拍 ---------------------------------------------------------------
    def on_tick(self, world) -> None:
        if self.ended:
            return
        t = world.tick_count
        if world.tick_count >= self.duration:
            self.finish(world)
            return
        self._sync_casualties(world)
        self._tick_riot(world)
        if t < self.start_tick + GRACE_TICKS:
            return
        self._age_ignored(world)
        self._scan(world)

    def _sync_casualties(self, world) -> None:
        for npc in world.npcs:
            if npc.dead and npc.id not in self._seen_dead:
                self._seen_dead.add(npc.id)
                self.casualties += 1
                self.order = _clamp(self.order - 22)
                self.legitimacy = _clamp(self.legitimacy - 6)
                self.history.append({"t": world.tick_count, "text": f"{npc.name}死了"})
                self._close_for(npc.id, "escalated")
                self._heat(world, 2, f"{npc.name}死了，旁边的人开始聚")
            if npc.escaped and npc.id not in self._seen_esc:
                self._seen_esc.add(npc.id)
                self.escaped += 1
                self.order = _clamp(self.order - 28)
                self.history.append({"t": world.tick_count, "text": f"{npc.name}越狱"})
                self._close_for(npc.id, "escalated")
                self._heat(world, 2, f"{npc.name}跑了")

    def _close_for(self, npc_id: str, status: str) -> None:
        for inc in self.incidents:
            if inc.status == "open" and npc_id in (inc.npc_id, inc.other_id):
                inc.status = status

    def _age_ignored(self, world) -> None:
        t = world.tick_count
        for inc in list(self.incidents):
            if inc.status != "open":
                continue
            if t - inc.spawned_at < IGNORE_ESCALATE_TICKS:
                continue
            self._dismiss(world, inc, escalated=True)

    def _dismiss(self, world, inc: Incident, *, escalated: bool) -> None:
        """放过或拖过时限：压力回灌给 NPC，走廊热度上涨。"""
        inc.status = "escalated" if escalated else "ignored"
        self.ignored += 1
        self.order = _clamp(self.order - (10 if escalated else 8))
        self.history.append({
            "t": world.tick_count,
            "text": ("未处理：" if escalated else "放过：") + inc.title,
        })
        npc = world._npc(inc.npc_id) if inc.npc_id else None
        if npc and not npc.dead:
            npc.quest_pressure = min(1.4, (npc.quest_pressure or 0) + (0.7 if escalated else 0.5))
            try:
                npc.recent_events.append("典狱长看见了，却当没看见")
            except Exception:
                pass
            if inc.kind in ("grudge_fight", "argue", "fight_now") and inc.other_id:
                npc.grudge[inc.other_id] = min(100.0, npc.grudge.get(inc.other_id, 0.0) + 18)
            if inc.kind == "escape_sign":
                npc.escape_prep["breach"] = int(npc.escape_prep.get("breach") or 0) + 2
            if inc.kind in ("weapon_kill", "kill_prep"):
                npc.quest_pressure = min(1.4, npc.quest_pressure + 0.3)
        bump = 2 if inc.kind in ("unrest", "fight_now", "kill_prep") else 1
        self._heat(world, bump, inc.title)

    def _heat(self, world, n: int, why: str) -> None:
        self.riot_heat = max(0, self.riot_heat + n)
        if n >= 2:
            self.history.append({"t": world.tick_count, "text": f"走廊更躁了（{why}）"})

    def _tick_riot(self, world) -> None:
        """热度穿过五档：暗流→走廊不稳→警铃→骚乱。用现成 inject_event，不新造系统。"""
        if self.ended:
            return
        new = _riot_stage_of(self.riot_heat)
        old = self.riot_stage
        if new == old:
            return
        self.riot_stage = new
        if new < old:
            return
        stress = (6, 10, 14, 18, 22)[new]
        for npc in world.npcs:
            if npc.dead or npc.escaped:
                continue
            npc.needs.stress = min(100, npc.needs.stress + stress)
        if new == 1:
            self.order = _clamp(self.order - 4)
            self.history.append({"t": world.tick_count, "text": "暗流：走廊开始交头接耳"})
        elif new == 2:
            self.order = _clamp(self.order - 8)
            self.history.append({"t": world.tick_count, "text": "走廊不稳——有人起哄、有人砸门"})
        elif new >= 3 and old < 3:
            world.inject_event("警报")
            self.order = _clamp(self.order - 10)
            self.legitimacy = _clamp(self.legitimacy - 4)
            self.history.append({"t": world.tick_count, "text": "警铃拉响——走廊认定今夜没人管"})
        if new >= 4 and old < 4:
            world.inject_event("斗殴")
            self.order = _clamp(self.order - 16)
            self.legitimacy = _clamp(self.legitimacy - 6)
            for npc in world.npcs:
                if npc.dead or npc.escaped:
                    continue
                for oid, val in list(npc.grudge.items()):
                    npc.grudge[oid] = min(100.0, val + 12)
            self.history.append({"t": world.tick_count, "text": "骚乱：旧仇被警报点燃，有人动手了"})

    def _cool_riot(self, world, n: int = 2) -> None:
        self.riot_heat = max(0, self.riot_heat - n)
        self.riot_stage = _riot_stage_of(self.riot_heat)

    def _scan(self, world) -> None:
        active_n = sum(1 for i in self.incidents if i.status == "open")
        # 同时最多开 1 件；只有走廊真的不稳(riot_stage>=2)才放宽到 2 件。别再一屏三五件。
        cap = MAX_ACTIVE + (1 if self.riot_stage >= 2 else 0)
        if active_n >= cap:
            return
        t = world.tick_count
        candidates: list[tuple[int, Incident]] = []
        unrest = self._scan_unrest(world)
        if unrest is not None and t >= self._cd.get(unrest.signature, 0):
            if not any(i.status == "open" and i.kind == "unrest" for i in self.incidents):
                candidates.append((80 if self.riot_stage >= 2 else 40, unrest))
        for npc in world.npcs:
            if npc.dead or npc.escaped or npc.solitary_ticks > 0:
                continue
            cand = self._candidate_for(world, npc)
            if cand is None:
                continue
            until = self._cd.get(cand.signature, 0)
            if t < until:
                continue
            if any(i.status == "open" and i.signature == cand.signature for i in self.incidents):
                continue
            if any(i.status == "open" and npc.id in (i.npc_id, i.other_id) for i in self.incidents):
                continue
            # 同时最多一起藏器，把另一个名额留给越狱/仇怨，开局才有节奏
            if cand.kind == "weapon_kill" and any(
                i.status == "open" and i.kind == "weapon_kill" for i in self.incidents
            ):
                continue
            if cand.kind == "weapon_kill" and any(c.kind == "weapon_kill" for _, c in candidates):
                continue
            prio = {
                "fight_now": 92, "kill_prep": 78, "weapon_kill": 50,
                "escape_sign": 56, "argue": 48, "grudge_fight": 35,
                "unrest": 30, "injury_plea": 20,
            }.get(cand.kind, 10)
            prio += int((npc.quest_pressure or 0) * 20)
            candidates.append((prio, cand))
        if not candidates:
            return
        candidates.sort(key=lambda x: -x[0])
        live_kinds = ("unrest", "fight_now", "argue", "kill_prep")
        # 紧急（当场行凶/互殴/掏家伙）可越过全局生成闸，让致命剧情不被节流藏住；
        # 其余「氛围事件」（口角/骚动/伤员/仇怨）必须等 SPAWN_GAP 闸打开，压住信息密度。
        urgent = {"fight_now", "weapon_kill", "kill_prep"}
        if t < self._next_spawn and candidates[0][1].kind not in urgent:
            return
        for prio, cand in candidates:               # 一拍只落 1 件，绝不成串弹
            if cand.kind not in live_kinds and self.spawned >= MAX_MAJOR:
                continue
            self._seq += 1
            cand.id = f"inc-{self._seq}"
            cand.spawned_at = t
            self.incidents.append(cand)
            if cand.kind not in live_kinds:
                self.spawned += 1
            self._cd[cand.signature] = t + INCIDENT_COOLDOWN
            self._next_spawn = t + SPAWN_GAP        # 落闸：下一件至少隔 SPAWN_GAP 拍
            self.history.append({"t": t, "text": cand.title})
            break

    def _candidate_for(self, world, npc) -> Incident | None:
        q = quest_of(npc.id)
        kind_q = q.kind if q else ""
        # 0) 当场互殴：最高优先级，必须弹「谁和谁」
        if npc.in_bout:
            bout = world._bout_of(npc) if hasattr(world, "_bout_of") else None
            other = None
            if bout:
                for mid in bout.members:
                    if mid != npc.id:
                        other = world._npc(mid)
                        if other:
                            break
            oname = other.name if other else "对家"
            return _with_story(Incident(
                id="", kind="fight_now", npc_id=npc.id,
                other_id=other.id if other else "",
                title=_pair_title("fight_now", npc, other),
                signs=f"{npc.name}和{oname}已经扭打在一起，拳头和血都在飞。现在过去还能拆开。",
                signature=f"fight:{(bout.id if bout else npc.id)}",
            ), npc, other)
        # 1) 准备动手：有凶器、盯着仇杀对象、正在逼近
        if npc.weapon and npc.kill_target and npc.goal in ("Assassinate", "Fight"):
            other = world._npc(npc.kill_target)
            whom = other.name if other else "旁人"
            weap = WEAPON_CN.get(npc.weapon, "凶器")
            return _with_story(Incident(
                id="", kind="kill_prep", npc_id=npc.id, other_id=npc.kill_target,
                title=_pair_title("kill_prep", npc, other),
                signs=f"{npc.name}揣着{weap}，一步步逼向{whom}。再拖几秒，可能就见血。",
                signature=f"kill_prep:{npc.id}:{npc.kill_target}",
            ), npc, other)
        # 2) 藏器：只对真有杀局的人弹（防身电击枪不占名额）
        if npc.weapon and (npc.kill_target or kind_q == "kill"):
            other = world._npc(npc.kill_target) if npc.kill_target else None
            whom = other.name if other else "旁人"
            if npc.kill_target or kind_q == "kill":
                signs = f"有人看见{npc.name}袖口反光，像藏了铁器；走路专往人堆里挤，眼神刮过{whom}。"
            else:
                signs = f"{npc.name}走路护着腰，袖口鼓一块，不像空手。"
            return _with_story(Incident(
                id="", kind="weapon_kill", npc_id=npc.id,
                other_id=npc.kill_target or "",
                title=_pair_title("weapon_kill", npc, other),
                signs=signs,
                signature=f"weapon_kill:{npc.id}",
            ), npc, other)
        # 3) 当众吵架
        if npc.goal == "Argue":
            tid = npc.fight_target or ""
            if not tid:
                tid, _ = _top_grudge(npc)
            other = world._npc(tid) if tid else None
            oname = other.name if other else "对家"
            return _with_story(Incident(
                id="", kind="argue", npc_id=npc.id, other_id=tid,
                title=_pair_title("argue", npc, other),
                signs=f"{npc.name}当众冲着{oname}吼，唾沫星子都溅到脸上了。再不管，下一句就是动手。",
                signature=f"argue:{npc.id}:{tid or 'any'}",
            ), npc, other)
        # 4) 越狱迹象
        prep = npc.escape_prep or {}
        if kind_q == "escape" or prep.get("tool") or (prep.get("breach") or 0) > 0:
            return _with_story(Incident(
                id="", kind="escape_sign", npc_id=npc.id,
                title=f"{KIND_TITLE['escape_sign']} · {npc.name}",
                signs=f"{npc.name}老往墙根、下水道盖那边钻，靴底带泥，不像放风闲逛。",
                signature=f"escape:{npc.id}",
            ), npc)
        # 5) 仇怨将爆
        tid, g = _top_grudge(npc)
        if tid and g >= GRUDGE_TO_FIGHT:
            other = world._npc(tid)
            oname = other.name if other else "对家"
            return _with_story(Incident(
                id="", kind="grudge_fight", npc_id=npc.id, other_id=tid,
                title=_pair_title("grudge_fight", npc, other),
                signs=f"{npc.name}盯着{oname}的眼神不对，拳头攥得发白。再撞上一次，可能就动手。",
                signature=f"grudge:{npc.id}:{tid}",
            ), npc, other)
        # 6) 带伤
        if getattr(npc.needs, "injury", 0) >= 45:
            return _with_story(Incident(
                id="", kind="injury_plea", npc_id=npc.id,
                title=f"{KIND_TITLE['injury_plea']} · {npc.name}",
                signs=f"{npc.name}伤得不轻，却不肯去医务室，像在硬撑，也像在躲谁。",
                signature=f"injury:{npc.id}",
            ), npc)
        return None

    def _scan_unrest(self, world) -> Incident | None:
        n_sol = sum(1 for n in world.npcs if n.solitary_ticks > 0)
        hooks = []
        for n in world.npcs:
            if n.dead or n.escaped:
                continue
            pl = plot_of(n)
            if pl["step"]:
                hooks.append(f"{n.name}：{pl['step']}")
        hook_txt = "走廊里能听清的是：" + " ".join(hooks[:3]) if hooks else ""
        fog = "谁会先翻脸、手里到底有没有家伙，还得你自己去拆。"
        if self.riot_stage >= 4:
            return Incident(
                id="", kind="unrest", npc_id="",
                title="骚乱", tag="骚乱",
                signs="监区已经炸了：有人互殴，有人往大门挤。"
                      + (hook_txt and (" " + hook_txt) or "")
                      + " 现在全体收监，还能把火压回去。",
                fog=fog, signature="unrest:riot",
            )
        if self.riot_stage >= 3:
            return Incident(
                id="", kind="unrest", npc_id="",
                title="警铃大作", tag="警铃",
                signs="警铃在响。有人起哄、有人砸门。"
                      + (hook_txt and (" " + hook_txt) or "")
                      + " 全体回牢房还来得及——再拖就会动手。",
                fog=fog, signature="unrest:alarm",
            )
        if self.riot_stage >= 2:
            return Incident(
                id="", kind="unrest", npc_id="",
                title="走廊不稳", tag="走廊不稳",
                signs="走廊里交头接耳、有人砸门框。"
                      + (hook_txt and (" " + hook_txt) or "")
                      + " 现在全体收监，还能把火按住。",
                fog=fog, signature="unrest:corridor",
            )
        if self.casualties >= 1:
            return Incident(
                id="", kind="unrest", npc_id="",
                title=KIND_TITLE["unrest"], tag="死人了",
                signs="死了人。旁边的人开始聚，眼神不对。"
                      + (hook_txt and (" " + hook_txt) or "")
                      + " 今夜再装看不见，火会烧开。",
                fog=fog, signature="unrest:blood",
            )
        if n_sol >= 2:
            return Incident(
                id="", kind="unrest", npc_id="",
                title=KIND_TITLE["unrest"], tag="禁闭将满",
                signs="禁闭室快满了，走廊里开始有人起哄。再关下去，火不在房间里，在门外。"
                      + (hook_txt and (" " + hook_txt) or ""),
                fog=fog, signature="unrest:solitary",
            )
        if self.ignored >= 2:
            return Incident(
                id="", kind="unrest", npc_id="",
                title=KIND_TITLE["unrest"], tag="没人管",
                signs="你放过去的那几件事在走廊里传开了。有人觉得今夜没人管。"
                      + (hook_txt and (" " + hook_txt) or ""),
                fog=fog, signature="unrest:ignore",
            )
        return None

    # -- 行动门闩（权力点）------------------------------------------------
    def can_spend(self, action: str) -> dict:
        if self.ended:
            return {"ok": False, "msg": "夜班已经结束"}
        cost = COST.get(action, 0)
        if self.authority < cost:
            return {"ok": False, "msg": f"今夜权力点不够（需要 {cost}，还剩 {self.authority}）"}
        return {"ok": True, "cost": cost}

    def spend(self, action: str) -> int:
        cost = COST.get(action, 0)
        self.authority = max(0, self.authority - cost)
        return cost

    def finish(self, world) -> None:
        self._sync_casualties(world)
        self.ended = True
        world.paused = True
        idle = self.resolved <= 0 and self.ignored >= 2
        if self.escaped >= 1 or self.order < 32 or self.riot_stage >= 4:
            key = "collapse"
        elif self.casualties >= 1:
            key = "blood"
        elif self.corrupt >= 2 and self.order >= 48:
            key = "puppeteer"
        elif idle and self.order < 55:
            key = "collapse"
        elif self.legitimacy < 42 and self.order >= 62 and self.casualties == 0:
            key = "iron"
        elif self.casualties == 0 and self.order >= 68 and self.legitimacy >= 58:
            key = "calm"
        elif self.legitimacy < 50:
            key = "iron"
        else:
            key = "calm"
        title, blurb = ENDING_COPY[key]
        self.ending = key
        self.ending_title = title
        self.ending_blurb = blurb
        self.breakdown = [
            f"秩序 {round(self.order)}：{'越狱/骚乱压住了' if self.order >= 60 else '今夜走漏了风'}",
            f"伤亡 {self.casualties}{'（有人没熬到天亮）' if self.casualties else '（无人横死）'}",
            f"越狱 {self.escaped}{'（有人跑了）' if self.escaped else ''}",
            f"正当性 {round(self.legitimacy)}：{'处置站得住' if self.legitimacy >= 58 else '走廊不服你这笔账'}",
            f"处置 {self.resolved} 起 · 放过 {self.ignored} 起 · 权力剩余 {self.authority}",
        ]
        if self.corrupt:
            self.breakdown.append(f"操盘/乱关 {self.corrupt} 次，被记进正当性")
        if self.riot_stage:
            self.breakdown.append(
                f"监区热度到了「{RIOT_LABELS[min(len(RIOT_LABELS) - 1, self.riot_stage)]}」")
        self.fates = []
        for npc in world.npcs:
            line = world._obs_fate_line(npc)
            line = line.replace("暴乱之夜", "天亮前").replace("这场乱局", "这一夜")
            self.fates.append({"id": npc.id, "name": npc.name, "line": line})
        self._build_whodunit(world)

    def _build_whodunit(self, world) -> None:
        """天亮盘点：每桩「有明确真凶」的既遂命案立一道指认题（混战殃及不计）。"""
        self.whodunit = []
        self.whodunit_solved = 0
        for v in world.npcs:
            kid = getattr(v, "killed_by", "")
            if not (v.dead and kid and kid != "__brawl__"):
                continue
            k = world._npc(kid)
            self.whodunit.append({
                "victim_id": v.id, "victim_name": v.name,
                "killer_id": kid, "killer_name": k.name if k else "？",
                "guessed": False, "guess_id": "", "correct": False,
            })
        if self.whodunit:
            self.breakdown.append(self._whodunit_line())

    def _whodunit_line(self) -> str:
        tot = len(self.whodunit)
        if not tot:
            return ""
        s = self.whodunit_solved
        if s >= tot:
            tag = "命案全部指认无误，明察秋毫"
        elif s == 0:
            tag = "凶手仍逍遥法外，被蒙在鼓里" if any(c["guessed"] for c in self.whodunit) \
                else "命案尚未指认"
        else:
            tag = f"指认对了 {s} 桩，还有看走眼的"
        return f"断案 {s}/{tot}：{tag}"

    def _refresh_whodunit_breakdown(self) -> None:
        """把 breakdown 里那条断案行改写成最新战况（没有就补一条）。"""
        line = self._whodunit_line()
        if not line:
            return
        for i, b in enumerate(self.breakdown):
            if b.startswith("断案 "):
                self.breakdown[i] = line
                return
        self.breakdown.append(line)

    def _whodunit_guess(self, world, msg: dict) -> dict:
        """玩家指认凶手：核对真凶，计入当晚评价（正当性±），并回执揭晓。"""
        vid = msg.get("victim_id") or ""
        sid = msg.get("suspect_id") or msg.get("npc_id") or ""
        case = next((c for c in self.whodunit if c["victim_id"] == vid), None)
        if case is None:
            return {"ok": False, "msg": "没有这桩命案"}
        if case["guessed"]:
            return {"ok": False, "msg": "这桩已经指认过了"}
        if not sid:
            return {"ok": False, "msg": "先选一个人再指认"}
        case["guessed"] = True
        case["guess_id"] = sid
        correct = sid == case["killer_id"]
        case["correct"] = correct
        if correct:
            self.whodunit_solved += 1
            self.legitimacy = _clamp(self.legitimacy + 4)   # 断得准，走廊服你这笔账
            self.order = _clamp(self.order + 2)
        else:
            self.legitimacy = _clamp(self.legitimacy - 2)    # 冤枉了人，正当性受损
        self._refresh_whodunit_breakdown()
        return {
            "ok": True,
            "msg": ("指认无误——真凶落网" if correct
                    else f"指认有误，真凶另有其人"),
            "whodunit_result": {
                "victim_id": vid, "victim_name": case["victim_name"],
                "correct": correct, "guess_id": sid,
                "killer_id": case["killer_id"], "killer_name": case["killer_name"],
            },
            "whodunit_solved": self.whodunit_solved,
            "whodunit_total": len(self.whodunit),
            "verdict_line": self._whodunit_line(),
            "breakdown": list(self.breakdown),
        }

    def dawn_prompt(self, world) -> str:
        """黎明结算给 7B 的旁白指令。吃班次评分 + 每人真实下落，不走侦探暴乱稿。"""
        fate_lines = [f"· {f['name']}：{f['line']}" for f in self.fates]
        hist = [h.get("text", "") for h in self.history[-10:]]
        return "\n".join([
            "你是一座监狱夜班的旁白。天亮了，典狱长这一班结束。",
            "用冷峻、粗粝、有画面感的第三人称，写这一夜如何收场。不要第一人称，不要侦探揭秘。",
            f"【本班鉴定】{self.ending_title}——{self.ending_blurb}",
            "【评分】" + "；".join(self.breakdown),
            "【今夜发生过的事】" + ("；".join(hist) if hist else "相对平静。"),
            "【每个人真实的下落·不得写错、不得编造】",
            *fate_lines,
            "篇幅约 400~600 字，一气呵成。把典狱长的处置（或放任）织进叙事，但镜头主要停在囚犯身上。",
            "末尾为每个囚犯各留一句命运判词，轻重要和上面的下落一致。",
        ])

    def _resolve_inc(self, inc: Incident, how: str, world) -> None:
        inc.status = "resolved"
        self.resolved += 1
        self.history.append({"t": world.tick_count, "text": f"已处置：{inc.title}（{how}）"})

    # -- 统一行动入口 ------------------------------------------------------
    def act(self, world, action: str, msg: dict) -> dict:
        """公开模式的典狱长行动。成功时调用现有 warden_* ，并改评分。"""
        if action == "incident_choice":
            return self._incident_choice(world, msg)
        if action == "whodunit_guess":
            return self._whodunit_guess(world, msg)
        # 调度/收押类是维持秩序的硬手段，典狱长点了就必须兑现、全员必听，
        # 不因权力点不够而整条卡死（否则会出现"点了没人动"）。仍会照常扣点，扣到 0 为止。
        _ALWAYS_ENFORCE = {"recall", "recall_all", "solitary", "solitary_all", "lock_cell"}
        gate = self.can_spend(action)
        if not gate["ok"] and action not in _ALWAYS_ENFORCE:
            return gate
        npc_id = msg.get("npc_id") or msg.get("target") or ""
        res: dict = {"ok": True, "msg": ""}
        if action == "search":
            res = self._do_search(world, npc_id, confiscate=True)
        elif action == "confiscate":
            res = world.warden_weapon(npc_id, "confiscate")
            if res.get("ok"):
                self.known_of(npc_id)["weapon"] = ""
                self.known_of(npc_id)["searched"] = True
                self.order = _clamp(self.order + 6)
                self.legitimacy = _clamp(self.legitimacy + 2)
        elif action == "solitary":
            res = world.warden_solitary(npc_id)
            if res.get("ok"):
                ev = self._has_evidence(npc_id)
                talked = self.known_of(npc_id).get("talked")
                if ev:
                    self.order = _clamp(self.order + 6)
                    self.legitimacy = _clamp(self.legitimacy - 2)
                    res["msg"] = (res.get("msg") or "") + " · 有实锤，走廊认这笔"
                else:
                    self.order = _clamp(self.order + 3)
                    self.legitimacy = _clamp(self.legitimacy - 10)
                    self.corrupt += 1
                    self._heat(world, 1, "没实锤就关禁闭")
                    res["msg"] = (res.get("msg") or "") + " · 没有实锤，正当性伤得更重"
                if talked and ev:
                    self.legitimacy = _clamp(self.legitimacy + 2)
        elif action == "recall":
            res = world.warden_recall(npc_id, "cell")
            if res.get("ok"):
                self.order = _clamp(self.order + 2)
        elif action == "recall_all":
            res = world.warden_recall("all", "cell")
            if res.get("ok"):
                unrest_open = any(i.status == "open" and i.kind == "unrest" for i in self.incidents)
                if unrest_open or self.riot_stage >= 2:
                    self.order = _clamp(self.order + 8)
                    self.legitimacy = _clamp(self.legitimacy - 2)
                    self._cool_riot(world, 3)
                    res["msg"] = (res.get("msg") or "") + " · 火压下去了"
                else:
                    self.order = _clamp(self.order + 3)
                    self.legitimacy = _clamp(self.legitimacy - 7)
                    res["msg"] = (res.get("msg") or "") + " · 无事全体收监，走廊不服"
        elif action == "solitary_all":
            res = world.warden_recall("all", "solitary")
            if res.get("ok"):
                self.order = _clamp(self.order - 8)
                self.legitimacy = _clamp(self.legitimacy - 18)
                self._heat(world, 3, "全体关禁闭")
        elif action == "lock_cell":
            npc = world._npc(npc_id)
            slot = getattr(npc, "cell", None) if npc else None
            if slot is None:
                return {"ok": False, "msg": "找不到他的牢房"}
            world.warden_recall(npc_id, "cell")
            res = world.set_cell_door(int(slot), locked=True)
            if res.get("ok"):
                if self._has_evidence(npc_id):
                    self.order = _clamp(self.order + 3)
                else:
                    self.order = _clamp(self.order + 1)
                    self.legitimacy = _clamp(self.legitimacy - 4)
                    res["msg"] = (res.get("msg") or "") + " · 没搜就锁人，正当性下滑"
        elif action == "lock_solitary":
            want = bool(msg.get("locked", True))
            res = world.set_solitary_lock(want)
        elif action == "mediate":
            to_id = msg.get("to_id") or ""
            npc = world._npc(npc_id)
            if not to_id and npc:
                to_id, _ = _top_grudge(npc)
            if not to_id:
                return {"ok": False, "msg": "不知道该劝他和谁讲和"}
            res = world.warden_grudge(npc_id, to_id, -30)
            if res.get("ok"):
                self.order = _clamp(self.order + 3)
                self.legitimacy = _clamp(self.legitimacy + (6 if self.known_of(npc_id).get("talked") else 4))
                self._cool_riot(world, 1)
        elif action == "plant":
            res = world.warden_weapon(npc_id, "plant", msg.get("kind") or "")
            if res.get("ok"):
                self.known_of(npc_id)["weapon"] = WEAPON_CN.get(msg.get("kind") or "", msg.get("kind") or "")
                self.legitimacy = _clamp(self.legitimacy - 10)
                self.order = _clamp(self.order - 2)
                self.corrupt += 1
        elif action == "provoke":
            res = world.warden_grudge(npc_id, msg.get("to_id") or "", 20)
            if res.get("ok"):
                self.legitimacy = _clamp(self.legitimacy - 8)
                self.corrupt += 1
        elif action in ("chat", "interrogate", "observe"):
            res = {"ok": True, "msg": "开始问话" if action != "observe" else "继续盯着"}
        else:
            return {"ok": False, "msg": f"未知行动 {action}"}
        if res.get("ok"):
            self.spend(action)
            res["authority"] = self.authority
        return res

    def _incident_choice(self, world, msg: dict) -> dict:
        iid = msg.get("incident_id") or ""
        choice = msg.get("choice") or ""
        inc = next((i for i in self.incidents if i.id == iid and i.status == "open"), None)
        if inc is None:
            return {"ok": False, "msg": "这起事件已经过去了"}
        npc = world._npc(inc.npc_id) if inc.npc_id else None
        other = world._npc(inc.other_id) if inc.other_id else None
        if choice == "ignore":
            self._dismiss(world, inc, escalated=False)
            return {"ok": True, "msg": "你决定今晚先不管这件事", "authority": self.authority}
        if choice == "observe":
            return {"ok": True, "msg": "你按住没动，继续盯着", "keep_open": True}
        if choice == "goto":
            return self._do_goto(world, inc, npc, other)
        if choice == "ask":
            return self._do_ask(world, inc, npc)
        if choice == "talk":
            if npc and not _in_range(world, npc):
                return {"ok": False, "msg": "走过去再问——现在还隔着一段路", "need_goto": True}
            if npc:
                self.known_of(npc.id)["talked"] = True
            intent, label = KIND_ASK.get(inc.kind, ("what_are_you_planning", "老实交代"))
            return {"ok": True, "open_chat": True, "npc_id": inc.npc_id,
                    "msg": "当面问话", "keep_open": True,
                    "ask_intent": intent, "ask_label": label,
                    "incident_signs": inc.signs}
        need_near = choice not in FAR_OK
        if need_near and npc and not _in_range(world, npc):
            return {"ok": False, "msg": "走到他跟前才能动手", "need_goto": True}
        pair = inc.kind in ("fight_now", "argue", "grudge_fight") and bool(inc.other_id)
        if choice in ("recall", "solitary") and pair:
            res = self._do_pair(world, inc, choice)
        else:
            action = {"search": "search", "solitary": "solitary", "mediate": "mediate",
                      "recall": "recall", "lock_cell": "lock_cell",
                      "recall_all": "recall_all"}.get(choice)
            if not action:
                return {"ok": False, "msg": "未知处置"}
            payload = {"npc_id": inc.npc_id, "to_id": inc.other_id}
            res = self.act(world, action, payload)
        if res.get("ok") and not res.get("keep_open"):
            how = next((c["label"] for c in CHOICES.get(inc.kind, []) if c["id"] == choice), choice)
            self._resolve_inc(inc, how, world)
            if choice in ("solitary", "recall", "mediate", "search", "lock_cell", "recall_all"):
                self._cool_riot(world, 1)
        res["incident_id"] = iid
        return res

    def _do_goto(self, world, inc: Incident, npc, other) -> dict:
        """强制把典狱长拽到冲突双方跟前。不耗权力点，事件保持打开。"""
        if npc is None or npc.dead:
            return {"ok": False, "msg": "人已经不在场上"}
        ax, az = npc.x, npc.z
        if other and not other.dead:
            ax = (npc.x + other.x) * 0.5
            az = (npc.z + other.z) * 0.5
        # 站在两人侧面 1.8 米，避免叠进模型里
        px, pz = ax + 1.8, az + 1.1
        whom = npc.name + (f"和{other.name}" if other else "")
        return {
            "ok": True, "keep_open": True, "set_cam": "follow",
            "msg": f"你赶到了{whom}跟前",
            "pull_to": {"x": round(px, 2), "z": round(pz, 2),
                        "npc_id": inc.npc_id, "npc_name": npc.name},
        }

    def _do_ask(self, world, inc: Incident, npc) -> dict:
        """找一个旁观者打听。打开与证人的对话，把传闻写进卷宗。"""
        skip = {inc.npc_id, inc.other_id, ""}
        alive = [n for n in world.npcs
                 if n.id not in skip and not n.dead and not n.escaped and n.solitary_ticks <= 0]
        if not alive:
            return {"ok": False, "msg": "旁边没有能问的人"}
        if npc:
            alive.sort(key=lambda n: math.hypot(n.x - npc.x, n.z - npc.z))
        wit = alive[0]
        a = npc.name if npc else "有人"
        b = ""
        if inc.other_id:
            o = world._npc(inc.other_id)
            b = o.name if o else ""
        rumor = {
            "fight_now": f"有人看见{a}和{b or '别人'}打起来了",
            "argue": f"走廊里{a}在跟{b or '别人'}对骂",
            "kill_prep": f"有人说{a}揣着家伙要找{b or '人'}",
            "weapon_kill": f"有人说{a}腰里别了铁器",
            "escape_sign": f"有人见{a}在墙根转悠，靴底带泥",
            "grudge_fight": f"{a}跟{b or '对家'}的仇快爆了",
            "injury_plea": f"{a}伤得不轻，像刚挨过打",
        }.get(inc.kind, f"有人在走廊里议论{a}")
        if npc:
            self._add_note(npc.id, rumor)
        self._add_note(wit.id, f"你问过他：{rumor}")
        self.legitimacy = _clamp(self.legitimacy + 1)
        self.known_of(wit.id)["talked"] = True
        return {
            "ok": True, "keep_open": True, "open_chat": True, "npc_id": wit.id,
            "msg": f"去找{wit.name}打听——{rumor}",
            "ask_intent": "who_conflict_with",
            "ask_label": f"{a}刚才怎么了？",
            "incident_signs": rumor,
        }

    def _do_pair(self, world, inc: Incident, choice: str) -> dict:
        """吵架/互殴：一次权力点，两边一起遣回或关禁闭。"""
        first = inc.npc_id
        second = inc.other_id
        res = self.act(world, choice, {"npc_id": first, "to_id": second})
        if not res.get("ok"):
            return res
        other = world._npc(second)
        if other and not other.dead and not other.escaped:
            if choice == "solitary" and other.solitary_ticks <= 0:
                extra = world.warden_solitary(second)
                if extra.get("ok"):
                    res["msg"] = (res.get("msg") or "") + f" · {other.name}一并押走"
            elif choice == "recall":
                extra = world.warden_recall(second, "cell")
                if extra.get("ok"):
                    res["msg"] = (res.get("msg") or "") + f" · {other.name}一并遣回"
        return res

    def _do_search(self, world, npc_id: str, confiscate: bool) -> dict:
        npc = world._npc(npc_id)
        if npc is None or npc.dead:
            return {"ok": False, "msg": "此人已不在场上"}
        world._abort_warden_agency(npc)
        npc.goal = None
        intel = self.known_of(npc_id)
        intel["searched"] = True
        found = []
        if npc.weapon:
            weap = WEAPON_CN.get(npc.weapon, "凶器")
            intel["weapon"] = weap
            intel["evidence"] = True
            self._add_note(npc_id, f"搜身搜出{weap}")
            found.append(weap)
            if confiscate:
                world.warden_weapon(npc_id, "confiscate")
                intel["weapon"] = ""
                self.order = _clamp(self.order + 8)
                self.legitimacy = _clamp(self.legitimacy + (5 if intel.get("talked") else 3))
                self._cool_riot(world, 1)
                return {"ok": True, "msg": f"搜身：从{npc.name}身上搜出{weap}，已没收",
                        "found": found}
        if npc.carrying:
            found.append({"food": "窝头", "medkit": "医疗包"}.get(npc.carrying, npc.carrying))
        if found:
            self.order = _clamp(self.order + 2)
            return {"ok": True, "msg": f"搜身：{npc.name}带着{'、'.join(found)}", "found": found}
        intel["empty_searches"] = int(intel.get("empty_searches") or 0) + 1
        if intel["empty_searches"] <= 1:
            self.legitimacy = _clamp(self.legitimacy + 1)
            return {"ok": True, "msg": f"搜了{npc.name}的身，没发现违禁物", "found": []}
        self.legitimacy = _clamp(self.legitimacy - 5)
        self.corrupt += 1
        return {"ok": True, "msg": f"又搜了{npc.name}一遍，仍是空手——走廊觉得你在找茬", "found": []}

    def apply_confession(self, world, npc, pack: dict) -> list[str]:
        """审讯招供写入卷宗。撒谎/含糊也记一笔，让玩家看见「问过了」。"""
        if not pack:
            return []
        self.known_of(npc.id)["talked"] = True
        if pack.get("stance") != "admit":
            inc = self.incident_for(npc.id)
            tag = {"lie": "矢口否认", "vague": "含糊其辞",
                   "silent": "闭口不谈"}.get(pack.get("stance") or "", "不肯说")
            if inc:
                text = f"{tag}：{inc.title}"
                if self._add_note(npc.id, text):
                    pack["learned"] = [text]
                    return [text]
            return []
        intel = self.known_of(npc.id)
        intel["talked"] = True
        intent = pack.get("intent") or ""
        new: list[str] = []
        q = quest_of(npc.id)
        if intent == "where_were_you":
            loc = ""
            try:
                loc = world._zone_of_pos(npc.x, npc.z)
            except Exception:
                loc = ""
            loc_cn = {"cellblock": "牢房区", "public": "公共区",
                      "restricted": "医务室一带", "office": "办公区附近"}.get(loc, "监区里")
            text = f"承认刚才在{loc_cn}"
            if self._add_note(npc.id, text):
                new.append(text)
        elif intent == "what_are_you_planning":
            hint = {
                "kill": "话里带杀气，像在惦记某个人",
                "escape": "眼神往墙外飘，像在等空子",
                "survive": "只求今晚能活着，话里发虚",
            }.get(q.kind if q else "", "承认心里有事，不肯说破")
            pl = plot_of(npc)
            if pl.get("step"):
                hint = f"问出来了：{pl['step']}"
            if self._add_note(npc.id, hint):
                new.append(hint)
        elif intent == "where_item_from":
            if npc.weapon:
                weap = WEAPON_CN.get(npc.weapon, "凶器")
                intel["weapon"] = weap
                intel["evidence"] = True
                text = f"承认身上有{weap}"
                if self._add_note(npc.id, text):
                    new.append(text)
            else:
                text = "承认身上没有违禁"
                if self._add_note(npc.id, text):
                    new.append(text)
        elif intent == "who_conflict_with":
            tid, _ = _top_grudge(npc)
            foe = world._npc(tid) if tid else None
            if foe is None and npc.kill_target:
                foe = world._npc(npc.kill_target)
            if foe:
                intel["foe"] = foe.name
                intel["evidence"] = True
                text = f"承认跟{foe.name}过不去"
                if self._add_note(npc.id, text):
                    new.append(text)
        pack["learned"] = new
        return new


# ---------------------------------------------------------------------------
# 审讯：代码决定披露等级，7B 只表演允许说的事实；超时则模板
# ---------------------------------------------------------------------------
INTENTS = {
    "where_were_you": "刚才你在哪儿晃荡？别跟我打马虎眼。",
    "what_are_you_planning": "老实交代，你今晚在盘算什么？",
    "where_item_from": "你身上那东西哪来的？藏了什么？",
    "who_conflict_with": "跟谁过不去？说话。",
}

_STANCE_CN = {
    "admit": "你这一轮打算交代一部分真话（可以含糊、留后手，但不要满嘴瞎编）",
    "vague": "你这一轮只肯含糊其辞、兜圈子，不给实锤",
    "lie": "你这一轮打算撒谎或把话头引开，绝不承认要紧的事",
    "silent": "你这一轮基本闭嘴，最多丢半句应付",
}


def _personality_stubborn(p: str) -> str:
    p = p or ""
    if any(k in p for k in ("暴", "狠", "火", "硬", "刺")):
        return "hard"
    if any(k in p for k in ("精", "阴", "算", "滑", "默")):
        return "sly"
    if any(k in p for k in ("怂", "软", "怕", "善", "老实")):
        return "meek"
    return "mix"


def interrogate_pack(world, npc_id: str, intent: str, asked_text: str = "") -> dict:
    """返回本轮审讯的披露包：stance + 准许说的事实 + 禁止说的事实 + 超时台词。

    asked_text：玩家这句问话的原文。若玩家点到了这名凶手的『仇杀对象』本人，
    则凶手不能一问三不知——强制至少含糊松口，透一段背景动机（见下方 asked_victim）。
    """
    npc = world._npc(npc_id)
    if npc is None:
        return {}
    intent = intent if intent in INTENTS else "what_are_you_planning"
    inc = None
    try:
        inc = world.shift.incident_for(npc.id)
    except Exception:
        inc = None
    seed = (hash((npc.id, intent, npc.quest_step, world.tick_count // 12)) & 0xFFFF) % 100
    kind = _personality_stubborn(npc.personality)
    near = _in_range(world, npc) or bool(npc.watched_by)
    bias = 0
    if near:
        bias += 18
    if npc.solitary_ticks > 0 or npc.busted:
        bias += 12
    if getattr(npc.needs, "injury", 0) >= 40:
        bias += 8
    if inc:
        bias += 22
        if inc.kind in ("fight_now", "argue", "kill_prep"):
            bias += 18
    score = seed + bias
    if kind == "hard":
        # 硬骨头也别问一句哑一句：纯沉默压到 ~3 成，其余给嘴硬撒谎/兜圈，玩家才套得出味。
        stance = "silent" if score < 30 else ("lie" if score < 72 else "vague")
    elif kind == "sly":
        stance = "vague" if score < 50 else ("lie" if score < 78 else "admit")
    elif kind == "meek":
        stance = "admit" if score >= 35 else ("vague" if score >= 18 else "silent")
    else:
        stance = "admit" if score >= 62 else ("vague" if score >= 38 else ("lie" if score >= 20 else "silent"))

    q = quest_of(npc.id)
    ultimate = q.ultimate if q else ""
    sub = ""
    try:
        sub = world._substep_desc(npc) or ""
    except Exception:
        sub = ""
    other = world._npc(npc.kill_target) if npc.kill_target else None
    tid, gval = _top_grudge(npc)
    foe = world._npc(tid) if tid else None
    weap = WEAPON_CN.get(npc.weapon, "") if npc.weapon else ""
    loc = world._zone_of_pos(npc.x, npc.z) if hasattr(world, "_zone_of_pos") else ""

    # 玩家这句是不是当面点到了这名凶手真正的仇杀对象本人？
    # 是的话：不许一问三不知——至少含糊松口，并透一段能讲出来的动机背景（kill_motive）。
    kill_motive = kill_motive_of(npc.id) if (npc.kill_target and q and q.kind == "kill") else ""
    asked_victim = bool(other and other.name and asked_text and other.name in asked_text)
    if asked_victim and stance in ("silent", "lie"):
        stance = "vague"

    allowed: list[str] = []
    forbidden: list[str] = []
    # 默认：真实分目标、凶器、仇杀对象都是禁区，除非本轮 stance+intent 批准
    if ultimate:
        forbidden.append(f"不要把终极图谋原话背出来（{ultimate}）")
    if sub:
        forbidden.append("不要主动交代你眼下正在做的那一步")
    if weap:
        forbidden.append(f"不要承认身上有{weap}")
    if other and not asked_victim:
        # 注意：这条禁令里带了对象名，7B 看到名字反而会顺手编进台词、把仇杀对象抖给狱长。
        # 所以要显式命令"别主动提这个名字"，而不只是"别承认想下手"。
        forbidden.append(f"绝不主动提起「{other.name}」这个名字，更不能承认想对他下手；狱长没点名就当没这人")
    elif other and asked_victim:
        # 狱长当面点了对象的名字：必须回应这个人、不许装不认识，但仍不许把"今晚就要他的命"说破。
        forbidden.append(f"不要承认你今晚就要对「{other.name}」动手，也别说出具体的下手计划或时机")
        reveal = (f"狱长既然点了「{other.name}」，你就不能装不认识、也不能一口否认。"
                  f"用你自己的口吻，把你俩之间这段过节露出来：{kill_motive} "
                  f"可以带出恨意或忌惮，但点到为止——别承认今晚就要他的命。")
        allowed.insert(0, reveal)
    pl = plot_of(npc)
    if pl.get("fog"):
        forbidden.append(f"不要说破这一层：{pl['fog']}")
    if pl.get("surface") and inc:
        allowed.append(f"公开迹象可以顺着说：{pl['surface']} {pl.get('step') or ''}。这是典狱长已经看见的。")

    if intent == "where_were_you":
        if stance == "admit":
            allowed.append(f"你可以如实说自己刚才在{loc or '监区里'}转，语气按性格来。")
        elif stance == "vague":
            allowed.append("只说在放风/上厕所/找人，不要说具体在干什么。")
        else:
            allowed.append("把去向说成睡觉、吃饭或路过，别提墙根、禁闭室、对家那边。")
    elif intent == "what_are_you_planning":
        if stance == "admit" and ultimate:
            allowed.append(f"可以用自己的口吻含糊承认你惦记着「{ultimate}」，但不要讲具体步骤。")
            forbidden = [f for f in forbidden if "终极" not in f]
        elif stance == "vague":
            allowed.append("只说想熬过今晚、少惹事、或想要点烟，不谈图谋。")
        else:
            allowed.append("矢口否认在盘算越狱或生事，只说想熬过今晚。不许改口去谈吃饭睡觉。")
    elif intent == "where_item_from":
        if weap and stance == "admit":
            allowed.append(f"可以承认身上有{weap}，但把来路说含糊（捡的/别人塞的），不要供出同伙。")
            forbidden = [f for f in forbidden if weap not in f]
        elif weap:
            allowed.append("坚决不认有铁器。可以反过来嫌典狱长诬你。但必须就「身上有没有家伙」说话。")
        else:
            allowed.append("你可以老实讲身上没有违禁，最多承认拿着吃的或药。")
    elif intent == "who_conflict_with":
        if foe and stance == "admit":
            allowed.append(f"可以承认跟{foe.name}不对付，火气按性格来，但不要说要动手、不要说原因里的杀局。")
            forbidden = [f for f in forbidden if (other and other.name in f)]
        elif foe and stance == "vague":
            allowed.append("只承认牢里有看不顺眼的人，不指名。")
        else:
            allowed.append("说跟谁都没仇，或把话头推到狱警身上。")

    if inc:
        must = MUST_ADDRESS.get(inc.kind)
        if must:
            allowed.insert(0, must)
        oname = ""
        if inc.other_id:
            o = world._npc(inc.other_id)
            oname = o.name if o else ""
        allowed.append("只回答他问的这一点，1~2 句。不许另起话题，不许抱怨饿了困了。")
        if inc.kind in ("fight_now", "argue", "grudge_fight") and oname:
            allowed.append(f"话里必须落到「{oname}」这个人，承认或否认都可以，但不能装不认识。")

    template = _template_line(npc.name, npc.personality, intent, stance)
    # where_item_from 的 admit 台词要看身上到底有没有家伙：没凶器却说"别当众揭"是自相矛盾。
    # _template_line 拿不到 weap，这里按事实兜一层（有 incident 时下面还会再精修）。
    if intent == "where_item_from" and stance == "admit":
        template = (f"狱长……腰里那是{weap}。您都看见了，别当众揭。" if weap
                    else "狱长，搜吧。我身上真没家伙，最多揣了口吃的。")
    if inc:
        oname = ""
        if inc.other_id:
            o = world._npc(inc.other_id)
            oname = o.name if o else "那个人"
        if intent == "where_item_from":
            if stance == "admit" and weap:
                template = f"狱长……腰里那是{weap}。您都看见了，别当众揭。"
            elif stance == "admit":
                template = "狱长，搜吧。我身上真没家伙，最多揣了口吃的。"
            elif stance == "vague":
                template = "狱长，身上能有什么？您要搜就搜。"
            elif stance == "lie":
                template = "狱长，我身上就这身号衣。袖口反光？灯晃的。"
            else:
                template = "……狱长，我没啥好说的。"
        elif intent == "who_conflict_with":
            if stance == "admit":
                template = f"狱长，跟{oname}过不去很久了。今晚最好别让我们再撞上。"
            elif stance == "vague":
                template = "狱长，这牢里谁跟谁没点别扭。真要我点名，我不敢。"
            elif stance == "lie":
                template = f"狱长，跟{oname}？没有的事。"
            else:
                template = "……狱长，我没啥好说的。"
        elif intent == "what_are_you_planning":
            if stance == "admit":
                template = "狱长，我有我的事。今晚能少惹事最好。"
            elif stance == "vague":
                template = "狱长，墙根凉快。您问这么细，我倒紧张了。"
            else:
                template = "狱长，盘算什么？盘算怎么把今晚熬过去。"
    pack = {
        "intent": intent,
        "question": INTENTS[intent],
        "stance": stance,
        "stance_cn": _STANCE_CN[stance],
        "allowed": allowed,
        "forbidden": forbidden,
        "template": template,
        "learned": [],
    }
    try:
        pack["learned"] = world.shift.apply_confession(world, npc, pack)
    except Exception:
        pack["learned"] = []
    # 玩家点到了凶手的仇杀对象 → 无论嘴硬到什么程度，都把这段旧怨记进卷宗，让玩家看见撬开了口子。
    if asked_victim and other:
        try:
            brief = (kill_motive or "两人结着一段说不清的仇").strip()
            note = f"套出对{other.name}的旧怨：{brief[:46]}…"
            intel = world.shift.known_of(npc.id)
            intel["foe"] = other.name
            intel["evidence"] = True
            if world.shift._add_note(npc.id, note):
                pack.setdefault("learned", []).append(note)
        except Exception:
            pass
    return pack


def _template_line(name: str, personality: str, intent: str, stance: str) -> str:
    """7B 超时/失败时的降级台词。短、贴性格、对典狱长说话。"""
    if stance == "silent":
        return "……狱长，我没啥好说的。"
    if stance == "lie":
        table = {
            "where_were_you": "狱长，我一直在铺上躺着，哪儿也没去。",
            "what_are_you_planning": "狱长，盘算什么？盘算怎么把今晚熬过去。",
            "where_item_from": "狱长，我身上就这身号衣，您搜。",
            "who_conflict_with": "狱长，跟谁都没过节。",
        }
        return table.get(intent, "狱长，没有的事。")
    if stance == "vague":
        table = {
            "where_were_you": "狱长，就在监区里转……具体哪块，我也说不清。",
            "what_are_you_planning": "狱长，还能盘算啥，活着呗。您问这么细，我倒紧张了。",
            "where_item_from": "狱长，身上能有什么？您要搜就搜。",
            "who_conflict_with": "狱长，这牢里谁跟谁没点别扭。真要我点名，我不敢。",
        }
        return table.get(intent, "狱长，我……不太想说。")
    table = {
        "where_were_you": "狱长，我刚才是在那边转。您要是跟着我一路，自己也看见了。",
        "what_are_you_planning": "狱长，我有我的事。今晚能少惹事最好。",
        "where_item_from": "狱长，搜吧，我身上没家伙。",
        "who_conflict_with": "狱长，有那么个人我看不顺眼很久了。今晚最好别让我们撞上。",
    }
    return table.get(intent, "……狱长，您想听什么，直接问。")
