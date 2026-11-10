"""物资层：让属性真的挂在世界内容上。

问题背景（改这版之前的状态）：
    「吃饭」= 走到食堂坐标 0.5 米内 → 每帧饥饿 -60。没有食物实体、没有取用、
    没有时长，也就不可能"没饭吃"。同理受伤只有纹身 +4 会涨，打架不产生任何伤害。
    属性是悬空的数字，跟世界里的东西没有因果关系。

这里补上被消耗/被占用的**实体**，让生存需求必须经过"取到东西"这一步：

    食物(food)   可消耗：食堂餐盘按份数掉，饭点补货 —— 错过饭点真的没饭吃
    饮水(water)  占用式：水龙头同时只能一个人用 —— 人多要排队
    床位(bed)    占用式：牢房铺位有限 —— 满了睡不成
    医疗包(medkit) 可消耗：打架伤员多了会用光 —— 伤势才有代价

两类语义差别（`consumable`）：
    可消耗  取走即 stock-1，用完等补货（食物、药品）
    占用式  stock 恒定，靠 user 字段独占，用完即释放（水龙头、床铺）

玩家和 NPC 抢的是**同一批** object：玩家把桌上窝头拿走，NPC 这顿就少一份。
所以这层必须在后端权威，不能像旧的前端道具那样各算一套。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 作息时段常量（与 context.py 一致，避免循环 import 这里写字面量）
PHASE_ROLLCALL = "点名"
PHASE_TRAIN = "训练"
PHASE_CLASS = "上课"
PHASE_YARD = "放风"
PHASE_SLEEP = "就寝"
PHASE_FREE = "自由活动"


@dataclass
class WorldObject:
    """世界里一件能被取用/占用的东西。"""

    id: str
    kind: str                 # food / water / bed / medkit
    name: str
    facility_id: str          # 挂在哪个设施里（前端据此摆放）
    x: float
    z: float
    cap: int                  # 满库存（占用式恒为并发上限）
    stock: int                # 当前剩余（占用式恒 = cap）
    consumable: bool          # True=取走就少；False=占用式
    slot: int = -1            # 牢房铺位：属于哪间隔间（其余物资 -1，谁都能用）
    users: list[str] = field(default_factory=list)   # 占用式：正在使用者的 npc id
    restock_phases: tuple[str, ...] = ()             # 进入这些时段时补满

    # -- 可用性 ------------------------------------------------------------
    @property
    def free_slots(self) -> int:
        return self.stock if self.consumable else self.cap - len(self.users)

    def available(self) -> bool:
        return self.free_slots > 0

    # -- 取用 / 释放 -------------------------------------------------------
    def take(self, npc_id: str) -> bool:
        """占一个名额。可消耗的直接扣库存，占用式的登记使用者。"""
        if not self.available():
            return False
        if self.consumable:
            self.stock -= 1
        else:
            self.users.append(npc_id)
        return True

    def release(self, npc_id: str) -> None:
        """放开占用（可消耗的东西没有"放回"——已经吃掉了）。"""
        if not self.consumable and npc_id in self.users:
            self.users.remove(npc_id)

    def restock(self) -> None:
        if self.consumable:
            self.stock = self.cap

    def to_message(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "name": self.name,
            "facility": self.facility_id, "pos": [self.x, self.z],
            "stock": self.stock, "cap": self.cap,
            "consumable": self.consumable, "busy": len(self.users),
            "available": self.available(),
        }


# ---------------------------------------------------------------------------
# 物资投放表：设施 kind -> 往里摆哪些东西
#
# 份数是按 6 个 NPC + 玩家调的：够用，但饭点前后会紧张，偶尔真的抢不到。
# 调这里就能调"稀缺程度"，是这套系统最主要的手感旋钮。
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObjectSpec:
    kind: str
    name: str
    count: int                      # 摆几个
    cap: int                        # 每个的容量/份数
    consumable: bool
    restock_phases: tuple[str, ...] = ()
    spread: float = 1.1             # 同类之间的摆放间距（前端照此散开）


SPAWN_TABLE: dict[str, tuple[ObjectSpec, ...]] = {
    # 食堂：两张餐盘各 6 份窝头，点名（早饭）和放风（午饭）补货。
    # 就寝时段不补 —— 夜里饿了只能忍到早上，这是"错过饭点"能被玩家感知的关键。
    "eat": (
        ObjectSpec("food", "窝头", count=2, cap=6, consumable=True,
                   restock_phases=(PHASE_ROLLCALL, PHASE_YARD)),
    ),
    # 水房：两个龙头，占用式 —— 同时最多两人喝，第三个人得等。
    "drink": (
        ObjectSpec("water", "水龙头", count=2, cap=1, consumable=False),
    ),
    # 牢房：一人一间隔间、一间一张铺，占用式。份数由 layout.CELLS 决定，
    # 见 spawn_objects 里的特例——铺位要落在各自那间隔间里，不能沿前墙排开。
    "sleep": (
        ObjectSpec("bed", "铺位", count=0, cap=1, consumable=False),
    ),
    # 医务室：一个药柜 6 份，每天点名补一次。打架多了就会见底。
    "cure": (
        ObjectSpec("medkit", "医疗包", count=1, cap=6, consumable=True,
                   restock_phases=(PHASE_ROLLCALL,)),
    ),
}

# 行动 -> 需要哪种物资。没列进来的行动不受物资约束（溜达、洗衣之类）。
REQUIRED_OBJECT: dict[str, str] = {
    "Eat": "food",
    "Drink": "water",
    "Sleep": "bed",
    "Cure": "medkit",
}

# 行动 -> 取到东西后要"做"多少帧才结算。
# 这是旧版本完全缺失的概念：以前到了就算完事、且每帧都吃一次。
PERFORM_TICKS: dict[str, int] = {
    "Eat": 6,
    "Drink": 3,
    "Sleep": 24,
    "Cure": 8,
}

# 行动完成时的一次性结算（不再是每帧 -60）。
PERFORM_EFFECT: dict[str, dict[str, int]] = {
    "Eat": {"hunger": -70},
    "Drink": {"thirst": -70},
    "Sleep": {"fatigue": -80, "stress": -10},
    "Cure": {"injury": -70, "stress": -6},
}

# 拿在手上就能直接用的东西（不必待在设施里）。
# 对应"从背包里掏出窝头就地吃"——也顺带修好了一个真实缺陷：
# 重伤/濒饿的 NPC 以前会因为食堂太远、Cost 把 Eat 压下去而饿着不动。
PORTABLE: dict[str, str] = {"Eat": "food", "Cure": "medkit"}


FACE_VEC = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}


def spawn_objects(facilities, cells=()) -> list[WorldObject]:
    """按投放表给每个设施生成物资实体。facilities 是 world.Facility 列表。

    牢房走特例：铺位一间一张、落在各自隔间的中心，并记下 slot——`_acquire_for` 据此
    只让人占自己那张铺。其余物资沿前墙排开（见下面的 off/face 计算）。
    """
    out: list[WorldObject] = []
    for f in facilities:
        if f.kind == "sleep":
            for c in cells:
                out.append(WorldObject(
                    id=f"{f.id}_bed{c['slot']}", kind="bed", name="铺位",
                    facility_id=f.id, x=c["x"], z=c["z"],
                    cap=1, stock=1, consumable=False, slot=c["slot"],
                ))
            continue
        for spec in SPAWN_TABLE.get(f.kind, ()):
            for i in range(spec.count):
                # 沿前墙方向摆开、再往门口挪 0.9m，前端直接用这个坐标放模型。
                # 必须跟着房间朝向算：房子贴墙成排之后各排朝向不同，
                # 再照死写 z+0.9 会把窝头摆到墙里去（西/东两排的 +z 是沿墙方向）。
                off = (i - (spec.count - 1) / 2) * spec.spread
                fx, fz = FACE_VEC.get(getattr(f, "face", "N"), (0.0, 1.0))
                out.append(WorldObject(
                    id=f"{f.id}_{spec.kind}{i}",
                    kind=spec.kind, name=spec.name, facility_id=f.id,
                    x=f.x - fz * off + fx * 0.9, z=f.z + fx * off + fz * 0.9,
                    cap=spec.cap,
                    stock=spec.cap,
                    consumable=spec.consumable,
                    restock_phases=spec.restock_phases,
                ))
    return out
