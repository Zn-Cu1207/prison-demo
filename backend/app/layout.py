"""监狱平面布局——唯一权威。

以前布局分三处各存一份：后端 `world.py` 存落点、前端 `main.js` 存房间尺寸、
体检脚本 `_overlap_probe.py` 又抄一份落点。三份各自演进，凑到一起就撞上：
禁闭室穿图书室、放风场铁栅栏整条穿透黑市，都是这么来的。现在只有这里一份——
落点由这里算，尺寸/朝向随世界帧发给前端，体检脚本直接 import 本模块。

本模块刻意不 import 任何东西：体检脚本要在不装后端依赖的情况下也能跑起来。

## 布局原则：贴墙连排，中间留院

四面围墙各一条 RUN（连排）。一条 RUN 里的房间背靠同一面墙、沿墙依次紧挨、
共用隔墙，门一律朝中庭开。每间房的**进深各不相同**，所以这排房子的内沿是参差的，
从斜上方看是一列凹凸的隔间群而不是一条平齐的板墙；宽度也各不相同。

### 为什么每条 RUN 都要躲开垂直方向那条的进深带

一条 RUN 的两端**不许伸进垂直方向那条 RUN 的进深带**（见 `_span`）。这不是为了好看：
拐角那间房的门在前墙居中、朝中庭开，要是它伸进了垂直那排的进深带，门外两三米处
正好是垂直那排的侧墙——门开出去就是一堵墙，人贴着墙根绕行，寻路还容易蹭进墙里。
把两端都退到进深带之外，每间房的门外就一定是一条直通中庭环路的空路。
体检脚本 `_overlap_probe.py` 把这条写成了断言（门到环路的直线通路检查）。

代价是四个角空着。这恰好也是真实监狱的样子：一圈实体围着一个院子，角上是通路。

## 场地为什么只有这么大

围墙半径从 30 收到 19.8。原来 30 的时候，单程能有五十多米，而后端 NPC 一律
3.5 m/s 走直线、前端按各人步幅配速只有 1.2~1.7 m/s，差距被行程放大成"前端画的人
落后后端十几米"——你走到眼前的犯人，后端并不认为他在跟前（敬礼/监视判定用后端坐标）。
收到 19.8 之后最长单程约 29 米，配上前端的追赶就能跟平。

房间也随之收窄了一圈（原来动辄 8~9 米宽，现在 4.4~5.6 米），只有食堂保持大间（11×7.5，
三排横向长餐桌）。收窄不影响用：一间房同时最多容下两三个人。
"""
from __future__ import annotations

WALL = 19.8                  # 围墙半径（前端 YARD/2）
BACK = 19.0                  # 房间外墙面离场地中心的距离，留 0.8 给围墙/台基模块

# 牢房连排：六间厂间式隔间，一间一人，门各自朝中庭开。
CELL_N = 6                   # 隔间数：五名囚犯各一间 + 主角（典狱长视角的出生点）那间
CELL_W = 2.35                # 每间沿墙宽度
CELL_D = 4.05                # 进深。与前端 openCell 的 4×4 净空 + 0.15 墙厚对齐

# 房间朝向：face 是"门朝哪个方向"。默认取背靠墙的反向。
_FACE_OF_SIDE = {"S": "N", "N": "S", "W": "E", "E": "W"}
FACE_VEC = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}

# 每条 RUN：(背靠哪面围墙, 房间序列)。沿墙起点由 _span 按"躲开垂直进深带"算，不再手填。
# 房间 = (id, 中文名, kind, 沿墙宽, 进深)
# kind == "gap" 是留给大门的一段空墙，不出设施。
RUNS: tuple[tuple[str, tuple[tuple[str, str, str, float, float], ...]], ...] = (
    ("S", (
        ("laundry",   "洗衣房", "laundry",  4.6, 5.0),
        ("canteen",   "食堂",   "eat",     11.0, 7.5),
        ("water",     "水房",   "drink",    4.2, 4.2),
        ("classroom", "教室",   "class",    5.0, 4.6),
    )),
    ("W", (
        ("cell",   "牢房", "sleep",  CELL_N * CELL_W, CELL_D),
        ("shower", "澡堂", "shower", 5.5,             5.0),
    )),
    ("E", (
        ("clinic",   "医务室",   "cure",     5.6, 5.2),
        ("market",   "黑市角落", "market",   4.4, 4.2),
        ("gym",      "健身区",   "gym",      5.0, 5.6),
        ("commons",  "娱乐室",   "commons",  5.0, 5.6),
        ("solitary", "禁闭室",   "solitary", 4.4, 4.2),
    )),
    ("N", (
        ("labor",     "苦役区",     "labor",     6.0, 6.0),
        ("visit",     "探视室",     "visit",     5.0, 5.0),
        ("library",   "图书室",     "library",   5.2, 5.0),
        ("",          "",           "gap",       3.0, 0.0),   # 大门那段空墙
        ("guardpost", "管教办公室", "guardpost", 5.4, 5.2),
    )),
)

# 不进连排的三处。放风场是露天场地（前端只铺跑道和铁栅栏，不盖房），
# 大门是围墙缺口里的一道铁闸，电话亭是中庭里一个人的地方。
# 大门朝 S：它是围墙缺口里的一道铁闸，本身没有门洞可言，face 在这里只用来
# 决定前端那条步道往哪边接——往院里接，而不是往墙外接。
FREE: tuple[tuple[str, str, str, float, float, float, float, str], ...] = (
    ("yard",  "放风场", "yard",   0.0, 1.0,  8.0, 2.0, "N"),
    ("gate",  "大门",   "gate",   5.1, 18.6, 2.0, 1.2, "S"),
    ("phone", "电话亭", "phone", -10.0, 9.5, 1.2, 1.2, "E"),
)
YARD_FENCE_R = 6.2           # 放风场铁栅栏半径（前端 buildYardProps，只圈北侧和东侧）


def _depth(side: str) -> float:
    """一条 RUN 的最大进深——垂直方向那两条要退开这么多。"""
    return max((d for _i, _n, _k, _w, d in dict(RUNS)[side]), default=0.0)


def _span(side: str) -> tuple[float, float]:
    """一条 RUN 沿墙方向可用的区间。两端各退开垂直方向那条 RUN 的进深带。"""
    if side in ("S", "N"):
        return -BACK + _depth("W"), BACK - _depth("E")
    return -BACK + _depth("S"), BACK - _depth("N")


def _slot(side: str, cursor: float, span: float, depth: float) -> tuple[float, float, float, float]:
    """一间房在世界系里占的矩形 (x0, x1, z0, z1)。

    span 是沿墙方向的宽度，depth 是从围墙往中庭伸进来的进深。
    """
    if side == "S":
        return cursor, cursor + span, -BACK, -BACK + depth
    if side == "N":
        return cursor, cursor + span, BACK - depth, BACK
    if side == "W":
        return -BACK, -BACK + depth, cursor, cursor + span
    return BACK - depth, BACK, cursor, cursor + span      # "E"


def build() -> list[dict]:
    """算出全部设施。每项含落点、朝向、以及房间的**局部**尺寸。

    局部尺寸的约定和前端 makeBuilding 一致：w 沿局部 x（前墙方向），d 沿局部 z
    （进深方向），门在 +z 那面。前端拿到 face 之后整组绕 Y 转，
    局部里写好的家具/门洞/模块化石墙就一起转过去了，不用为每个朝向各写一套。
    """
    out: list[dict] = []
    for side, rooms in RUNS:
        lo, hi = _span(side)
        total = sum(w for _i, _n, _k, w, _d in rooms)
        cursor = lo + (hi - lo - total) / 2          # 整排在可用区间里居中
        for fid, name, kind, span, depth in rooms:
            if kind == "gap":
                cursor += span
                continue
            x0, x1, z0, z1 = _slot(side, cursor, span, depth)
            cursor += span
            face = _FACE_OF_SIDE[side]
            # face 垂直于墙时前墙是沿墙那条边，局部 w=span；平行时两者互换。
            horiz = face in ("N", "S")
            out.append({
                "id": fid, "name": name, "kind": kind,
                "x": round((x0 + x1) / 2, 3), "z": round((z0 + z1) / 2, 3),
                "w": round(x1 - x0 if horiz else z1 - z0, 3),
                "d": round(z1 - z0 if horiz else x1 - x0, 3),
                "face": face, "door": 0.0,
            })
    for fid, name, kind, x, z, w, d, face in FREE:
        out.append({"id": fid, "name": name, "kind": kind, "x": x, "z": z,
                    "w": w, "d": d, "face": face, "door": 0.0})
    return out


FACILITIES: list[dict] = build()
BY_ID: dict[str, dict] = {f["id"]: f for f in FACILITIES}


def footprint(fac: dict) -> tuple[float, float, float, float]:
    """设施在世界系的占地矩形 (x0, x1, z0, z1)。"""
    horiz = fac["face"] in ("N", "S")
    ex = fac["w"] if horiz else fac["d"]
    ez = fac["d"] if horiz else fac["w"]
    return (fac["x"] - ex / 2, fac["x"] + ex / 2, fac["z"] - ez / 2, fac["z"] + ez / 2)


def contains(fac: dict, x: float, z: float, inset: float = 0.45) -> bool:
    """点是否在房间净空里。inset 往里收一圈，门口走廊不算进了屋。"""
    x0, x1, z0, z1 = footprint(fac)
    w, d = x1 - x0, z1 - z0
    if w < 0.5 or d < 0.5:
        cx, cz = fac["x"], fac["z"]
        return (x - cx) ** 2 + (z - cz) ** 2 <= 1.35 ** 2
    m = min(inset, w * 0.22, d * 0.22)
    return (x0 + m) <= x <= (x1 - m) and (z0 + m) <= z <= (z1 - m)


def door_rect(fac: dict, width: float = 1.5) -> tuple[float, float, float, float]:
    """门洞在世界系占的那一小段墙 (x0, x1, z0, z1)，用来查门有没有被邻居堵住。

    与前端 makeBuilding 同一套算法：门宽 min(1.5, w*0.42)，沿前墙按 door 偏移。
    """
    dw = min(width, fac["w"] * 0.42)
    fx, fz = FACE_VEC[fac["face"]]
    # 前墙中点：从房间中心沿朝向走半个进深
    cx = fac["x"] + fx * fac["d"] / 2
    cz = fac["z"] + fz * fac["d"] / 2
    # 沿前墙方向（把朝向转 90°）挪动 door
    ax, az = -fz, fx
    cx += ax * fac["door"]
    cz += az * fac["door"]
    ex, ez = abs(ax) * dw, abs(az) * dw
    return (cx - ex / 2, cx + ex / 2, cz - ez / 2, cz + ez / 2)


def courtyard() -> tuple[float, float, float, float]:
    """中庭净空 (x0, x1, z0, z1)：四条连排最深那间的内沿围出来的方框。"""
    return (-BACK + _depth("W"), BACK - _depth("E"),
            -BACK + _depth("S"), BACK - _depth("N"))


def patrol_ring(inset: float = 1.8) -> list[tuple[float, float]]:
    """巡逻环：中庭四角内缩一点。狱警/狱犬的路线由它派生。

    以前巡逻路点是照 ±30 的老场地手写的六个坐标。场地缩到 ±19.8 之后那些点
    落进了房子里，狱警会一头顶着墙走——巡逻线必须跟着布局算，不能手抄。
    环走在放风场铁栅栏和房子之间那条带子上（两侧各留三四米），所以既扫得到
    中庭，也不会插进铁栅栏。
    """
    x0, x1, z0, z1 = courtyard()
    return [(x0 + inset, z0 + inset), (x1 - inset, z0 + inset),
            (x1 - inset, z1 - inset), (x0 + inset, z1 - inset)]


def cells() -> list[dict]:
    """牢房连排的六间隔间，沿前墙从"沿墙负向"到正向依次排列。

    每间是一个独立的封闭小间，只有朝中庭那面留一道栅栏门（`door_x/door_z` 是门心）。
    NPC 睡觉/歇着去自己那间，所以这些门必须真能走通——不再像上一版那样是几块
    不可达的空腔。门能上锁，锁上之后寻路网格里那一格就封了（见 backend/app/nav.py）。

    `slot` 是编号，world.py 按囚犯名册顺序分房；最后一间留给主角。
    """
    fac = BY_ID["cell"]
    fx, fz = FACE_VEC[fac["face"]]
    ax, az = -fz, fx                       # 沿前墙方向
    out: list[dict] = []
    for i in range(CELL_N):
        off = -fac["w"] / 2 + CELL_W * (i + 0.5)
        cx, cz = fac["x"] + ax * off, fac["z"] + az * off
        out.append({
            "slot": i,
            "x": round(cx, 3), "z": round(cz, 3),
            "w": CELL_W, "d": CELL_D,
            "face": fac["face"],
            "door_x": round(cx + fx * CELL_D / 2, 3),
            "door_z": round(cz + fz * CELL_D / 2, 3),
        })
    return out


CELLS: list[dict] = cells()

SOLITARY_XZ: tuple[float, float] = (BY_ID["solitary"]["x"], BY_ID["solitary"]["z"])
