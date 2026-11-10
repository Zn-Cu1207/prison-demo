"""寻路网格：后端也得绕墙走。

## 为什么要有这个模块

在这之前，后端 `_move_toward` 是**朝目标走直线**、完全不认墙，而前端有自己的一套
A* 绕墙。同一个人，两边走的是两条路，于是同时出两种毛病：

  · 穿墙：后端权威坐标会落在墙里、屋里。前端有碰撞兜底，但它"卡住"时会朝
    权威坐标硬顶一步，硬顶的方向就是墙里 —— 于是肉眼可见地蹭进墙。
  · 落后：绕路的一方走得远、直线的一方走得近，行程越长偏差越大。贴墙连排那版
    量出来最多落后 19 米，人还在院子里走，决策卡已经写着他在健身；典狱长走到
    眼前，后端也不认为他在跟前（敬礼/监视判定用的是后端坐标）。

两边各走一条路，怎么调都是治标。所以把绕墙这件事下沉到后端：**后端出路径，
前端跟着走**。路线同一条，落后只剩前后端步速差那一点，穿墙则从源头上不再发生。

## 网格怎么建

·  0.35m 一格，覆盖整个场地（±WALL）。格子够细，1.5m 的门洞才留得出通路。
·  房间是**空心**的：只有墙那一圈算障碍，屋里可以站人。墙带按前端 makeBuilding
   的做法取"沿占地边界往里 WALL_T"，再朝两侧各膨胀一个身宽半径 BODY_R——
   身体是有粗细的，圆心离墙面不能小于半径。
·  门洞在墙带上凿开：净宽要扣掉两个身宽半径，所以 1.5m 的门只剩 0.66m 可走，
   刚好两格。牢房那六间的门是整面栅栏（2.05m），宽裕得多。
·  牢门可以锁。锁上就把那道门重新填实（`set_gate`），于是网格里出不去也进不来——
   典狱长的"关禁闭"因此是货真价实的物理封锁，不是状态标记。

坐标与格子的换算全部走 `_ix/_wx`，别在外面手算。
"""
from __future__ import annotations

import heapq
import math

from . import layout as L

GRID = 0.35            # 格边长（米）
WALL_T = 0.25          # 墙厚，与前端 makeBuilding 一致
BODY_R = 0.42          # 身宽半径，与前端 BODY_R 一致
MARGIN = WALL_T + BODY_R

NX = int(math.ceil(2 * L.WALL / GRID))          # 每边格数（正方形场地）

SOLITARY_GATE = -1    # 禁闭室的门在 _gates 里用这个哨兵键（牢房用 0..5）

_DIAG = math.sqrt(2.0)
# 八向邻居：(dx, dz, 代价)。斜向要求两个正交方向都通，否则会从墙角对角线穿过去。
_NEIGH = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, _DIAG), (1, -1, _DIAG), (-1, 1, _DIAG), (-1, -1, _DIAG))

Rect = tuple[float, float, float, float]


def _ix(v: float) -> int:
    """世界坐标 → 格号（钳在场地内）。"""
    return max(0, min(NX - 1, int((v + L.WALL) / GRID)))


def _wx(i: int) -> float:
    """格号 → 格心的世界坐标。"""
    return -L.WALL + (i + 0.5) * GRID


def _inflate(r: Rect, m: float) -> Rect:
    return (r[0] - m, r[1] + m, r[2] - m, r[3] + m)


def _in(r: Rect, x: float, z: float) -> bool:
    return r[0] <= x <= r[1] and r[2] <= z <= r[3]


class Nav:
    """场地寻路网格。整个进程一份（见模块末尾的 NAV），牢门状态可以随时改。"""

    def __init__(self) -> None:
        self._base = bytearray(NX * NX)          # 1 = 不可走
        self._grid = bytearray(NX * NX)          # base + 当前锁着的牢门
        self._gates: dict[int, Rect] = {}        # slot → 那道栅栏门要填实的格子范围
        self._locked: set[int] = set()
        self._build()

    # -- 建网格 -------------------------------------------------------------
    def _build(self) -> None:
        walls: list[Rect] = []       # 空心房间的占地（只有边界那圈算障碍）
        solids: list[Rect] = []      # 实心障碍（电话亭、铁闸）
        gaps: list[Rect] = []        # 要凿开的门洞（世界坐标下的一小块）

        for f in L.FACILITIES:
            kind = f["kind"]
            if kind == "yard":
                continue                          # 露天场地，不挡路
            if kind == "sleep":
                continue                          # 牢房按隔间单独处理
            fp = L.footprint(f)
            if kind in ("phone", "gate"):
                solids.append(fp)
                continue
            walls.append(fp)
            gap = self._door_gap(f)
            gaps.append(gap)
            if kind == "solitary":          # 禁闭室的门也能被典狱长锁上（见 set_gate）
                self._gates[SOLITARY_GATE] = gap

        # 牢房六间：各自三面实墙 + 整面栅栏门
        for c in L.CELLS:
            fp = (c["x"] - c["d"] / 2, c["x"] + c["d"] / 2,
                  c["z"] - c["w"] / 2, c["z"] + c["w"] / 2)
            if c["face"] in ("N", "S"):           # 朝向南北时宽度沿 x
                fp = (c["x"] - c["w"] / 2, c["x"] + c["w"] / 2,
                      c["z"] - c["d"] / 2, c["z"] + c["d"] / 2)
            walls.append(fp)
            gate = self._gate_gap(c)
            gaps.append(gate)
            self._gates[c["slot"]] = gate

        for i in range(NX):
            x = _wx(i)
            for j in range(NX):
                z = _wx(j)
                blocked = abs(x) > L.WALL - MARGIN or abs(z) > L.WALL - MARGIN
                if not blocked:
                    for r in solids:
                        if _in(_inflate(r, BODY_R), x, z):
                            blocked = True
                            break
                if not blocked:
                    for r in walls:
                        # 墙带 = 膨胀后的外框内、缩进后的内框外
                        if _in(_inflate(r, MARGIN), x, z) and not _in(_inflate(r, -MARGIN), x, z):
                            blocked = True
                            break
                if blocked:
                    self._base[j * NX + i] = 1

        for g in gaps:                            # 凿门：门洞上的格子一律放通
            self._carve(g)
        self._grid[:] = self._base

    def _door_gap(self, fac: dict) -> Rect:
        """常规房间的门洞：沿朝向穿透整条墙带，沿门宽扣掉两个身宽半径。"""
        dw = min(1.5, fac["w"] * 0.42)
        fx, fz = L.FACE_VEC[fac["face"]]
        cx = fac["x"] + fx * fac["d"] / 2
        cz = fac["z"] + fz * fac["d"] / 2
        half = max(GRID / 2, dw / 2 - BODY_R)     # 净宽；太窄就退到一格，保证不封死
        return self._slab(cx, cz, fx, fz, half)

    def _gate_gap(self, cell: dict) -> Rect:
        """牢房栅栏门：整面前墙，只在两侧各留下墙垛。"""
        fx, fz = L.FACE_VEC[cell["face"]]
        half = max(GRID / 2, cell["w"] / 2 - WALL_T - BODY_R)
        return self._slab(cell["door_x"], cell["door_z"], fx, fz, half)

    @staticmethod
    def _slab(cx: float, cz: float, fx: float, fz: float, half: float) -> Rect:
        """以 (cx,cz) 为门心、沿朝向穿透墙带的一小块。"""
        depth = MARGIN + GRID                     # 穿透整条墙带（含两侧膨胀）
        ax, az = -fz, fx                          # 沿门宽方向
        xs = (cx - abs(ax) * half - abs(fx) * depth, cx + abs(ax) * half + abs(fx) * depth)
        zs = (cz - abs(az) * half - abs(fz) * depth, cz + abs(az) * half + abs(fz) * depth)
        return (xs[0], xs[1], zs[0], zs[1])

    def _carve(self, r: Rect, blocked: int = 0) -> None:
        for i in range(_ix(r[0]), _ix(r[1]) + 1):
            for j in range(_ix(r[2]), _ix(r[3]) + 1):
                self._base[j * NX + i] = blocked

    # -- 牢门 ---------------------------------------------------------------
    def set_gate(self, slot: int, locked: bool) -> None:
        """锁上/打开某间牢房的栅栏门。锁上之后网格里那道门就是实墙。"""
        if locked:
            self._locked.add(slot)
        else:
            self._locked.discard(slot)
        self._grid[:] = self._base
        for s in self._locked:
            r = self._gates.get(s)
            if not r:
                continue
            for i in range(_ix(r[0]), _ix(r[1]) + 1):
                for j in range(_ix(r[2]), _ix(r[3]) + 1):
                    self._grid[j * NX + i] = 1

    # -- 查询 ---------------------------------------------------------------
    def walkable(self, x: float, z: float) -> bool:
        return not self._grid[_ix(z) * NX + _ix(x)]

    def nearest_walkable(self, x: float, z: float, radius: int = 8) -> tuple[float, float]:
        """把落在墙里的点拽到最近的可走格。被锁在牢里的人也能借此站稳。"""
        if self.walkable(x, z):
            return x, z
        ci, cj = _ix(x), _ix(z)
        for r in range(1, radius + 1):
            best = None
            bd = 1e9
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = ci + di, cj + dj
                    if not (0 <= i < NX and 0 <= j < NX) or self._grid[j * NX + i]:
                        continue
                    d = (di * GRID) ** 2 + (dj * GRID) ** 2
                    if d < bd:
                        bd, best = d, (_wx(i), _wx(j))
            if best:
                return best
        return x, z

    def _clear_line(self, x1: float, z1: float, x2: float, z2: float) -> bool:
        """两点之间能不能直着走（按格子采样）。"""
        d = math.hypot(x2 - x1, z2 - z1)
        n = int(d / (GRID * 0.5)) + 1
        for k in range(n + 1):
            t = k / n
            if not self.walkable(x1 + (x2 - x1) * t, z1 + (z2 - z1) * t):
                return False
        return True

    def path(self, x: float, z: float, tx: float, tz: float) -> list[tuple[float, float]]:
        """A* 求路径点（不含起点，最后一个是终点）。走不通时返回空列表。

        直线能到就直接返回终点——绝大多数在院子里的移动都走这条捷径，不进 A*。
        """
        if self._clear_line(x, z, tx, tz):
            return [(tx, tz)]

        sx, sz = self.nearest_walkable(x, z)
        gx, gz = self.nearest_walkable(tx, tz)
        s = (_ix(sx), _ix(sz))
        g = (_ix(gx), _ix(gz))
        if s == g:
            return [(tx, tz)]

        def h(i: int, j: int) -> float:
            return math.hypot(i - g[0], j - g[1])

        open_q: list[tuple[float, float, tuple[int, int]]] = [(h(*s), 0.0, s)]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        cost: dict[tuple[int, int], float] = {s: 0.0}
        found = False
        while open_q:
            _f, gc, cur = heapq.heappop(open_q)
            if cur == g:
                found = True
                break
            if gc > cost.get(cur, 1e18):
                continue
            ci, cj = cur
            for di, dj, w in _NEIGH:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < NX and 0 <= nj < NX) or self._grid[nj * NX + ni]:
                    continue
                if di and dj and (self._grid[cj * NX + ni] or self._grid[nj * NX + ci]):
                    continue                      # 不许贴着墙角斜穿
                ng = gc + w
                if ng < cost.get((ni, nj), 1e18):
                    cost[(ni, nj)] = ng
                    came[(ni, nj)] = cur
                    heapq.heappush(open_q, (ng + h(ni, nj), ng, (ni, nj)))
        if not found:
            return []

        cells = [g]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        cells.reverse()
        pts = [(_wx(i), _wx(j)) for i, j in cells[1:]]
        if pts and self.walkable(tx, tz):
            pts[-1] = (tx, tz)                    # 终点用真实坐标，别停在格心上
        # 终点本身在墙里（剧情落点算出来的"踩点大门""盯梢某人"就会这样）时，
        # 停在最近的可走格上——绝不能把不可走的原始坐标当终点，那等于走进墙里。
        return self._smooth(x, z, pts)

    def _smooth(self, x: float, z: float, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """拉直：能一步看到的中间点全部跳过，省下贴着格子走的锯齿。"""
        out: list[tuple[float, float]] = []
        cx, cz = x, z
        i = 0
        while i < len(pts):
            j = len(pts) - 1
            while j > i and not self._clear_line(cx, cz, *pts[j]):
                j -= 1
            out.append(pts[j])
            cx, cz = pts[j]
            i = j + 1
        return out


NAV = Nav()
