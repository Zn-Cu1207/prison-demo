"""看一眼 GLB 里到底有什么：尺寸、材质、片段、三角数。

用法：
    python frontend/tools/inspect_glb.py frontend/assets/env/*.glb
    python frontend/tools/inspect_glb.py frontend/assets/env/SM_Wall_PrisonStone_05_Straight.glb -v

引环境资产时先跑它：模块的真实尺寸（贴合网格用）、材质基色（配色/打光用）都在这儿。
"""

from __future__ import annotations

import glob
import json
import struct
import sys


def load(path: str) -> dict:
    data = open(path, "rb").read()
    if data[:4] != b"glTF":
        raise SystemExit(f"{path} 不是 glb")
    json_len = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + json_len])


def mat_of(node: dict) -> list[list[float]]:
    """节点局部变换 → 4x4（行优先）。glTF 的 matrix 是列优先，需转置。"""
    if "matrix" in node:
        m = node["matrix"]
        return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    tx, ty, tz = node.get("translation", [0.0, 0.0, 0.0])
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    sx, sy, sz = node.get("scale", [1.0, 1.0, 1.0])
    rot = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    scale = (sx, sy, sz)
    out = [[rot[r][c] * scale[c] for c in range(3)] + [(tx, ty, tz)[r]] for r in range(3)]
    return out + [[0.0, 0.0, 0.0, 1.0]]


def mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def apply(m: list[list[float]], p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    return tuple(m[r][0] * x + m[r][1] * y + m[r][2] * z + m[r][3] for r in range(3))  # type: ignore[return-value]


def world_boxes(gltf: dict) -> tuple[list[float], list[float], int, list[str]]:
    nodes = gltf.get("nodes", [])
    parent: dict[int, int] = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    tris = 0
    mesh_names: list[str] = []
    for i, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        m = mat_of(n)
        k = i
        while k in parent:
            k = parent[k]
            m = mul(mat_of(nodes[k]), m)
        mesh = gltf["meshes"][n["mesh"]]
        mesh_names.append(mesh.get("name") or n.get("name") or "?")
        for prim in mesh["primitives"]:
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            if "indices" in prim:
                tris += gltf["accessors"][prim["indices"]]["count"] // 3
            else:
                tris += acc["count"] // 3
            a, b = acc["min"], acc["max"]
            for cx in (a[0], b[0]):
                for cy in (a[1], b[1]):
                    for cz in (a[2], b[2]):
                        p = apply(m, (cx, cy, cz))
                        for d in range(3):
                            lo[d] = min(lo[d], p[d])
                            hi[d] = max(hi[d], p[d])
    return lo, hi, tris, mesh_names


def hexcol(c: list[float]) -> str:
    """glTF 基色是线性空间，转 sRGB 再给十六进制——照抄进前端才不会偏色。"""
    def srgb(v: float) -> int:
        v = max(0.0, min(1.0, v))
        s = 12.92 * v if v <= 0.0031308 else 1.055 * (v ** (1 / 2.4)) - 0.055
        return round(s * 255)
    r, g, b = (srgb(v) for v in c[:3])
    return f"0x{r:02x}{g:02x}{b:02x}"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    paths: list[str] = []
    for a in args:
        paths.extend(sorted(glob.glob(a)) or [a])
    if not paths:
        raise SystemExit(__doc__)
    for path in paths:
        gltf = load(path)
        lo, hi, tris, mesh_names = world_boxes(gltf)
        size = [hi[d] - lo[d] for d in range(3)]
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        print(f"{name:<38} {size[0]:6.2f} x {size[1]:6.2f} x {size[2]:6.2f} m"
              f"  底 y={lo[1]:+.3f}  三角 {tris:5d}  节点 {len(gltf.get('nodes', []))}"
              f"  材质 {len(gltf.get('materials', []))}"
              + ("  贴图 %d" % len(gltf["images"]) if gltf.get("images") else ""))
        if verbose:
            print(f"    中心 x={(lo[0]+hi[0])/2:+.3f} z={(lo[2]+hi[2])/2:+.3f}"
                  f"  x∈[{lo[0]:+.2f},{hi[0]:+.2f}] z∈[{lo[2]:+.2f},{hi[2]:+.2f}]")
            for mat in gltf.get("materials", []):
                pbr = mat.get("pbrMetallicRoughness", {})
                base = pbr.get("baseColorFactor", [1, 1, 1, 1])
                print(f"    材质 {mat.get('name','?'):<28} 基色 {hexcol(base)}"
                      f" rough={pbr.get('roughnessFactor', 1.0):.2f}"
                      f" metal={pbr.get('metallicFactor', 1.0):.2f}"
                      + ("  自发光 %s" % hexcol(mat["emissiveFactor"])
                         if mat.get("emissiveFactor", [0, 0, 0]) != [0, 0, 0] else ""))
            if mesh_names:
                print("    网格 " + ", ".join(mesh_names[:12])
                      + (f" …共 {len(mesh_names)}" if len(mesh_names) > 12 else ""))


if __name__ == "__main__":
    main()
