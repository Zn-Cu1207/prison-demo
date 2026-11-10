"""报告 GLB 的骨架/动画/贴图概况——接入新角色前的体检单。

inspect_glb.py 只看几何（包围盒、三角数）。但每接一个 Mixamo 导出的角色，
真正会出问题的是另外三件事，每一件都在之前的角色上翻过车：

  1. 多具骨架：Mixamo 每个动作导进来各带一具 Armature。动画绑在看不见的那几具上，
     人物就成了 T 字形平移（hero.glb 当时有 4 具）。
  2. 骨骼名前缀不一致：同一个文件里可能混着 mixamorig: 和 mixamorig5:，
     按原名重定向会静默丢通道（madman.glb 踩过）。
  3. 根位移：Hips 的 translation 通道会把人物推离脚下位置，
     再被游戏逻辑拉回来，看起来就是瞬移。

另外报告每个片段的原始步幅与时长——这是算 pace / clipTimeScale 的输入，
不对齐就会出现"脚在地上贴着平移"。

用法：python inspect_glb_rig.py <glb>
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

BONE_PREFIX = re.compile(r"^mixamorig\d*[:_]?")


def load(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise SystemExit(f"{path} 不是 GLB")
    gltf, bin_chunk = None, b""
    off = 12
    while off < len(raw):
        length, kind = struct.unpack_from("<II", raw, off)
        body = raw[off + 8 : off + 8 + length]
        if kind == 0x4E4F534A:
            gltf = json.loads(body)
        elif kind == 0x004E4942:
            bin_chunk = body
        off += 8 + length + (-length % 4)
    if gltf is None:
        raise SystemExit("缺少 JSON chunk")
    return gltf, bin_chunk


_COMP = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_accessor(gltf: dict, blob: bytes, index: int) -> list[tuple[float, ...]]:
    acc = gltf["accessors"][index]
    n = _COUNT[acc["type"]]
    fmt = _COMP[acc["componentType"]]
    view = gltf["bufferViews"][acc["bufferView"]]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    size = struct.calcsize("<" + fmt)
    stride = view.get("byteStride") or size * n
    out = []
    for i in range(acc["count"]):
        at = base + i * stride
        out.append(struct.unpack_from("<" + fmt * n, blob, at))
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    gltf, blob = load(path)
    nodes = gltf.get("nodes", [])
    name_of = {i: n.get("name", f"#{i}") for i, n in enumerate(nodes)}

    print(f"== {path.name} ==")

    skins = gltf.get("skins", [])
    print(f"\n[骨架] {len(skins)} 具" + ("  ← 多于 1 具需要清理" if len(skins) > 1 else ""))
    for i, sk in enumerate(skins):
        joints = sk.get("joints", [])
        prefixes = sorted({BONE_PREFIX.match(name_of[j]).group(0)
                           if BONE_PREFIX.match(name_of[j]) else "(无前缀)"
                           for j in joints})
        print(f"  skin{i}: {len(joints)} 根骨骼  前缀={prefixes}")

    meshes = gltf.get("meshes", [])
    print(f"\n[网格] {len(meshes)} 个")
    for m in meshes:
        for p in m.get("primitives", []):
            attrs = sorted(p.get("attributes", {}))
            missing = [a for a in ("NORMAL", "TEXCOORD_0") if a not in attrs]
            flag = f"  ← 缺 {', '.join(missing)}（贴图会渲不出来）" if missing else ""
            print(f"  {m.get('name','?')}: {attrs}{flag}")

    anims = gltf.get("animations", [])
    print(f"\n[动画] {len(anims)} 段")
    for a in anims:
        name = a.get("name", "?")
        dur = 0.0
        hips_disp = None
        targets: set[int] = set()
        for ch in a.get("channels", []):
            node = ch["target"].get("node")
            if node is None:
                continue
            targets.add(node)
            samp = a["samplers"][ch["sampler"]]
            times = read_accessor(gltf, blob, samp["input"])
            if times:
                dur = max(dur, times[-1][0])
            bone = BONE_PREFIX.sub("", name_of.get(node, ""))
            if bone == "Hips" and ch["target"]["path"] == "translation":
                vals = read_accessor(gltf, blob, samp["output"])
                if len(vals) > 1:
                    hips_disp = vals
        # 动画影响到哪些骨架：跨骨架就是"动画绑在看不见的骨架上"那个坑
        touched = []
        for i, sk in enumerate(skins):
            if targets & set(sk.get("joints", [])):
                touched.append(f"skin{i}")
        extra = ""
        if hips_disp:
            # 竖直轴：静置时 Hips 离原点最远的那个分量（Mixamo 有 Y-up / Z-up 两种约定）
            rest = hips_disp[0]
            up = max(range(3), key=lambda i: abs(rest[i]))
            flat = [i for i in range(3) if i != up]
            flat_path = sum(
                sum((b[i] - a2[i]) ** 2 for i in flat) ** 0.5
                for a2, b in zip(hips_disp, hips_disp[1:])
            )
            hi = [v[up] for v in hips_disp]
            mean_h = sum(hi) / len(hi)
            # 认片段的判据（Mixamo 导出后名字全丢了，只能靠数字认）：
            #   平均髋高低一截 → 跪姿；首末髋高从高掉到低 → 倒地/死亡；
            #   髋高平稳且水平路径长 → 走/跑。
            #
            # 但"走"和"挥砍/受击"都可能既有水平位移又有髋部起伏，光看上面三条会认错。
            # 再加一条步态判据：数髋高的方向反转次数，两次反转＝一个起伏周期。
            # 走/跑是周期步态，每秒约 2 个周期；挥砍、倒地这种单次动作全程只有 0~1 个。
            # 死区取幅值的 5%，否则采样噪声会把一条平线数出几十次反转。
            extra = (f"  髋高 均{mean_h:6.1f} 首{hi[0]:6.1f}→末{hi[-1]:6.1f}"
                     f" 幅{max(hi)-min(hi):5.1f}  水平路径{flat_path:7.1f}")
            amp = max(hi) - min(hi)
            dead = amp * 0.05
            sign, flips = 0, 0
            for a3, b in zip(hi, hi[1:]):
                v = b - a3
                if abs(v) < dead:
                    continue
                s = 1 if v > 0 else -1
                if sign and s != sign:
                    flips += 1
                sign = s
            if dur > 0.1:
                extra += (f"  起伏{flips / 2:4.1f}周期({flips / 2 / dur:4.1f}/s)"
                          f"  速度{flat_path / dur:6.1f}/s")
        print(f"  {name:<18} {dur:6.2f}s 通道{len(a.get('channels',[])):4d} "
              f"{','.join(touched) or '无骨架':<6}{extra}")

    imgs = gltf.get("images", [])
    print(f"\n[贴图] {len(imgs)} 张")
    for im in imgs:
        bv = im.get("bufferView")
        kb = 0
        if bv is not None:
            kb = gltf["bufferViews"][bv].get("byteLength", 0) / 1024
        print(f"  {im.get('name','?'):<32} {im.get('mimeType','?'):<12} {kb:8.0f} KB")
    total = sum(gltf["bufferViews"][im["bufferView"]].get("byteLength", 0)
                for im in imgs if im.get("bufferView") is not None)
    print(f"  贴图合计 {total/1024/1024:.1f} MB / 文件 {Path(sys.argv[1]).stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
