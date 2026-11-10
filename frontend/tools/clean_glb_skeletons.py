"""把 Mixamo + Blender 导出的"多骨架" GLB 收敛成单骨架。

在 Blender 里逐个导入动作 FBX 时，每个动作都会带进一具自己的骨架。若导出前没删掉，
glb 里就会有多具骨架：带皮的网格只绑在其中一具上，其余动画驱动的是"隐形骨架"，
表现为人物停在 T-pose 被平移拖着走。

本脚本按骨骼名把所有动画通道重定向到网格真正绑定的那具骨架，再删掉多余骨架与 skin，
并支持顺手重命名/丢弃动画片段。二进制块原样保留（孤立的 accessor 字节量极小）。

用法：
    python frontend/tools/clean_glb_skeletons.py frontend/assets/hero.glb \
        --rename "Armature.001|mixamo.com|Layer0=idle" \
        --drop "Armature|mixamo.com|Layer0"
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path

# Mixamo 同一套骨骼在不同次导出里前缀可能带编号（mixamorig: / mixamorig5: ...），
# 甚至同一个 blend 文件里本体和动作骨架的前缀都不一样，按名匹配前先去掉前缀。
BONE_PREFIX = re.compile(r"^mixamorig\d*:")


def bone_key(name: str | None) -> str:
    return BONE_PREFIX.sub("", name or "")

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, _version, total = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:
        raise SystemExit(f"{path} 不是 GLB 文件")
    off, gltf, binary = 12, None, b""
    while off < total:
        length, ctype = struct.unpack("<II", data[off : off + 8])
        chunk = data[off + 8 : off + 8 + length]
        if ctype == JSON_CHUNK:
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype == BIN_CHUNK:
            binary = chunk
        off += 8 + length
    if gltf is None:
        raise SystemExit("GLB 缺少 JSON 块")
    return gltf, binary


def write_glb(path: Path, gltf: dict, binary: bytes) -> None:
    js = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    bn = binary + b"\x00" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(js) + (8 + len(bn) if bn else 0)
    out = bytearray(struct.pack("<III", 0x46546C67, 2, total))
    out += struct.pack("<II", len(js), JSON_CHUNK) + js
    if bn:
        out += struct.pack("<II", len(bn), BIN_CHUNK) + bn
    path.write_bytes(bytes(out))


def drop_meshes(gltf: dict, patterns: set[str]) -> int:
    """按网格名或材质名删掉整段几何，并摘掉随之无人引用的材质/贴图/图片。

    用途是角色自带的道具。疤脸这具 MaleBruteA 手里攥着一把斧头（6664 面 + 5.4MB 贴图），
    但游戏里凶器只有毒药/匕首/电击枪三种、且靠决策卡文字呈现，场上没有武器模型；
    他的任务链还是空手开局、第二步才去弄家伙。一把常驻的斧头会同时废掉
    "去弄凶器"这条剧情节拍和典狱长"没收凶器"这个操作。

    只摘引用，字节由后面的 gc_buffers 统一回收。
    """
    meshes = gltf.get("meshes", [])
    mats = gltf.get("materials", [])
    hit = 0
    keep_mesh: list[int] = []
    for mi, mesh in enumerate(meshes):
        prims = []
        for p in mesh.get("primitives", []):
            mat_name = mats[p["material"]].get("name", "") if "material" in p else ""
            if (mesh.get("name") or "") in patterns or mat_name in patterns:
                print(f"  删掉网格 {mesh.get('name')!r} / 材质 {mat_name!r}")
                hit += 1
                continue
            prims.append(p)
        mesh["primitives"] = prims
        if prims:
            keep_mesh.append(mi)
    if not hit:
        return 0

    # 网格重排：整段空掉的网格移出列表，引用它的节点摘掉 mesh/skin（后续 clean 就不会留它）
    mesh_remap = {old: new for new, old in enumerate(keep_mesh)}
    gltf["meshes"] = [meshes[i] for i in keep_mesh]
    for node in gltf.get("nodes", []):
        if "mesh" not in node:
            continue
        if node["mesh"] in mesh_remap:
            node["mesh"] = mesh_remap[node["mesh"]]
        else:
            node.pop("mesh", None)
            node.pop("skin", None)

    # 材质 → 贴图 → 图片，逐层摘掉没人引用的
    used_mat = {p["material"] for m in gltf["meshes"] for p in m["primitives"] if "material" in p}
    mat_remap = {old: new for new, old in enumerate(sorted(used_mat))}
    gltf["materials"] = [mats[i] for i in sorted(used_mat)]
    for mesh in gltf["meshes"]:
        for p in mesh["primitives"]:
            if "material" in p:
                p["material"] = mat_remap[p["material"]]

    def tex_refs(node, out: set[int]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and key.endswith("Texture") and "index" in value:
                    out.add(value["index"])
                tex_refs(value, out)
        elif isinstance(node, list):
            for v in node:
                tex_refs(v, out)

    used_tex: set[int] = set()
    for m in gltf["materials"]:
        tex_refs(m, used_tex)
    texs = gltf.get("textures", [])
    tex_remap = {old: new for new, old in enumerate(sorted(used_tex))}
    gltf["textures"] = [texs[i] for i in sorted(used_tex)]

    def retarget(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and key.endswith("Texture") and "index" in value:
                    value["index"] = tex_remap[value["index"]]
                retarget(value)
        elif isinstance(node, list):
            for v in node:
                retarget(v)

    for m in gltf["materials"]:
        retarget(m)

    used_img = {t["source"] for t in gltf["textures"] if "source" in t}
    imgs = gltf.get("images", [])
    img_remap = {old: new for new, old in enumerate(sorted(used_img))}
    for name in {(imgs[i].get("name") or "?") for i in range(len(imgs)) if i not in used_img}:
        print(f"  摘掉贴图 {name}")
    gltf["images"] = [imgs[i] for i in sorted(used_img)]
    for t in gltf["textures"]:
        if "source" in t:
            t["source"] = img_remap[t["source"]]
    return hit


def gc_buffers(gltf: dict, binary: bytes) -> bytes:
    """回收没人引用的 accessor / bufferView，重建 BIN 块。

    删网格、删动画之后它们的 accessor 就成了孤儿。原先的做法是"二进制块原样保留"，
    孤儿只有几十 KB 时无所谓；但删掉一把带 4K 贴图的斧头之后孤儿有好几 MB，
    下载下来纯属浪费。所以做一次可达性回收：从网格/动画/skin/图片出发标记，其余丢弃。
    """
    accs = gltf.get("accessors", [])
    views = gltf.get("bufferViews", [])

    used_acc: set[int] = set()
    for mesh in gltf.get("meshes", []):
        for p in mesh.get("primitives", []):
            used_acc.update(p.get("attributes", {}).values())
            if "indices" in p:
                used_acc.add(p["indices"])
            for t in p.get("targets", []):
                used_acc.update(t.values())
    for anim in gltf.get("animations", []):
        for s in anim.get("samplers", []):
            used_acc.add(s["input"])
            used_acc.add(s["output"])
    for skin in gltf.get("skins", []):
        if "inverseBindMatrices" in skin:
            used_acc.add(skin["inverseBindMatrices"])

    used_view = {accs[i]["bufferView"] for i in used_acc if "bufferView" in accs[i]}
    used_view |= {im["bufferView"] for im in gltf.get("images", []) if "bufferView" in im}

    acc_remap = {old: new for new, old in enumerate(sorted(used_acc))}
    view_remap = {old: new for new, old in enumerate(sorted(used_view))}

    out = bytearray()
    new_views = []
    for old in sorted(used_view):
        view = dict(views[old])
        start = view.get("byteOffset", 0)
        blob = bytes(binary[start : start + view["byteLength"]])
        pad = (4 - len(out) % 4) % 4          # bufferView 之间四字节对齐，否则 accessor 读错位
        out += b"\x00" * pad
        view["byteOffset"] = len(out)
        out += blob
        new_views.append(view)

    new_accs = []
    for old in sorted(used_acc):
        acc = dict(accs[old])
        if "bufferView" in acc:
            acc["bufferView"] = view_remap[acc["bufferView"]]
        new_accs.append(acc)

    for mesh in gltf.get("meshes", []):
        for p in mesh.get("primitives", []):
            p["attributes"] = {k: acc_remap[v] for k, v in p.get("attributes", {}).items()}
            if "indices" in p:
                p["indices"] = acc_remap[p["indices"]]
            if p.get("targets"):
                p["targets"] = [{k: acc_remap[v] for k, v in t.items()} for t in p["targets"]]
    for anim in gltf.get("animations", []):
        for s in anim.get("samplers", []):
            s["input"] = acc_remap[s["input"]]
            s["output"] = acc_remap[s["output"]]
    for skin in gltf.get("skins", []):
        if "inverseBindMatrices" in skin:
            skin["inverseBindMatrices"] = acc_remap[skin["inverseBindMatrices"]]
    for im in gltf.get("images", []):
        if "bufferView" in im:
            im["bufferView"] = view_remap[im["bufferView"]]

    print(f"字节回收：accessor {len(accs)} -> {len(new_accs)}，"
          f"bufferView {len(views)} -> {len(new_views)}，"
          f"BIN {len(binary) / 1048576:.2f} MB -> {len(out) / 1048576:.2f} MB")
    gltf["accessors"] = new_accs
    gltf["bufferViews"] = new_views
    gltf["buffers"] = [{"byteLength": len(out)}]
    return bytes(out)


def clean(gltf: dict, renames: dict[str, str], drops: set[str]) -> dict:
    nodes, skins = gltf["nodes"], gltf.get("skins", [])
    if not skins:
        raise SystemExit("没有 skin，无需处理")

    # 带皮网格绑在哪具骨架上 —— 这是唯一要保留的那具
    mesh_nodes = [i for i, n in enumerate(nodes) if "mesh" in n and "skin" in n]
    if not mesh_nodes:
        raise SystemExit("找不到带 skin 的网格节点")
    keep_skin = nodes[mesh_nodes[0]]["skin"]
    joints = skins[keep_skin]["joints"]
    name2joint = {bone_key(nodes[j].get("name")): j for j in joints}
    joint_set = set(joints)
    print(f"保留 skin {keep_skin}（{len(joints)} 根骨骼），网格节点 {mesh_nodes}")

    # 1) 动画通道按骨骼名重定向到保留骨架；保留骨架里没有的骨骼，通道直接丢弃
    kept_anims, dropped_ch = [], 0
    for anim in gltf.get("animations", []):
        name = anim.get("name", "")
        if name in drops:
            print(f"  丢弃片段 {name!r}")
            continue
        channels = []
        for ch in anim["channels"]:
            target = ch["target"].get("node")
            if target is None:
                continue
            if target not in joint_set:
                mapped = name2joint.get(bone_key(nodes[target].get("name")))
                if mapped is None:
                    dropped_ch += 1
                    continue
                ch["target"]["node"] = mapped
            channels.append(ch)
        anim["channels"] = channels
        if name in renames:
            print(f"  片段改名 {name!r} -> {renames[name]!r}")
            anim["name"] = renames[name]
        kept_anims.append(anim)
    gltf["animations"] = kept_anims
    if dropped_ch:
        print(f"  丢弃了 {dropped_ch} 条指向不存在骨骼的通道")

    # 2) 计算要保留的节点：网格 + 保留骨架的骨骼 + 它们的所有祖先
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    keep = joint_set | set(mesh_nodes)
    for start in list(keep):
        cur = parent.get(start)
        while cur is not None and cur not in keep:
            keep.add(cur)
            cur = parent.get(cur)

    # 3) 重建索引
    order = sorted(keep)
    remap = {old: new for new, old in enumerate(order)}
    print(f"节点 {len(nodes)} -> {len(order)}；skin {len(skins)} -> 1")

    new_nodes = []
    for old in order:
        node = dict(nodes[old])
        children = [remap[c] for c in node.get("children", []) if c in remap]
        if children:
            node["children"] = children
        else:
            node.pop("children", None)
        if "skin" in node:
            node["skin"] = 0
        new_nodes.append(node)
    gltf["nodes"] = new_nodes

    skin = dict(skins[keep_skin])
    skin["joints"] = [remap[j] for j in skin["joints"]]
    if "skeleton" in skin:
        skin["skeleton"] = remap[skin["skeleton"]]
    gltf["skins"] = [skin]

    for scene in gltf.get("scenes", []):
        scene["nodes"] = [remap[n] for n in scene.get("nodes", []) if n in remap]
    for anim in gltf["animations"]:
        for ch in anim["channels"]:
            ch["target"]["node"] = remap[ch["target"]["node"]]

    return gltf


def bake_in_place(gltf: dict, binary: bytes, names: set[str]) -> bytes:
    """把指定片段的根骨水平位移锁到静置位置，只保留竖直方向的重心起伏。

    人物在场景里的位置由游戏逻辑（后端权威位置）驱动，动画若再带根位移，人就会走出
    自己脚下的位置、循环回头时又弹回原点，看起来像瞬移。

    竖直轴不能假定是 Y：Blender 导出的骨架带轴向转换，实测这类文件里根骨的竖直分量在
    Z 上（rest 约 -108，即胯部离地高度），前后位移反而在 Y 上。因此按 rest 平移里
    绝对值最大的分量判定竖直轴（胯高远大于左右/前后偏移），锁掉另外两个水平轴。
    """
    nodes, accs, bvs = gltf["nodes"], gltf["accessors"], gltf["bufferViews"]
    buf = bytearray(binary)
    for anim in gltf.get("animations", []):
        if anim.get("name") not in names:
            continue
        for ch in anim["channels"]:
            if ch["target"].get("path") != "translation":
                continue
            node_index = ch["target"]["node"]
            if not (nodes[node_index].get("name") or "").endswith("Hips"):
                continue
            out = accs[anim["samplers"][ch["sampler"]]["output"]]
            if out.get("componentType") != 5126 or out.get("type") != "VEC3":
                raise SystemExit("根骨位移不是 float VEC3，脚本需扩展")
            view = bvs[out["bufferView"]]
            stride = view.get("byteStride") or 12
            base = view.get("byteOffset", 0) + out.get("byteOffset", 0)
            rest = list(nodes[node_index].get("translation", [0.0, 0.0, 0.0]))
            up = max(range(3), key=lambda i: abs(rest[i]))
            frames = []
            for frame in range(out["count"]):
                off = base + frame * stride
                v = list(struct.unpack_from("<fff", buf, off))
                frames.append(v)
                locked = [rest[0], rest[1], rest[2]]
                locked[up] = v[up]
                struct.pack_into("<fff", buf, off, *locked)
            span = [max(f[i] for f in frames) - min(f[i] for f in frames) for i in range(3)]
            if "min" in out and "max" in out:
                lo, hi = list(rest), list(rest)
                lo[up] = min(f[up] for f in frames)
                hi[up] = max(f[up] for f in frames)
                out["min"], out["max"] = lo, hi
            axis = "XYZ"[up]
            print(
                f"  {anim['name']}: 幅度 X={span[0]:.2f} Y={span[1]:.2f} Z={span[2]:.2f}"
                f" | 竖直轴={axis}(rest {rest[up]:.2f}) -> 锁住另外两轴"
            )
    return bytes(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", type=Path)
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--drop", action="append", default=[], metavar="NAME")
    ap.add_argument("--in-place", action="append", default=[], metavar="NAME",
                    help="锁掉该片段的根骨水平位移（可多次指定）")
    ap.add_argument("--drop-mesh", action="append", default=[], metavar="NAME",
                    help="按网格名或材质名删掉整段几何，如角色自带的武器（可多次指定）")
    args = ap.parse_args()

    renames = dict(r.split("=", 1) for r in args.rename)
    gltf, binary = read_glb(args.glb)

    backup = args.glb.with_suffix(args.glb.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(args.glb, backup)
        print(f"已备份原文件 -> {backup}")

    # 顺序要紧：先删几何，否则 clean() 可能拿武器网格那一具当"要保留的骨架"
    if args.drop_mesh:
        drop_meshes(gltf, set(args.drop_mesh))
    clean(gltf, renames, set(args.drop))
    if args.in_place:
        binary = bake_in_place(gltf, binary, set(args.in_place))
    binary = gc_buffers(gltf, binary)
    write_glb(args.glb, gltf, binary)
    print(f"已写回 {args.glb}（{args.glb.stat().st_size / 1048576:.2f} MB）")
    print("片段：", [a.get("name") for a in gltf.get("animations", [])])


if __name__ == "__main__":
    main()
