"""给 GLB 里的贴图降分辨率并重编码，用于把 Mixamo 原始资产压到能上网的体积。

Mixamo 导出的角色常带 4K PNG（单张十几 MB），浏览器解码后每张要占几十 MB 显存，
在网页里既拖慢首屏又容易把低端设备撑爆。游戏里角色只有屏幕上几十像素高，
1K 贴图完全够用。

同时可以丢掉次要贴图（如 glossiness/specular），它们对这种远景小人物几乎没有观感贡献。

用法：
    python frontend/tools/shrink_glb_textures.py frontend/assets/doctor.glb --max 1024
    python frontend/tools/shrink_glb_textures.py x.glb --max 1024 --drop-suffix Glossiness Specular
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
from pathlib import Path

from PIL import Image

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    _magic, _version, total = struct.unpack("<III", data[:12])
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


def shrink(gltf: dict, binary: bytes, max_side: int, drop_suffix: list[str], quality: int) -> bytes:
    """重建 BIN：非图片的 bufferView 原样搬过去，图片则缩放后重编码。"""
    views = gltf["bufferViews"]
    image_view = {img["bufferView"]: i for i, img in enumerate(gltf.get("images", [])) if "bufferView" in img}

    # 被丢弃的贴图：从材质里摘掉引用，图片本身也不再写入
    dropped: set[int] = set()
    if drop_suffix:
        for i, img in enumerate(gltf.get("images", [])):
            if any((img.get("name") or "").endswith(s) for s in drop_suffix):
                dropped.add(i)
    tex_dropped = {t for t, tex in enumerate(gltf.get("textures", [])) if tex.get("source") in dropped}

    def prune_material(node: dict) -> None:
        for key in list(node.keys()):
            value = node[key]
            if isinstance(value, dict):
                if value.get("index") in tex_dropped and key.endswith("Texture"):
                    node.pop(key)
                else:
                    prune_material(value)

    for material in gltf.get("materials", []):
        prune_material(material)

    out = bytearray()
    new_views = []
    for vi, view in enumerate(views):
        start = view.get("byteOffset", 0)
        blob = bytes(binary[start : start + view["byteLength"]])
        img_index = image_view.get(vi)
        if img_index is not None:
            if img_index in dropped:
                print(f"  丢弃贴图 {gltf['images'][img_index].get('name')}")
                blob = b""
            else:
                blob = recode(gltf["images"][img_index], blob, max_side, quality)
        # bufferView 之间按 4 字节对齐，avoid 让后续 accessor 读到错位数据
        pad = (4 - len(out) % 4) % 4
        out += b"\x00" * pad
        new_view = dict(view)
        new_view["byteOffset"] = len(out)
        new_view["byteLength"] = max(len(blob), 1)
        out += blob or b"\x00"
        new_views.append(new_view)

    gltf["bufferViews"] = new_views
    gltf["buffers"] = [{"byteLength": len(out)}]
    return bytes(out)


def recode(image_def: dict, blob: bytes, max_side: int, quality: int) -> bytes:
    name = image_def.get("name") or "?"
    with Image.open(io.BytesIO(blob)) as im:
        before = im.size
        im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
        if max(im.size) > max_side:
            ratio = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * ratio)), max(1, round(im.height * ratio))), Image.LANCZOS)
        buf = io.BytesIO()
        # WebP：体积远小于 PNG，且 three.js 的 GLTFLoader 原生支持（EXT_texture_webp 之外，
        # 单纯把 image/webp 放进 GLB 也能被浏览器解码）
        im.save(buf, format="WEBP", quality=quality, method=6)
        data = buf.getvalue()
    image_def["mimeType"] = "image/webp"
    print(
        f"  {name:<28} {before[0]}x{before[1]} -> {im.width}x{im.height}"
        f"  {len(blob) / 1048576:.2f} MB -> {len(data) / 1048576:.2f} MB"
    )
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", type=Path)
    ap.add_argument("--max", type=int, default=1024, help="贴图最长边上限")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--drop-suffix", nargs="*", default=[], help="按贴图名后缀丢弃，如 Glossiness Specular")
    args = ap.parse_args()

    gltf, binary = read_glb(args.glb)
    backup = args.glb.with_suffix(args.glb.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(args.glb, backup)
        print(f"已备份原文件 -> {backup}")

    before = args.glb.stat().st_size
    binary = shrink(gltf, binary, args.max, args.drop_suffix, args.quality)
    write_glb(args.glb, gltf, binary)
    after = args.glb.stat().st_size
    print(f"{args.glb}: {before / 1048576:.2f} MB -> {after / 1048576:.2f} MB")


if __name__ == "__main__":
    main()
