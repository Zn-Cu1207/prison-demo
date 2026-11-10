"""人工抽检标注质量：把 preferences.jsonl 渲染成人读的清单。

    python review_labels.py --in data/preferences.jsonl --n 15 --only hy --out review.txt
"""
from __future__ import annotations

import argparse
import json


def brief_ctx(ctx: str) -> str:
    parts = [p.strip() for p in ctx.split(";")]
    keep = [parts[0]]
    for p in parts[1:]:
        if any(k in p for k in ("威胁", "事件", "时段", "监视度", "苦役", "玩家指令", "仇人")):
            if not p.endswith(":无") and not p.endswith("仇人:无"):
                keep.append(p)
    return "\n    ".join(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/preferences.jsonl")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--only", default="", help="只看 hy 或 rule")
    ap.add_argument("--out", default="review.txt")
    args = ap.parse_args()

    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if not args.only or r.get("label_by") == args.only:
                    rows.append(r)
    rows = rows[: args.n]

    blocks = []
    for i, r in enumerate(rows, 1):
        c, j = r["chosen"], r["rejected"]
        blocks.append(
            f"[{i}] ({r['label_by']}, conf={r['confidence']})\n"
            f"    {brief_ctx(r['context'])}\n"
            f"  ✔ 选中: {c['goal']}({c['desc']}) 来源={c['source']}\n"
            f"  ✘ 落选: {j['goal']}({j['desc']}) 来源={j['source']}\n"
            f"  理由: {r.get('reason', '')}\n"
        )
    text = "\n".join(blocks)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"已写出 {len(rows)} 条到 {args.out}")


if __name__ == "__main__":
    main()
