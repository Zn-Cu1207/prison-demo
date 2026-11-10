"""阶段0-b：把每条情境的候选两两配对 → pairs.jsonl

RA §5：优先跨来源(跨类别)配对，让"哪类更该做"的偏好被充分采到；
同时保留少量同来源配对（学"两个需求谁更急"这种连续偏好）。
每条情境限量取样，避免候选多的情境刷屏。a/b 顺序随机化，消除位置偏见。

用法：
    python pair.py --in data/situations.jsonl --out data/pairs.jsonl --per 4 --seed 7
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from guardrails import candidate_tier


def build_pairs(rec: dict, cross: int, same: int, rng: random.Random) -> list[dict]:
    """每条情境产出两类成对样本：
    - 跨档位(cross-tier)：教优先级阶梯，几乎都能被护栏免费裁定（不烧 API）。
    - 同档位(same-tier)：教"两个需求谁更急/哪个消遣更合适"这类连续偏好，交大模型细判。
    两者都要，才既有宏观排序又有细腻手感。
    """
    cands = rec["candidates"]
    ctx = rec["ctx"]
    if len(cands) < 2:
        return []
    tiers = [candidate_tier(c, ctx) for c in cands]
    all_pairs = list(itertools.combinations(range(len(cands)), 2))
    cross_pairs = [p for p in all_pairs if tiers[p[0]] != tiers[p[1]]]
    same_pairs = [p for p in all_pairs if tiers[p[0]] == tiers[p[1]]
                  and cands[p[0]]["goal"] != cands[p[1]]["goal"]]

    def cross_key(pair):
        i, j = pair
        return abs(tiers[i] - tiers[j]) + rng.random()  # 档差越大越先取（阶梯两端）

    def same_key(pair):
        i, j = pair
        diff_src = cands[i]["source"] != cands[j]["source"]
        return (1 if diff_src else 0) + rng.random()

    cross_pairs.sort(key=cross_key, reverse=True)
    same_pairs.sort(key=same_key, reverse=True)

    chosen: list[tuple[int, int]] = []
    # 必含：最高档 vs 最低档（阶梯两端一定采到）
    hi = max(range(len(cands)), key=lambda k: tiers[k])
    lo = min(range(len(cands)), key=lambda k: tiers[k])
    if hi != lo:
        chosen.append((hi, lo))
    for p in cross_pairs:
        if sum(1 for c in chosen if tiers[c[0]] != tiers[c[1]]) >= cross:
            break
        if p not in chosen and (p[1], p[0]) not in chosen:
            chosen.append(p)
    for p in same_pairs[:same]:
        if p not in chosen and (p[1], p[0]) not in chosen:
            chosen.append(p)

    out = []
    for idx, (i, j) in enumerate(chosen):
        a, b = cands[i], cands[j]
        if rng.random() < 0.5:
            a, b = b, a
        out.append({
            "situation_id": rec["situation_id"],
            "pair_id": f"{rec['situation_id']}_p{idx}",
            "context": rec["context"],
            "ctx": ctx,
            "cand_a": a,
            "cand_b": b,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/situations.jsonl")
    ap.add_argument("--out", default="data/pairs.jsonl")
    ap.add_argument("--cross", type=int, default=2, help="每情境跨档位对数(护栏免费裁)")
    ap.add_argument("--same", type=int, default=2, help="每情境同档位对数(交大模型细判)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    n_sit = 0
    n_pair = 0
    with open(args.inp, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_sit += 1
            for p in build_pairs(rec, args.cross, args.same, rng):
                g.write(json.dumps(p, ensure_ascii=False) + "\n")
                n_pair += 1
    print(f"{n_sit} 情境 → {n_pair} 成对样本 → {args.out}")


if __name__ == "__main__":
    main()
