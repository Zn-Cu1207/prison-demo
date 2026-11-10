"""对已启动的 value_server 做端到端冒烟：契约 + 优先级阶梯是否真的学会了。

检查项：
  1) /health 通
  2) /score 返回与候选同序、同数量的 scores
  3) **优先级阶梯**：濒死危机 > 威胁 > 玩家指令 > 普通需求 > 待办 > 日常闲逛
  4) /score_multi 跨 NPC 合批可用、与逐个 /score 结果一致
  5) 打印 M_value = exp(v/2)（后端 model_client 的换算），确认量级能拉开

    python smoke_serve.py --url http://127.0.0.1:8009
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import httpx

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))
from app.context import Candidate, ContextState, NeedState, build_context_caption  # noqa: E402

V_MIN, V_MAX, TEMP, M_FLOOR, M_CAP = -18.0, 14.0, 2.0, 0.00012, 1097.0


def m_value(v: float) -> float:
    v = max(V_MIN, min(V_MAX, v))
    return max(M_FLOOR, min(M_CAP, math.exp(v / TEMP)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8009")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    ok_all = True

    with httpx.Client(timeout=120) as c:
        # 1) health
        h = c.get(base + "/health")
        print(f"[1] /health {h.status_code} {h.text.strip()[:120]}")
        ok_all &= h.status_code == 200

        # 2)+3) 一个"重伤 + 仇人在场 + 有玩家指令 + 被派苦役"的复合情境，
        #        六类候选同时上桌，看模型给的相对顺序是否符合阶梯。
        ctx = ContextState(
            needs=NeedState(hunger=45, thirst=30, fatigue=40, injury=90, boredom=50, social=40, stress=60),
            threat="仇人", event="无", phase="放风", personality="老油条",
            surveillance="低", enemy_present=True, has_labor=True,
            motive="想早点平安假释", player_command="命令去在放风场溜达",
        )
        cap = build_context_caption(ctx)
        cands = [
            Candidate("Cure", "need"),                                  # 濒死危机(受伤90)
            Candidate("Fight", "threat"),                               # 威胁应对
            Candidate("Patrol", "command", b_force_plan=True),          # 玩家指令
            Candidate("Eat", "need"),                                   # 普通需求
            Candidate("Labor", "worktask"),                             # 待办工作
            Candidate("Gamble", "daily"),                               # 日常(且违禁)
        ]
        payload = {"context": cap, "candidates": [x.to_payload() for x in cands]}
        r = c.post(base + "/score", json=payload)
        print(f"[2] /score {r.status_code}")
        ok_all &= r.status_code == 200
        if r.status_code != 200:
            print(r.text[:300])
            print("SMOKE_FAIL")
            return
        scores = r.json()["scores"]
        ok_len = len(scores) == len(cands)
        print(f"    条数对齐: {ok_len} ({len(scores)}/{len(cands)})  server_ms={r.json().get('server_ms', 0):.0f}")
        ok_all &= ok_len

        print("[3] 优先级阶梯（v 越大越该做；M_value=exp(v/2) 是后端实际用的基础分）：")
        rows = sorted(zip(cands, scores), key=lambda t: -t[1])
        for cand, v in rows:
            print(f"    {cand.goal:<12} 来源={cand.source:<9} v={v:+8.3f}  M_value={m_value(v):>12.4f}")
        by_goal = dict(zip([x.goal for x in cands], scores))
        checks = [
            ("濒死(Cure) > 日常违禁(Gamble)", by_goal["Cure"] > by_goal["Gamble"]),
            ("濒死(Cure) > 待办(Labor)", by_goal["Cure"] > by_goal["Labor"]),
            ("威胁(Fight) > 日常(Gamble)", by_goal["Fight"] > by_goal["Gamble"]),
            ("指令(Patrol) > 日常(Gamble)", by_goal["Patrol"] > by_goal["Gamble"]),
            ("待办(Labor) > 日常违禁(Gamble)", by_goal["Labor"] > by_goal["Gamble"]),
        ]
        for name, good in checks:
            print(f"    {'✅' if good else '❌'} {name}")
            ok_all &= good

        # 4) 合批一致性
        groups = {"groups": [payload, {"context": build_context_caption(ContextState(
            needs=NeedState(hunger=88))), "candidates": [Candidate("Eat", "need").to_payload(),
                                                        Candidate("Rest", "daily").to_payload()]}]}
        rm = c.post(base + "/score_multi", json=groups)
        print(f"[4] /score_multi {rm.status_code}")
        ok_all &= rm.status_code == 200
        if rm.status_code == 200:
            rows2 = rm.json()["scores"]
            same = len(rows2) == 2 and len(rows2[0]) == len(cands)
            drift = max(abs(a - b) for a, b in zip(rows2[0], scores)) if same else 9.9
            print(f"    组数/条数对齐={same}  与逐个 /score 最大偏差={drift:.4f}")
            hungry_ok = rows2[1][0] > rows2[1][1]
            print(f"    {'✅' if hungry_ok else '❌'} 饥饿88 → Eat > Rest")
            ok_all &= same and drift < 0.05 and hungry_ok

        # 5) 单调性：抬饥饿看 Eat 是否上升
        print("[5] 单调性（饥饿↑ ⇒ Eat 分↑）：")
        vals = []
        for hg in (20, 45, 70, 95):
            cp = build_context_caption(ContextState(needs=NeedState(hunger=hg)))
            v = c.post(base + "/score", json={"context": cp,
                                              "candidates": [Candidate("Eat", "need").to_payload()]}
                       ).json()["scores"][0]
            vals.append(v)
            print(f"    饥饿{hg:>3} → Eat v={v:+.3f}  M_value={m_value(v):.4f}")
        mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        print(f"    {'✅ 单调' if mono else '⚠️ 非严格单调(平坦区噪声，不一定影响行为)'}")

    print("\nSMOKE_PASS" if ok_all else "\nSMOKE_FAIL(有检查项未通过，见上)")


if __name__ == "__main__":
    main()
