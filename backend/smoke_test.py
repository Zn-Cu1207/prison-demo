"""M0 冒烟测试：验证 context 拼装 + ModelClient → value_server 打通。

用法：先起 mock_value_server（或真 value_server），再 `python backend/smoke_test.py`。
"""
import asyncio

from app.context import Candidate, ContextState, NeedState, build_context_caption
from app.model_client import ModelClient


async def main() -> None:
    ctx = ContextState(needs=NeedState(hunger=90, thirst=20, fatigue=10, boredom=15))
    caption = build_context_caption(ctx)
    print("caption:", caption)

    candidates = [
        Candidate(goal="Eat", source="need"),
        Candidate(goal="Drink", source="need"),
        Candidate(goal="Patrol", source="daily"),
    ]
    client = ModelClient()
    try:
        scored = await client.score(caption, candidates)
    finally:
        await client.aclose()

    print("\n候选打分（应：Eat 最高，Patrol 垫底）：")
    for sc in scored:
        print(f"  {sc.candidate.goal:16s} source={sc.candidate.source:7s} "
              f"v={sc.v:8.3f}  M_value={sc.m_value:10.4f}  score={sc.score:10.4f}")

    best = max(scored, key=lambda s: s.score)
    print(f"\n选中：{best.candidate.goal}")
    assert best.candidate.goal == "Eat", "预期饥饿90时应选 Eat"
    print("OK ✅ /score 打通，决策链正常")


if __name__ == "__main__":
    asyncio.run(main())
