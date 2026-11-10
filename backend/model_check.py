"""直接问模型：不同饥饿下 Eat/Sleep/Drink/Patrol 的原始分，隔离模型行为。"""
import asyncio

from app.context import Candidate, ContextState, NeedState, build_context_caption
from app.model_client import ModelClient


async def probe(model, hunger):
    ctx = ContextState(needs=NeedState(hunger=hunger))
    cap = build_context_caption(ctx)
    cands = [Candidate("Eat", "need"), Candidate("Sleep", "need"),
             Candidate("Drink", "need"), Candidate("Patrol", "daily")]
    scored = await model.score(cap, cands)
    print(f"饥饿={hunger}:  " + "  ".join(
        f"{s.candidate.goal} v={s.v:.2f} m={s.m_value:.2f}" for s in scored))


async def main():
    model = ModelClient()
    for h in (100, 70, 40, 8):
        await probe(model, h)
    await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
