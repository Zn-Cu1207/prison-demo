"""泛化探针：用现有 RA 价值模型，喂几个"监狱原生"情境，看它对陌生词/新目标的反应。

结论直接打印，用来判断哪些监狱语义模型"碰巧能泛化"、哪些"完全懵"（→ 重训清单）。
用法：python backend/probe_generalization.py
"""
import asyncio

from app.context import (
    CROWD_PACKED,
    PHASE_CLASS,
    PHASE_SLEEP,
    PHASE_YARD,
    Candidate,
    ContextState,
    NeedState,
    SURVEIL_LOW,
    build_context_caption,
)
from app.model_client import ModelClient


async def probe(client: ModelClient, title: str, ctx: ContextState, cands: list[Candidate]) -> None:
    caption = build_context_caption(ctx)
    scored = await client.score(caption, cands)
    print(f"\n### {title}")
    print(f"context: {caption}")
    for sc in sorted(scored, key=lambda s: s.score, reverse=True):
        print(f"   {sc.candidate.label:6s}({sc.candidate.goal:12s}) v={sc.v:8.3f}  score={sc.score:10.3f}")


async def main() -> None:
    client = ModelClient()
    try:
        # 1. 放风 + 仇人在场（暴躁）：模型会不会把"打架"顶上去？
        await probe(
            client, "放风+仇人在场+暴躁 → 打架 该不该赢",
            ContextState(needs=NeedState(hunger=20, boredom=40, stress=60),
                         threat="仇人", phase=PHASE_YARD, personality="暴躁",
                         enemy_present=True, backer_present=True, yard_crowd=CROWD_PACKED, event="斗殴"),
            [Candidate("Fight", "threat"), Candidate("DeclineBoredom", "need"),
             Candidate("Socialize", "need"), Candidate("Patrol", "daily")],
        )

        # 2. 就寝+监视低：假释临近 vs 尚早，越狱分会不会因"假释"变化？（模型没学过"假释"）
        await probe(
            client, "就寝+监视低+假释【临近】 → 越狱该不该被压住",
            ContextState(needs=NeedState(fatigue=70), phase=PHASE_SLEEP,
                         personality="普通", surveillance=SURVEIL_LOW, parole_near=True),
            [Candidate("PrisonBreak", "daily"), Candidate("Sleep", "need"), Candidate("Patrol", "daily")],
        )
        await probe(
            client, "就寝+监视低+假释【尚早】 → 越狱分对比上一条",
            ContextState(needs=NeedState(fatigue=70), phase=PHASE_SLEEP,
                         personality="暴躁", surveillance=SURVEIL_LOW, parole_near=False),
            [Candidate("PrisonBreak", "daily"), Candidate("Sleep", "need"), Candidate("Patrol", "daily")],
        )

        # 3. 上课：新目标"学习"怎么打分（模型没见过 Study）
        await probe(
            client, "上课时段 → 新目标「学习」的分数（泛化观察）",
            ContextState(needs=NeedState(boredom=60), phase=PHASE_CLASS, personality="书呆"),
            [Candidate("Study", "daily"), Candidate("DeclineBoredom", "need"), Candidate("Patrol", "daily")],
        )

        # 4. 高饥饿基准：确认模型对"需求"这套仍然听得懂（对照组）
        await probe(
            client, "对照：饥饿90 → 吃饭该碾压（验证需求轴仍有效）",
            ContextState(needs=NeedState(hunger=90, thirst=30), phase=PHASE_YARD, personality="老油条"),
            [Candidate("Eat", "need"), Candidate("Drink", "need"), Candidate("Patrol", "daily")],
        )
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
