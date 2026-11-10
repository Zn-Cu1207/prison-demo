"""探活 /score_multi：确认 value_server 支持跨 NPC 合批，以及返回结构。"""
from __future__ import annotations

import asyncio
import time

import httpx

from app.config import MODEL_URL
from app.context import Candidate, ContextState, build_context_caption

MULTI_URL = MODEL_URL.rsplit("/", 1)[0] + "/score_multi"
HEALTH_URL = MODEL_URL.rsplit("/", 1)[0] + "/health"


async def main() -> None:
    ctx = build_context_caption(ContextState())
    cands = [Candidate("Eat", "need"), Candidate("Sleep", "need"), Candidate("Patrol", "daily")]
    payload_one = {"context": ctx, "candidates": [c.to_payload() for c in cands]}
    payload_multi = {"groups": [payload_one, payload_one, payload_one, payload_one, payload_one]}

    async with httpx.AsyncClient(timeout=30.0) as cli:
        try:
            r = await cli.get(HEALTH_URL)
            print("health:", r.status_code, r.text[:120])
        except Exception as e:
            print("health 失败:", e)

        t0 = time.perf_counter()
        r = await cli.post(MODEL_URL, json=payload_one)
        one_ms = (time.perf_counter() - t0) * 1000
        print(f"/score        -> {r.status_code}  {one_ms:.0f}ms  {r.text[:160]}")

        t0 = time.perf_counter()
        try:
            r = await cli.post(MULTI_URL, json=payload_multi)
            multi_ms = (time.perf_counter() - t0) * 1000
            print(f"/score_multi  -> {r.status_code}  {multi_ms:.0f}ms  {r.text[:220]}")
            if r.status_code == 200:
                data = r.json()
                sc = data.get("scores")
                print("groups:", len(sc) if sc else None,
                      "每组候选数:", [len(g) for g in sc] if sc else None)
                print(f"合批收益：5 组单发约 {one_ms * 5:.0f}ms vs 合批 {multi_ms:.0f}ms")
        except Exception as e:
            print("/score_multi 失败:", type(e).__name__, e)


if __name__ == "__main__":
    asyncio.run(main())
