"""快速验证 WebSocket 帧 + 观察真模型对监狱原生词的泛化表现。

用法（后端已在 8000 运行时）：python backend/ws_smoke.py
"""
import asyncio
import json

import websockets


async def run() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        world = None
        decisions: dict[str, dict] = {}
        for _ in range(40):
            msg = json.loads(await ws.recv())
            if msg["type"] == "world_state":
                world = msg
            elif msg["type"] == "npc_decision":
                decisions[msg["npc_id"]] = msg
            if world and len(decisions) >= len(world["npcs"]):
                break

        assert world is not None and decisions, "未收到完整帧"
        print(f"时段={world['phase']}  放风拥挤={world['yard_crowd']}  tick={world['tick']}\n")
        for npc in world["npcs"]:
            d = decisions.get(npc["id"])
            print(f"── {npc['name']}（{npc['personality']}）独白: 「{npc.get('monologue')}」")
            if d:
                print(f"   为什么: {d.get('explain')}")
                ranked = sorted(d["candidates"], key=lambda c: c["score"], reverse=True)
                cs = " | ".join(f"{c['goal_label']}={c['score']:.2f}" + ("✅" if c["chosen"] else "")
                                for c in ranked)
                print(f"   候选: {cs}")
        print("\nOK ✅ M2 独白/解释/候选 帧正常")


if __name__ == "__main__":
    asyncio.run(run())
