"""本地 mock 价值模型服务（M0 fallback 用）。

作用：当真 value_server（3060/T4 上的 Qwen1.5B+LoRA）不可用时（如 SSH 隧道断开），
本地起一个**同契约**的桩，让"后端 → 模型服务"整条链路仍能端到端跑通。

注意：真模型的 SSH 隧道占用 8009，所以本 mock 默认监听 **8010**，避免抢端口。
启用 mock 时把后端环境变量设为 `MODEL_URL=http://127.0.0.1:8010/score` 再启动后端即可，
后端代码零改动。真隧道在线时，保持默认 MODEL_URL(8009) 即直连真模型。

契约（与 shadow-integration-cpp.md §3 一致）：
    POST /score        {"context": "...", "candidates": [...]}  ->  {"scores": [v...], "server_ms": f}
    POST /score_multi  {"items": [{"context","candidates"}...]} ->  {"results": [{"scores":[...]}...], "server_ms": f}

打分是**启发式**的（不是真模型）：从 caption 解析需求值，需求越高该目标 v 越大，
复刻真模型"分随需求单调上升"的行为，好让 M0 的 NPC 看起来是"活的"。真模型分布见文档，
数量级不同没关系——价值头只看相对排序。
"""
from __future__ import annotations

import re
import time

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="mock value_server")

# goal → 它由哪个需求驱动（用于读 caption 里对应的数值）
GOAL_NEED = {
    "Eat": "饥饿",
    "Drink": "口渴",
    "Sleep": "疲劳",
    "DeclineBoredom": "无聊",
}

_NEED_RE = {name: re.compile(rf"{name}(\d+)/100") for name in
            ["饥饿", "口渴", "疲劳", "受伤", "无聊", "社交欲", "压力"]}


def _parse_need(context: str, need_name: str) -> int:
    m = _NEED_RE[need_name].search(context)
    return int(m.group(1)) if m else 0


def _score_one(context: str, cand: dict) -> float:
    """返回一个近似 Bradley-Terry logit 的标量 v。"""
    goal = cand.get("goal", "")
    source = cand.get("source", "need")

    # 玩家指令兜底（command 中位约 -4.6，但强制指令给高一点）
    if source == "command":
        return 6.0 if cand.get("bForcePlan") else -3.0
    # 日常行为（daily 中位 ~ -12）
    if source == "daily":
        return -12.0

    need_name = GOAL_NEED.get(goal)
    if need_name is not None:
        val = _parse_need(context, need_name)
        # 需求 0→100 平滑映射到 v ≈ [-9, +11]，单调上升（复刻文档 B1/B2）
        return (val - 45) / 5.0

    # 其余未知目标给个中性偏低值
    return -6.0


class ScoreRequest(BaseModel):
    context: str
    candidates: list[dict]


class MultiItem(BaseModel):
    context: str
    candidates: list[dict]


class ScoreMultiRequest(BaseModel):
    items: list[MultiItem]


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    t0 = time.perf_counter()
    scores = [_score_one(req.context, c) for c in req.candidates]
    return {"scores": scores, "server_ms": (time.perf_counter() - t0) * 1000.0}


@app.post("/score_multi")
def score_multi(req: ScoreMultiRequest) -> dict:
    t0 = time.perf_counter()
    results = [{"scores": [_score_one(it.context, c) for c in it.candidates]} for it in req.items]
    return {"results": results, "server_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("MOCK_PORT", "8010"))
    uvicorn.run(app, host="127.0.0.1", port=port)
