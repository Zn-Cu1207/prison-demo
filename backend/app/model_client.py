"""价值模型服务客户端：异步 HTTP 调 value_server 的 /score，并把标量 v 换算成分。

对接契约（来自 shadow-integration-cpp.md §3）：
    请求 POST body: {"context": "<caption>", "candidates": [{goal,desc,source,priority,bForcePlan}, ...]}
    响应:            {"scores": [v1, v2, ...], "server_ms": float}   顺序与 candidates 对齐

M_value 映射（来自 integration-design.md §6，已在 RA 真实分布上标定）：
    M_value = clamp( exp( clamp(v, -18.0, +14.0) / 2.0 ),  0.00012,  1097.0 )
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass

import httpx

from .config import MODEL_TIMEOUT_S, MODEL_URL
from .context import Candidate, ContextState, build_context_caption

log = logging.getLogger(__name__)

# M_value 映射常量（照抄标定结果，勿改）
_V_MIN = -18.0
_V_MAX = 14.0
_TEMPERATURE = 2.0
_M_FLOOR = 0.00012
_M_CAP = 1097.0


def v_to_m_value(v: float) -> float:
    """把模型原始标量 v（Bradley-Terry logit）换算成正的乘数 M_value。"""
    clamped = max(_V_MIN, min(_V_MAX, v))
    m = math.exp(clamped / _TEMPERATURE)
    return max(_M_FLOOR, min(_M_CAP, m))


@dataclass
class ScoredCandidate:
    """一个候选的完整打分拆解。

    最终分 = M_value × M_light × M_inertia ÷ (1+Cost)
             × M_interruption × M_failure × M_timeout × M_repetition
             × M_crowd × M_command        （后两项是监狱 Demo 的引擎侧扩展）

    每一项都单独留字段，前端面板照着展开，"为什么它赢了"一眼可查。
    """

    candidate: Candidate
    v: float          # 模型原始分（Bradley-Terry logit）
    m_value: float    # exp(v/T) 换算后的基础分
    score: float      # 叠完所有乘数的最终分
    # RA 打分公式的运行时项
    m_light: float = 1.0          # 环境/状态（熄灯后户外行动打折等）
    m_inertia: float = 1.0        # 惯性：正在做的目标加成，防一拍一变
    cost: float = 0.0             # 距离成本，最终除以 (1+Cost)
    m_interruption: float = 1.0   # 被打断过的目标给补偿，鼓励回去做完
    m_failure: float = 1.0        # 做不成的目标衰减
    m_timeout: float = 1.0        # 长时间到不了的目标衰减
    m_repetition: float = 1.0     # 短期内反复做同一件事衰减
    # 监狱 Demo 的引擎侧扩展
    m_crowd: float = 1.0     # 放风拥挤预期的性格化乘数（重训后由模型内化）
    m_command: float = 1.0   # 玩家指令保底胜的乘数（非紧急时指令占优）
    m_watch: float = 1.0     # 被狱警/狱犬盯上时对违禁动作的压制（≈0）
    m_drive: float = 1.0     # 宏观动机乘数：推进长期心思的行为抬、损耗的压、禁忌≈0
    cached: bool = True      # False = 这条还没拿到模型分（基础分按 0 处理）


class ModelClient:
    """封装对 value_server 的异步调用。整个后端共用一个实例（复用连接池）。"""

    def __init__(self, url: str = MODEL_URL, timeout_s: float = MODEL_TIMEOUT_S) -> None:
        self.url = url
        # 跨 NPC 合批端点。实测 5 个 NPC：单发 ~3900ms，合批 ~460ms。
        self.multi_url = url.rsplit("/", 1)[0] + "/score_multi"
        self._multi_ok = True          # 一旦 /score_multi 不可用就永久回退到逐个 /score
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ping(self) -> dict:
        """探活：拿一个最小请求打一下 /score，验证"/score 打通"。"""
        payload = {
            "context": build_context_caption(ContextState()),
            "candidates": [Candidate(goal="Patrol", source="daily").to_payload()],
        }
        r = await self._client.post(self.url, json=payload)
        r.raise_for_status()
        return r.json()

    async def score(self, context: str, candidates: list[Candidate]) -> list[ScoredCandidate]:
        """给一组候选打分。返回与输入同序的 ScoredCandidate 列表。"""
        if not candidates:
            return []
        payload = {
            "context": context,
            "candidates": [c.to_payload() for c in candidates],
        }
        r = await self._client.post(self.url, json=payload)
        r.raise_for_status()
        data = r.json()
        scores = data.get("scores", [])
        if len(scores) != len(candidates):
            raise ValueError(
                f"value_server 返回 scores 数({len(scores)}) 与候选数({len(candidates)}) 不一致"
            )
        result: list[ScoredCandidate] = []
        for cand, v in zip(candidates, scores):
            m = v_to_m_value(float(v))
            # M0：Score 就是 M_value（模型管"想做什么"）。
            # M2 起再补引擎侧运行时项：Score = M_value × M_light × M_inertia / (1+Cost) × 惩罚
            result.append(ScoredCandidate(candidate=cand, v=float(v), m_value=m, score=m))
        return result

    async def score_multi(
        self, groups: list[tuple[str, list[Candidate]]]
    ) -> list[list[ScoredCandidate]]:
        """跨 NPC 合批打分：一次请求覆盖所有 NPC，显存里凑成一个 batch。"""
        payload = {
            "groups": [
                {"context": cap, "candidates": [c.to_payload() for c in cands]}
                for cap, cands in groups
            ]
        }
        r = await self._client.post(self.multi_url, json=payload)
        r.raise_for_status()
        rows = r.json().get("scores", [])
        if len(rows) != len(groups):
            raise ValueError(f"/score_multi 返回组数({len(rows)}) 与请求组数({len(groups)}) 不一致")
        out: list[list[ScoredCandidate]] = []
        for (_cap, cands), scores in zip(groups, rows):
            if len(scores) != len(cands):
                raise ValueError("/score_multi 某组的分数个数与候选数不一致")
            out.append([
                ScoredCandidate(candidate=c, v=float(v), m_value=v_to_m_value(float(v)),
                                score=v_to_m_value(float(v)))
                for c, v in zip(cands, scores)
            ])
        return out

    async def evaluate(
        self, items: list[tuple[str, list[Candidate], str]]
    ) -> tuple[list[list[ScoredCandidate]], list[str | None]]:
        """统一大脑接口。优先合批，端点不可用时退回逐个并发。says 为 None（用模板台词）。"""
        groups = [(cap, cands) for cap, cands, _p in items]
        if self._multi_ok and len(groups) > 1:
            try:
                return await self.score_multi(groups), [None] * len(items)
            except httpx.HTTPStatusError as e:
                # 只有"这个端点根本不存在"才永久降级；超时/连接重置属于抖动，
                # 一次网络打嗝就把合批关掉、之后一直走慢 8 倍的路径，不划算。
                if e.response.status_code in (404, 405, 501):
                    self._multi_ok = False
                    log.warning("value_server 不支持 /score_multi，永久回退逐个 /score")
                else:
                    log.warning("/score_multi 本次失败(%s)，本拍回退逐个 /score",
                                e.response.status_code)
            except (httpx.RequestError, ValueError) as e:
                log.warning("/score_multi 本次失败，本拍回退逐个 /score：%s: %s",
                            type(e).__name__, e)
        scored = await asyncio.gather(*(self.score(cap, cands) for cap, cands in groups))
        return list(scored), [None] * len(items)

    @property
    def label(self) -> str:
        return "价值模型(Qwen1.5B+LoRA)"
