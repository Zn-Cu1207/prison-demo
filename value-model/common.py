"""train-serve 唯一真相源：把 (情境, 候选) 渲染成喂给价值模型的输入文本。

RA §1.2 的铁律——**这个模板一个字都不能改**，否则训练时和推理时模型看到的
文本对不齐，分数全是噪声。value_server / train_value / eval 全部只经这一处渲染。

模板：
    情境：{context}
    候选行动：{goal}（{desc}；来源：{中文来源}{extra}）
    请评估该行动在此情境下的紧急/重要程度。

- 中文来源：need→自身需求 / threat→威胁应对 / command→玩家指令 /
             worktask→待办工作 / event→突发事件 / daily→日常行为
- extra：仅 command 时拼「，优先级{priority}、强制执行」
"""
from __future__ import annotations

# 与 backend/app/context.py 的 SOURCE_CN 保持完全一致（有 test_consistency.py 兜底校验）
SOURCE_CN = {
    "need": "自身需求",
    "threat": "威胁应对",
    "command": "玩家指令",
    "worktask": "待办工作",
    "event": "突发事件",
    "daily": "日常行为",
}


def _cand_get(candidate: dict, *keys, default=""):
    for k in keys:
        if k in candidate and candidate[k] is not None:
            return candidate[k]
    return default


def build_input(context: str, candidate: dict) -> str:
    """渲染单个候选的模型输入文本（train / serve 共用，勿另写一份）。

    candidate 字段对齐 backend 的 Candidate.to_payload()：
        {goal, desc, source, priority, bForcePlan}
    """
    goal = str(_cand_get(candidate, "goal"))
    desc = str(_cand_get(candidate, "desc", default=goal))
    source = str(_cand_get(candidate, "source", default="daily"))
    source_cn = SOURCE_CN.get(source, source)
    extra = ""
    if source == "command":
        priority = str(_cand_get(candidate, "priority", default="Precise"))
        extra = f"，优先级{priority}、强制执行"
    return (
        f"情境：{context}\n"
        f"候选行动：{goal}（{desc}；来源：{source_cn}{extra}）\n"
        f"请评估该行动在此情境下的紧急/重要程度。"
    )
