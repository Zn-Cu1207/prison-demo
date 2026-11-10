"""RA 护栏：领域优先级硬规则，用来纠正大模型标注的系统性偏差。

RA 报告 §4.3 的铁律阶梯（跨类别宏观排序）：
    濒死级生存危机 > 玩家指令 > 一切非生存危机需求 > worktask > 休闲
    宏观：threat > event > command > need > worktask > daily

实现为「候选优先级档位 tier」——**只在跨档位时强制裁决**（高档必胜），
同档位（例如两个 need 谁更急）交给大模型细判，正是我们想让价值模型学到的连续偏好。

另加一条监狱专属护栏：**违禁动作(illicit) 在狱警/狱犬盯防或高监视度下被压到最低档**
（对应引擎侧 m_watch≈0），保证"当着管教面老实收敛"这一行为被模型内化。
"""
from __future__ import annotations

# 濒死级需求阈值：到这个值该行动升到最高档（压过玩家指令）
CRISIS_NEED = {
    "Cure": ("injury", 70),
    "Eat": ("hunger", 85),
    "Drink": ("thirst", 85),
    "Sleep": ("fatigue", 90),
}

TIER_NAME = {5: "濒死危机", 4: "应急威胁/事件", 3: "玩家指令", 2: "普通需求", 1: "待办工作", 0: "日常休闲", -1: "违禁被压制"}


def candidate_tier(cand: dict, ctx: dict) -> int:
    """给候选在此情境下的优先级档位（越大越该做）。"""
    g = cand.get("goal", "")
    src = cand.get("source", "daily")
    illicit = bool(cand.get("illicit", False))
    needs = ctx.get("needs", {}) or {}
    threat = ctx.get("threat", "无")
    event = ctx.get("event", "无")
    surv = ctx.get("surveillance", "中")
    enemy = bool(ctx.get("enemy_present", False))

    # 夜班关键：典狱长走近 = caption 威胁:狱警。下杀手/打架虽标 threat，
    # 当着管教面仍是见不得光——沉底，把位置让给 LayLow。指令通道除外（引擎会强制）。
    watched = surv == "高" or threat in ("狱警", "狱犬")
    armed = bool(ctx.get("has_weapon", False))
    grudge = ctx.get("grudge", "无")

    # 搜牢时 Hide 就是事件应对，不能当普通违禁沉底。
    if g == "Hide" and event == "狱警搜牢":
        return 4

    if illicit and watched and src != "command":
        return -1

    # 没刀的 Assassinate 不是有效威胁应对（v4 护栏按 source=threat 给了最高档，
    # 把「空挥 > 去弄刀」写进偏好，凶器轴才学反）。
    if g == "Assassinate" and not armed and src != "command":
        return 0

    if (g == "GetWeapon" and not armed and src != "command"
            and (enemy or threat == "仇人" or grudge == "深")):
        return 2

    if src == "threat":
        return 5 if (enemy or threat in ("仇人", "狱犬")) else 4
    if src == "event":
        return 5 if event == "警报" else 4
    if src == "command":
        return 3
    if src == "need":
        key_thr = CRISIS_NEED.get(g)
        if key_thr and needs.get(key_thr[0], 0) >= key_thr[1]:
            return 5  # 濒死，压过指令
        return 2
    if src == "worktask":
        return 1
    return 0  # daily 休闲


def guardrail(ctx: dict, cand_a: dict, cand_b: dict) -> tuple[int, str] | None:
    """跨档位强制裁决。

    返回 (winner_idx, reason)：winner_idx ∈ {0,1} 指向 cand_a/cand_b；同档位返回 None（交大模型）。
    """
    ta = candidate_tier(cand_a, ctx)
    tb = candidate_tier(cand_b, ctx)
    if ta == tb:
        return None
    win = 0 if ta > tb else 1
    hi, lo = (ta, tb) if win == 0 else (tb, ta)
    reason = f"护栏:{TIER_NAME[hi]}(档{hi}) > {TIER_NAME[lo]}(档{lo})"
    return win, reason
