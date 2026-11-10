"""train-serve 一致性护栏：确保 common.build_input 的渲染 = 后端会发出的候选。

价值模型的分数只有在"训练时看到的文本"和"推理时看到的文本"逐字一致时才有意义。
这个测试把后端真实的 SOURCE_CN / Candidate.to_payload 拿来对照 common.py，任何一处
漂移都会在这里红掉。CI/改动后务必跑：  python test_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import SOURCE_CN, build_input

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))
from app.context import SOURCE_CN as BE_SOURCE_CN  # noqa: E402
from app.context import Candidate, ContextState, NeedState, build_context_caption  # noqa: E402


def test_source_cn_matches():
    assert SOURCE_CN == BE_SOURCE_CN, f"SOURCE_CN 漂移: {SOURCE_CN} vs {BE_SOURCE_CN}"


def test_render_matches_expected():
    ctx = ContextState(needs=NeedState(hunger=70))
    cap = build_context_caption(ctx)

    # need 候选
    need = Candidate("Eat", "need")
    got = build_input(cap, need.to_payload())
    want = (
        f"情境：{cap}\n"
        f"候选行动：Eat（去食堂吃饭；来源：自身需求）\n"
        f"请评估该行动在此情境下的紧急/重要程度。"
    )
    assert got == want, f"\nGOT:\n{got}\nWANT:\n{want}"

    # command 候选（带 优先级X、强制执行）
    cmd = Candidate("Patrol", "command", b_force_plan=True)
    got_c = build_input(cap, cmd.to_payload())
    assert "来源：玩家指令，优先级Precise、强制执行）" in got_c, got_c


def test_all_sources_render():
    cap = build_context_caption(ContextState())
    for src in ("need", "threat", "command", "worktask", "event", "daily"):
        c = Candidate("Patrol", src)  # goal 任意，只测来源渲染不炸
        txt = build_input(cap, c.to_payload())
        assert SOURCE_CN[src] in txt


def test_caption_new_axes_pinned():
    """把 v2 新增四轴的措辞和顺序钉死在 caption 末尾。

    这四轴是 v2 重标重训的依据。一旦有人改了字段顺序或用词（比如把"食物"改成"口粮"），
    线上 caption 就和训练数据不再逐字一致，模型的分立刻失去意义——而且不会报错，
    只会表现为"决策莫名变差"，极难排查。所以在这里红一次，好过上线后猜一周。

    注：默认已切 v3（尾部还会再追加社交四轴），本测试专测 v2 那段尾巴，故显式置 v2。
    """
    import os

    old = os.environ.get("PRISON_CAPTION")
    os.environ["PRISON_CAPTION"] = "v2"
    try:
        cap = build_context_caption(ContextState())
        assert cap.endswith("食物:充足; 仇怨:无; 表现:一般; 近期被抓:否"), \
            f"新四轴的措辞/顺序漂移了：\n{cap[-80:]}"

        # 非默认值也要能正确渲染（训练数据里大量是非默认值）
        ctx = ContextState(food_supply="断供", grudge="深", perform="好", busted_recently=True)
        cap2 = build_context_caption(ctx)
        assert cap2.endswith("食物:断供; 仇怨:深; 表现:好; 近期被抓:是"), cap2[-80:]
    finally:
        if old is None:
            os.environ.pop("PRISON_CAPTION", None)
        else:
            os.environ["PRISON_CAPTION"] = old


def test_caption_v3_social_axes_pinned():
    """v3/P2：社交四轴按固定措辞/顺序追加在 v2 尾部之后。

    2026-08-06 起默认已切 v3（线上服务的就是 v3 社交模型）；显式 v2 可回滚且不渲染社交轴，
    保护"若临时切回老模型"的场景不吃 OOD。两种模式都在这里钉死，防止训练/推理再次漂移。
    """
    import os

    from app.context import caption_version

    old = os.environ.get("PRISON_CAPTION")
    try:
        # 默认应为 v3（线上现役）：带社交四轴
        os.environ.pop("PRISON_CAPTION", None)
        assert caption_version() == "v3"
        assert "邻近:" in build_context_caption(ContextState())

        # 显式回滚 v2：不渲染社交轴（保护老模型）
        os.environ["PRISON_CAPTION"] = "v2"
        assert caption_version() == "v2"
        assert "邻近:" not in build_context_caption(ContextState())

        # v3：末尾恰好追加四轴，措辞/顺序钉死
        os.environ["PRISON_CAPTION"] = "v3"
        assert caption_version() == "v3"
        cap = build_context_caption(ContextState())
        assert cap.endswith("近期被抓:否; 邻近:无; 对方意图:无; 社交近况:无; 人情:无; 凶器:无"), cap[-100:]

        ctx = ContextState(near_rel="盟友", other_intent="结盟",
                           social_recent="新结盟", obligation="有人欠我", has_weapon=True)
        cap2 = build_context_caption(ctx)
        assert cap2.endswith("邻近:盟友; 对方意图:结盟; 社交近况:新结盟; 人情:有人欠我; 凶器:有"), cap2[-100:]

        # v3 下 build_input 仍逐字包住新 caption（train-serve 包装不变）
        txt = build_input(cap2, Candidate("Ally", "daily").to_payload())
        assert "邻近:盟友" in txt and "来源：日常行为" in txt
    finally:
        if old is None:
            os.environ.pop("PRISON_CAPTION", None)
        else:
            os.environ["PRISON_CAPTION"] = old


if __name__ == "__main__":
    test_source_cn_matches()
    test_render_matches_expected()
    test_all_sources_render()
    test_caption_new_axes_pinned()
    test_caption_v3_social_axes_pinned()
    print("✅ train-serve 一致性测试全部通过")
