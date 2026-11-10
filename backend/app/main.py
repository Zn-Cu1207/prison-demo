"""FastAPI 入口：仿真循环后台任务 + WebSocket 实时同步 + 健康/探活接口。

三层解耦中的"后端·游戏大脑"。前端经 WebSocket 收世界状态帧，后端经 HTTP 调价值模型。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import BRAIN, CHAT_BACKEND, MODEL_URL, TICK_INTERVAL_S
from .deepseek_client import (
    LLMBrain,
    convo_director,
    monologue_director,
    roleplay_system,
    roleplay_warden_brief,
)
from .model_client import ModelClient
from .plot import ANCHOR_ID, SOFT_SPOT, outcome_lines, wants_escape_tool
from .shift import FREE_CHAT_TIMEOUT_S, INTERROGATE_TIMEOUT_S, interrogate_pack
from .world import World

# 对话触发的世界改变里，值得单独给前端弹 toast 的强事件（其余靠黑板同步即可）
_STRONG_NOTES = {"smuggler_deal", "doctor_alcohol", "guard_bribed", "guard_offended",
                 "oldtooth_exposed", "warden_cornered", "doctor_exposed", "smuggler_busted"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prison-demo")


class Hub:
    """管理所有 WebSocket 连接，负责广播世界帧。"""

    def __init__(self) -> None:
        self._conns: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._conns.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._conns.discard(ws)

    async def broadcast(self, message: dict) -> None:
        dead: list[WebSocket] = []
        # 快照迭代：send_json 会 await，让出事件循环；此时客户端断开会调 disconnect
        # 改动 _conns，直接迭代原集合会 "Set changed size during iteration"（刷新页面时高发）。
        for ws in list(self._conns):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def count(self) -> int:
        return len(self._conns)


hub = Hub()
if BRAIN == "local":
    model = LLMBrain.local()
elif BRAIN == "deepseek":
    model = LLMBrain.deepseek()
else:
    model = ModelClient()
world = World(model)

# Configure the provider endpoint and optional capabilities for your own environment.
# 否则用本地 Ollama。只管 NPC↔NPC 对话/开场白/结局的发声，不碰决策（决策始终价值模型）。
if CHAT_BACKEND == "opus":
    chat_llm = LLMBrain.opus()
else:
    chat_llm = model if isinstance(model, LLMBrain) else LLMBrain.local()
log.info("对话大脑：%s", chat_llm.label)

# 对比大脑：裸的本地大模型（未微调），只在玩家查看某 NPC 决策时按需给同一套候选打分，
# 与微调价值模型并排。这是整个 Demo 的核心输出点——微调相对"裸本地大模型"多了什么，
# 直接摆在屏幕上让玩家看见：护栏是否守序、速度/成本差多少。**始终留在本地**：
# 即便对话切到 Opus，这一路也不能变成 Opus，否则"微调 vs 裸本地"的对照就被污染了。
compare_llm = model if isinstance(model, LLMBrain) else LLMBrain.local()

# NPC↔NPC 对话懒生成的在途任务：一次只放一场，避免把本地 7B 打爆
_convo_task: asyncio.Task | None = None

# LLM 独白（自言自语气泡）交给对话脑：一次产 8 句缓存逐句放，生成频率很低。
# 对话/开场白/结局/独白全部走对话脑——本地模式=全程无 API（微调模型也练过独白，格式一致）。
# 本地单卡还要同时扛 NPC↔NPC 对话，独白并发压到 1，别和台词抢显存把节奏拖慢。
LLM_MONO = True
MONO_MAX_INFLIGHT = 1 if CHAT_BACKEND == "local" else 2
MONO_USER = "开始，直接输出含 8 句内心独白的 JSON 数组。"
_mono_inflight: set[str] = set()
_mono_tasks: set[asyncio.Task] = set()   # 独白在途任务引用，供玩家开口时整批掐断

# 玩家对话优先窗口：玩家一开口，接下来这么多秒内暂停所有后台 NPC 对话/独白，
# 把本地 7B 全留给玩家这一路。多轮追问因此都能瞬间拿到 GPU，不被 760-token 后台生成挡住。
# 窗口在每轮问答后刷新；玩家停下超过这个时长，周围 NPC 才恢复自说自话。
PLAYER_CHAT_HOLD_S = 8.0
_player_chat_hold_until = 0.0


def _mono_mood(npc) -> str:
    """按 NPC 此刻状态挑一个独白心境（喂给 monologue_director）——让心里话贴合处境。"""
    n = npc.needs
    if getattr(npc, "in_bout", False):
        return "刚跟人动了手，肾上腺素还没退，脑子里全是狠劲和后账"
    if npc.quest_pressure >= 0.85:
        return "主线大事被眼前的吃喝拉撒一再耽搁，你按捺不住、心里正拧着劲，惦记着该去办的那件事"
    if max(getattr(n, "injury", 0), getattr(n, "hunger", 0), getattr(n, "thirst", 0)) >= 75:
        return "身子熬得快散架，可有件事怎么都撂不下"
    if npc.grudge and max(npc.grudge.values(), default=0) >= 60:
        return "心里记着一笔账、惦记着一个人，越想越咽不下这口气"
    if npc.quest_pressure >= 0.4:
        return "此刻还算安生，可心思一刻没停，盘算着往后那步棋怎么走"
    return "一个人待着，脑子里过着这地方的人和事、自己的盘算"


def _parse_mono(raw: str) -> list[str]:
    """从 Opus 回复里抠出独白：优先 JSON 字符串数组，退化则按行切。每句 6~40 字。"""
    t = (raw or "").strip()
    lines: list[str] = []
    i, j = t.find("["), t.rfind("]")
    if 0 <= i < j:
        with contextlib.suppress(Exception):
            arr = json.loads(t[i:j + 1])
            if isinstance(arr, list):
                for s in arr:
                    if isinstance(s, dict):
                        s = s.get("text") or s.get("say") or s.get("line") or ""
                    lines.append(str(s))
    if not lines:                                  # 没有合法数组：按行兜底
        lines = [ln.strip(" 　-·•\"'“”[]") for ln in t.splitlines()]
    out, seen = [], set()
    for s in lines:
        s = s.strip().strip("“”\"'")
        if 6 <= len(s) <= 40 and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:8]


async def _run_mono(npc_id: str) -> None:
    """给一个 NPC 灌一池 Opus 独白（8 句心里话）。目标在生成途中变了就丢弃（配音一致优先）。"""
    try:
        ctx = world.npc_chat_context(npc_id)
        npc = world._npc(npc_id)
        if not ctx or npc is None:
            return
        goal = npc.goal
        raw = await chat_llm.chat(monologue_director(ctx, _mono_mood(npc)), [],
                                  MONO_USER, max_tokens=520)
        lines = _parse_mono(raw)
        if lines and world._npc(npc_id) is not None:
            world.set_mono(npc_id, goal, lines)
    except Exception as e:  # noqa: BLE001
        log.debug("独白生成失败 %s：%s", npc_id, e)
    finally:
        _mono_inflight.discard(npc_id)


def _parse_convo(raw: str, a_id: str, b_id: str, a_name: str, b_name: str) -> list[dict]:
    """把 LLM 产出的 JSON 台词脚本解析成 [{who,name,text}]。健壮兜底：取首个[到末个]。"""
    if not raw:
        return []
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        arr = json.loads(raw[i:j + 1])
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict):
            continue
        who = str(item.get("who", "")).strip().lower()
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if who in ("a", a_name.lower()) or who == a_id:
            wid, wname = a_id, a_name
        elif who in ("b", b_name.lower()) or who == b_id:
            wid, wname = b_id, b_name
        else:                                   # who 认不出 → 按已产出条数交替兜底
            wid, wname = (a_id, a_name) if len(out) % 2 == 0 else (b_id, b_name)
        out.append({"who": wid, "name": wname, "text": text[:80]})
        if len(out) >= 6:
            break
    return out


async def _run_convo(pending: dict) -> None:
    """懒生成一场 NPC↔NPC 对话：调一次本地 LLM 出整段脚本，落地后广播给前端逐字放。"""
    a_id, b_id = pending["a"], pending["b"]
    a_ctx = world.npc_chat_context(a_id, other_id=b_id)
    b_ctx = world.npc_chat_context(b_id, other_id=a_id)
    if not a_ctx or not b_ctx:
        world.convo_fail(8)
        return
    system = convo_director(a_ctx, b_ctx, pending["topic"], pending["kind"])
    world.chat_active += 1              # 生成期间让决策打分让路（和玩家对话共用背压）
    world.interrupt_scoring()
    raw = ""
    try:
        raw = await chat_llm.chat(system, [], "开始，直接输出含 6 句的 JSON 数组。", max_tokens=760)
    except Exception as e:  # noqa: BLE001
        log.debug("NPC对话生成失败：%s", e)
    finally:
        world.chat_active = max(0, world.chat_active - 1)
    lines = _parse_convo(raw, a_id, b_id, a_ctx["name"], b_ctx["name"])
    if len(lines) < 2:            # 单句不成对话，丢弃重来（本地小模型偶尔只吐一句）
        world.convo_fail(10)
        return
    # 生成 LLM 脚本要好几秒，这期间可能有人被杀/进禁闭/打起来——绝不能让死人开口说话。
    # 生成后复检：任一方已不再"可攀谈"就整场丢弃（既不写回世界，也不广播）。
    a_npc, b_npc = world._npc(a_id), world._npc(b_id)
    if (not a_npc or not b_npc
            or not world._convo_active(a_npc) or not world._convo_active(b_npc)):
        world.convo_fail(8)
        log.debug("对话作废：生成期间 %s/%s 有人身亡或离场", a_id, b_id)
        return
    # P1：把这场对白解析成社交事件并写回世界（谈成军火→真给刀、放狠话→梁子更深…），
    # 后果既进命运账本(前端已渲染)，也随 convo 消息发给前端提示"这次谈话改变了什么"。
    writeback = world.ingest_conversation(a_id, b_id, pending["kind"], lines)
    msg = world.start_conversation(a_id, b_id, pending["topic"], lines)
    msg["consequences"] = writeback.get("consequences", [])
    log.info("NPC对话：%s↔%s（%s）%d句%s", a_ctx["name"], b_ctx["name"], pending["kind"],
             len(lines), "，后果:" + "；".join(writeback["consequences"]) if writeback["consequences"] else "")
    with contextlib.suppress(Exception):
        await hub.broadcast({"type": "convo", **msg})


async def _sim_loop() -> None:
    """后台仿真循环：每 TICK_INTERVAL_S 推进一步并广播。"""
    log.info("仿真循环启动，tick=%.2fs，大脑=%s，MODEL_URL=%s",
             TICK_INTERVAL_S, getattr(model, "label", "?"), MODEL_URL)
    # 预热：LLM 大脑首次调用要把模型载入显存（可能数十秒），先 ping 一下避免首帧长时间空白
    if hasattr(model, "ping"):
        try:
            log.info("预热决策脑…%s", await model.ping())
        except Exception:  # noqa: BLE001
            log.exception("决策脑预热失败")
    # 关键：也预热『语言脑』(本地 7B)。冷加载要把 4.5G 装回显存 ≈4s 首字，热态仅 ~0.4s。
    # 之前这里只 ping 了决策脑，7B 从没被预热 → 后端起来后第一次搭话/独白必吃 4 秒。
    if CHAT_BACKEND != "opus" and chat_llm is not model and hasattr(chat_llm, "ping"):
        try:
            log.info("预热语言脑…%s", await chat_llm.ping())
        except Exception:  # noqa: BLE001
            log.exception("语言脑预热失败")
    global _convo_task
    ka = 0                       # 保活计数：本地 7B 空闲 >5min 会被 Ollama 卸载 → 首字回落到冷加载
    while True:
        try:
            await world.tick(TICK_INTERVAL_S)
            await hub.broadcast(world.world_state_message())
            # M0 单 NPC：顺带把它的决策明细也广播出去（M2 会做成切视角面板）
            for npc in world.npcs:
                await hub.broadcast(world.npc_decision_message(npc))
            # 玩家正在打字对话时，本地语言脑要全力伺候这一路（首字/流式速度）——
            # 别再新起 NPC↔NPC 对话或独白去抢同一块 GPU，否则玩家回话被排到几百 token 后面，奇慢。
            # （chat_active 在玩家对话与 NPC 对话生成期间都会 +1，这里只在完全空闲时才放新活。）
            # 再叠加"玩家对话优先窗口"：每轮问答后 PLAYER_CHAT_HOLD_S 秒内也不放后台活，
            # 让玩家连续追问都吃满 GPU；玩家停下超过这个时长，周围 NPC 才恢复自说自话。
            llm_free = world.chat_active == 0 and time.monotonic() >= _player_chat_hold_until
            # 观察者：NPC↔NPC 对话懒生成——有排队、有观众、没暂停、上一场已收尾、语言脑空闲，才起新的一场
            if (llm_free and world.convo_pending and hub.count > 0 and not world.paused
                    and (_convo_task is None or _convo_task.done())):
                p = world.claim_convo()
                if p:
                    _convo_task = asyncio.create_task(_run_convo(p))
            # 观察者：自言自语气泡也让 Opus 来写（一次产 8 句缓存逐句放）。有观众、没暂停、语言脑空闲时才刷，
            # 并发封顶 MONO_MAX_INFLIGHT，别把网关打爆。目标刚变的 NPC 优先（画面上最突兀）。
            if LLM_MONO and llm_free and hub.count > 0 and not world.paused:
                free = MONO_MAX_INFLIGHT - len(_mono_inflight)
                if free > 0:
                    for npc in world.mono_candidates():
                        if npc.id in _mono_inflight:
                            continue
                        _mono_inflight.add(npc.id)
                        mt = asyncio.create_task(_run_mono(npc.id))
                        _mono_tasks.add(mt)
                        mt.add_done_callback(_mono_tasks.discard)
                        free -= 1
                        if free <= 0:
                            break
            # 保活：每 ~60s，若此刻语言脑空闲（没在生成对话/独白），补一发极小 ping。
            # 目的是把 Ollama 的 5min 空闲卸载计时器一直按住 → 模型常驻显存 → 首字恒 ~0.4s，
            # 哪怕演示中途暂停讲解好几分钟，回来点 NPC 也不再吃 4 秒冷加载。
            # 只在空闲时打，绝不和正在跑的生成抢本地 GPU。
            ka += 1
            if (ka % 120 == 0 and CHAT_BACKEND != "opus" and hasattr(chat_llm, "ping")
                    and not _mono_inflight and (_convo_task is None or _convo_task.done())):
                asyncio.create_task(chat_llm.ping())
        except Exception:
            log.exception("tick 出错（可能是价值模型服务未启动）")
        await asyncio.sleep(TICK_INTERVAL_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_sim_loop())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await world.aclose()      # 等在途打分批次落地，再关连接
    await model.aclose()
    if chat_llm is not model:
        await chat_llm.aclose()


app = FastAPI(title="监狱 NPC Demo · 后端", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "tick": world.tick_count, "clients": hub.count,
            "brain": getattr(model, "label", "?"), "model_url": MODEL_URL}


@app.get("/metrics")
async def metrics() -> dict:
    """M5：导出遥测（放风人数序列 + 各性格选择分布 + 统计），供出数据图。"""
    return world.metrics()


@app.get("/model_ping")
async def model_ping() -> dict:
    """探活价值模型服务，验证 M0 的 "/score 打通"。"""
    try:
        result = await model.ping()
        return {"ok": True, "model_url": MODEL_URL, "response": result}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "model_url": MODEL_URL, "error": str(e)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await hub.connect(ws)
    log.info("客户端接入，当前 %d 个连接", hub.count)
    try:
        # 接入即先推一帧，避免前端等一个 tick 才有画面
        await ws.send_json(world.world_state_message())
        while True:
            raw = await ws.receive_text()
            await _handle_client_message(ws, raw)
    except WebSocketDisconnect:
        hub.disconnect(ws)
        log.info("客户端断开，当前 %d 个连接", hub.count)


async def _handle_client_message(ws: WebSocket, raw: str) -> None:
    """处理前端玩家输入（下指令 / 挑事 / 招牌片段 / 对话）。"""
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return
    kind = msg.get("type")
    if kind == "player_cmd":
        world.set_command(msg.get("npc_id", ""), msg.get("goal"))
        log.info("玩家指令：%s → %s", msg.get("npc_id"), msg.get("goal"))
    elif kind == "event":
        world.inject_event(msg.get("kind"))
        log.info("玩家挑事：注入事件 %s", msg.get("kind"))
    elif kind == "scenario":
        desc = world.run_scenario(msg.get("name", ""))
        log.info("招牌片段：%s（%s）", msg.get("name"), desc)
    elif kind == "case_progress":
        world.set_case_progress(msg.get("found", 0))
    elif kind == "take_object":
        # 玩家从世界里拿走一份物资。走的是和 NPC 完全同一批 object，
        # 所以"玩家把最后一个窝头拿走了"会真的让 NPC 吃不上饭（还会被记恨）。
        res = world.take_object(msg.get("id", ""))
        await ws.send_json({"type": "take_result", **res})
    elif kind == "player_pos":
        # 玩家位置：物理层要用（群殴误伤/被记恨的人贴上来动手都按距离判）
        world.set_player_pos(msg.get("x", 0.0), msg.get("z", 0.0))
    elif kind == "use_item":
        res = world.player_use(msg.get("kind", ""))
        await ws.send_json({"type": "use_result", **res})
    elif kind == "clue_found":
        # 检视到一条线索：记进证据库、推进案件、检查逐级解锁（可能解锁受限/办公室区）
        eff = world.on_clue_found(msg.get("id", ""))
        if eff.get("new") or eff.get("unlocked"):
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "bb_event", **eff})
    elif kind == "incite_riot":
        # 冲动路线（大胆分叉）：证据够多就铤而走险，不做全套准备直接挑起暴乱、硬闯办公室
        eff = world.incite_riot()
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "bb_event", **eff})
    elif kind == "observer_finale":
        # 观察者模式：上帝随时发起最终暴乱并进入结算（不受证据/分区门槛）
        eff = world.observer_finale()
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "bb_event", **eff})
            await hub.broadcast(world.world_state_message())
    elif kind == "drug_dog":
        # 阶段三：把镇静肉丢给哮天 → 迷晕它（消耗镇静剂）
        eff = world.drug_dog()
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "bb_event", **eff})
    elif kind == "plead":
        # 阶段三潜行：被撞见时狡辩，判定借口够不够严密
        res = world.plead(msg.get("text", ""))
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "plead_result", **res})
    elif kind == "chat":
        # 对话可能耗时数秒（LLM），放到后台任务里，别阻塞接收循环与仿真
        asyncio.create_task(_handle_chat(ws, msg))
    elif kind == "generate_ending":
        # 阶段五：读整块黑板，让 DM 大模型生成本局专属结局（流式吐字）
        asyncio.create_task(_handle_ending(ws, msg))
    elif kind == "phase_intro":
        # 分阶段开场白：指定核心 NPC 读人设+黑板，即兴生成一段开场台词（流式）
        asyncio.create_task(_handle_phase_intro(ws, msg))
    elif kind == "compare_brains":
        # Demo 核心：同一情境同一候选，微调价值模型 vs 裸本地大模型 并排出结果
        asyncio.create_task(_handle_compare(ws, msg))
    elif kind == "pause":
        # 观察者模式·上帝暂停/继续：定格世界逐个查看 NPC
        world.set_paused(bool(msg.get("value", True)))
        with contextlib.suppress(Exception):
            await hub.broadcast(world.world_state_message())
    elif kind == "warden_weapon":
        # 典狱长手段：没收 / 栽赃某 NPC 的杀人工具（改的是决策脑下一拍看到的处境）
        res = world.warden_weapon(msg.get("npc_id", ""), msg.get("op", ""), msg.get("kind", ""))
        with contextlib.suppress(Exception):
            await hub.broadcast(world.world_state_message())   # 先刷新，前端 warden_result 里能读到新凶器态
            await ws.send_json({"type": "warden_result", **res})
        log.info("典狱长·凶器：%s %s %s → %s", msg.get("npc_id"), msg.get("op"), msg.get("kind"), res.get("msg"))
    elif kind == "warden_grudge":
        # 典狱长手段：挑拨(正) / 调解(负) 两名 NPC 之间的仇怨
        try:
            delta = float(msg.get("delta", 0.0))
        except (TypeError, ValueError):
            delta = 0.0
        res = world.warden_grudge(msg.get("from_id", ""), msg.get("to_id", ""), delta)
        with contextlib.suppress(Exception):
            await hub.broadcast(world.world_state_message())
            await ws.send_json({"type": "warden_result", **res})
        log.info("典狱长·仇怨：%s→%s %+.0f → %s", msg.get("from_id"), msg.get("to_id"), delta, res.get("msg"))
    elif kind == "warden_solitary":
        # 典狱长手段：把某 NPC 直接传送进禁闭室关禁闭（狱警职权）
        res = world.warden_solitary(msg.get("npc_id", ""))
        with contextlib.suppress(Exception):
            await hub.broadcast(world.world_state_message())
            await ws.send_json({"type": "warden_result", **res})
        log.info("典狱长·禁闭：%s → %s", msg.get("npc_id"), res.get("msg"))
    elif kind == "cell_door":
        # 牢门：关/开谁都能做，**锁**是典狱长专属（前端只在典狱长面板给这个入口）。
        # 锁上之后后端寻路网格里那道门就是实墙，人真的出不来（见 nav.set_gate）。
        try:
            slot = int(msg.get("slot", -1))
        except (TypeError, ValueError):
            slot = -1
        kw: dict[str, bool] = {}
        if "locked" in msg:
            kw["locked"] = bool(msg.get("locked"))
        if "closed" in msg:
            kw["closed"] = bool(msg.get("closed"))
        res = world.set_cell_door(slot, **kw)
        with contextlib.suppress(Exception):
            await hub.broadcast(world.world_state_message())
            await ws.send_json({"type": "warden_result", **res})
        log.info("牢门：#%s %s → %s", slot, kw, res.get("msg"))
    elif kind == "solitary_lock":
        # 禁闭室的门：典狱长专属的开关锁（和牢门一套逻辑，锁上后 nav 网格里那道门变实墙）
        res = world.set_solitary_lock(bool(msg.get("locked")))
        with contextlib.suppress(Exception):
            await hub.broadcast(world.world_state_message())
            await ws.send_json({"type": "warden_result", **res})
        log.info("禁闭室门：locked=%s → %s", msg.get("locked"), res.get("msg"))
    elif kind == "warden_recall":
        # 典狱长点名：把某 NPC 或全体遣送回牢房 / 关进禁闭室
        res = world.warden_recall(msg.get("target", ""), msg.get("dest", "cell"))
        with contextlib.suppress(Exception):
            await hub.broadcast(world.world_state_message())
            await ws.send_json({"type": "warden_result", **res})
        log.info("典狱长·点名：%s → %s（%s）", msg.get("target"), msg.get("dest"), res.get("msg"))
    elif kind == "shift_act":
        # 公开夜班：权力点门闩 + 事件处置。研究模式仍走上面的 warden_* 直通车。
        res = world.shift_act(msg.get("action", ""), msg)
        with contextlib.suppress(Exception):
            await hub.broadcast(world.world_state_message())
            await ws.send_json({"type": "shift_result", **res})
        log.info("夜班行动：%s → %s", msg.get("action"), res.get("msg"))


def _sniff_warden_intent(text: str) -> str:
    """自由输入也尽量对上四个审讯槽，好让招供能进卷宗。"""
    t = text or ""
    if any(k in t for k in ("东西", "藏", "凶器", "刀", "药", "身上", "哪来")):
        return "where_item_from"
    if any(k in t for k in ("过不去", "仇", "打架", "跟谁", "谁啊")):
        return "who_conflict_with"
    if any(k in t for k in ("哪儿", "哪里", "刚才", "去哪")):
        return "where_were_you"
    if any(k in t for k in ("盘算", "打算", "干什么", "图什么", "今夜",
                            "老实交代", "交代", "越狱", "跑路", "图谋", "计划", "想干嘛")):
        return "what_are_you_planning"
    return ""


async def _handle_chat(ws: WebSocket, msg: dict) -> None:
    global _convo_task, _player_chat_hold_until
    npc_id = msg.get("npc_id", "")
    text = (msg.get("text") or "").strip()
    history = msg.get("history") or []
    ctx = world.npc_chat_context(npc_id)
    if not ctx:
        return
    # 玩家开口＝独占语言脑：立刻掐掉所有正在跑的后台生成（NPC↔NPC 对话一次 760 token、
    # 独白 520 token，都很占 GPU），并开启优先窗口暂停后续后台活，把 7B 全留给玩家这一句。
    # 否则玩家这句流式排在它们后面，必然撞上超时、只能回退模板。
    _player_chat_hold_until = time.monotonic() + PLAYER_CHAT_HOLD_S
    if _convo_task is not None and not _convo_task.done():
        _convo_task.cancel()
    for mt in list(_mono_tasks):
        if not mt.done():
            mt.cancel()
    as_warden = bool(msg.get("as_warden", True)) or bool(getattr(world, "warden_active", False))
    ctx["as_warden"] = as_warden
    # 典狱长审讯：代码先定披露等级，7B 只表演准许说的事实。超时走模板。
    intent = (msg.get("intent") or "").strip()
    if not intent and as_warden:
        intent = _sniff_warden_intent(text)
    if not intent and as_warden:
        brief = (ctx.get("incident") or {}) if ctx else {}
        intent = (brief.get("ask_intent") or "").strip()
        if intent and not text:
            text = brief.get("ask_label") or ""
    pack: dict = {}
    if intent:
        pack = interrogate_pack(world, npc_id, intent, asked_text=text)
        ctx["warden_disclose"] = pack
        text = text or (pack.get("question") or "")
    if not text:
        return
    if as_warden and not str(text).startswith("【典狱长"):
        text = f"【典狱长当面问你】{text}"
    ps = msg.get("player_state")
    if isinstance(ps, dict):
        ctx["player_state"] = ps        # 玩家此刻在干嘛/拿着啥/在哪，供 NPC 就地反应
    # 阶段一·向导破冰：跟老牙流露出想出这间牢房/要开锁工具 → 他顺水推舟给磨尖牙刷，
    # 后台开牢门、解锁公共区（俯视图随即多长出一片）。事实库已让他在台词里自然递出来。
    anchor_gave = False
    if npc_id == ANCHOR_ID and wants_escape_tool(text):
        eff = world.anchor_give_tool()
        if eff.get("new"):
            anchor_gave = True
            with contextlib.suppress(Exception):
                await ws.send_json({
                    "type": "bb_event", **eff,
                    "toast": "老牙压低声音，把一把磨尖的牙刷塞进你手心："
                             "「先拿着，牢门那把锁我教你怎么撬。」",
                })
            log.info("阶段一：老牙把牙刷给了玩家，解锁分区=%s", eff.get("unlocked_zone"))
    # 阶段二/三·对话改世界（两轨第一轨）：解析这句话的意图，据角色/证据/道具改黑板
    #   ——换磁卡镇静剂、收买铁面、揭穿老牙、逐级解锁，全在这里被锁进确定性状态机。
    ieff = world.resolve_interaction(npc_id, text)
    # 引擎已经把后果落地了（给道具/解锁/翻幕），把"已成事实"喂给对话，
    # 否则 LLM 不知道自己刚做了什么，就会出现嘴上说不给、东西却已经到手。
    ctx["outcome"] = outcome_lines(ieff)
    if anchor_gave:
        ctx["outcome"].insert(0, "你刚刚已经把那把磨尖的牙刷塞进他手里了，这是既成事实。"
                                 "回话建立在「东西已经给了」之上，顺势教他两句怎么撬那把锁，"
                                 "别拒绝、也别说要再想想。")
    notes = ieff.get("notes") or []
    if ieff.get("toast") or ieff.get("unlocked") or (set(notes) & _STRONG_NOTES):
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "bb_event", **ieff})
        log.info("对话改世界：%s intent=%s notes=%s unlocked=%s",
                 npc_id, ieff.get("intent"), notes, ieff.get("unlocked"))
    # 对话即指令：玩家在聊天里直接吩咐"去做X" → 下正规指令，NPC 接下来真会去做
    order = None
    # 典狱长的调度令（回牢房/关禁闭）硬编码兑现，不走模型软指令、点了就全员必听。
    if as_warden:
        wcmd = world.warden_command_from_text(npc_id, text)
        if wcmd and wcmd.get("ok"):
            order = wcmd["label"]
            ctx["player_order"] = order
            with contextlib.suppress(Exception):
                await hub.broadcast(world.world_state_message())
                await ws.send_json({"type": "warden_result", **wcmd})
                await ws.send_json({"type": "chat_sys", "npc_id": npc_id,
                                    "text": f"（典狱长一声令下：{wcmd.get('msg') or order}）"})
            log.info("典狱长·聊天硬指令：%s → %s（%s）", npc_id, wcmd.get("dest"), wcmd.get("msg"))
    if order is None:
        order = world.try_command_from_text(npc_id, text)
        if order:
            ctx["player_order"] = order
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "chat_sys", "npc_id": npc_id,
                                    "text": f"（你吩咐 {ctx['name']} 去{order}）"})
    # 典狱长问话用短提示，长稿预填充会卡好几秒。
    system = roleplay_warden_brief(ctx) if as_warden else roleplay_system(ctx)
    name = ctx["name"]
    world.chat_active += 1              # 对话期间让决策打分给模型让路，回话更快
    world.interrupt_scoring()          # 打断已在跑的那一批，别让对话首字干等它
    buf: list[str] = []
    template = (pack.get("template") if pack else "") or ""
    fallback = template or "……狱长，我一时不知怎么说。"
    used_llm = False
    max_tok = 48 if intent else 64
    wait_s = INTERROGATE_TIMEOUT_S if intent else FREE_CHAT_TIMEOUT_S

    # 不再"模板先行"：以前先秒回一句模板、几秒后再替换成 7B，屏幕上一闪一换观感很差。
    # 现在统一等 7B（前端显示"正在想…"），模板只在超时/失败时才作为兜底出现（见下方 except）。

    async def _consume_stream() -> None:
        nonlocal used_llm
        async for delta in chat_llm.chat_stream(system, history, text, max_tokens=max_tok):
            if not used_llm:
                used_llm = True
                buf.clear()
                buf.append(delta)
                with contextlib.suppress(Exception):
                    await ws.send_json({"type": "chat_delta", "npc_id": npc_id,
                                        "name": name, "delta": delta, "replace": True})
            else:
                buf.append(delta)
                with contextlib.suppress(Exception):
                    await ws.send_json({"type": "chat_delta", "npc_id": npc_id,
                                        "name": name, "delta": delta})

    try:
        try:
            async with asyncio.timeout(wait_s):
                await _consume_stream()
        except TimeoutError:
            if not used_llm:
                if not "".join(buf).strip():
                    buf.append(fallback)
                    with contextlib.suppress(Exception):
                        await ws.send_json({"type": "chat_delta", "npc_id": npc_id,
                                            "name": name, "delta": fallback,
                                            "replace": True})
                log.info("对话超时，用模板兜底：%s intent=%s", npc_id, intent or "-")
            else:
                log.info("对话截断：%s 已有部分生成", npc_id)
    except Exception as e:  # noqa: BLE001
        log.warning("对话失败：%s", e)
        if not "".join(buf).strip():
            buf.append(fallback)
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "chat_delta", "npc_id": npc_id,
                                    "name": name, "delta": fallback, "replace": True})
    finally:
        world.chat_active = max(0, world.chat_active - 1)
        # 刷新优先窗口：本轮答完后再压 PLAYER_CHAT_HOLD_S 秒，让紧接着的追问仍吃满 GPU。
        _player_chat_hold_until = time.monotonic() + PLAYER_CHAT_HOLD_S
    reply = "".join(buf).strip() or fallback
    learned = (pack.get("learned") if pack else None) or []
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "chat_reply", "npc_id": npc_id,
                            "name": name, "text": reply, "intel": learned})
    if learned:
        with contextlib.suppress(Exception):
            await hub.broadcast(world.world_state_message())
    # 关系判官再调一次本地模型，会卡住下一句问话。典狱长夜班不跑；研究模式后台跑。
    softspot = SOFT_SPOT.get(npc_id)
    if softspot and isinstance(chat_llm, LLMBrain) and not as_warden:
        async def _softspot_later() -> None:
            try:
                eff = await chat_llm.parse_effect(
                    name, softspot, history + [{"role": "assistant", "content": reply}], text)
                if eff.get("trust_delta") or eff.get("hit") != "none":
                    rel = world.apply_relationship(npc_id, eff.get("trust_delta", 0))
                    with contextlib.suppress(Exception):
                        await ws.send_json({"type": "bb_event", **rel,
                                            "hit": eff.get("hit"), "note": eff.get("reason", "")})
                    log.info("关系判官：%s Δ好感=%+d hit=%s（%s）",
                             npc_id, rel.get("trust_delta", 0),
                             eff.get("hit"), eff.get("reason"))
            except Exception as e:  # noqa: BLE001
                log.debug("关系判官流程异常：%s", e)

        asyncio.create_task(_softspot_later())


# —— 两脑对比（Demo 核心输出点）——————————————————————————————
# 濒死级需求 → 唯一正解的救命动作。裸模型在这上头最容易失守（命都顾不上还去干别的）。
_CRISIS_ACTION = {"hunger": ("Eat", "吃饭"), "thirst": ("Drink", "喝水"),
                  "fatigue": ("Sleep", "睡觉"), "injury": ("Cure", "治伤")}
_SURVIVAL_GOALS = {"Eat", "Drink", "Sleep", "Cure"}
_CRISIS_LEVEL = 82   # 与 world.EMERGENCY_NEED 对齐


def _compare_verdict(needs: dict, value_top, base_top, value_label: str, base_label: str) -> dict:
    """判定两脑分歧、尤其裸模型有没有违背『濒死优先』护栏——这是核心卖点的自动佐证。

    只要有需求濒死：微调选了任一救命动作、而裸模型没有，就算护栏失守（不苛求它一定
    选那个"最高"的需求——先保命这件事本身才是护栏）。
    """
    dying = sorted(((needs.get(k, 0), k) for k in _CRISIS_ACTION
                    if needs.get(k, 0) >= _CRISIS_LEVEL), reverse=True)
    agree = value_top == base_top
    if dying:
        worst = dying[0][1]
        correct, correct_cn = _CRISIS_ACTION[worst]
        v_surv, b_surv = value_top in _SURVIVAL_GOALS, base_top in _SURVIVAL_GOALS
        if v_surv and not b_surv:
            same = "" if value_top == correct else f"（选了『{value_label}』，同属救命动作）"
            return {"agree": False, "guardrail": "violated",
                    "note": f"该需求已濒死（{correct_cn}才是正解）。微调价值模型守住了"
                            f"『濒死优先』护栏{same}；裸模型却想去「{base_label}」——这正是"
                            f"本地裸模型最典型的失守：优先级会漂、顾不上命。"}
        if v_surv and b_surv:
            return {"agree": agree, "guardrail": "both_ok",
                    "note": "危机当前两脑都落在救命动作上；差别在别处——速度、成本、一致性。"}
        return {"agree": agree, "guardrail": "value_miss",
                "note": "本例微调侧也没落在救命动作上（可能缓存滞后/引擎另有裁决），留意复查。"}
    if agree:
        return {"agree": True, "guardrail": "n/a",
                "note": "非危机情形两脑判断一致；此处对比看的是每次决策的速度与成本。"}
    return {"agree": False, "guardrail": "n/a",
            "note": f"非危机情形两脑倾向不同：微调→「{value_label}」，裸模型→「{base_label}」。"}


async def _handle_compare(ws: WebSocket, msg: dict) -> None:
    snap = world.brain_compare_snapshot(msg.get("npc_id", ""))
    if not snap:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "brain_compare", "ok": False,
                                "reason": "该 NPC 这一拍还没有可对比的决策（可能刚重置或在禁闭）。"})
        return
    cands = snap["candidates"]
    world.chat_active += 1          # 让本地 7B 给对比让路的那套背压也管住它
    world.interrupt_scoring()
    t0 = time.perf_counter()
    base_scored = base_say = None
    try:
        scored_all, says = await compare_llm.evaluate(
            [(snap["caption"], cands, snap["personality"])])
        base_scored, base_say = scored_all[0], says[0]
    except Exception as e:  # noqa: BLE001
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "brain_compare", "ok": False,
                                "reason": f"裸本地大模型未就绪：{e}"})
        return
    finally:
        world.chat_active = max(0, world.chat_active - 1)
    base_ms = (time.perf_counter() - t0) * 1000.0
    base_rank = sorted(
        [{"goal": sc.candidate.goal, "label": sc.candidate.label, "want": round(sc.v, 1)}
         for sc in base_scored], key=lambda r: -r["want"])
    base_top = base_rank[0]["goal"] if base_rank else None
    base_label = base_rank[0]["label"] if base_rank else "?"
    value_top = snap["value_top"]
    value_label = next((r["label"] for r in snap["value_rank"] if r["goal"] == value_top), "?")
    verdict = _compare_verdict(snap["needs"], value_top, base_top, value_label, base_label)
    n = max(1, world.n_agents)
    with contextlib.suppress(Exception):
        await ws.send_json({
            "type": "brain_compare", "ok": True,
            "npc_id": snap["npc_id"], "name": snap["name"], "personality": snap["personality"],
            "caption": snap["caption"], "needs": snap["needs"], "drivers": snap["drivers"],
            "value": {
                "label": "微调价值模型 · Qwen2.5-1.5B+LoRA",
                "top": value_top, "top_label": value_label,
                "rank": snap["value_rank"][:6],
                # 价值模型是跨 NPC 合批一次前向：整批耗时摊到每人，几乎可忽略
                "ms_per_npc": round(world.score_ms / n, 1), "batch_ms": round(world.score_ms),
            },
            "base": {
                "label": f"裸本地大模型 · {getattr(compare_llm, 'model', '?')}（未微调）",
                "top": base_top, "top_label": base_label, "rank": base_rank[:6],
                "say": base_say, "ms": round(base_ms),
            },
            "verdict": verdict,
        })
    log.info("两脑对比 %s：微调→%s 裸→%s 护栏=%s",
             snap["npc_id"], value_top, base_top, verdict.get("guardrail"))


async def _handle_phase_intro(ws: WebSocket, msg: dict) -> None:
    """分阶段开场白：核心 NPC 主动开口。读其人设+黑板+导演指令，LLM 即兴生成并流式下发。"""
    zone = msg.get("zone", "")
    setup = world.phase_intro_setup(zone)
    if not setup:
        return
    guide, ctx, directive = setup
    system = roleplay_system(ctx) + "\n" + directive
    name = ctx["name"]
    log.info("阶段开场白：zone=%s 向导=%s(%s)", zone, guide, name)
    world.chat_active += 1              # 生成期间让决策打分让路，开场白更快蹦出来
    world.interrupt_scoring()
    buf: list[str] = []
    try:
        async for delta in chat_llm.chat_stream(
                system, [], "（新阶段开始，此刻你主动开口，对他说第一句话。）", max_tokens=150):
            buf.append(delta)
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "phase_intro_delta", "zone": zone,
                                    "npc_id": guide, "name": name, "delta": delta})
    except Exception as e:  # noqa: BLE001
        log.warning("开场白生成失败：%s", e)
    finally:
        world.chat_active = max(0, world.chat_active - 1)
    text = "".join(buf).strip()
    if text:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "phase_intro_reply", "zone": zone,
                                "npc_id": guide, "name": name, "text": text})


async def _handle_ending(ws: WebSocket, msg: dict) -> None:
    """专属结局：夜班走黎明旁白；研究模式仍走观察者暴乱稿。"""
    last = (msg.get("text") or "").strip()
    shift_dawn = bool(world.shift.ended) or bool(msg.get("shift"))
    if shift_dawn:
        if not world.shift.ended:
            world.shift.finish(world)
        system = world.shift.dawn_prompt(world)
        sim = world.sim_epilogue_lines()
        if sim:
            system += "\n\n【仿真补遗·不得冲突】\n" + "\n".join(sim)
        seed = "（天色将亮。就着今夜典狱长的处置和每个人真实的下落，收束这一班。）"
        log.info("生成夜班黎明结局：%s", world.shift.ending)
        extra = {"shift": True, "ending": world.shift.ending,
                 "ending_title": world.shift.ending_title}
    else:
        res = world.observer_resolution()
        system = world.ending_prompt(last)
        sim = world.sim_epilogue_lines()
        if sim:
            system += (
                "\n\n【这一局每个人真实的下落·按此为每个人定制不同结局，与上面的既成事实不得冲突】\n"
                + "\n".join(sim)
                + "\n\n收尾铁律：正文末尾必须为上面出现的每一个囚犯各写一句独立的命运判词，"
                "且要吃他各自的『终极目标完成度、结没结成仇、刑期/假释、是否横死或越狱』——"
                "把毕生图谋走到头的、功亏一篑的、横死的、越狱的、假释在望的，结局措辞与轻重都要明显不同，"
                "不能给所有人套同一句模板。谁的图谋越接近达成，谁的判词就越浓墨重彩。")
        seed = "（警铃炸响，探照灯扫过一间间号子。就着此刻牢里每个人真实的处境，收束这一夜。）"
        log.info("生成专属结局：tier=%s score=%s factors=%s sim=%d",
                 res["tier"], res["score"], res["factors"], len(sim))
        extra = res
    world.chat_active += 1
    world.interrupt_scoring()
    buf: list[str] = []
    try:
        async def _consume() -> None:
            async for delta in chat_llm.chat_stream(system, [], seed, max_tokens=900):
                buf.append(delta)
                with contextlib.suppress(Exception):
                    await ws.send_json({"type": "ending_delta", "delta": delta})
        if shift_dawn:
            try:
                async with asyncio.timeout(20.0):
                    await _consume()
            except TimeoutError:
                log.info("黎明旁白超时，用鉴定稿兜底")
        else:
            await _consume()
    except Exception as e:  # noqa: BLE001
        log.warning("结局生成失败：%s", e)
    finally:
        world.chat_active = max(0, world.chat_active - 1)
    text = "".join(buf).strip()
    if not text:
        text = world.shift.ending_blurb if shift_dawn else "（狱墙沉默，故事在黑暗里戛然而止。）"
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "ending_reply", "text": text, **extra})
