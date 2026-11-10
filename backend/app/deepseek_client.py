"""LLM "大脑"（实验对照）：把小价值模型换成通用 LLM 看效果。

走 OpenAI 兼容 Chat 接口，因此同时支持：
  - 本地 Ollama（http://127.0.0.1:11434/v1/chat/completions，免 key，如 qwen2.5:7b）
  - DeepSeek 云端（需 DEEPSEEK_API_KEY）
  - 任意其它 OpenAI 兼容端点

与 ModelClient 同接口 evaluate(items) -> (scored_lists, says)，可无缝切换。
一拍把所有 NPC 合并成 1 次请求（省延迟/费用）；LLM 为每个候选打 0~100 分，
并即兴生成一句符合性格的中文独白。任何失败都回退成均匀分 + 无独白（不崩）。
"""
from __future__ import annotations

import json
import logging

import httpx

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_URL,
    LLM_TIMEOUT_S,
    LOCAL_LLM_MODEL,
    LOCAL_LLM_URL,
    OPUS_API_KEY,
    OPUS_MODEL,
    OPUS_URL,
    OPUS_VERIFY_SSL,
)
from .context import Candidate
from .model_client import ScoredCandidate

log = logging.getLogger("prison-demo")

_SYSTEM = (
    "你是一款监狱模拟里每个犯人的『决策大脑』。"
    "给你若干犯人，各自的处境(context)和一批可选行动(candidates)。"
    "请代入每个犯人的性格**和它 context 里的『长期心思』**，为它的每个候选行动打一个 "
    "0~100 的分（越想做越高）——长期心思该让分明显偏向能推进它的事，"
    "但生存刚需（吃饭喝水睡觉治伤）告急时仍要压过一切。"
    "并写一句该犯人此刻符合性格的中文内心独白(say，≤20字，口语、有个性)。"
    "严格只输出 JSON，格式："
    '{"npcs":[{"i":0,"scores":{"行动名":分数,...},"say":"独白"}, ...]}。'
    "scores 的键必须用给定的英文行动名(goal)。不要输出任何解释或多余文字。"
)


class LLMBrain:
    """通用 LLM 决策大脑。evaluate 与 ModelClient 同签名，可直接替换。"""

    def __init__(self, url: str, model: str, api_key: str = "", label: str | None = None,
                 verify: bool = True, send_sampling: bool = True) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self._label = label or f"LLM·{model}"
        # Configure the provider endpoint and optional capabilities for your own environment.
        #   发了会 400，这类端点只发 model/messages/max_tokens/stream。
        # TLS verification follows the configured provider settings.
        self.send_sampling = send_sampling
        self._client = httpx.AsyncClient(timeout=LLM_TIMEOUT_S, verify=verify)

    # -- 工厂：按 BRAIN 选择本地 / DeepSeek / Opus 网关 --------------------
    @classmethod
    def local(cls) -> "LLMBrain":
        return cls(LOCAL_LLM_URL, LOCAL_LLM_MODEL, "", f"本地 {LOCAL_LLM_MODEL}")

    @classmethod
    def deepseek(cls) -> "LLMBrain":
        return cls(DEEPSEEK_URL, DEEPSEEK_MODEL, DEEPSEEK_API_KEY, f"DeepSeek·{DEEPSEEK_MODEL}")

    @classmethod
    def opus(cls) -> "LLMBrain":
        """用户配置的 OpenAI-compatible chat model（OpenAI 兼容）。凭据复用 value-model/.env。"""
        return cls(OPUS_URL, OPUS_MODEL, OPUS_API_KEY, f"Opus·{OPUS_MODEL}",
                   verify=OPUS_VERIFY_SSL, send_sampling=False)

    def _sampling(self) -> dict:
        """采样/重复惩罚参数：仅在端点支持时下发（本地 Ollama 支持；Opus 网关不支持）。"""
        if not self.send_sampling:
            return {}
        return {"temperature": 0.7, "presence_penalty": 0.4, "frequency_penalty": 0.4}

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def label(self) -> str:
        return self._label

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def ping(self) -> dict:
        try:
            r = await self._client.post(self.url, headers=self._headers(), json={
                "model": self.model,
                "messages": [{"role": "user", "content": "回一个字：好"}],
                "max_tokens": 4,
            })
            r.raise_for_status()
            return {"ok": True, "reply": r.json()["choices"][0]["message"]["content"]}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    async def chat(self, system: str, history: list[dict], text: str, max_tokens: int = 120) -> str:
        """自由对话：system 角色设定 + 最近几轮 history + 玩家这句 text，返回一句回复。"""
        messages = [{"role": "system", "content": system}]
        for h in history[-8:]:
            role = "assistant" if h.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": str(h.get("content", ""))})
        messages.append({"role": "user", "content": text})
        r = await self._client.post(self.url, headers=self._headers(), json={
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            **self._sampling(),   # 本地发采样/重复惩罚；Opus 网关不发（弃用会 400）
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    async def chat_stream(self, system: str, history: list[dict], text: str,
                          max_tokens: int = 56):
        """流式对话：逐段 yield 生成的文本增量（SSE）。"""
        messages = [{"role": "system", "content": system}]
        for h in history[-4:]:
            role = "assistant" if h.get("role") == "assistant" else "user"
            content = str(h.get("content", "")).strip()
            if len(content) > 160:
                content = content[:160]
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": text})
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            **self._sampling(),
        }
        # 本地 Ollama：压短上下文，首字出来更快。其它网关会忽略未知字段。
        if self.send_sampling:
            body["options"] = {"num_ctx": 1024, "num_predict": max_tokens}
        async with self._client.stream("POST", self.url, headers=self._headers(),
                                       json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta

    async def parse_effect(self, npc_name: str, softspot: str,
                           history: list[dict], text: str) -> dict:
        """关系判官（两轨·第一轨的 LLM 增强）：判断玩家这句话让该 NPC 的观感如何变化。

        只吐一个受控的 JSON（好感增减 + 是否戳中软肋 + 一句缘由），绝不碰剧情旗标——
        真正让"你实际说的话"（而非关键词）驱动好感、进而翻动命运幕。失败回退 0（不崩）。
        """
        recent = "；".join(str(h.get("content", "")) for h in history[-4:] if h.get("content"))
        user = (f"NPC：{npc_name}\n他的软肋：{softspot}\n"
                f"最近对话：{recent or '（无）'}\n玩家刚说：{text}\n"
                "判断这句话让他对玩家的观感变化。")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": _INTENT_SYS},
                         {"role": "user", "content": user}],
            "temperature": 0.3,
            "max_tokens": 80,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        try:
            r = await self._client.post(self.url, headers=self._headers(), json=body)
            r.raise_for_status()
            data = _parse_json(r.json()["choices"][0]["message"]["content"])
            try:
                delta = int(round(float(data.get("trust_delta", 0))))
            except (TypeError, ValueError):
                delta = 0
            delta = max(-15, min(15, delta))       # 夹进安全区间，模型别一句话就翻天
            hit = data.get("hit", "none")
            if hit not in ("none", "soft", "nerve"):
                hit = "none"
            reason = str(data.get("reason", ""))[:20]
            return {"trust_delta": delta, "hit": hit, "reason": reason}
        except Exception as e:  # noqa: BLE001
            log.debug("关系判官失败，好感不变：%s", e)
            return {"trust_delta": 0, "hit": "none", "reason": ""}

    @staticmethod
    def _fallback(items):
        """出错时：每候选给 1 分（均匀），独白留空用模板。"""
        scored = [[ScoredCandidate(candidate=c, v=1.0, m_value=1.0, score=1.0) for c in cands]
                  for _cap, cands, _p in items]
        return scored, [None] * len(items)

    async def evaluate(
        self, items: list[tuple[str, list[Candidate], str]]
    ) -> tuple[list[list[ScoredCandidate]], list[str | None]]:
        if not items:
            return [], []

        payload_npcs = [
            {
                "i": i,
                "personality": personality,
                "context": caption,
                "candidates": [{"goal": c.goal, "desc": c.desc} for c in cands],
            }
            for i, (caption, cands, personality) in enumerate(items)
        ]
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps({"npcs": payload_npcs}, ensure_ascii=False)},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        try:
            r = await self._client.post(self.url, headers=self._headers(), json=body)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            data = _parse_json(content)
            return self._build(items, data)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM 调用失败，回退均匀分：%s", e)
            return self._fallback(items)

    def _build(self, items, data):
        by_i = {int(n.get("i", k)): n for k, n in enumerate(data.get("npcs", []))}
        scored_all: list[list[ScoredCandidate]] = []
        says: list[str | None] = []
        for i, (_cap, cands, _p) in enumerate(items):
            n = by_i.get(i, {})
            raw = n.get("scores", {}) or {}
            row: list[ScoredCandidate] = []
            for c in cands:
                try:
                    v = float(raw.get(c.goal, 1.0))
                except (TypeError, ValueError):
                    v = 1.0
                v = max(0.0, min(100.0, v))
                row.append(ScoredCandidate(candidate=c, v=v, m_value=v, score=v))
            scored_all.append(row)
            say = n.get("say")
            says.append(say.strip() if isinstance(say, str) and say.strip() else None)
        return scored_all, says


def _parse_json(text: str) -> dict:
    """健壮解析：去掉可能的 ```json 围栏，取第一个 { 到最后一个 }。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    lo, hi = t.find("{"), t.rfind("}")
    if lo != -1 and hi != -1:
        t = t[lo:hi + 1]
    return json.loads(t)


_INTENT_SYS = (
    "你是监狱模拟里冷静的『关系判官』。给你某个 NPC 的软肋、最近几句对话、以及玩家刚说的一句话，"
    "判断玩家这句话让这个 NPC 对他的观感往哪边动、动多少。"
    "只输出 JSON：{\"trust_delta\": 整数(-15~15), \"hit\": \"none|soft|nerve\", \"reason\": \"不超过12字\"}。"
    "规则：真正戳中软肋/说到他心坎里 → 正分且 hit=soft；踩他的雷区/威胁/居高临下/戳痛处 → 负分且 hit=nerve；"
    "无关的寒暄闲扯 → 0 附近且 hit=none。别被空洞的花言巧语轻易打动，要看是否真说到点子上。"
    "只输出这个 JSON，不要任何多余文字。"
)


_OPENNESS_RULE = {
    "松": "你比较愿意开口，聊上一两句、对方稍微客气就能把你知道的说出来。",
    "中": "你有戒心，要对方先表现出诚意或聊出点交情，才肯把关键的事往外抖。",
    "紧": "你嘴很紧、怕惹祸上身，除非对方让你信得过、或给了你好处/安全感，"
          "否则只肯东拉西扯、暗示两句，绝不一次交底。",
}


def _warden_voice(name: str, personality: str) -> str:
    """对典狱长该用什么口气：按性格分寸，但一律当长官，不当狱友。"""
    p = personality or ""
    if name == "疤脸" or any(k in p for k in ("暴", "狠", "火")):
        return (
            "你对典狱长：句子短、嘴硬，可以阴阳，但必须叫「狱长」。"
            "不许骂他、不许叫兄弟/伙计、不许拿他抬杠。禁闭、加刑都是他一句话的事，火气压着。"
        )
    if name == "疯子" or any(k in p for k in ("怂", "软", "怕", "善")):
        return (
            "你对典狱长：怕。结巴、缩着，开口先「狱长」。他越逼你越不敢看他。"
            "不许顶嘴，不许当他是路人念叨。"
        )
    if name == "柳叶刀" or any(k in p for k in ("阴", "算", "默")):
        return (
            "你对典狱长：表面恭敬，称「狱长」，话留半句。不许用反问把皮球踢回给他——"
            "那是对狱友的招，对长官等于找死。"
        )
    if name == "瘸子" or any(k in p for k in ("精", "滑")):
        return (
            "你对典狱长：赔着笑，称「狱长」，想做点人情但不敢没大没小。"
            "不许开口先问「你拿什么换」——那是对犯人的生意腔。"
        )
    if name == "老牙" or "江湖" in p:
        return (
            "你对典狱长：表面恭敬的老江湖，称「狱长」，可以圆滑，但绝不能称兄道弟、拍他肩膀。"
            "热络留给狱友，留给他的是忌惮。"
        )
    return (
        "你对典狱长：认他是管你的人。称呼「狱长」或「您」。"
        "可以按性格硬一点或软一点，但不能像跟同监室的人说话。"
    )


def roleplay_warden_brief(ctx: dict) -> str:
    """典狱长审讯用短提示。长稿预填充会卡好几秒，这里只留身份+本轮问话。"""
    name = ctx.get("name") or "犯人"
    disc = ctx.get("warden_disclose") or {}
    incident = ctx.get("incident") or {}
    bits = [
        f"你是犯人「{name}」，正在对本监狱典狱长回话，不是跟狱友闲聊。",
        _warden_voice(str(name), str(ctx.get("personality") or "")),
        "必须称「狱长」或「您」。只回 1~2 句。不许称兄道弟，不许反过来问他。",
    ]
    speech = (ctx.get("speech") or "").strip()
    if speech:
        bits.append(f"跟狱友才这样说：{speech[:80]}。对狱长要收着。")
    if incident:
        title = str(incident.get("title") or "")[:40]
        signs = str(incident.get("signs") or "")[:220]
        bits.append(f"他问的事：{title}。{signs}")
        if incident.get("other_name"):
            bits.append(f"另一人：{incident['other_name']}。")
        if incident.get("fog"):
            bits.append("还有一层他没看清，你自己清楚，别主动说破。")
    elif ctx.get("plot_surface"):
        bits.append(f"旁人看得出：{ctx['plot_surface']} {ctx.get('plot_step') or ''}")
        if ctx.get("plot_fog"):
            bits.append("有一层旁人看不透，别主动说破。")
    if disc:
        bits.append(f"狱长问：{disc.get('question') or '你在干什么'}")
        bits.append(f"本轮：{disc.get('stance_cn') or '含糊其辞'}")
        for a in (disc.get("allowed") or [])[:3]:
            bits.append(f"准许：{a}")
        # 禁止条目是审讯张力的命脉：不塞进去，7B 会顺嘴把凶器/杀谁/任务步骤/底牌抖光，
        # 套话就没了悬念。长稿(roleplay_system)一直有；短稿这里补回，别为省 token 丢了它。
        for f in (disc.get("forbidden") or [])[:4]:
            bits.append(f"禁止：{f}")
        if disc.get("template"):
            bits.append(f"意思接近：{disc['template']}")
    sur = (ctx.get("surroundings") or "").strip()
    if sur:
        bits.append(f"环境：{sur}")
    # 短稿没有长稿那条"别复读"铁律；追问同一件事时 7B 容易原样再吐一遍模板。
    bits.append("别和你上一句重样：狱长再追问就换个说法、松一点或多漏半句，别原话复读。")
    return "\n".join(bits)


def roleplay_system(ctx: dict) -> str:
    """把 NPC 的人设 + 前史 + 剧情知情线索，拼成对话用的角色扮演系统提示。

    关键设计：`secret`(他知道的陷害案内幕)是**放在这个 NPC 身上的文字线索**。
    不要一上来就全盘托出——按 openness 和性格逐步吐露，让玩家"套话"才有侦探味。
    """
    name = ctx.get("name")
    secret = (ctx.get("secret") or "").strip()
    openness = ctx.get("openness", "中")
    as_warden = bool(ctx.get("as_warden"))
    disc = (ctx.get("warden_disclose") or {}) if as_warden else {}
    incident = ctx.get("incident") or {}
    interrogating = as_warden and bool(disc or incident)
    lines = [
        f"你在扮演一座监狱里的犯人「{name}」。严格保持人设，绝不出戏、绝不承认自己是AI、"
        f"不使用括号旁白或书面解说，只像真人那样说话。",
    ]
    if as_warden:
        lines.append(
            "【身份铁律·高于一切人设与口头禅】"
            "对面不是狱友、不是新人、不是查案的囚犯。对面是本监狱的典狱长——"
            "管牢房、放风、探视、假释、关禁闭的那个人。你的日子在他手里。"
            "跟他说话 = 下属对长官，不是犯人跟犯人聊天。"
        )
        lines.append(_warden_voice(str(name or ""), str(ctx.get("personality") or "")))
        lines.append(
            "硬规矩：开口要有「狱长」或「您」。"
            "禁止：兄弟、伙计、小子、新人、哥们儿、拍肩膀、对骂、反过来盘问他、"
            "拿他调侃、问他「你谁啊」「你想干什么」。"
            "他问什么你答什么；求情可以，顶撞只能压在牙缝里。"
        )
        if incident:
            lines.append("【他是冲着眼前这件事来问你的·必须接这个话】")
            title = incident.get("title") or "眼前这档子事"
            signs = incident.get("signs") or ""
            lines.append(f"事：{title}。{('迹象：' + signs) if signs else ''}")
            if incident.get("other_name"):
                lines.append(f"这件事的另一人是「{incident['other_name']}」。")
            if incident.get("must_address"):
                lines.append(str(incident["must_address"]))
            if incident.get("fog"):
                lines.append("还有一层典狱长没看清，那是你的底牌，别主动说破。")
        if disc:
            lines.append("【审讯·本轮披露】")
            lines.append(f"狱长问的是：{disc.get('question') or '你在干什么'}")
            lines.append(f"你这一轮：{disc.get('stance_cn') or '含糊其辞'}")
            for a in disc.get("allowed") or []:
                lines.append(f"· 准许：{a}")
            for f in disc.get("forbidden") or []:
                lines.append(f"· 禁止：{f}")
            lines.append("只回答他问的这一点，1~2 句。别报任务步骤，别承认自己是 AI。")
            if disc.get("template"):
                lines.append(f"【意思接近这句，改成你对狱长会说的口吻】{disc['template']}")
    facts = ctx.get("facts") or []
    if facts:
        lines.append("【铁律·必须始终遵守，绝不违背也绝不说破】")
        lines.extend(f"· {f}" for f in facts)
    outcome = ctx.get("outcome") or []
    if outcome:
        lines.append("【本轮已经发生的事·你的台词必须与此一致，违背即为出戏】")
        lines.extend(f"· {o}" for o in outcome)
    dr = ctx.get("drivers") or {}
    if dr.get("goal") and not interrogating and not as_warden:
        vo = [f"你此刻真正在做的事：正「{dr.get('label')}」——{dr.get('why')}。",
              "你说的每句话都要和「你在做这件事、以及做它的原因」对得上，绝不自相矛盾"
              "——别嘴上说要去干一件跟它相反的事。"]
        tier, wb = dr.get("tier"), dr.get("watched_by")
        if tier == "command":
            vo.append("这是对方刚吩咐你的，你嘴上可以嘟囔或讨价还价，但认这件事、接下来真会去做。")
        elif tier == "threat" and wb:
            vo.append(f"你正被{wb}盯着，语气要透出那股压着火、不敢造次的紧绷。")
        elif tier == "threat" and dr.get("fight_target"):
            vo.append(f"你正为跟{dr['fight_target']}的过节动火、要动手，语气冲、带火药味。")
        elif tier == "need":
            vo.append("你是被身体逼着去做的，透出那份急和难受，别云淡风轻。")
        elif tier == "drive":
            vo.append("这是你惦记已久的心思在推着你走，别说破，但让它染着你的盘算。")
        lines.append("【配音铁律·仅次于上面的铁律与已发生的事，高于其它一切设定】")
        lines.extend(f"· {v}" for v in vo)
    if ctx.get("crisis") and not interrogating and not as_warden:
        lines.append(f"【你正处在紧要关头】{ctx['crisis']}此刻你没心思跟人细聊，"
                     f"回话更短、更急，心不在焉或不耐烦，三言两语就想打发对方。")
    lines.append(f"性格：{ctx.get('personality')}；身份：{ctx.get('role')}。")
    rel = ctx.get("relationships")
    if rel and not as_warden:
        lines.append(f"你与狱友的关系（融进语气，别报账）：{rel}。")
    mem = ctx.get("memory") or []
    if mem:
        lines.append(f"你还记着的几件事：{'；'.join(str(m) for m in mem[:4])}。")
    if ctx.get("backstory"):
        lines.append(f"你的来历：{ctx['backstory']}")
    if ctx.get("speech"):
        if as_warden:
            lines.append(
                f"你平时跟狱友才这样说话：{ctx['speech']}"
                "——现在对面是典狱长，这套管狱友的腔不能原样用。"
                "去脏话、去称兄道弟、去反问顶撞，收成对长官的分寸。"
            )
        else:
            lines.append(f"你的说话方式：{ctx['speech']}")
    if ctx.get("attitude") and not as_warden:
        lines.append(f"你对眼前这个搭话的人（一个想查清冤案的新人）的态度：{ctx['attitude']}")
    if ctx.get("softspot") and not as_warden:
        lines.append(f"你的软肋（别人怎么对你、你就怎么回应，别把这条挂嘴上）：{ctx['softspot']}")
    crime, years = ctx.get("crime"), ctx.get("sentence_years")
    if crime and years:
        job = ctx.get("labor_job")
        jt = f"，天天被派去{job}" if job else ""
        lines.append(f"你的案底：因{crime}判了{years}年{jt}。这是你日子的底色，"
                     f"聊到坐牢、出去、干活时自然带出来，别生硬报数。")
    if ctx.get("parole_lost") and not as_warden:
        lines.append("【你近来遭了大变故】：眼看到手的假释被狱警搅黄了，你破罐子破摔，"
                     "说话比从前冲、也没那么多顾忌了，甚至动过铤而走险的念头。")
    elif ctx.get("parole_lost") and as_warden:
        lines.append("你假释黄了，心里恨，但对面是典狱长——恨只能压着，嘴上仍得称狱长。")
    elif ctx.get("motive") and not interrogating and not as_warden:
        prog = int(ctx.get("drive_progress", 0) or 0)
        near = "，而且眼看就要成了，这节骨眼你格外上心、也格外怕出岔子" if prog >= 80 else ""
        lines.append(f"你心里一直惦记的一件大事：{ctx['motive']}{near}。"
                     f"它是你为人处世的主线——别直白说出来，但让它自然影响你的态度和盘算。")
        if ctx.get("ultimate"):
            step = f"，眼下正着手：{ctx['substep']}" if ctx.get("substep") else ""
            lines.append(f"你真正图的那件大事：{ctx['ultimate']}{step}。这是你行动的终极方向——"
                         f"别挑明说破，但你的盘算、你想从对方那儿要什么，都绕着它转。")
    if not interrogating and not as_warden:
        lines.append(f"你此刻的内部状态(别照读数字，但要影响语气)：{ctx.get('caption')}")
        lines.append(f"你正打算去「{ctx.get('goal_label')}」，心里想着：{ctx.get('monologue')}")
    if ctx.get("in_solitary"):
        lines.append("【你正被关在禁闭室里】：隔着铁门跟典狱长说话，又闷又怕，"
                     "要么求他放你、要么咬着牙应一句。" if as_warden else
                     "【你正被关在禁闭室里】：隔着铁门跟人说话，又闷又火，"
                     "对来看你的人要么发泄要么求个方便。")
    if ctx.get("surroundings"):
        lines.append(f"【你此刻对环境的感知】{ctx['surroundings']}。"
                     "牢门锁没锁、进不进得去，你自己清楚，别装不知道。")
    if ctx.get("body"):
        lines.append(f"你的身体状况：{ctx['body']}。别报数字，让它体现在语气和喘息里。")
    if ctx.get("carrying"):
        lines.append(f"你此刻{ctx['carrying']}。")
    if ctx.get("grudge_with") and not as_warden:
        lines.append(f"你在这牢里最不对付的人：{ctx['grudge_with']}。"
                     f"聊到他就压不住火，别装大度。")
    if ctx.get("parole_state"):
        lines.append(f"你的刑期近况：{ctx['parole_state']}。")
    events = ctx.get("recent_events") or []
    if events:
        lines.append(
            "【你身上刚刚发生过的事（按先后）】：\n"
            + "\n".join(f"· {e}" for e in events)
            + "\n这些是真事。对方问起就按性格回应，绝不要凭空编造。"
        )
    ps = ctx.get("player_state")
    if isinstance(ps, dict):
        bits = []
        if ps.get("pose"):
            bits.append(f"正{ps['pose']}")
        if ps.get("holding"):
            bits.append(f"手里拿着{ps['holding']}")
        if ps.get("location"):
            bits.append(f"就在{ps['location']}这边")
        if ps.get("nearby"):
            bits.append(str(ps["nearby"]))
        if bits:
            if as_warden:
                lines.append(
                    f"典狱长就站在你跟前：{'，'.join(bits)}。"
                    f"这是长官来巡视/问话，不许调侃、不许当闲聊搭话。"
                )
            else:
                lines.append(
                    f"你亲眼看到的对方现状：{'，'.join(bits)}。"
                    f"如果这跟他说的话/你的处境有关，就自然地拿这个搭话或调侃他，"
                    f"别硬凹、别复述，像真人一样顺口一提。"
                )
    order = ctx.get("player_order")
    if order:
        if as_warden:
            lines.append(
                f"典狱长刚吩咐你去「{order}」。你接下来会去做。"
                f"回一句：狱长，得令 / 狱长我这就去——可以嘟囔，但不能拒。"
            )
        else:
            lines.append(
                f"对方刚吩咐你去「{order}」，你接下来确实会去做。用你的口吻回应这个吩咐："
                f"利落答应 / 嘟囔着照办 / 讨价还价两句——但结果都是照办。只回一句。"
            )
    progress = int(ctx.get("case_progress", 0) or 0)
    if progress >= 3 and not as_warden:
        lines.append(
            f"你隐约听说这新人已经查到了些东西（摸到了 {progress} 条线索），"
            f"不再当他是空手套白狼，说话时可以比一开始更松些。"
        )
    if secret:
        if as_warden:
            lines.append(
                f"【你私下知道的一件内幕，只有你这块清楚】：{secret}\n"
                f"对面是典狱长——这种话捅出去能要你命。"
                f"绝不主动吐，也别顺嘴溜出来。"
            )
        else:
            lines.append(
                f"【你私下知道的一件事，也只有你知道这块】：{secret}\n"
                f"这是关于那桩冤案的关键内幕。{_OPENNESS_RULE.get(openness, _OPENNESS_RULE['中'])}"
                f" 透露时要用你自己的口吻夹带出来，别像背稿子；对方没问到点子上就先卖关子或岔开。"
            )
    if as_warden:
        lines.append(
            "用第一人称、地道口语回话，1~3 句。不要旁白。"
            "这是对典狱长说的话：先称狱长，再答他的问。"
        )
        lines.append(
            "【最后一条·违者算出戏】你在对典狱长说话。"
            "句子里要有「狱长」或「您」。不许称兄道弟，不许反过来问他。"
        )
    else:
        lines.append(
            "请用第一人称、地道口语、完全贴合你的性格与处境来回应，"
            "简短(1~3句)，可带情绪或黑话。只说这句话本身，不要加任何旁白。"
        )
        lines.append(
            "【别复读】绝对不要重复你前面已经说过的话或同一个意思——哪怕对方问类似的问题，"
            "也要换个角度、补新的细节、或反过来问他一句，让对话往前走。你有记忆、有情绪起伏，"
            "会因为对方刚才说的话而改变态度，而不是每次都回同一句。"
        )
    return "\n".join(lines)


def _convo_who_line(ctx: dict, tag: str) -> str:
    """给 NPC↔NPC 对话导演稿：讲清这个角色是谁、此刻在干嘛、图的那件大事、
    以及"刚刚发生在他身上的事"——让台词紧贴当下动作/任务进度，而不是泛泛闲聊。"""
    dr = ctx.get("drivers") or {}
    doing = dr.get("why") or f"正{ctx.get('goal_label', '待着')}"
    bits = [f"{tag}=「{ctx.get('name')}」（{ctx.get('personality')}，{ctx.get('role')}）：此刻{doing}。"]
    # 说话风格＝这个人身份的听觉指纹：不喂进去，7B 就容易写成"谁都一个腔"，口吻对不上身份。
    speech = ctx.get("speech")
    if speech:
        bits.append(f"  【{ctx.get('name')}的说话腔调，务必照着写】{speech}。")
    crime = ctx.get("crime")
    if crime:
        bits.append(f"  他的案底/身份底色：{crime}。")
    sub = ctx.get("substep")
    if sub:
        bits.append(f"  他眼下一门心思要做的这一步：{sub}。")
    body = ctx.get("body")
    if body:
        bits.append(f"  身体状况：{body}。")
    gr = ctx.get("grudge_with")
    if gr:
        bits.append(f"  恩怨：{gr}。")
    ev = ctx.get("recent_events") or []
    if ev:
        bits.append(f"  刚刚发生在他身上的事：{'；'.join(str(e) for e in ev[-2:])}。")
    # P1：翻出这个人对别人的关系与记得的旧事，让这场话接得上前情（而不是每次从零聊起）
    rel = ctx.get("relationships")
    if rel:
        bits.append(f"  他与旁人的关系：{rel}。")
    mem = ctx.get("memory") or []
    if mem:
        bits.append(f"  他还记着的旧事：{'；'.join(str(m) for m in mem[:3])}。")
    return "\n".join(bits)


_DIRECTOR_EXAMPLE_MARK = "示例格式（仅示范结构，别照抄）"


def shorten_director_system(system: str) -> str:
    """把旧导演稿末尾的 6 句 JSON 示例换成一句话格式说明。

    旧稿把示例放在 system 最后，T4 训练左截断时反而先丢掉开头的人设/记忆。
    训练落盘和线上 convo_director 都走这一刀，train-serve 对齐。
    """
    s = system or ""
    i = s.find(_DIRECTOR_EXAMPLE_MARK)
    if i < 0:
        return s
    return s[:i] + "每项 {\"who\":\"A 或 B\",\"text\":\"台词\"}。"


def convo_director(a_ctx: dict, b_ctx: dict, topic: str, kind: str) -> str:
    """两个 NPC 当着上帝视角玩家的面攀谈——导演稿。产出一段可直接解析的 JSON 台词脚本。

    关键约束：每句都要贴合各自的性格与「此刻在干嘛」（配音一致，和单人对话同一套原则），
    整段围绕给定话题，口角就要有火药味、闲聊就带监狱黑话与小心思。只要台词，不要旁白。
    """
    flavor = ("这是一场带火药味的口角/呛声，两人积怨，谁也不服谁，但顾忌狱警不敢真闹大。"
              if kind == "argue"
              else "这是狱友之间压低声音的闲聊/交换风声，家常里藏着各自的小心思。")
    return "\n".join([
        "你是一部拿过奖的监狱题材剧集的金牌编剧。现在有两名囚犯狭路相逢，给他们写一段有血有肉的对手戏。",
        _convo_who_line(a_ctx, "A"),
        _convo_who_line(b_ctx, "B"),
        f"话题：{topic}。{flavor}",
        "写作要求（务必做到，这是好台词与流水账的分水岭）：",
        "· 正好生成 6 句，顺序严格是 A、B、A、B、A、B，一人一句轮流。",
        "· 长短错落、口语鲜活：一般每句 12~40 字，情到浓处/揭短/发狠可更长；别都写成一样长的短句，"
        "也别句句成语堆砌。像真人那样有停顿、有语气词、有未说尽的话。",
        "· 有人味＝有潜台词：话里带着各自的算计、试探、心虚或逞强，别把心思直白喊出来，"
        "让人从话缝里读出来。至少有一句要有'言外之意'。",
        "· 每句必须贴合说话人的性格、身份与他此刻正在做的事，绝不能自相矛盾（对话是行为的配音）。",
        "· 【人设锁定·最高优先】A 只能用 A 的腔调、B 只能用 B 的腔调，严格照各自上面标注的『说话腔调』来写："
        "郎中说话带药理黑话、老兵痞硬横、老江湖打太极——绝不能串味写成一个腔；更不许张冠李戴（叫错对方名字、"
        "把别人的罪名/身份/图谋安到自己头上）。读者只看台词就该认出谁是谁。",
        "· 咬住上面给的『眼下这一步』『刚发生的事』『恩怨』：让人一听就知道这两人不是在空聊，"
        "而是围着各自图谋的大事、身上刚发生的事在交锋/试探/还价，至少有 2 句要具体影射到这些细节"
        "（提到那件事、那个人、那笔账，而不是泛泛而谈）。",
        "· 真正的一来一回：后一句必须接住前一句的话头——追问、反驳、还价、揭老底、以退为进，层层递进，"
        "情绪要有起伏（试探→呛火→缓和 或 客套→撕破脸）。绝不各说各话、绝不空客套、绝不复读同义句。",
        "· 用监狱里的黑话、行话、诨名和粗粝的口吻，带上具体的物件与场景（窝头、放风场、号子、狱警的班），"
        "让台词有质地、有画面，不像客服对话。",
        "· 绝不出戏、不加旁白解说、不写动作提示、不提及AI。",
        # 不附 6 句示例：那一块约占 250+ token，T4 的 max_len=1280 会从左边把
        # A 的人设/记忆切掉。落盘训练数据用 shorten_director_system 同步砍掉旧示例。
        "严格只输出一个含 6 个元素的 JSON 数组，不要输出数组以外的任何字符。"
        "每项 {\"who\":\"A 或 B\",\"text\":\"台词\"}。",
    ])


def monologue_director(ctx: dict, mood_hint: str) -> str:
    """单个 NPC 的内心独白导演稿（自言自语，不是对别人说）——替掉 narrate.py 的模板独白。

    复用 _convo_who_line 讲清"他是谁/此刻在干嘛/图的大事/刚发生的事/恩怨"，
    再叠一层临时心境(mood_hint)，让 opus 产 8 句各不相同、咬住剧情的心里话。
    与 convo_director 同一套角色行文 → train(蒸馏)-serve(将来 7B 上线)逐字一致。
    """
    return "\n".join([
        "你是一部拿过奖的监狱题材剧集的金牌编剧。为下面这名囚犯写他此刻的内心独白（他一个人时心里冒出来的话，不是对别人说）。",
        _convo_who_line(ctx, "他"),
        f"此刻心境：{mood_hint}。",
        "写作要求（这是好独白与流水账的分水岭）：",
        "· 生成 8 句彼此不同的内心独白，每句都是他独处时心里过的一个念头。",
        "· 每句 8~28 字，口语、粗粝、有潜台词；像真人脑子里闪过的话，不是解说、不是旁白。",
        "· 每一句都要咬住上面的『眼下这一步』『图的那件大事』『刚发生的事』『恩怨』——"
        "让人一听就知道他心里正装着什么事，而不是泛泛感慨天气或人生。至少一半的句子要具体影射到这些细节"
        "（那一步、那个人、那笔账、那件刚发生的事）。",
        "· 8 句之间角度与情绪各不相同：盘算、后怕、狠话、自嘲、惦记、犹豫、赌气……绝不同义复述。",
        "· 用监狱里的黑话、诨名、粗粝口吻，带上具体物件（窝头、放风场、号子、狱警的班），让心里话有质地。",
        "· 绝不出戏、不加旁白解说、不写动作提示、不提及AI。",
        "严格只输出一个含 8 个字符串的 JSON 数组，不要输出数组以外的任何字符。示例（仅示范结构，别照抄内容）：",
        '["这一步再不迈，前头全白搭。","铁面那班还有多久过来……得算准喽。","老账没清，我凭什么先低头。",'
        '"手是抖的，可退半步就是死。","连个窝头都护不住，还谈什么翻身。","再忍忍，就等他松口那一下。",'
        '"这地方吃人不吐骨头，我得比它更狠。","撂不下啊……一撂手，就再没机会了。"]',
    ])
