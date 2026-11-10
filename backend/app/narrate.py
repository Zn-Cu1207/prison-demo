"""决策叙事：把"模型选了啥、为什么选"翻译成人话（决策可视化的文字部分）。

产物：
  - activity(goal)：行动的动词 + 图标（前端头顶清晰展示"在干嘛"）。
  - monologue(...)：头顶独白。按 目标×性格 多条台词轮换，并随情境
    （被指挥/抗命、警报/搜牢/狱犬、饥渴程度）动态切换，避免翻来覆去几句。
  - explain(...)：一句解释——胜者分数 + 关键情境 + 压过第二名多少。

都在后端算好随 npc_decision / world_state 下发，前端保持薄。
"""
from __future__ import annotations

from .context import GOAL_LABEL, ContextState, NeedState
from .model_client import ScoredCandidate

# ---------------------------------------------------------------------------
# 行动 → (动词, 图标)：给前端头顶"正在做什么"用
# ---------------------------------------------------------------------------
ACTIVITY: dict[str, tuple[str, str]] = {
    "Eat": ("吃饭", "🍚"),
    "Drink": ("喝水", "💧"),
    "Sleep": ("睡觉", "💤"),
    "Cure": ("包扎", "🩹"),
    "DeclineBoredom": ("找乐子", "🎲"),
    "Socialize": ("唠嗑", "💬"),
    "Fight": ("打架", "🥊"),
    "Patrol": ("放风", "🚶"),
    "Study": ("学习", "📖"),
    "PrisonBreak": ("越狱", "🧗"),
    "Labor": ("苦役", "🔨"),
    "Rest": ("歇着", "🛏️"),
    "Bury": ("埋葬", "⚰️"),
    "Mourn": ("哀悼", "🕯️"),
    "Workout": ("健身", "🏋️"),
    "Shower": ("洗澡", "🚿"),
    "Visit": ("探视", "📞"),
    "Read": ("看书", "📚"),
    "Gamble": ("赌钱", "🎴"),
    "Smoke": ("抽烟", "🚬"),
    "Clean": ("打扫", "🧹"),
    "Pray": ("祷告", "🙏"),
    "Phone": ("打电话", "☎️"),
    "Laundry": ("洗衣", "🧺"),
    "KitchenDuty": ("帮厨", "🍳"),
    "Barter": ("换货", "🤝"),
    "Snitch": ("告密", "🤫"),
    "Chess": ("下棋", "♟️"),
    "Music": ("唱歌", "🎸"),
    "Tattoo": ("纹身", "💉"),
    "Watch": ("望风", "👀"),
    "Argue": ("吵架", "🗯️"),
    "Hide": ("藏东西", "📦"),
    "LayLow": ("收敛", "🤐"),
    "Tunnel": ("撬门", "⛏️"),
    # 宏观动机的签名动作
    "CurryFavor": ("刷表现", "🙇"),
    "Extort": ("立威", "💪"),
    "SellInfo": ("卖消息", "🕵️"),
    "ReviewFiles": ("核卷宗", "🔎"),
    "SeekProtector": ("贴靠山", "🫂"),
}


def activity(goal: str | None) -> dict:
    verb, icon = ACTIVITY.get(goal or "", (GOAL_LABEL.get(goal or "", "发呆"), "❓"))
    return {"verb": verb, "icon": icon}


# ---------------------------------------------------------------------------
# 台词池：goal -> {性格: [多条], "default": [多条]}
# ---------------------------------------------------------------------------
_MONO: dict[str, dict[str, list[str]]] = {
    "Eat": {
        "暴躁": ["妈的，饿了，谁挡我路削谁。", "开饭了，老子先来。"],
        "老油条": ["点子上了，去食堂占个好位。", "吃饭这事，急不得也少不得。"],
        "书呆": ["按时吃饭，身体是革命本钱。", "边吃边把今天的书过一遍。"],
        "新人": ["食堂……我该坐哪儿啊。", "低调点去打饭，别惹人。"],
        "default": ["去食堂垫吧垫吧。", "肚子叫了，开饭去。"],
    },
    "Drink": {
        "暴躁": ["渴死了，去灌两口。"],
        "老油条": ["水房打点水，顺便看看风声。"],
        "default": ["嗓子冒烟，去打点水。", "喝口水润润。"],
    },
    "Sleep": {
        "老油条": ["熬一天是一天，睡了。", "眯一会儿，养足精神好过日子。"],
        "新人": ["躺下也睡不踏实……", "熄灯了，赶紧闭眼别出岔子。"],
        "default": ["困得眼皮打架，睡了。", "回牢房躺平。"],
    },
    "Cure": {
        "暴躁": ["挂彩了，简单包一下接着来。"],
        "default": ["伤口得处理下，别发炎。", "去医务室包扎包扎。"],
    },
    "DeclineBoredom": {
        "暴躁": ["闲得发慌，去投两个篮泄泄火。"],
        "书呆": ["闷了，翻两页闲书解解乏。"],
        "老油条": ["无聊啊，找点乐子打发时间。"],
        "新人": ["不敢乱走，就近待着找点事做。"],
        "default": ["关久了闷得慌，找点乐子。", "无聊透顶，活动活动。"],
    },
    "Socialize": {
        "暴躁": ["找兄弟们扯两句，谁不服说出来。"],
        "老油条": ["去套套近乎，探点消息，混脸熟。", "多个朋友多条路，唠两句。"],
        "书呆": ["跟人搭句话……聊聊也好。"],
        "新人": ["找个面善的说说话，壮壮胆。", "想找人聊聊，又怕说错话。"],
        "default": ["找狱友唠两句。", "凑过去聊聊天。"],
    },
    "Fight": {
        "暴躁": ["这口气咽不下，揍他！", "敢惹我？往死里打！", "手痒了，正好活动筋骨。"],
        "老油条": ["躲不过就先下手，速战速决。"],
        "书呆": ["我……我不想打，可他逼我。"],
        "新人": ["他要动手？我、我也不能怂！", "腿有点抖，但不能认怂……"],
        "default": ["敢惹我？来啊！", "干就完了。"],
    },
    "Patrol": {
        "暴躁": ["放风场热闹，去凑个热闹！", "出去溜达，看谁不顺眼。"],
        "老油条": ["放风时间，出去走走透透气。"],
        "新人": ["跟着大伙儿去放风，别落单。"],
        "default": ["去放风场溜达溜达。", "出去晒晒太阳。"],
    },
    "Rest": {
        "书呆": ["人太多，回牢房清静看会儿书。", "躲开吵闹，安静待着。"],
        "新人": ["外面太挤，我还是躲屋里。", "人多我心慌，回铺上待着。"],
        "老油条": ["这时候不凑热闹，回去歇着。"],
        "default": ["人多闹腾，回牢房歇着。", "懒得动，躺会儿。"],
    },
    "Study": {
        "书呆": ["上课去，多学点出去有用。", "知识改变命运，认真听讲。"],
        "老油条": ["混个学分减刑，划算。"],
        "default": ["去教室待着学点东西。", "上课时间，进去坐着。"],
    },
    "Labor": {
        "老油条": ["苦役躲不掉，磨洋工吧。", "干活悠着点，别累着自己。"],
        "暴躁": ["又是苦役，烦死了，干完拉倒。"],
        "default": ["去干苦役。", "派到活了，干吧。"],
    },
    "PrisonBreak": {
        "暴躁": ["老子受够了，翻墙跑！", "拼一把，冲出去！"],
        "老油条": ["瞅准空子，悄悄摸出去……"],
        "新人": ["真要跑吗……被抓到就完了。"],
        "default": ["趁没人盯着，翻出去……", "找机会溜出这鬼地方。"],
    },
    "Bury": {"default": ["把人埋了吧，晦气。"]},
    "Mourn": {"default": ["唉……走好。"]},
    "Workout": {
        "暴躁": ["举两组铁，把火气砸出去！"],
        "default": ["撸铁去，练结实点没人敢惹。", "健身房走起，出出汗。"],
    },
    "Shower": {"default": ["身上馊了，冲个澡。", "趁热水还有，去洗洗。"]},
    "Visit": {
        "新人": ["家里人来看我了……鼻子有点酸。"],
        "老油条": ["会客去，顺便带点消息进来。"],
        "default": ["有人探视，去会客室。", "去见见家里人。"],
    },
    "Read": {
        "书呆": ["图书室安静，正好啃两章。", "书里天地宽，躲进去。"],
        "default": ["翻两页书打发时间。"],
    },
    "Gamble": {
        "老油条": ["来两把，愿赌服输。", "牌桌上见真章，跟不跟？"],
        "暴躁": ["赌一把，输了别赖账！"],
        "default": ["蹲角落摸两把牌。"],
    },
    "Smoke": {
        "老油条": ["靠墙点一根，眯着眼看戏。"],
        "default": ["抽根烟压压。", "来根烟，缓一缓。"],
    },
    "Clean": {"default": ["轮到我打扫，磨蹭着干。"]},
    "Pray": {
        "新人": ["求菩萨保佑我平安出去……"],
        "default": ["静下心，念叨两句。", "闭眼祷告，图个心安。"],
    },
    "Phone": {
        "新人": ["妈，我挺好的……真的挺好的。", "就三分钟，我想听听家里的声音。"],
        "暴躁": ["排队打个电话，谁插队削谁。"],
        "老油条": ["打个电话，外面的事得有人办。"],
        "书呆": ["跟家里报个平安，说说进度。"],
        "default": ["去电话亭给家里挂个电话。", "攒了半天话，打回去说两句。"],
    },
    "Laundry": {
        "老油条": ["洗衣房热乎，还能躲会儿懒。"],
        "暴躁": ["搓这破囚服，搓得手都烂了。"],
        "default": ["囚服馊了，去洗衣房搓搓。", "轮到我洗衣，去吧。"],
    },
    "KitchenDuty": {
        "老油条": ["后厨的活抢着干——离饭最近。", "帮厨的人从不挨饿，懂的都懂。"],
        "新人": ["让我帮厨？听话干活总没错。"],
        "default": ["去后厨打下手。", "帮厨去，顺便捞两口。"],
    },
    "Barter": {
        "老油条": ["拿两包烟换点门路，值。", "黑市走一趟，行情得摸着。"],
        "暴躁": ["换东西可以，敢坑我试试。"],
        "新人": ["听说那边能换东西……我该带什么去？"],
        "default": ["去角落里换点用得上的。", "手头这玩意儿能换更好的。"],
    },
    "Snitch": {
        "老油条": ["递个话给管教，换我几天清净。", "这种事，说出去比压着划算。"],
        "新人": ["我……我要不要告诉管教？"],
        "书呆": ["按规矩，这事该报上去。"],
        "default": ["去管教那儿说两句。", "这事得让上面知道。"],
    },
    "Chess": {
        "书呆": ["杀一盘，脑子不能生锈。"],
        "老油条": ["棋盘上让你三子，话我先说。"],
        "default": ["来一盘，输了别耍赖。", "娱乐室摆棋去。"],
    },
    "Music": {
        "暴躁": ["吼两嗓子，痛快！"],
        "新人": ["跟着哼两句，心里松快点。"],
        "default": ["拨两下弦，唱两句解闷。", "来段老歌，谁爱听谁听。"],
    },
    "Tattoo": {
        "暴躁": ["再纹一条，看谁还敢瞪我。"],
        "新人": ["纹身……疼吗？算了，纹个小的。"],
        "老油条": ["身上这些图，一道一道都是账。"],
        "default": ["找人纹一个。", "闲着也是闲着，纹点花样。"],
    },
    "Watch": {
        "老油条": ["我在这儿盯着，管教一露头就咳嗽。"],
        "新人": ["让我望风……被抓到怎么办。"],
        "default": ["我把风，你们快点。", "盯着点门口，别出岔子。"],
    },
    "Argue": {
        "暴躁": ["先骂两句，不服再动手。"],
        "老油条": ["吵吵就得了，动手不划算。"],
        "书呆": ["讲道理，我不跟你动手。"],
        "新人": ["你、你凭什么这么说我！"],
        "default": ["有话说清楚，别动手。", "吵两句出出气。"],
    },
    "Hide": {
        "老油条": ["东西塞马桶后头，他们懒得摸。"],
        "新人": ["快藏起来快藏起来……在哪儿藏啊！"],
        "暴躁": ["藏好了，搜出来就翻脸。"],
        "default": ["趁没进来，把东西藏了。", "违禁品得赶紧处理掉。"],
    },
    # —— 宏观动机的签名动作台词 ——
    "CurryFavor": {
        "普通": ["管教，这活儿我来干，您歇着。", "表现分攒够了，就能准点出去。",
                "离假释就差这口气，不能松劲。"],
        "老油条": ["管教跟前多露露脸，出去快。"],
        "default": ["去管教那儿表现表现。", "勤快点，人家看在眼里。"],
    },
    "Extort": {
        "暴躁": ["这片儿归我管，该交的交。", "不长眼的，敢不给面子？",
                "地盘是打出来的，谁不服？"],
        "老油条": ["立个规矩，往后好办事。"],
        "default": ["去把该收的收上来。", "这地界儿得有人说了算。"],
    },
    "SellInfo": {
        "老油条": ["这条消息值两包烟，卖了。", "谁想知道，拿东西来换。",
                "外头那债，一条条消息填。"],
        "default": ["手里这点消息能换钱。", "去兜售点听来的事儿。"],
    },
    "ReviewFiles": {
        "书呆": ["卷宗第三页那笔账，日期对不上。", "数字不会骗人，我再核一遍。",
                "总能找出他们做假的破绽。"],
        "default": ["去把卷宗再对一遍。", "翻翻案卷，找找漏洞。"],
    },
    "SeekProtector": {
        "新人": ["石头哥……能不能罩着我点。", "跟紧了大哥，就没人敢欺负我。",
                "找个靠得住的，心里才踏实。"],
        "default": ["凑过去，找个靠山。", "得抱条粗腿才活得下去。"],
    },
}

# 被指挥且服从 / 抗命
_OBEY: dict[str, list[str]] = {
    "暴躁": ["行，看你面子，这次听你的。", "哼，就这一回。"],
    "老油条": ["队长发话了，那就走一趟。", "听指挥，稳当。"],
    "书呆": ["好的，我这就去。"],
    "新人": ["是！我马上去！", "听、听你的！"],
    "default": ["得令，这就去办。", "行，听你的。"],
}
_DEFY: dict[str, list[str]] = {
    "暴躁": ["少废话，命都要没了还听你的？", "滚开，老子得先顾自己！"],
    "老油条": ["这节骨眼，恕不奉陪，保命要紧。"],
    "书呆": ["抱歉，我现在实在顾不上……"],
    "新人": ["对不起对不起，我快撑不住了！"],
    "default": ["现在不行，我得先顾自己！", "命令先放放，救急要紧！"],
}

# 情境反应（警报/搜牢/狱犬），优先于普通台词
_EVENT_REACT: dict[str, dict[str, list[str]]] = {
    "警报": {
        "暴躁": ["警报？谁跑了？别赖到我头上！"],
        "老油条": ["警报响了，都别乱动，装没事人。"],
        "新人": ["警、警报！我什么都没干啊！"],
        "default": ["越狱警报？赶紧规矩点。", "警报拉响，别撞枪口上。"],
    },
    "狱警搜牢": {
        "老油条": ["搜牢了，东西该藏的都藏好。"],
        "新人": ["搜牢……我铺底下没东西吧？"],
        "default": ["狱警搜牢，老实站着。"],
    },
}
_DOG_REACT: dict[str, list[str]] = {
    "新人": ["狱、狱犬！别过来……"],
    "暴躁": ["放狗吓唬谁呢。"],
    "default": ["狱犬盯上了，别乱动。", "那狗龇牙了，慢慢退。"],
}

# 生存需求极高时的加料台词
_NEED_FIELD = {"Eat": "hunger", "Drink": "thirst", "Sleep": "fatigue", "Cure": "injury"}
_NEED_FLAVOR: dict[str, list[str]] = {
    "Eat": ["饿得前胸贴后背，眼冒金星，必须吃！", "再不吃要晕过去了……"],
    "Drink": ["渴得嗓子快冒烟了！"],
    "Sleep": ["困得站着都能睡着……"],
    "Cure": ["血止不住，再不包要出事！"],
}

_IDLE = ["……", "发会儿呆。", "这日子啥时候是个头。"]


def _pick(table: dict[str, list[str]], personality: str, seed: int) -> str:
    lst = table.get(personality) or table.get("default") or _IDLE
    return lst[seed % len(lst)]


def monologue(
    goal: str | None,
    personality: str,
    ctx: ContextState | None = None,
    commanded_goal: str | None = None,
    seed: int = 0,
) -> str:
    if not goal:
        return _IDLE[seed % len(_IDLE)]
    # 1) 玩家指令：服从 / 抗命
    if commanded_goal:
        return _pick(_OBEY if goal == commanded_goal else _DEFY, personality, seed)
    # 2) 戏剧性情境反应（打架本身已表达，故排除 Fight）
    if ctx and goal != "Fight":
        if ctx.event in _EVENT_REACT:
            return _pick(_EVENT_REACT[ctx.event], personality, seed)
        if ctx.threat == "狱犬":
            return _pick(_DOG_REACT, personality, seed)
    # 3) 生存需求极高时的加料（_NEED_FLAVOR 是纯列表，直接按 seed 取）
    if ctx and goal in _NEED_FIELD:
        if getattr(ctx.needs, _NEED_FIELD[goal], 0) >= 85:
            flavor = _NEED_FLAVOR[goal]
            return flavor[seed % len(flavor)]
    # 4) 常规 目标×性格 台词池
    table = _MONO.get(goal)
    if not table:
        return GOAL_LABEL.get(goal, goal)
    return _pick(table, personality, seed)


def _reason(goal: str, ctx: ContextState, n: NeedState) -> str:
    if goal == "Eat":
        return f"饥饿{n.hunger}"
    if goal == "Drink":
        return f"口渴{n.thirst}"
    if goal == "Sleep":
        return f"疲劳{n.fatigue}"
    if goal == "DeclineBoredom":
        return f"无聊{n.boredom}"
    if goal == "Socialize":
        return f"社交欲{n.social}"
    if goal == "Cure":
        return f"受伤{n.injury}"
    if goal == "Fight":
        bits = ["仇人在场"] if ctx.enemy_present else ["有威胁"]
        if ctx.event == "斗殴":
            bits.append("正在斗殴")
        if ctx.backer_present:
            bits.append("有靠山撑腰")
        return "、".join(bits)
    if goal == "Study":
        return "上课时段"
    if goal == "PrisonBreak":
        return f"就寝+监视{ctx.surveillance}" + ("（快假释了本不该跑）" if ctx.parole_near else "")
    if goal == "Labor":
        return "被派了苦役"
    if goal == "Patrol":
        return "没别的急事，去放风场"
    if goal == "Rest":
        return "躲开放风场的拥挤"
    if goal == "Argue":
        return "仇人在场，但选择只斗嘴不动手"
    if goal == "Phone":
        return f"社交欲{n.social}，想家"
    if goal == "Hide":
        return "狱警搜牢，先处理违禁品"
    if goal == "Snitch":
        bits = []
        if ctx.event == "斗殴":
            bits.append("有斗殴可报")
        if ctx.has_corpse:
            bits.append("有尸体")
        if ctx.enemy_present:
            bits.append("想借管教收拾仇人")
        return "、".join(bits) or "有事想报给管教"
    if goal == "Watch":
        return f"监视度{ctx.surveillance}，有人需要放哨"
    if goal in ("Laundry", "KitchenDuty"):
        return "被派了工，挑个轻省的活"
    if goal == "Barter":
        return f"监视度{ctx.surveillance}，黑市能走货"
    if goal == "CurryFavor":
        return "快假释了，抢着表现攒分"
    if goal == "Extort":
        return "想立威抢地盘"
    if goal == "SellInfo":
        return "缺钱，卖消息换钱"
    if goal == "ReviewFiles":
        return "一心翻案，核对卷宗"
    if goal == "SeekProtector":
        return "新人怕事，想找靠山"
    if goal == "LayLow":
        return "狱警/狱犬盯着，先收敛"
    return "综合当前情境"


def explain(goal: str | None, scored: list[ScoredCandidate], ctx: ContextState) -> str:
    if not goal or not scored:
        return ""
    ranked = sorted(scored, key=lambda s: s.score, reverse=True)
    win = ranked[0]
    reason = _reason(goal, ctx, ctx.needs)
    win_label = GOAL_LABEL.get(win.candidate.goal, win.candidate.goal)
    if len(ranked) == 1:
        return f"选「{win_label}」({win.score:.1f})：{reason}；当前无其它候选"
    second = ranked[1]
    sec_label = GOAL_LABEL.get(second.candidate.goal, second.candidate.goal)
    ratio = win.score / second.score if second.score > 1e-6 else float("inf")
    gap = "碾压" if ratio >= 5 else ("明显高于" if ratio >= 1.5 else "略高于")
    return f"选「{win_label}」({win.score:.1f})：{reason}；{gap}「{sec_label}」({second.score:.1f})"
