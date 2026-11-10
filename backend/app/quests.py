"""终极目标 + 分目标任务链（macro goal → sub-goals）。

为什么要有这层（在 drives.py 之上再叠一层）：
    drives.py 给每个犯人一句"长期心思"和一个签名动作，但它是**扁平**的——
    一个 0~100 的进度条、一套固定的 advance/boost。结果 NPC 的动机"看起来有方向，
    但没有剧情弧线"：一直在刷同几个动作，玩家看不出他在**一步步逼近一个终极目标**。

    quests.py 把每个人的长期心思展开成一条**有序的任务链**：
      · 一个**终极目标**（ultimate）——他这一辈子在牢里图的那件大事；
      · 5~10 个**分目标**（Step）——为达成终极目标必须一步步啃下来的关卡。

    引擎每完成一个分目标就：推进到下一步、往命运账本记一条叙事化的 beat、
    并把"当前正在推进的分目标"喂给决策(候选/乘数)和对话(配音)——
    于是 NPC 的每个举动都指向"此刻这一步"，整条链又指向那个终极目标。
    行为、剧情、对话三方读同一份 Step，一致性天然成立。

注入通道（与 drives 三通道一致，避免"照 prompt 复读"）：
    1) 候选上桌：当前 Step.focus 里的动作稳定进候选（world._gen_candidates）。
    2) 引擎乘数：当前 Step.focus 的动作 M_drive 额外抬一档（world._m_drive），
       让"这一步该干的事"在中间宽带里胜出——但压不过告急的生存刚需。
    3) 对话配音 + 命运账本：ultimate/当前分目标进 drivers 与 npc_chat_context，
       完成分目标时的 beat 进 bb.fate_log（前端命运账本 + 关键节点字幕）。

focus 里的 goal 必须是 actions.ACTIONS 里真实存在的动作，否则永远推进不了。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# 完成一个"推进当前分目标"的动作加多少进度点；step.need 是本步所需的总点数。
STEP_GAIN = 10.0
# 当前分目标的 focus 动作，M_drive 在 drive 基础上再乘这一档，让"这一步"胜出。
# 再调高：配合需求门槛上调/漂移减半，让终极目标主线真正成为 NPC 行为的主色调，
# 而不是被"吃饭喝水解闷"的日常需求淹没（五个人动作雷同的根因）。
QUEST_FOCUS_BOOST = 3.2


@dataclass(frozen=True)
class Step:
    desc: str                     # 当前分目标一句话（进 drivers/对话/命运账本）
    focus: frozenset[str]         # 推进本步的动作集合（上桌 + 加成 + 计进度）
    need: int = 20                # 完成本步所需的进度点（STEP_GAIN 的倍数）
    beat: str = ""                # 完成本步时记入命运账本的叙事句
    big: bool = False             # 是否作为关键节点走电影字幕
    surface: str = ""             # 标签给玩家看的这一步（大半剧情）
    fog: str = ""                 # 这一步里留给玩家自己去问/搜的一小截


@dataclass(frozen=True)
class Quest:
    ultimate: str                 # 终极目标（他在牢里图的那件大事）
    steps: tuple[Step, ...] = field(default_factory=tuple)
    kind: str = "survive"         # kill / survive / escape —— 终极目标的类型
    target: str = ""              # kind=kill 时的下手对象 NPC id
    weapon: str = ""              # 开局就持有的凶器（"" 表示需自己弄到）：poison/knife/taser
    surface: str = ""             # 旁观就能读到的大半剧情（冲突弹窗/卷宗用）
    fog: str = ""                 # 整条链里故意不摊开的那一小截
    kill_motive: str = ""         # 为什么想杀对方：一段能讲出来的背景故事（审讯被问到对象时透露）


def plot_of(npc) -> dict:
    """把总任务+当前分目标拆成『能写在标签上的』和『留给玩家的雾』。"""
    q = quest_of(getattr(npc, "id", "") or "")
    if q is None:
        return {"surface": "", "step": "", "fog": ""}
    step_s, step_f = "", q.fog
    idx = int(getattr(npc, "quest_step", 0) or 0)
    if q.steps and 0 <= idx < len(q.steps):
        st = q.steps[idx]
        step_s = st.surface or ""
        if st.fog:
            step_f = st.fog
    return {"surface": q.surface, "step": step_s, "fog": step_f}


def conflict_story(kind: str, npc, other=None) -> dict:
    """冲突文案：signs 摊开大半剧情（接到总任务/分目标），fog 只留一小截。"""
    a = plot_of(npc)
    b = plot_of(other) if other is not None else {"surface": "", "step": "", "fog": ""}
    an = getattr(npc, "name", "") or ""
    bn = getattr(other, "name", "") if other is not None else ""

    def _join(parts: list[str]) -> str:
        return "".join(p for p in parts if p)

    fog = a["fog"] or b["fog"] or "还有一层隔这么远看不真切，得走到跟前问、或者搜。"
    tag = ""
    signs = ""

    if kind == "fight_now":
        tag = "旧账见血"
        signs = _join([
            f"{an}和{bn}已经扭打在一起。",
            f"{a['surface']}" if a["surface"] else "",
            f"{b['surface']}" if b["surface"] else "",
            f"眼下{an}正做着这一步：{a['step']}。" if a["step"] else "",
            "现在过去还能拆开。",
        ])
        fog = a["fog"] or b["fog"] or "这一下是教训还是要命，旁人不敢赌。"
    elif kind == "kill_prep":
        tag = "要动手了"
        signs = _join([
            f"{an}正一步步逼向{bn or '旁人'}。",
            f"{a['surface']}" if a["surface"] else "",
            f"他眼下这一步：{a['step']}。" if a["step"] else "",
            f"{bn}这边：{b['surface']}" if bn and b["surface"] else "",
            "再拖几秒，可能就见血。",
        ])
    elif kind == "weapon_kill":
        tag = "袖口有东西"
        signs = _join([
            f"有人看见{an}走路护着腰、袖口反光，不像空手。",
            f"{a['surface']}" if a["surface"] else "",
            f"他眼下：{a['step']}。" if a["step"] else "",
            f"眼神刮过{bn}。" if bn else "",
        ])
    elif kind == "argue":
        tag = "当众翻旧账"
        signs = _join([
            f"{an}当众冲着{bn or '人'}吼，唾沫星子都溅到脸上了。",
            f"{a['surface']}" if a["surface"] else "",
            f"这一步他在做：{a['step']}。" if a["step"] else "",
            f"{bn}呢：{b['surface']}" if bn and b["surface"] else "",
            "再不管，下一句就是动手。",
        ])
        fog = a["fog"] or b["fog"] or "他们过的究竟是仇、是货、还是一条命，隔着人声听不全。"
    elif kind == "escape_sign":
        tag = "在找活路"
        signs = _join([
            f"{an}老往墙根、大门那边钻，靴底带泥，不像放风闲逛。",
            f"{a['surface']}" if a["surface"] else "",
            f"眼下这一步：{a['step']}。" if a["step"] else "",
        ])
    elif kind == "grudge_fight":
        tag = "仇怨将爆"
        signs = _join([
            f"{an}盯着{bn or '对家'}的眼神不对，拳头攥得发白。",
            f"{a['surface']}" if a["surface"] else "",
            f"他正做：{a['step']}。" if a["step"] else "",
            f"{bn}：{b['surface']}" if bn and b["surface"] else "",
            "再撞上一次，可能就动手。",
        ])
    elif kind == "injury_plea":
        tag = "带伤硬撑"
        signs = _join([
            f"{an}伤得不轻，却不肯去医务室，像在硬撑，也像在躲谁。",
            f"{a['surface']}" if a["surface"] else "",
            f"他眼下：{a['step']}。" if a["step"] else "",
        ])
        fog = a["fog"] or "伤是自己摔的还是被人下手，他不说。"
    else:
        tag = "出事了"
        signs = _join([
            a["surface"],
            f"眼下：{a['step']}。" if a["step"] else "",
        ]) or f"{an}这边不对劲。"

    return {"signs": signs.strip(), "fog": fog, "tag": tag}


# 凶器中文名（前端/命运账本/对话用）
WEAPON_CN: dict[str, str] = {"poison": "毒药", "knife": "刀子", "taser": "电击枪"}


# ---------------------------------------------------------------------------
# 五人任务链（key = NPC id，与 personas.ROSTER 对齐）——没有玩家本人的自洽群像剧。
#
# 生死网升级为「人人皆有仇杀对象」的五角闭环（顺时针一圈）：
#   疤脸 → 老牙 → 瘸子 → 柳叶刀 → 疯子 → 疤脸
# 每个人既是猎手、也是猎物；每条边配一段能讲出来的背景故事(kill_motive)——
# 或因某人当年把他害进来、或因入狱后结下的死仇、或为灭口除掉知情人。
#   · 疤脸→老牙：老牙当年卖了他顶罪，白蹲好几年。
#   · 老牙→瘸子：瘸子攥着能钉死老牙『真凶』身份的把柄，要灭口。
#   · 瘸子→柳叶刀：黑医拿医疗档案常年拿捏他、还克扣药害死过他的兄弟。
#   · 柳叶刀→疯子：疯子撞破了他的勾当，要在说漏前灭口。
#   · 疯子→疤脸：疯子偏执地认定是疤脸把他打进禁闭室、害成今天这样。
# 谁能得手取决于：凶器到没到手、目标落没落单、有没有被拦下/被狱警撞破——
# 一夜死 0~4 人不等（既遂命案见 world._do_kill；未遂/拦截/殃及见 combat 分支）。
# 每条链的 focus 动作都取自 actions.py（含 GetWeapon 弄凶器 / Assassinate 下杀手）。
# ---------------------------------------------------------------------------
QUESTS: dict[str, Quest] = {
    # 老牙：向导/真凶。要灭掉攥着他老底的走私贩子瘸子。手无寸铁，得先弄把见血的家伙。
    "laochen": Quest(
        ultimate="赶在东窗事发前，除掉那个攥着自己老底的走私贩子瘸子",
        kind="kill", target="laok", weapon="",
        kill_motive="瘸子手里攥着半张旧欠条，还有老牙当年在外欠下命债、买凶顶罪的旧事——"
                    "这条线一旦捅出去，老牙这个『真凶』就钉死了。他要赶在事发前让瘸子永远闭嘴。",
        surface="老牙是这牢里的老江湖，路子最广。今夜他不像闲逛，眼神总往瘸子那边刮。",
        fog="他软底鞋走路没声，靴筒里藏没藏见血的家伙，只有贴身搜才清楚。",
        steps=(
            Step("不动声色摸清瘸子的门路、何时落单，顺手弄把见血的家伙",
                 frozenset({"Investigate", "GetWeapon", "Barter"}), 10,
                 "老牙从相熟的路子上摸回一把磨尖的铁片，掖进袖管，脸上还是那副和气样。", True,
                 surface="他跟在瘸子后头打听行踪，像在张罗一件见不得光的东西。",
                 fog="他弄到手没有，旁人看不见。"),
            Step("缀着瘸子，等他那条瘸腿挪到没人的角落",
                 frozenset({"Watch", "Investigate"}), 10,
                 "老牙远远吊着瘸子，一步不近一步不远，就等他身边没人。",
                 surface="他不动声色缀着瘸子走，像在等他落单。",
                 fog="他打算什么时候下手，只有他自己清楚。"),
            Step("逮住落单的瘸子，无声无息了结他",
                 frozenset({"Assassinate", "Fight"}), 10,
                 "老牙贴到了瘸子背后，软底鞋没发出一点声响……", True,
                 surface="他贴着瘸子逼上去了，今晚那条瘸腿怕是走不出这道墙。",
                 fog="这一下是吓唬还是要命，旁人不敢赌。"),
        ),
    ),
    # 疤脸：弄刀 → 动手。找瘸子弄把刀，了结坑过他的老牙。
    "shitou": Quest(
        ultimate="亲手了结那个坑过他、害他多蹲好几年的老牙",
        kind="kill", target="laochen", weapon="",
        kill_motive="老牙当年在外头把疤脸卖了，让他替一桩不属于自己的案子顶了罪，白蹲了好几年。"
                    "这笔账疤脸记到今天，今夜火气再也压不住。",
        surface="疤脸记着老牙害他多蹲的那些年，今夜火气压不住，像要了结这桩旧账。",
        fog="他腰里那一鼓究竟是不是见血的家伙，还得搜了才算数。",
        steps=(
            Step("放出狠话，找瘸子弄一把见血的家伙",
                 frozenset({"Argue", "GetWeapon", "Barter"}), 10,
                 "疤脸从瘸子那儿摸回一把磨得锃亮的刀，掖进裤腰，眼里全是狠劲。", True,
                 surface="他当众放狠话，往瘸子那边凑，像在要一件能见血的东西。",
                 fog="货有没有到手、瘸子卖没卖，还没实锤。"),
            Step("盯死老牙的行踪",
                 frozenset({"Watch", "Investigate"}), 10,
                 "疤脸远远缀着老牙，像头等着扑食的兽——就等他身边没人。",
                 surface="他不吵了，改成缀着老牙走，像在等落单。",
                 fog="他准备什么时候下手，只有他自己清楚。"),
            Step("逮住落单的老牙，一刀了结他",
                 frozenset({"Assassinate", "Fight"}), 10,
                 "疤脸攥紧了刀，一步步逼向落了单的老牙……", True,
                 surface="他贴着老牙逼上去了，今晚可能见血。",
                 fog="这一下是教训还是要命，旁人不敢赌。"),
        ),
    ),
    # 瘸子：军火贩子，自己揣电击枪。表面陪笑，暗地想除掉常年拿捏他的黑医柳叶刀。
    "laok": Quest(
        ultimate="除掉那个拿医疗档案常年拿捏自己、还害死过弟兄的黑医柳叶刀",
        kind="kill", target="yanjing", weapon="taser",
        kill_motive="柳叶刀攥着瘸子的医疗档案当把柄，年年逼他低价过药、断他财路，"
                    "还克扣过救命药、害死了瘸子拜把的兄弟。瘸子表面陪着笑，心里恨得早想让这黑医今夜下不了台。",
        surface="瘸子两边都做生意谁的钱都赚，可一提起医务室就眼神发冷，今夜总往柳叶刀那边瞟。",
        fog="他贴身藏的那把电击枪、和他跟黑医的这段旧怨，都得问出来、搜出来才算数。",
        steps=(
            Step("借过药之名接近柳叶刀，把防身的电击枪贴身藏好",
                 frozenset({"Barter", "LayLow", "SellInfo"}), 10,
                 "瘸子揣着电击枪往医务室那边凑，脸上堆着笑，袖口却绷得发紧。", True,
                 surface="他一瘸一拐往医务室凑，像去过药，又像在踩点。",
                 fog="他怀里那硬邦邦的一块是什么，得搜才知道。"),
            Step("盯着柳叶刀，等这冷面黑医落单",
                 frozenset({"Watch", "Investigate"}), 10,
                 "瘸子远远盯着柳叶刀进出药柜，把他的空当一一记在心里。",
                 surface="他缀着柳叶刀走，像在等他身边没人。",
                 fog="他打算什么时候动手，只有他自己清楚。"),
            Step("趁柳叶刀落单，一下撂倒这个拿捏他的人",
                 frozenset({"Assassinate", "Fight"}), 10,
                 "瘸子摸出电击枪，一瘸一拐地逼近落了单的柳叶刀……", True,
                 surface="他贴着柳叶刀逼上去了，那把电棍今晚怕要见真章。",
                 fog="这一下是泄愤还是要命，旁人不敢赌。"),
        ),
    ),
    # 柳叶刀：开局就有毒药，三步内就能下手。灭口撞破他秘密的疯子。
    "yanjing": Quest(
        ultimate="神不知鬼不觉地毒死那个撞破他秘密的疯子，永绝后患",
        kind="kill", target="xiaodou", weapon="poison",
        kill_motive="陷害那晚用的迷药出自柳叶刀的医务室，疯子撞见过他不该被人看见的勾当。"
                    "柳叶刀要赶在疯子那张破嘴说漏之前，把这个知情人神不知鬼不觉地灭口。",
        surface="黑医柳叶刀跟疯子不对劲。疯子撞见过他不该看见的事，今夜柳叶刀总往疯子那边靠。",
        fog="他白大褂里那一小包是安神药还是别的，隔着人看不清。",
        steps=(
            Step("不动声色地摸清疯子的作息、何时落单",
                 frozenset({"Investigate", "Watch"}), 10,
                 "柳叶刀在病历本背面记下了疯子的作息——几点放风、几点犯病。", True,
                 surface="他不急着下手，先跟在疯子后面记作息。",
                 fog="本子上写了什么，只有他看得懂。"),
            Step("借复诊接近疯子、把那包东西贴身藏着",
                 frozenset({"Socialize", "LayLow"}), 10,
                 "柳叶刀破天荒对疯子和颜悦色，白大褂内袋里却多了一小包白粉。", True,
                 surface="他破天荒对疯子和颜悦色，像要给人看病。",
                 fog="内袋那一小包，不是走到跟前搜，你看不到。"),
            Step("趁疯子落单，悄无声息地下手",
                 frozenset({"Assassinate"}), 10,
                 "柳叶刀端着那杯『安神药』，脸上没有一丝表情，向疯子走去……", True,
                 surface="他端着杯子向疯子走去，脸上没有表情。",
                 fog="那杯东西倒底是药还是要命的，问他他也不会直说。"),
        ),
    ),
    # 疯子：目击者，被熬垮的偏执者。认定是疤脸把他害成这样，攥着一片铁皮想先下手。
    "xiaodou": Quest(
        ultimate="在那影子再找上门之前，先把害他成这样的疤脸做掉",
        kind="kill", target="shitou", weapon="",
        kill_motive="疯子精神早被禁闭熬垮，一口咬定是疤脸当年带人把他打进禁闭室、害他成了今天这副"
                    "人不人鬼不鬼的样子。今夜那『影子』又来了，恐惧逼出了狠劲——真凶是不是他，疯子自己也分不清了。",
        surface="疯子这两日惶惶的，嘴里念叨着『影子』，眼睛却直勾勾黏在疤脸身上，像认定了仇人。",
        fog="他攥在袖子里那片冰凉的东西、和他究竟撞破过谁的什么事，他自己都说不利索。",
        steps=(
            Step("在惶恐里把满腔的恨认到疤脸头上，摸到一片磨尖的铁皮",
                 frozenset({"Pray", "GetWeapon", "LayLow"}), 10,
                 "疯子缩在墙角，把一片磨尖的铁皮攥进掌心，一遍遍念：『是他……就是他……』", True,
                 surface="他缩在墙角念叨『就是他』，眼神黏着疤脸不放。",
                 fog="他掌心攥着的那片冰凉，不搜看不见。"),
            Step("远远盯着疤脸，等那影子再靠近",
                 frozenset({"Watch", "SeekProtector"}), 10,
                 "疯子躲在暗处死盯着疤脸，嘴里数着谁也听不懂的数。",
                 surface="他躲着疤脸又盯着疤脸，忽远忽近，像在等一个只有他知道的时机。",
                 fog="他到底想干什么，连他自己都未必清楚。"),
            Step("被恐惧逼到头，扑向疤脸",
                 frozenset({"Assassinate", "Fight"}), 10,
                 "疯子突然从暗处窜出，攥着那片铁皮尖叫着扑向疤脸……", True,
                 surface="他疯魔了一样冲向疤脸，今晚这条命悬了。",
                 fog="他扑上去是要命还是求死，谁也说不清。"),
        ),
    ),
}


def kill_motive_of(npc_id: str) -> str:
    q = QUESTS.get(npc_id)
    return q.kill_motive if q else ""


def quest_of(npc_id: str) -> Quest | None:
    return QUESTS.get(npc_id)
