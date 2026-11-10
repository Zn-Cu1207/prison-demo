"""世界状态 + 仿真循环（监狱原生版）。

每拍做的事：
  漂移需求 → 算拥挤预期 → 拼监狱原生 context → 生成候选（按需求/时段/处境，不看性格）
  → 读分缓存算分 → 决策主流程选目标 → 移动 → 帧末合批把打分请求丢给后台。

打分与决策照「RA → 监狱 Demo · 可迁移逻辑与实现清单」移植：
  §2.2 完整乘数链（M_light / M_inertia / 距离成本 / 打断·失败·超时·重复四惩罚）
  §2.3 决策主流程（当前目标回灌、全≤0 保持、切换迟滞 margin、指令必胜）
  §2.4 异步分缓存 + 跨 NPC 合批 /score_multi —— 模拟不再同步等模型

分工的分界线始终是：**模型管"想做什么"，引擎管"能不能去、值不值得跑、要不要改主意"**。
M_crowd（放风拥挤）和 M_command（玩家指令）是本 Demo 在 RA 公式外加的两项，
等监狱专属轴重训完，这两项应当收回模型内部。
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace

from .actions import ACT_RELIEF as ACT_RELIEF_TABLE
from .actions import GOAL_TO_KIND as ACT_GOAL_TO_KIND
from .actions import ILLICIT_GOALS, PREP_BY_GOAL
from .drives import get_drive, AIMLESS
from . import layout
from . import nav
from .nav import NAV
from .quests import QUEST_FOCUS_BOOST, STEP_GAIN, WEAPON_CN, plot_of, quest_of
from .shift import ShiftState
from .social import REL_AXES, REL_CN, SocialEvent, blank_rel, parse_conversation
from . import nlu
from .plot import (
    ALCOHOL,
    ANCHOR_ID,
    CLUE_GRANTS,
    UNLOCK_ORDER,
    DOCTOR_ID,
    DOG_ID,
    EXPOSE_EVIDENCE,
    FACT_BASE,
    GANG_ID,
    GUARD_ID,
    MADMAN_ID,
    KEYCARD,
    PENSION_LIST,
    SEDATIVE,
    PHASE_GUIDE,
    SMUGGLER_ID,
    SOFT_SPOT,
    TOOTHBRUSH,
    WARDEN_ID,
    Blackboard,
    ending_system_prompt,
    fact_lines,
    phase_intro_directive,
    resolution_score,
    wants_escape_tool,
    zone_of_kind,
)
from .context import (
    CROWD_EMPTY,
    CROWD_MID,
    CROWD_PACKED,
    EVENT_NONE,
    GOAL_LABEL,
    # v2 caption 四新轴的取值常量（_supply_axis / _grudge_axis / _perform_axis 用）
    GRUDGE_DEEP,
    GRUDGE_NONE,
    GRUDGE_SOME,
    PERFORM_BAD,
    PERFORM_GOOD,
    PERFORM_MID,
    SUPPLY_OK,
    SUPPLY_OUT,
    SUPPLY_TIGHT,
    PHASE_CLASS,
    PHASE_FREE,
    PHASE_ROLLCALL,
    PHASE_SLEEP,
    PHASE_TRAIN,
    PHASE_YARD,
    SURVEIL_HIGH,
    SURVEIL_LOW,
    SURVEIL_MID,
    THREAT_NONE,
    # v3 社交四轴取值常量（_social_axes 用，P2）
    NEAR_ALLY,
    NEAR_CREDITOR,
    NEAR_DEBTOR,
    NEAR_FOE,
    NEAR_NONE,
    NEAR_STRANGER,
    INTENT_ALLY,
    INTENT_DEAL,
    INTENT_FRIENDLY,
    INTENT_HOSTILE,
    INTENT_NONE,
    INTENT_UNKNOWN,
    INTENT_WEAK,
    SOC_ALLIED,
    SOC_DEAL,
    SOC_NONE,
    SOC_PROTECTED,
    SOC_RECONCILED,
    SOC_SNITCHED,
    SOC_THREATENED,
    OBLIG_NONE,
    OBLIG_OWE,
    OBLIG_OWED,
    DESC_MAP,
    Candidate,
    ContextState,
    NeedState,
    build_context_caption,
    caption_version,
)
from .model_client import ModelClient, ScoredCandidate
from .narrate import activity, explain, monologue
from .personas import ROSTER
from .combat import (
    SEED_GRUDGE,
    SEED_REL,
    SEED_OBLIGATION,
    FIGHT_MAX_TICKS,
    FIGHT_RANGE,
    GRUDGE_ON_HIT,
    GRUDGE_ON_LOSS,
    GRUDGE_ON_STOLEN_SUPPLY,
    GRUDGE_TO_FIGHT,
    JOIN_CHANCE,
    JOIN_RANGE,
    KO_INJURY,
    STRESS_PER_TICK,
    FightBout,
    roll_damage,
    strength,
)
from .world_objects import (
    PERFORM_EFFECT,
    PERFORM_TICKS,
    REQUIRED_OBJECT,
    WorldObject,
    spawn_objects,
)

log = logging.getLogger(__name__)

# 需求触发阈值
# 需求进入候选的门槛：抬高"填充型"需求(无聊/社交/饥渴)的门槛，让日常需求别再霸屏——
# 只有真正压上来时才排进候选，把决策桌面让给任务/剧情/互动动作。生存危急仍由模型在
# 85+ 高区裁决(见 guardrails)，与这里的候选门槛无关，故不动 injury/fatigue 的裁决逻辑。
THRESHOLDS = {"hunger": 42, "thirst": 42, "fatigue": 55, "boredom": 26, "social": 42, "injury": 30}

# 作息时段顺序 + 每段时长（拍）。一天 = 6×50 = 300 拍 ≈ 150s
PHASE_ORDER = [PHASE_ROLLCALL, PHASE_TRAIN, PHASE_CLASS, PHASE_YARD, PHASE_FREE, PHASE_SLEEP]
PHASE_TICKS = 50

# 各时段的监视度
PHASE_SURVEIL = {
    PHASE_ROLLCALL: SURVEIL_HIGH,
    PHASE_TRAIN: SURVEIL_MID,
    PHASE_CLASS: SURVEIL_MID,
    PHASE_YARD: SURVEIL_MID,
    PHASE_FREE: SURVEIL_LOW,
    PHASE_SLEEP: SURVEIL_LOW,
}

# goal → 设施 kind / 需求缓解：都从行动全集 actions.py 派生（单一真相源）
GOAL_TO_KIND = ACT_GOAL_TO_KIND

# 去放风场的"休闲"候选（受拥挤预期调节的那批；Fight 虽在放风场但不压制）
YARD_LEISURE = {"Patrol", "DeclineBoredom", "Socialize"}

# 各时段额外提供的"丰富日常活动"菜单（让选择更多样，也喂给 LLM 大脑发挥）
PHASE_ACTIVITIES = {
    PHASE_TRAIN: ("Workout", "Clean", "Laundry"),
    PHASE_CLASS: ("Read", "Chess"),
    PHASE_YARD: ("Workout", "Gamble", "Smoke", "Music", "Barter"),
    PHASE_FREE: ("Workout", "Shower", "Read", "Gamble", "Smoke", "Visit", "Pray",
                 "Chess", "Music", "Tattoo", "Barter"),
    PHASE_SLEEP: ("Pray", "Tattoo"),
}

# 每拍从时段菜单里最多取几项。菜单越长选择越丰富，但候选全量喂给 LLM 会明显变慢，
# 所以按 (tick, npc) 轮换采样：单拍候选可控，跨时间仍能把整张菜单走遍。
ACT_MENU_MAX = 4

# 丰富活动到达后的需求缓解：从行动全集派生（生存/放风类走 _drift_needs 的 kind 分支）
ACT_RELIEF = ACT_RELIEF_TABLE

# El Farol：放风场"舒适阈值"（个人容忍大多聚在 2 附近，取 2 作曲线参考线）
YARD_COMFORT = 2

# 性格 → (对拥挤的偏好, 个人容忍人数)。这是"性格差异"的引擎侧来源，重训后交回模型。
#   like   = 爱扎堆（暴躁），人越多越想去
#   avoid  = 怕挤/怕事（书呆、新人），预期超过容忍就躲开
#   neutral= 看情况（老油条、普通）
PERSONALITY_CROWD = {
    "暴躁": ("like", 5),
    "书呆": ("avoid", 2),
    "老油条": ("neutral", 2),
    "新人": ("avoid", 2),
    "普通": ("neutral", 2),
}

# 各犯人的拥挤预测策略（异质预测器 = El Farol 有限理性的核心）
PREDICTOR_BY_ID = {
    "laok": "avg4",      # 老油条：看近 4 拍平均，稳
    "shitou": "last",    # 暴躁：只看上一拍，反应快
    "yanjing": "trend",  # 书呆：看趋势，提前预判
    "xiaodou": "contra", # 新人：逆向猜（越怕越猜相反）
    "laochen": "avg2",   # 普通：近 2 拍平均
}

# M4 玩家挑事：事件 kind -> (作用字段, 取值)。注入后影响所有 NPC 的 context。
EVENT_INJECT = {
    "狱犬": ("threat", "狱犬"),
    "警报": ("event", "警报"),
    "搜牢": ("event", "狱警搜牢"),
    "斗殴": ("event", "斗殴"),
}
EVENT_DURATION = 30       # 注入事件持续拍数
COMMAND_DURATION = 24     # 玩家指令有效拍数（未到达则到期自动解除）
# 指令保底乘数：非紧急时指令占优；有生存紧急(饥饿/口渴/受伤>阈)时减弱，允许合理抗命
COMMAND_BOOST = 4.0
COMMAND_BOOST_EMERGENCY = 1.6
EMERGENCY_NEED = 82

# ---------------------------------------------------------------------------
# 主线意图·承诺压力（Plot Pressure）—— 治"嘴上念叨主线、身子迟迟不动"的根因。
#
# 病根：价值模型逐拍最优，非危机需求(中度饿/渴)常压过主线；主线只在模型偶然选中
#   focus 动作时才推进，于是 NPC 能把"找瘸子弄刀"念叨一整天却永远不去 → 言行脱节、
#   关键剧情永不发生。
# 解法：给每个还没啃完任务链的 NPC 一条"承诺压力"。安全时它每拍不推进主线就攒压，
#   推进(做了 focus 动作)就清零。压力越高：① focus 动作的 m_drive 越水涨船高；
#   ② 攒满(≥FORCE)且此刻安全(没被盯/没濒死/没被指挥)时，硬规则直接把 focus 顶到最高，
#   保证"关键一步"必然在有限的安全时间内落地。护栏阶梯不破：濒死/威胁/指令仍先于它，
#   它只压过"非危机需求 + 日常闲逛"——一个正谋划大事的人，本就不会为每顿饭无限推迟正事。
# 节拍：tick=0.5s。RISE=0.02 → 约 50 拍(~25s)安全时间攒满，念叨几句后必然起身去办。
PRESSURE_RISE = 0.045       # 安全且主线停滞时，每拍攒多少压（~11s 攒满，夜班里赶得上）
PRESSURE_MAX = 1.4          # 压力上限
PRESSURE_GAIN = 7.0         # 折进 focus 动作 m_drive 的系数（额外乘数 = 压力×此值）
PRESSURE_FORCE = 1.0        # 压力到此且安全 → 硬规则强制推进当前分目标（关键一步必落地）
PRESSURE_TENSION = 0.5      # 压力到此：忙需求时的独白改口成"惦记着主线"的纠结话（诚实配音）

# LLM 独白：一句心里话在气泡里停几拍再换下一句（8 拍≈4s，配 8 句池子≈一句Opus管32s）
MONO_HOLD_TICKS = 8

# 生死网：下杀手需要贴到这么近；行凶落网的后果（加刑年数 / 长禁闭拍数）
KILL_RANGE = 3.5
KILL_SENTENCE_ADD = 10.0
KILL_SOLITARY_TICKS = 120

# —— 涌现命运：把"必杀"改成"较量+分支"，让每局结局不再固定 ——
# 中等随机(倾向剧情但有可观翻盘率)标定：干净得手约 55%，有保护者时压到 ~15%。
# 只有毒药(柳叶刀)/刀(疤脸)会真用来杀；电击枪(瘸子)是求生者防身，基础低=多半只是制服。
KILL_BASE = {"poison": 0.70, "knife": 0.62, "taser": 0.30}
KILL_GUARD_SIGHT = 7.0       # 行凶时这么近内有巡逻者 → 有概率当场被撞破
KILL_CAUGHT_CHANCE = 0.40    # 撞破概率（被按住＝行凶未遂、凶手落网、目标活）
KILL_RETRY_CD = 20           # 一次失手后约 10s 再下手（夜班里要真见血，窗口不能拖太长）
KILL_ATTEMPT_SOLITARY = 34   # 未遂/被撞破的禁闭拍数（短于既遂的长禁闭）
KILL_ATTEMPT_SENTENCE = 3.0  # 未遂被抓的加刑年数（轻于既遂）
PROTECT_RANGE = 4.6          # 保护者在受害者这么近才拦得住
POISON_SPILL_ON_FAIL = 0.5   # 毒药下手失败时一半概率被打翻用光（柳叶刀可能白忙一场）
COLLATERAL_LETHAL = 0.07     # 火并/行凶混乱中当场丧命的概率（调低：让死亡更多来自剧情化仇杀，
                             # 少来自随机群殴。真正的贴身行凶(_do_kill)不走这条，不受影响）

# NPC↔NPC 对话（观察者模式）触发/节流参数
CONVO_RANGE = 6.0            # 两人多近才会攀谈（触发后前端会让二人凑到面对面）
CONVO_GLOBAL_CD = 120        # 全局：两场对话至少隔这么多拍（~60s）。原来 28(~14s) 太密，
                             # 场上一直有人在对话、💬 满屏。拉长后同一时刻基本只有一对在聊，
                             # 其余 NPC 安静地由动作模型做日常，信息密度降下来。
CONVO_PAIR_CD = 360          # 同一对：这么多拍(~3min)内不重复攀谈（原 150 太短）
CONVO_GRUDGE_MIN = 35        # 仇怨到这个值 → 触发口角（有戏，优先）
# 对话钩子的优先级档（_convo_plan 返回，_plan_convo 先比档再比分数）。
# 各档的 score 量纲不可比，所以必须靠这个显式档位来定序，见 _convo_plan 的说明。
HOOK_TIER_KILL = 4           # 生死网：一方要另一方的命
HOOK_TIER_DEBT = 3           # 军火债：义务账本里有未结的债
HOOK_TIER_GRUDGE = 2         # 旧仇：积怨到口角线
HOOK_TIER_ALLY = 1           # 结盟：拴在一条绳上
HOOK_TIER_FALLBACK = 0       # 无钩子：泛泛闲聊
CONVO_SOCIAL_GOALS = {"Socialize", "Chess", "Barter", "Patrol", "Rest",
                      "Phone", "Study", "Laundry", "Tattoo", "LayLow"}

# ---------------------------------------------------------------------------
# RA 打分公式 / 决策主流程移植（见「可迁移逻辑与实现清单」§2.2 §2.3 §2.4）
#
#   最终分 = M_value × M_light × M_inertia ÷ (1+Cost)
#            × M_interruption × M_failure × M_timeout × M_repetition
#            × M_crowd × M_command      （后两项是本 Demo 的引擎侧扩展）
# ---------------------------------------------------------------------------
INERTIA_BONUS = 0.35        # M_inertia = 1+该值，只给"正在做的目标"。防一拍一个主意
# 刚下的决心额外加一层、随拍数衰减到 0。
# RA 里 M_light/M_crowd 这类环境项摆幅小，光靠 1.35 的惯性就够；本 Demo 的拥挤项
# 为了演示 El Farol 摆幅高达 12 倍，只有基础惯性会让人在放风场门口反复横跳。
# 让"刚决定"的那几拍更难被撼动，人会先把路走完，拥挤震荡也回到更自然的周期上。
COMMIT_BONUS = 4.0
COMMIT_TICKS = 8            # 决心加成衰减到 0 所需的拍数
# 执行承诺：已经走到位、正在"做这件事"(act_stage=perform)时给一层强惯性，
# 让动作有始有终，不再"才做一两秒就被顶替"。生存危机因其模型分极高，仍能压过它打断
# （危机档 m_value 远大于日常动作，即便乘上承诺也拦不住），所以只挡住无谓的日常摇摆。
PERFORM_COMMIT = 2.6
SWITCH_MARGIN = 0.30        # 切换迟滞：新目标要高出当前目标这么多比例才换（略调高，行为更连贯）
# 距离成本尺度：Cost = 距离/该值，场地对角约 30m → Cost 最大 ≈2.1。
# 放风场在地图东侧、食堂水房在西侧，这一项会给"去放风"加一份纯几何的固定罚分，
# 调大调小都试过：锁定放风时段跑 100 拍，14 时出勤均值 1.88（舒适阈值 2 附近来回摆），
# 22 时塌成一条直线不再自组织。别为了"让人少走路"把这个数往上调。
COST_SCALE = 14.0
NIGHT_OUTDOOR_M = 0.5       # M_light：熄灯（就寝时段）后户外行动打折

# 四项惩罚的记账参数
INTERRUPT_BONUS = 0.6       # 目标没做完就被换掉时，给它记的补偿（下次更容易回去收尾）
INTERRUPT_DECAY = 0.85      # 补偿每拍衰减
PENALTY_DECAY = 0.92        # 失败/超时/重复惩罚每拍衰减
TIMEOUT_TICKS = 40          # 认定"这么久还没到达"的拍数
TIMEOUT_PENALTY = 0.35
FAILURE_PENALTY = 0.50      # 目标压根没有对应设施 → 去不了
REPEAT_PENALTY = 0.12       # 近窗口内每多完成一次同一目标累加
REPEAT_PENALTY_CAP = 0.60   # 封顶，别把"饿了就得吃"这种刚需也压死
REPEAT_WINDOW = 10          # 重复统计窗口（最近完成的 N 个目标）

# —— P1 · 社交写回（对白→世界）的关系轴变化量与阈值 ————————————————————————
MEMORY_MAX = 24             # 每个 NPC 的情景记忆容量（按显著度检索，不是全量回放）
# Generative Agents 三因子检索。全场 5 人、每人最多 24 条、每条已带 {tick,kind,other,w}，
# 先符号打分即可，上向量库/BGE-Reranker 会把"语义相似但对象不对"的废话捞回来。
MEM_RECENCY_HALFLIFE = 200  # 半衰期（拍）：200 拍前的事 recency=0.5
MEM_W_CEIL = 2.4            # 写入端最高显著度（betrayed_by=2.4），用来把 w 归一到 0~1
MEM_ALPHA, MEM_BETA, MEM_GAMMA = 0.30, 0.30, 0.40  # recency / importance / relevance
REL_MAX = 100.0
# 一场对话触发某类社交事件时，各关系轴的增量（保守，单场改不爆）
DELTA_THREAT_FEAR = 16.0    # 被威胁 → 怕对方
DELTA_THREAT_GRUDGE = 8.0   # 威胁 → 双方梁子都更深（施威者更甚）
DELTA_DEAL_DEBT = 20.0      # 收了军火 → 欠对方人情/钱
DELTA_DEAL_TRUST = 8.0      # 做成买卖 → 生意上的信任
DELTA_ALLY = 18.0           # 结盟/收人
DELTA_PROTEGE_TRUST = 14.0  # 被罩者 → 信任靠山
DELTA_RECONCILE_GRUDGE = -22.0   # 讲和 → 敌意回落（可能压到打架线以下）
DELTA_RECONCILE_TRUST = 10.0
DELTA_INFO_TRUST = 6.0
ALLY_NOFIGHT = 15.0         # 结盟到这个值：engine 侧不再把对方选成动手对象（一次结盟即够）
# v3/P2 定向社交动作：由决策脑在社交轴齐备时选择，引擎按"近旁最该打交道的人"落地。
SOCIAL_ACTION_GOALS = {"Propose", "Ally", "Protect", "Repay", "Threaten", "Betray"}
SOCIAL_RANGE = 13.0         # 多近才算"近旁那个人"（社交动作作用半径）。8 太窄，同房也常够不着，
                            # 社交半天不冒一条；放到 13≈同一活动区就算近旁，让互动真的发生。
SOCIAL_COOLDOWN = 10        # 社交动作冷却拍数（别一秒一个"结盟/背叛"刷屏）
# 关系钩子加成：有真实关系(仇敌/盟友/债主/欠我)时，把定向社交抬到能压过纯发呆式日常
# (Patrol/Rest)，但仍低于需求/威胁的高基分——护栏层级不破，只是别让"有戏的社交"被闲逛淹没。
SOCIAL_HOOK_BOOST = 2.3

# M_light 打折的户外目标。越狱本来就该趁夜黑，单独豁免。
_OUTDOOR_KINDS = {"yard", "gym", "gate", "market"}
OUTDOOR_GOALS = {g for g, k in GOAL_TO_KIND.items() if k in _OUTDOOR_KINDS} - {"PrisonBreak", "Tunnel"}
# 必须走进这间功能室、停够拍数才算做成。yard/gate 是露天落点，另用距离。
_ROOM_KINDS = {
    "eat", "drink", "sleep", "cure", "shower", "gym", "commons", "library",
    "class", "laundry", "labor", "visit", "market", "guardpost", "phone",
}
_CELL_GOALS = {"Sleep", "Rest", "Pray", "Hide"}
_SKIP_DWELL = SOCIAL_ACTION_GOALS | {
    "LayLow", "Fight", "Argue", "Assassinate", "Tunnel", "PrisonBreak", "Bury", "Mourn",
}
# 进了屋还得待一会儿。生存类用 world_objects.PERFORM_TICKS；这里补日常。
ROOM_DWELL = {
    "Shower": 12, "Workout": 14, "Read": 12, "Chess": 12, "Music": 10,
    "Laundry": 10, "Study": 12, "Visit": 10, "Phone": 8,
    "KitchenDuty": 12, "Labor": 14, "Clean": 10, "Barter": 8,
    "Tattoo": 10, "Investigate": 10, "GetWeapon": 8, "Snitch": 8,
    "ReviewFiles": 12, "SeekProtector": 8, "Extort": 8, "SellInfo": 8,
    "CurryFavor": 8, "Hide": 10, "Pray": 10, "Rest": 10,
    "Patrol": 10, "Socialize": 10, "DeclineBoredom": 8, "Gamble": 10,
    "Smoke": 8, "Watch": 8,
}

# ---------------------------------------------------------------------------
# 狱警 / 狱犬 巡逻 + 视线（动机的环境来源）
#   被盯上 → 威胁字段变狱警/狱犬、监视度拉满、违禁动作被 m_watch 压没。
#   这逼着 NPC "伺机而动"：等巡逻走开、或让同伙望风拖住注意力，才敢干见不得人的事。
# ---------------------------------------------------------------------------
GUARD_VIEW_RANGE = 8.0      # 狱警视野距离（收窄一档：别让 NPC 老被盯着被迫躲，腾出时间做任务）
GUARD_POINT_BLANK = 1.5     # 贴到这么近就一定被发现（原来 d≈0 会因视锥算不出而漏判）
GUARD_VIEW_COS = 0.5        # 视锥半角 60°（cos60°=0.5）：只盯正前方扇形，会被绕后
DOG_SMELL_RANGE = 5.0       # 狱犬嗅觉半径（360°，近但无死角；收窄一档，减少无谓的收敛）
M_WATCH_ILLICIT = 0.02      # 被盯上时违禁动作的乘数（几乎压没，但非硬删，面板看得见）
# 越狱链参数
ESCAPE_BREACH_TICKS = 3     # 撬门累计到这个数算"挖通"（夜班窗口短，不能攒太久）
GATE_REACH_DIST = 1.2       # 到大门多近算到位

# ---------------------------------------------------------------------------
# 假释：从"静态布尔"改成真实的表现分（改这版之前的问题）
#   原来 parole_near 只是种子里的一个 bool，而且**全场没有一个 NPC 是 True**，
#   整套假释机器（drives 的 parole 心思、CurryFavor 签名动作、越狱门禁）都是死的。
#   现在它是 0~100 的表现分：干活/刷表现涨，打架被抓、越狱失败掉并加刑期。
#   parole_near 由分数派生 → 立刻接通已经写好的 _m_drive 与越狱门禁。
# ---------------------------------------------------------------------------
PAROLE_NEAR_AT = 70.0        # 表现分到这个线算"假释临近"
PAROLE_GAIN_WORK = 3.0       # 完成一次苦役/帮厨/打扫
PAROLE_GAIN_CURRY = 6.0      # 完成一次刷表现
PAROLE_FIGHT_PENALTY = 25.0  # 聚众斗殴被抓
PAROLE_ESCAPE_PENALTY = 60.0 # 越狱失败被抓
EXTRA_DAYS_FIGHT = 90        # 加刑天数
EXTRA_DAYS_ESCAPE = 365
SOLITARY_TICKS = 40          # 关禁闭的拍数（期间完全不能行动）
RECALL_TICKS = 200           # 典狱长点名回牢房后乖乖待着的拍数（约 100 秒）。
                             # 原来 ~40 秒太短，NPC 很快又自主乱跑，观感像"没听命令"。
RECALL_SNAP_TICKS = 40       # 连续走不通这么多拍(~20s)才兜底落格。原来 10 拍太急，稍一被人
                             # 挡道就硬传回牢房，观感就是"瞬移"。放宽后优先让他自己寻路走回去，
                             # 只有真卡死（门锁、堵死）才落格兜底。
# 从禁闭放出来之后的"老实期"拍数。只在人自由时递减（关着的时候不走），
# 所以它就是**出来以后**还会收敛多久，而不是被禁闭时长吃掉。
# 这就是 caption 里「近期被抓」那一轴的来源：刚吃过处罚的人该躲着违禁行为。
BUSTED_RECENT_TICKS = 45
FIGHT_COOLDOWN = 10          # 散架后多久不会被再拖进新的架
FIGHT_COOLDOWN_LOSER = 30    # 输家更久：得给他去医务室的时间
SOLITARY_POS = layout.SOLITARY_XZ   # 禁闭室位置：跟着布局走，不再手抄一份
# 押送：被抓的人由最近的狱警/狱犬押去禁闭室。给一个拍数上限当兜底——万一路上出了岔子
# （押送者被藏进未解锁分区、或人被卡住），也不会永远押不到、把人锁在"既不自由也没关进去"。
# 走到门口就提前结束，所以正常情况下这个数用不满。
# 放得宽：正常情况该由"走到了"来结束押送，而不是由这个预算截断——被预算截断就会
# 在最后几米上瞬移一下，正好是这次要修掉的毛病。真卡住则由停滞检测提前收尾。
ESCORT_TICKS = 90
ESCORT_ARRIVE_R = 1.6        # 离禁闭室这么近就算押到了
ESCORT_GRAB_R = 1.8          # 押送者走到这么近才算"抓到手"，此前犯人站着等
ESCORT_SPEED_MUL = 0.8       # 押送时的步速：被扭着走，比自己赶路慢一点
ESCORT_STALL_MAX = 8         # 连续这么多拍没能靠近禁闭室 → 判定卡住，直接收监
SALUTE_SECONDS = 2.6         # 狱警/狱犬向典狱长行礼的时长（秒）
SALUTE_R = 6.0               # 典狱长走进这个半径 → 触发敬礼
SALUTE_COOLDOWN = 14.0       # 同一人再次敬礼的间隔，避免贴着典狱长走时一路刷礼
# 干这些活算"表现好"
PAROLE_WORK_GOALS = {"Labor": PAROLE_GAIN_WORK, "KitchenDuty": PAROLE_GAIN_WORK,
                     "Clean": PAROLE_GAIN_WORK, "Laundry": PAROLE_GAIN_WORK,
                     "CurryFavor": PAROLE_GAIN_CURRY}

# ---------------------------------------------------------------------------
# 宏观动机（drives.py）的运行时参数
#   - 签名动作低频出现（冷却），避免"每拍复读同一个专属动作"的不自然。
#   - 动机进度是性格的量化弧线：干推进它的事就涨，翻车就掉。
#   - 狱警手里有真权力：抓到揣着越狱工具/挖过洞的，加刑 + 加派更重的苦役，
#     快假释的直接把假释搞黄 → 破罐破摔（drives.DESPERATE），行为整体转向。
# ---------------------------------------------------------------------------
SIG_COOLDOWN = 45           # 签名动作做完后的冷却拍数（低频=不复读）
DRIVE_MILESTONE = 80        # 动机进度到这算"眼看就要成"，进 motive 措辞
GUARD_BUST_TICKS = 5        # 被狱警连续盯这么多拍且身上有越狱痕迹 → 当场搜出被抓
BUST_SENTENCE_ADD = 1.5     # 每次被抓加的刑期（年）
BUST_PROGRESS_DROP = 40     # 被抓时动机进度的跌幅
HARSH_LABOR = "看守所苦力"  # 被罚时加派的重苦役工种


@dataclass
class Patroller:
    """狱警/狱犬：沿 waypoints 巡逻的实体。kind ∈ guard/dog。"""

    id: str
    kind: str                       # guard / dog
    name: str
    waypoints: list[tuple[float, float]]
    speed: float
    x: float = 0.0
    z: float = 0.0
    wp: int = 0                     # 当前朝向的路点
    hx: float = 0.0                 # 朝向单位向量（判视锥用）
    hz: float = 1.0
    dwell: float = 0.0              # 到点后驻留倒计时（秒）：停下来扫视一会儿，不再一路直线窜
    escorting: str = ""             # 正在押送的 NPC id：非空则脱离巡逻线，改押人去禁闭室
    salute: float = 0.0             # 向典狱长敬礼的剩余秒数：期间站定行礼，不巡逻


@dataclass
class Facility:
    id: str
    name: str
    kind: str
    x: float
    z: float
    # 房间的局部尺寸与朝向。后端本身只用得到落点（NPC 走直线奔中心），
    # 这几项是随世界帧发给前端建房用的——布局只在 layout.py 存一份，
    # 前端不再自己抄一套尺寸。w 沿前墙方向，d 是进深，face 是门朝向。
    w: float = 0.0
    d: float = 0.0
    face: str = "N"
    door: float = 0.0


@dataclass
class NPC:
    id: str
    name: str
    personality: str
    role: str
    x: float = 0.0
    z: float = 0.0
    speed: float = 3.5
    needs: NeedState = field(default_factory=NeedState)
    backer_present: bool = False
    enemy_present: bool = False
    parole_near: bool = False
    has_labor: bool = False
    goal: str | None = None
    target: Facility | None = None
    # 专属牢房编号（layout.CELLS 的 slot）。休息/睡觉去自己那间，不再五个人挤一间。
    cell: int = -1
    # 绕墙路径缓存：后端也走 A*（见 nav.py），不再朝目标走直线穿墙。
    # path 是剩余路点，path_key 是它对应的终点——终点变了才重算。
    path: list[tuple[float, float]] = field(default_factory=list)
    path_key: tuple[float, float] = (1e9, 1e9)
    last_context: str = ""
    last_ctx: ContextState | None = None
    last_scored: list[ScoredCandidate] = field(default_factory=list)
    # M3 拥挤自组织
    crowd_pref: str = "neutral"     # like / avoid / neutral
    crowd_tol: int = 3              # 个人容忍人数
    predictor: str = "avg2"        # 拥挤预测策略
    expected_crowd: int = 0        # 本拍对放风人数的预期
    # M4 玩家指令
    commanded_goal: str | None = None   # 玩家当前下达的指令目标
    command_ticks: int = 0              # 指令剩余有效拍数
    llm_say: str | None = None          # LLM 大脑即兴生成的独白（value 模式为 None）
    llm_say_goal: str | None = None     # 那句独白是冲着哪个目标说的（见 _say）
    # LLM 独白池（Opus 一次产 8 句心里话，缓存起来每句显示几拍再换下一句）：
    # 一次调用够一个 NPC 说 ~30s，把 API 量压到很低，同时每个气泡都是活的。见 main._run_mono。
    llm_say_pool: list[str] = field(default_factory=list)
    llm_say_next_tick: int = 0          # 到这一拍才换下一句（每句多停几拍，别 0.5s 就跳）
    # 异步分缓存（清单 §2.4）：goal -> (v, m_value, 写入时的 tick)。
    # 决策只读缓存立即出结果，模型请求在后台跑，模拟不再被推理延迟拖住。
    score_cache: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    # 打分公式的运行时记账（清单 §2.2 四项惩罚）
    interrupt_bonus: dict[str, float] = field(default_factory=dict)
    failure_pen: dict[str, float] = field(default_factory=dict)
    timeout_pen: dict[str, float] = field(default_factory=dict)
    repeat_pen: dict[str, float] = field(default_factory=dict)
    goal_ticks: int = 0                 # 当前目标已持续拍数（判超时）
    goal_reached: bool = False          # 当前目标是否到达过（判打断）
    # 行动执行分阶段（world_objects）：以前"到了就算做完、且每帧结算一次"，
    # 现在生存类行动必须走 前往→取用→执行N帧→结算，取不到物资就是真的做不成。
    act_stage: str = "goto"             # goto / perform / done / failed
    act_ticks: int = 0                  # perform 已进行帧数
    act_obj: str = ""                   # 占用/取用的 WorldObject id（占用式要释放）
    act_fail: str = ""                  # 最近一次失败原因（播报/前端提示）
    carrying: str = ""                  # 手里拿着的物资 kind（"food"/"medkit"）
    # 打架（combat.py）：以前 enemy_present 是静态布尔、Fight 是单人动作。
    # 现在仇怨是**对具体人**的，且必须那个人在场才打得起来。
    grudge: dict[str, float] = field(default_factory=dict)  # 对方 npc id -> 仇怨 0~100
    # P1 关系图：对方 id -> {trust/fear/debt/ally/respect}（grudge=敌意 仍单列在上）。
    # 由事件（打架/抢物资）与对白（威胁/交易/结盟/讲和）共同增减——是"对话能改世界"的落点。
    rel: dict[str, dict[str, float]] = field(default_factory=dict)
    # P1 情景记忆：按显著度检索，喂进对话/决策 prompt，让一场接一场地累积成剧情。
    # 每条 {tick, kind, other, text, w(显著度)}。比 recent_events 更长、可检索、带来源。
    memory: deque[dict] = field(default_factory=lambda: deque(maxlen=MEMORY_MAX))
    fight_target: str = ""              # 本拍锁定的动手对象（视线内最恨的人）
    in_bout: str = ""                   # 正在参与的 bout id
    solitary_ticks: int = 0             # 关禁闭剩余拍数（打架被抓/越狱失败）
    recall_ticks: int = 0               # 典狱长"点名回牢房"剩余拍数：这段时间只回本人牢房待着
    recall_stuck: int = 0               # 回牢房走不通的拍数：到点就硬传回格子，不让寻路卡死指令
    lock_aware: str = ""                # 牢门感知："" | "in"(锁在里面) | "out"(锁在外面进不去)
    # 押送段：被抓后不再瞬移进禁闭室，而是由狱警/狱犬押着走过去，到门口才开始计禁闭。
    escort_ticks: int = 0               # 押送剩余拍数（走到即提前结束，纯兜底）
    escort_by: str = ""                 # 押送他的巡逻者 id（前端据此把两人连起来看）
    escort_grabbed: bool = False        # 押送者是否已走到跟前：False 时人站着等，不自己往禁闭室走
    escort_stall: int = 0               # 连续没能靠近禁闭室的拍数（卡住的判据）
    self_march: bool = False            # 没狱警可派时：自己走去禁闭室（不瞬移），走到门口才落锁
    pending_solitary: int = 0           # 押到后要关的拍数（押送途中先寄存在这儿）
    busted_cd: int = 0                  # 出禁闭后的"老实期"剩余拍数（喂 caption 的「近期被抓」轴）
    fight_cd: int = 0                   # 散架后的冷却：这段时间不会被再拖进新的架
    recent_done: deque[str] = field(default_factory=lambda: deque(maxlen=REPEAT_WINDOW))
    kept_by: str = ""                   # 本拍为什么没换目标（迟滞/无分/…），给前端解释用
    # 狱警/狱犬视线
    watched_by: str = ""                # "" / "狱警" / "狱犬"：本拍谁盯着它（进威胁字段）
    guard_watch_ticks: int = 0          # 被狱警连续盯的拍数（判"搜出违禁被抓"）
    # 越狱链
    escape_prep: dict[str, int] = field(default_factory=dict)  # "tool":1 / "breach":累计进度
    escaped: bool = False               # 已越狱出逃（退出场上活动）
    # 宏观动机（drives.py）
    drive: str = "safety"               # 长期心思的 key（唯一真相源）
    drive_progress: float = 0.0         # 动机推进度 0~100（性格的量化弧线）
    # 终极目标·任务链（quests.py）：quest_step=正在啃的第几个分目标，step_progress=本步进度点
    quest_step: int = 0
    step_progress: float = 0.0
    # 主线意图·承诺压力：安全时主线停滞就攒、推进就清零；驱动"念叨→必然起身去办"
    quest_pressure: float = 0.0
    pressure_telegraphed: int = -1      # 已为哪一步预告过"再也按捺不住"（避免重复刷字幕）
    # 生死网（无玩家版剧情）：weapon=持有的凶器（poison/knife/taser），kill_target=仇杀对象 id，
    # dead=已被杀身亡（退出场上活动，同 escaped 一样不再决策/移动）。
    weapon: str = ""
    kill_target: str = ""
    kill_cd: int = 0                    # 行凶未遂后的冷却：期间不再贴上去下手（给目标逃/求援窗口）
    dead: bool = False
    killed_by: str = ""                 # 真凶 id（既遂命案）；混战殃及记 "__brawl__"（无单独凶手）
    parole_lost: bool = False           # 假释被狱警搞黄 → 破罐破摔态
    sig_cooldown: int = 0               # 签名动作冷却（低频出现）
    # 判决 / 苦役（狱警行权的对象）
    crime: str = ""                     # 案由
    sentence_years: float = 5.0         # 判几年（被抓会加刑）
    parole_score: float = 45.0          # 假释表现分 0~100（干活涨、犯事掉）
    extra_days: int = 0                 # 因违纪累计加的刑期（天）
    parole_log: list[str] = field(default_factory=list)   # 假释变动流水（给前端解释）
    # 身上刚发生过的事（打架/受伤/禁闭/加刑/没饭吃/结仇）。
    # 存在的理由：对话以前只看得到 caption 的当前数值，看不到"经历"，
    # 于是你问疤脸"你脸怎么了"，他并不知道自己十分钟前跟瘸子打过一架。
    # 这条流水会注入角色扮演 prompt，让台词和世界里真实发生过的事对得上。
    recent_events: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    labor_job: str = ""                 # 苦役工种（被罚会加派更重的）
    busted: int = 0                     # 累计被抓次数（遥测/播报）
    # 出生分区（固定不变）。前端据此决定"这人何时登场"：所在分区解锁前一律藏起来，
    # 免得 NPC 顺着长期心思溜达到未解锁区、live 分区一翻就忽隐忽现。
    home_zone: str = "public"
    # —— 共享状态"脊梁"（观察者模式）：决策的结构化归因 + 重大决策流水 ——
    # 价值模型只定 goal；decision_drivers 把"为什么定它"按护栏优先级结构化出来。
    # 它是"对话是行为的配音"的唯一事实源（据它发声，绝不与行为矛盾），也供决策卡展开、
    # 暂停面板查看、逐 NPC 结局收场。三方读同一份，一致性天然成立、不靠事后对齐。
    decision_drivers: dict = field(default_factory=dict)
    # 重大决策流水（只记有张力/触及优先级阶梯的转折，不记换个消遣这种琐事）
    decision_log: deque[dict] = field(default_factory=lambda: deque(maxlen=12))
    laylow_log_tick: int = -999         # 上次把"躲狱警"记进流水的拍——用于限流，别刷屏


@dataclass
class PlayerState:
    """玩家也有身体。

    改这版之前玩家是纯旁观者：能把桌上最后一个窝头拿走让 NPC 挨饿，自己却不会饿、
    不会伤、也不会被打。物理层只对 NPC 生效，等于玩家在看别人受规则约束。

    现在玩家同样受约束：会饿会渴、靠吃拿到的东西缓解、站在群殴里会被误伤、
    抢了别人最后一口饭会被记恨、被记恨的人会真的动手打你。
    """

    x: float = 0.0
    z: float = 0.0
    hunger: float = 20.0
    thirst: float = 20.0
    injury: float = 0.0
    bag: dict[str, int] = field(default_factory=dict)   # kind -> 数量（food/water/medkit）
    hits: list[str] = field(default_factory=list)       # 最近挨打的播报
    drift_tick: int = 0

    def to_message(self) -> dict:
        return {
            "pos": [round(self.x, 2), round(self.z, 2)],
            "hunger": round(self.hunger), "thirst": round(self.thirst),
            "injury": round(self.injury),
            "bag": dict(self.bag), "hits": self.hits[-3:],
        }


# 玩家物理层参数
PLAYER_ID = "player"
PLAYER_DRIFT_EVERY = 3       # 每 N 拍涨 1 点需求（比 NPC 慢，免得变成吃饭模拟器）
PLAYER_SPLASH_RANGE = 2.6    # 站这么近群殴会被误伤
WARDEN_SIGHT = 4.6           # 典狱长在这么近的全向范围内 → NPC 视同"被狱警盯住"（僵住/收敛）
PLAYER_SPLASH_DMG = 2.4
PLAYER_BEAT_RANGE = 2.2      # 记恨你的人贴这么近就动手
PLAYER_BEAT_DMG = 3.6
PLAYER_USE = {               # 用掉一件东西的效果
    "food": {"hunger": -60},
    "water": {"thirst": -60},
    "medkit": {"injury": -60},
}


class World:
    def __init__(self, model: ModelClient) -> None:
        self.model = model
        self.tick_count = 0
        # 落点、尺寸、朝向全部来自 layout.py（贴墙连排布局的唯一权威）。
        # 以前这里手写一份坐标、前端手写一份尺寸，两边各自演进就穿模——禁闭室穿图书室、
        # 放风场铁栅栏穿黑市都是那么来的。改布局只改 layout.py，
        # 然后跑 `python backend/_overlap_probe.py` 体检。
        # 禁闭室依旧没有任何行动的 kind 指向它，NPC 永远不会主动"去禁闭"，
        # 只会被押进来（见 SOLITARY_POS / _escort_to_solitary）。
        self.facilities: list[Facility] = [
            Facility(f["id"], f["name"], f["kind"], f["x"], f["z"],
                     f["w"], f["d"], f["face"], f["door"])
            for f in layout.FACILITIES
        ]
        # 物资实体（食物份数/水龙头/铺位/医疗包）。玩家和 NPC 抢同一批，故后端权威。
        self.objects: list[WorldObject] = spawn_objects(self.facilities, layout.CELLS)
        self._last_phase: str = ""     # 用来检测时段切换 → 补货
        self.supply_log: list[str] = []   # 最近的物资事件（补货/告罄），给前端播报
        self.bouts: list[FightBout] = []  # 正在进行的打架
        self.fight_log: list[str] = []    # 打架/禁闭播报
        self.player = PlayerState()       # 玩家的身体（同样受物理层约束）
        self.warden_active = False        # 前端上报过玩家(典狱长)位置 → 典狱长在场，其站位起狱警监视之效
        self._salute_cd: dict[str, float] = {}   # 巡逻者 id → 距下次可向典狱长行礼的秒数
        self.npcs: list[NPC] = []
        for s in ROSTER:
            pref, tol = PERSONALITY_CROWD.get(s.personality, ("neutral", YARD_COMFORT))
            self.npcs.append(NPC(
                id=s.id, name=s.name, personality=s.personality, role=s.role,
                # needs 必须复制一份：直接传 s.needs 会让 NPC 和花名册**共用同一个对象**，
                # 于是每拍的需求漂移都在改花名册本身。后果是 reset() 从花名册取回来的
                # 「初始需求」早就被这一局污染了——重置永远还原不到真正的开局值。
                x=s.x, z=s.z, needs=replace(s.needs),
                backer_present=s.backer_present, enemy_present=s.enemy_present,
                parole_near=s.parole_near, has_labor=s.has_labor,
                crowd_pref=pref, crowd_tol=tol,
                predictor=PREDICTOR_BY_ID.get(s.id, "avg2"),
                expected_crowd=YARD_COMFORT,
                drive=s.drive, crime=s.crime,
                sentence_years=s.sentence_years, labor_job=s.labor_job,
                home_zone=self._zone_of_pos(s.x, s.z),
            ))
        for npc in self.npcs:      # 生死网：按任务链发凶器 + 定仇杀对象
            self._seed_kill_state(npc)
        self.n_agents = len(self.npcs)
        # 一人一间牢房。花名册里那几个出生坐标是照老场地（±30）挑的"某设施旁边"，
        # 场地缩小、房间重排之后它们要么落进墙里、要么指向别的房间，所以一律改成
        # 各自那间牢房——监狱本来就该这么开场，顺带保证初始位置一定在可走格上。
        self.solitary_locked: bool = False   # 禁闭室的门是否被典狱长锁上（同牢门一套逻辑）
        self.cell_state: list[dict] = []
        for i, c in enumerate(layout.CELLS):
            npc = self.npcs[i] if i < len(self.npcs) else None
            if npc is not None:
                npc.cell = c["slot"]
                npc.x, npc.z = c["x"], c["z"]
            self.cell_state.append({
                "slot": c["slot"], "npc": npc.id if npc else "",
                "name": npc.name if npc else "空房",
                "x": c["x"], "z": c["z"],
                "door_x": c["door_x"], "door_z": c["door_z"],
                "face": c["face"], "w": c["w"], "d": c["d"],
                # 常闭：牢门平时是合上的，有人靠近才推开（前端 updateCellDoors）。
                # locked 才是真封锁——只有典狱长能扭，锁上后 nav 网格里那道门变实墙。
                "closed": True, "locked": False,
            })
        # 铺初始仇怨（剧情对齐，见 combat.SEED_GRUDGE）
        ids = {n.id for n in self.npcs}
        for (a, b), v in SEED_GRUDGE.items():
            if a in ids and b in ids:
                self._npc(a).grudge[b] = v      # type: ignore[union-attr]
        # 狱警/狱犬巡逻队。路线由 layout.patrol_ring() 派生——原来是照 ±30 老场地
        # 手写的六个坐标，场地缩小后那些点落进了房子里，狱警会一头顶着墙走。
        # 《囚笼悖论》：只留一名巡逻狱警『铁面』(可对话·可被抚恤金名单收买)——另一名巡警的活
        # 由玩家扮演的典狱长亲自来干（见 warden_active / _watcher_of / warden_solitary）；
        # 狱犬『哮天』(嗅觉巡逻·可被镇静肉丢翻)。id 用 plot.GUARD_ID/DOG_ID 保持一致。
        ring = layout.patrol_ring()
        self.patrollers: list[Patroller] = [
            # 狱警走整圈；狱犬走内圈的一条对角捷径，两条线错开才有"被绕过去"的空档
            Patroller(GUARD_ID, "guard", "狱警·铁面", ring + [ring[0]], 3.4),
            Patroller(DOG_ID, "dog", "狱犬·哮天",
                      [ring[0], ring[2], ring[1], ring[3], ring[0]], 4.2),
        ]
        for p in self.patrollers:
            p.x, p.z = p.waypoints[0]
        self.escaped_names: list[str] = []   # 已越狱名单（遥测/播报）
        self.busts: list[dict] = []          # 最近被狱警抓现行的记录（前端 toast）
        # P1 社交写回：全局义务账本（谁欠谁一把刀/一条命/一次护）+ 最近社交事件（遥测/上帝视角）
        self.obligations: list[dict] = []    # {debtor, creditor, kind, text, tick, settled}
        self.social_events: deque[dict] = deque(maxlen=30)
        self._seed_social()                  # 铺初始关系图 + 义务账本（社交动作的开局燃料）
        # 侦探案件进展：前端找到几条线索就同步过来，越高知情 NPC 越松口、越敢帮着越狱
        self.case_progress = 0
        # 全局黑板：一局游戏的唯一真相源（分区解锁/道具/好感/证据）。阶段一只用到
        # zones_unlocked + inventory：跟老牙要到牙刷 → 开牢门 → 解锁公共区。
        self.bb = Blackboard()
        # 遥测：放风人数历史 + 各性格累计选择分布（喂给前端 El Farol 曲线/分布图）
        self.attendance_history: deque[int] = deque(maxlen=600)
        self.choice_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # M4 玩家挑事：全局注入事件
        self.event_kind: str | None = None
        self.event_ticks: int = 0
        # M5 招牌片段：时段覆盖 + 花名册快照（供 reset）
        self.phase_override: str | None = None
        self._seed_by_id = {s.id: s for s in ROSTER}
        # 异步打分：在途批次 + 遥测（帧率是否被推理拖住，一看便知）
        self._score_inflight = False
        self._score_task: asyncio.Task | None = None
        self.score_ms = 0.0        # 最近一批打分耗时
        self.score_lag = 0         # 缓存落后当前拍数
        self.score_rounds = 0      # 完成的批次数
        self.score_skipped = 0     # 因在途而跳过的拍数
        self.score_fail = 0        # 失败批次数
        # 对话优先：LLM 大脑下，对话和逐拍打分抢同一个模型。对话进行时先让路，
        # 世界照样靠分缓存跑（可能滞后几拍），但玩家的对话会明显更快回来。
        self.chat_active = 0
        # 观察者模式·上帝暂停：玩家可随时定格世界，逐个查看 NPC 的处境/归因/决策流水。
        # 暂停时 tick 直接空转（不漂移需求、不动位、不决策），但 _sim_loop 照常广播这帧
        # 冻结状态，所以画面不黑、对话/对比仍可发起。
        self.paused = False
        # 观察者模式·NPC↔NPC 对话（事件触发·懒生成）：靠得近 + 有关系钩子(仇怨/同伙)时
        # 攒一场 3~4 句的短对话。LLM 只在真要开口时被调一次（生成整段脚本），前端逐字放，
        # 不逐字回后端，省得把本地 7B 打爆。全局/同对 都有冷却，保证偶发、不刷屏。
        self.convo_pending: dict | None = None    # 待生成：{"a","b","topic","kind"}
        self.convo_cd = 24                        # 全局冷却剩余拍（开局先按住一会儿）
        self._convo_pair_cd: dict[frozenset, int] = {}
        self._convo_seq = 0
        # 典狱长夜班：时钟 / 权力点 / 风险事件。reset() 时一并清掉。
        self.shift = ShiftState()

    def unlock_all(self) -> None:
        """沙盒/演示/探针：一次性解锁全部分区，关掉阶段门禁，让全监狱都活起来。
        （阶段门禁只服务于第一人称剧情流；M 系列招牌片段/自检要的是满场活动。）"""
        self.bb.zones_unlocked = set(UNLOCK_ORDER)

    def interrupt_scoring(self) -> None:
        """立刻打断在途的逐拍打分批次，把模型让给对话（分缓存兜底，世界不冻结）。
        7B 上一整批打分可能要好几秒，对话若排在它后面首字就得干等这么久。"""
        task = self._score_task
        if task is not None and not task.done():
            task.cancel()
        self._score_inflight = False

    # -- M4 玩家输入 --------------------------------------------------------
    def set_command(self, npc_id: str, goal: str | None) -> None:
        """玩家给某个 NPC 下指令（goal=None 表示取消）。"""
        npc = next((n for n in self.npcs if n.id == npc_id), None)
        if npc is None:
            return
        # 以前要求「必须有对应设施」，结盟/下杀手/收敛这类 kind="" 的动作会被静默清掉。
        # 现在：只要是行动全集里的 goal 就能下令；取消指令仍走 goal=None。
        if goal and goal in GOAL_TO_KIND:
            npc.commanded_goal = goal
            npc.command_ticks = COMMAND_DURATION
        else:
            npc.commanded_goal = None
            npc.command_ticks = 0

    # 对话即指令：玩家在聊天里直接吩咐 → 关键词映射到 goal → 走正规指令通道
    CHAT_CMD_KEYWORDS = {
        "Eat": ["吃饭", "吃点", "去吃", "食堂", "吃东西"],
        "Drink": ["喝水", "喝点", "口渴", "水房", "去喝"],
        "Sleep": ["睡觉", "去睡", "睡会", "回去睡", "睡一觉"],
        "Rest": ["歇着", "回牢房", "歇会", "歇一下", "回房", "别乱跑"],
        "Cure": ["治伤", "看病", "医务室", "上药", "包扎"],
        "Socialize": ["社交", "唠嗑", "聊聊", "找人说话", "去交际"],
        "Workout": ["健身", "练两组", "举铁", "锻炼", "健身房", "去练"],
        "Shower": ["洗澡", "澡堂", "冲个澡"],
        "Study": ["学习", "上课", "去学", "念书"],
        "Read": ["看书", "读书", "图书室"],
        "Phone": ["打电话", "电话亭", "给家里打"],
        "Visit": ["探视", "会见", "见家属", "探视室"],
        "Labor": ["苦役", "干活", "干苦力"],
        "KitchenDuty": ["帮厨", "后厨", "打下手"],
        "Clean": ["打扫", "扫地", "搞卫生"],
        "Laundry": ["洗衣", "洗衣服", "洗衣房"],
        "Pray": ["祷告", "祈祷"],
        "GetWeapon": ["弄凶器", "弄把刀", "去弄家伙", "搞把刀", "弄刀"],
        "Barter": ["换货", "黑市", "换点东西"],
        "Chess": ["下棋", "来一盘", "象棋", "对弈"],
        "Music": ["弹琴", "唱歌", "来点音乐"],
        "Patrol": ["放风", "溜达", "去院子", "出去转", "放风场"],
        "Assassinate": ["下杀手", "去动手", "做掉", "干掉他"],
        "LayLow": ["收敛", "老实待着", "别出格", "站好", "别乱动"],
        "Ally": ["结盟", "拜把子", "结成同盟"],
        "Propose": ["去提议", "谈一笔", "提个条件"],
        "Threaten": ["放狠话", "威胁他", "拿捏"],
        "Protect": ["护着", "罩着", "护住"],
        "Hide": ["藏东西", "藏起来", "藏违禁"],
        "Fight": ["打架", "教训他", "揍一顿"],
    }

    def try_command_from_text(self, npc_id: str, text: str) -> str | None:
        """从玩家对话里解析"去做某事"的吩咐。命中就下正规指令，返回中文标签供回话确认。

        只在像"吩咐"时才触发：带问号/『吗』的当成提问，不当命令，免得"你吃饭了吗"被误判。
        """
        if not text or any(q in text for q in "?？吗"):
            return None
        for goal, kws in self.CHAT_CMD_KEYWORDS.items():
            if any(k in text for k in kws) and goal in GOAL_TO_KIND:
                self.set_command(npc_id, goal)
                return GOAL_LABEL.get(goal, goal)
        return None

    # 典狱长的调度指令（回牢房/关禁闭）必须硬编码兑现：直接改状态、锁死决策脑，
    # 绝不走 LLM 打分那套软指令（那套需求高时能抗命、还得等模型），点了就一定执行、全员必听。
    _WARDEN_ALL_KWS = ("全体", "所有人", "大家", "都给我", "统统", "全部",
                       "一个不留", "都回", "全都", "谁都")
    _WARDEN_SOLITARY_KWS = ("关禁闭", "进禁闭", "禁闭室", "小黑屋", "收监",
                            "关起来", "单独关", "关小黑")
    _WARDEN_CELL_KWS = ("回牢房", "回房", "回笼", "回监区", "回自己", "关回",
                        "赶回", "回去待", "回去睡", "都回去", "滚回去", "回去", "回窝")

    def warden_command_from_text(self, npc_id: str, text: str) -> dict | None:
        """典狱长聊天里的硬指令：命中『回牢房/关禁闭』就直接 warden_recall，不经模型。
        返回 {label, dest, **recall_result}；没命中返回 None。"""
        if not text or any(q in text for q in "?？吗"):
            return None
        to_all = any(k in text for k in self._WARDEN_ALL_KWS)
        target = "all" if to_all else npc_id
        if any(k in text for k in self._WARDEN_SOLITARY_KWS):
            res = self.warden_recall(target, "solitary")
            return {"label": "进禁闭室", "dest": "solitary", **res}
        if any(k in text for k in self._WARDEN_CELL_KWS):
            res = self.warden_recall(target, "cell")
            return {"label": "回牢房", "dest": "cell", **res}
        return None

    def inject_event(self, kind: str | None) -> None:
        """玩家挑事：注入/解除一个全局事件（狱犬/警报/搜牢/斗殴）。"""
        if kind in EVENT_INJECT:
            self.event_kind = kind
            self.event_ticks = EVENT_DURATION
        else:
            self.event_kind = None
            self.event_ticks = 0

    def set_case_progress(self, found: int) -> None:
        """前端把已找到的线索数同步过来：查得越深，知情 NPC 越松口、越敢帮忙。"""
        self.case_progress = max(0, int(found))

    # -- M5 招牌片段 --------------------------------------------------------
    def _npc(self, npc_id: str) -> NPC | None:
        return next((n for n in self.npcs if n.id == npc_id), None)

    def reset(self) -> None:
        """恢复到花名册初始状态（清空指令/事件/时段覆盖）。"""
        for npc in self.npcs:
            s = self._seed_by_id.get(npc.id)
            if s is None:
                continue
            npc.needs = replace(s.needs)
            # 位置归位。漏掉这条的话，重置后所有人还站在上一局结束时的地方
            # （有人在禁闭室、有人在大门口），录像开头就不成立。
            npc.x, npc.z = s.x, s.z
            # 物理层状态：这些是"打架/禁闭/假释"落地后新增的，原来的 reset 不认识它们，
            # 于是重置后上一局的伤、仇、刑期、禁闭剩余拍数会全部带进新一局。
            npc.act_stage, npc.act_ticks, npc.act_obj, npc.act_fail = "goto", 0, "", ""
            npc.carrying = ""
            npc.grudge.clear()
            npc.rel.clear()          # P1 关系图归零（重铺初始仇怨在下方，rel 一律新开局）
            npc.memory.clear()       # P1 情景记忆清空
            npc.fight_target = ""
            npc.in_bout = ""
            npc.solitary_ticks = 0
            npc.recall_ticks = 0
            npc.recall_stuck = 0
            npc.escort_ticks = 0
            npc.escort_by = ""
            npc.escort_grabbed = False
            npc.escort_stall = 0
            npc.self_march = False
            npc.pending_solitary = 0
            npc.fight_cd = 0
            npc.busted_cd = 0
            npc.parole_score = 45.0
            npc.extra_days = 0
            npc.parole_log.clear()
            npc.recent_events.clear()
            npc.drive_progress = 0.0
            npc.backer_present = s.backer_present
            npc.enemy_present = s.enemy_present
            npc.parole_near = s.parole_near
            npc.has_labor = s.has_labor
            npc.commanded_goal = None
            npc.command_ticks = 0
            # 记账清账：不然上一个片段的超时/重复惩罚会跟着进下一个片段
            npc.interrupt_bonus.clear()
            npc.failure_pen.clear()
            npc.timeout_pen.clear()
            npc.repeat_pen.clear()
            npc.recent_done.clear()
            npc.goal_ticks = 0
            npc.goal_reached = False
            npc.watched_by = ""
            npc.guard_watch_ticks = 0
            npc.escaped = False
            npc.escape_prep.clear()
            # 宏观动机复位（drive_key 从花名册来，不变；进度/崩塌/刑期/苦役归位）
            npc.drive = s.drive
            npc.drive_progress = 0.0
            npc.quest_step = 0          # 终极目标任务链：回到第一个分目标
            npc.step_progress = 0.0
            npc.quest_pressure = 0.0    # 主线承诺压力归零
            npc.pressure_telegraphed = -1
            npc.llm_say = None          # LLM 独白池随局清空，别把上一局的心里话带过来
            npc.llm_say_goal = None
            npc.llm_say_pool = []
            npc.llm_say_next_tick = 0
            self._seed_kill_state(npc)  # 生死网：凶器/仇杀对象/生死状态复位
            npc.parole_lost = False
            npc.sig_cooldown = 0
            npc.crime = s.crime
            npc.sentence_years = s.sentence_years
            npc.labor_job = s.labor_job
            npc.busted = 0
            npc.decision_drivers = {}
            npc.decision_log.clear()
            npc.laylow_log_tick = -999
            # 分缓存保留：情境变了旧分只是偏，清掉反而让片段开头集体呆立
        self.escaped_names.clear()
        # 巡逻者归位：押送/敬礼是跨拍状态，不清掉的话新一局开局就有人押着空气走
        for p in self.patrollers:
            p.x, p.z = p.waypoints[0]
            p.wp = 0
            p.dwell = 0.0
            p.escorting = ""
            p.salute = 0.0
        self._salute_cd.clear()
        self.busts.clear()
        self.obligations.clear()     # P1 义务账本 + 社交事件流水随局清空
        self.social_events.clear()
        self.event_kind = None
        self.event_ticks = 0
        self.phase_override = None
        # 重新铺初始仇怨：上面把 grudge 清空了，剧情预设的恩怨（combat.SEED_GRUDGE）
        # 得跟着回来，否则重置后谁都不记恨谁，打架线索整条消失。
        ids = {n.id for n in self.npcs}
        for (a, b), v in SEED_GRUDGE.items():
            if a in ids and b in ids:
                self._npc(a).grudge[b] = v      # type: ignore[union-attr]
        self._seed_social()                     # 关系图/义务账本随局重铺（rel/obligations 上面已清空）
        # 世界物资重新铺满（上一局吃空的窝头/占着的铺位/用掉的医疗包）
        self.objects = spawn_objects(self.facilities)
        self.supply_log.clear()
        self.bouts.clear()
        self.fight_log.clear()
        self._last_phase = ""
        self.player = PlayerState()      # 玩家的身体也归零（饿/渴/伤/背包）
        self.convo_pending = None        # NPC↔NPC 对话：待生成/冷却全部归零
        self.convo_cd = 24
        self._convo_pair_cd.clear()
        # 上一班 shift.finish() 把世界定格了（world.paused=True，给结算/断案画面留一帧冻结）。
        # 重开一局必须解冻，否则新一班 tick() 一进门就 `if self.paused: return`，整座监狱僵死——
        # NPC 不动、不决策、不出事件，玩家会看到"打完一晚再开就是空场"。
        self.paused = False
        # 异步打分状态随局清零：上一班收尾时往往还压着一个在途批次（_score_inflight=True）。
        # 不清掉的话新一班要么被这个 stale 批次卡着不再派新活，要么它落地后把旧分写乱缓存。
        t = self._score_task
        if t is not None and not t.done():
            t.cancel()
        self._score_task = None
        self._score_inflight = False
        self.chat_active = 0
        self.bb.reset()
        self.shift.reset()

    def shift_act(self, action: str, msg: dict) -> dict:
        """公开夜班的典狱长行动（扣权力点、改评分、必要时走现有 warden_*）。"""
        return self.shift.act(self, action, msg)

    def run_scenario(self, name: str) -> str:
        """一键触发招牌片段。返回一句给前端展示的说明。"""
        if name == "new_game":
            # 观察者（上帝视角）重开：整座监狱一开局就全部活着——不再逐区解锁。
            # reset() 现在默认就把 zones_unlocked 铺满（见 Blackboard.reset），
            # 这里再兜底 unlock_all 一次，保证录像/演示一睁眼全狱皆活、众生各行其是。
            self.reset()
            self.unlock_all()
            self.tick_count = 0
            return "已重新开局：全狱皆活（上帝观察模式）"
        if name == "reset":
            self.reset()
            self.unlock_all()   # 招牌片段/沙盒：全场解锁，保证 M 系列演示满场活动
            return "已重置到正常作息"

        self.reset()  # 每个片段都从干净基线开始
        self.unlock_all()

        if name == "cmd_vs_need":
            laok, yanjing = self._npc("laok"), self._npc("yanjing")
            if laok:
                laok.needs = NeedState(hunger=95)   # 快饿死（生存紧急），其余清零
                self.set_command("laok", "Sleep")
            if yanjing:
                yanjing.needs = NeedState(hunger=8)  # 不饿
                self.set_command("yanjing", "Sleep")
            return "指令 vs 需求：都命令回牢房睡觉——饿疯的老K抗命先吃，没需求的眼镜乖乖去睡"

        if name == "alarm_over_sleep":
            self.phase_override = PHASE_SLEEP
            for npc in self.npcs:
                npc.needs.fatigue = 78           # 都很困，本该睡觉
            self.inject_event("警报")            # 但警报盖过疲劳
            return "情境盖过需求：深夜都困得要死，突然拉响越狱警报，看谁还睡得着"

        if name == "personality":
            self.phase_override = PHASE_YARD
            for npc in self.npcs:
                npc.needs = NeedState(hunger=15, thirst=15, fatigue=15,
                                      boredom=45, social=45, stress=15, injury=0)
                npc.enemy_present = False
                npc.backer_present = False
                npc.has_labor = False
            return "同境不同性：所有人需求完全一样、只有性格不同，看放风时各选各的"

        if name == "yard_riot":
            self.phase_override = PHASE_YARD
            for npc in self.npcs:
                npc.needs.boredom = 55
            for nid in ("laok", "shitou", "yanjing", "laochen"):
                self.set_command(nid, "Patrol")   # 硬凑一堆人去放风
            return "扎堆诱发斗殴：命令四个人一起去放风，放风场瞬间爆满，怕挤的新人躲开、火药桶开打"

        if name == "multi_need":
            laok = self._npc("laok")
            if laok:
                laok.needs.hunger = 72
                laok.needs.thirst = 68
                laok.needs.fatigue = 62
                laok.needs.boredom = 58          # 又饿又渴又困又闷
            return "多需求缠身：老K饥渴困闷全上头，看模型先解决哪个、再依次收拾"

        return f"未知片段：{name}"

    # -- 遥测/导出 ----------------------------------------------------------
    def metrics(self) -> dict:
        h = list(self.attendance_history)
        comfort = YARD_COMFORT
        mean = sum(h) / len(h) if h else 0.0
        within = sum(1 for v in h if abs(v - comfort) <= 1) / len(h) * 100 if h else 0.0
        reversals = sum(
            1 for i in range(2, len(h))
            if (h[i] - h[i - 1]) * (h[i - 1] - h[i - 2]) < 0
        )
        return {
            "tick": self.tick_count,
            "n_agents": self.n_agents,
            "yard_comfort": comfort,
            "attendance": h,
            "choice_dist": self._choice_dist(),
            "stats": {
                "mean": round(mean, 3),
                "min": min(h) if h else 0,
                "max": max(h) if h else 0,
                "within_band_pct": round(within, 1),
                "reversals": reversals,
                "samples": len(h),
            },
        }

    # -- 设施 ---------------------------------------------------------------
    def _facility_for_goal(self, goal: str) -> Facility | None:
        kind = GOAL_TO_KIND.get(goal)
        return next((f for f in self.facilities if f.kind == kind), None) if kind else None

    # -- 物资层：取用 / 释放 / 补货 ------------------------------------------
    def _target_for_goal(self, npc: NPC, goal: str) -> Facility | None:
        """这次行动实际要走到哪。

        手里已经有窝头/医疗包时，取用层仍认 carrying（到了直接用、不再占第二份），
        但落点必须还是食堂/医务室——否则人会在空地上坐下吃饭、就地包扎，看着像没去。
        """
        f = self._facility_for_goal(goal)
        # 睡觉/歇着/祷告/藏东西：回**自己**那间牢房，铺位落在后墙（前端 bunk 同坐标）。
        # id 仍沿用牢房设施的（"cell"），铺位物资挂在同一个设施 id 上才找得到。
        if f is not None and f.kind == "sleep" and npc.cell >= 0:
            c = layout.CELLS[npc.cell]
            fx, fz = layout.FACE_VEC.get(c["face"], (0.0, 1.0))
            # 门朝 face，床靠后墙：沿 -face 退 (进深/2 - 1.15)，与前端 bunk 对齐
            bx = c["x"] - fx * (c["d"] / 2 - 1.15)
            bz = c["z"] - fz * (c["d"] / 2 - 1.15)
            return Facility(f.id, f.name, f.kind, bx, bz)
        return f

    # -- 剧情染色：让"无所事事"的填充动作也指向各自的终极目标 -----------------
    @staticmethod
    def _plot_reachable(n: NPC | None) -> bool:
        """这人此刻能不能被凑上去（没死、没越狱成功、没关禁闭）。"""
        return bool(n and not n.dead and not n.escaped and n.solitary_ticks <= 0)

    def _near_facility(self, fid: str, name: str, npc: NPC,
                       other: NPC, dist: float) -> Facility:
        """在 npc→other 连线上、离 other 约 dist 处停下（盯梢/贴靠，不与对方重叠）。
        返回一个**新** Facility（绝不能改共享的 self.facilities，那会挪乱全局落点）。"""
        dx, dz = other.x - npc.x, other.z - npc.z
        d = math.hypot(dx, dz) or 1.0
        return Facility(fid, name, "yard", other.x - dx / d * dist,
                        other.z - dz / d * dist)

    def _killer_targeting(self, victim_id: str) -> NPC | None:
        """谁的终极目标是干掉这个人（生死网里的下手方）。"""
        for n in self.npcs:
            q = quest_of(n.id)
            if q and q.kind == "kill" and q.target == victim_id and not n.dead:
                return n
        return None

    def _best_ally_present(self, npc: NPC) -> NPC | None:
        """当前在场、关系最铁的盟友（供疯子往靠山身边凑）。"""
        best, bv = None, ALLY_NOFIGHT
        for oid, e in npc.rel.items():
            o = self._npc(oid)
            if not self._plot_reachable(o):
                continue
            if e.get("ally", 0.0) >= bv:
                best, bv = o, e["ally"]
        return best

    def _plot_idle_target(self, npc: NPC) -> Facility | None:
        """把闲逛/歇着的落点改到剧情上：返回新落点，或 None（沿用默认闲逛点）。

        - 杀手(疤脸/柳叶刀)：盯梢自己的猎物，停在几步远处——玩家一眼看出"他在缀着谁"。
        - 越狱者(老牙)：踩点大门，把闲晃变成"数狱警、量出路"。
        - 目击者(疯子)：黑医逼近就往人堆里躲，否则往最铁的靠山身边凑。
        """
        q = quest_of(npc.id)
        if q is None:
            return None
        if q.kind == "kill" and q.target:
            tgt = self._npc(q.target)
            if self._plot_reachable(tgt):
                return self._near_facility("stalk", f"盯梢{tgt.name}", npc, tgt, 3.5)  # type: ignore[arg-type]
            return None
        if q.kind == "escape":
            gate = self._facility_for_goal("Tunnel") or self._facility_for_goal("PrisonBreak")
            if gate is not None:
                return Facility("casegate", "踩点大门", gate.kind, gate.x, gate.z)
            return None
        if q.kind == "survive" and npc.id == "xiaodou":
            killer = self._killer_targeting(npc.id)
            if (self._plot_reachable(killer)
                    and math.hypot(killer.x - npc.x, killer.z - npc.z) < 14):  # type: ignore[union-attr]
                crowd = self._facility_for_goal("Patrol")
                if crowd is not None:
                    return Facility("flee", "躲开黑医", crowd.kind, crowd.x, crowd.z)
            prot = self._best_ally_present(npc)
            if prot is not None:
                return self._near_facility("cling", f"贴着{prot.name}", npc, prot, 2.5)
        return None

    def _objects_at(self, facility_id: str, kind: str) -> list[WorldObject]:
        return [o for o in self.objects
                if o.facility_id == facility_id and o.kind == kind]

    def _object(self, oid: str) -> WorldObject | None:
        return next((o for o in self.objects if o.id == oid), None)

    def _release_object(self, npc: NPC) -> None:
        """放开占用式物资（铺位/龙头）。换目标、被打断、行动结束都要调。"""
        if npc.act_obj:
            o = self._object(npc.act_obj)
            if o is not None:
                o.release(npc.id)
            npc.act_obj = ""

    def _restock_if_phase_changed(self, phase: str) -> None:
        """时段切换时给该补的物资补满。

        食堂只在点名/放风补 → 夜里饿了真的没饭吃，"错过饭点"才有分量。
        """
        if phase == self._last_phase:
            return
        self._last_phase = phase
        for o in self.objects:
            if phase in o.restock_phases and o.stock < o.cap:
                o.restock()
                self.supply_log.append(f"{phase}：{o.name}补货({o.cap})")
        del self.supply_log[:-8]

    def take_object(self, oid: str, taker: str = "player") -> dict:
        """玩家从世界里拿走一份物资（桌上的窝头等）。

        走的是和 NPC 完全同一批 object —— 玩家把窝头拿走，NPC 这顿就少一份。
        """
        o = self._object(oid)
        if o is None:
            return {"ok": False, "reason": "没有这件东西"}
        if not o.available():
            return {"ok": False, "reason": f"{o.name}已经没了"}
        o.take(taker)
        if taker == PLAYER_ID:
            self.player_take(o.kind)          # 进玩家背包，之后能真的吃/用
        if o.consumable and o.stock == 0:
            self.supply_log.append(f"{o.name}被取空了")
            # 你抢走了最后一份，旁边正饿着的人会记你的仇 —— 而且真的会来揍你
            self._blame_for_last_portion(
                taker, "你" if taker == PLAYER_ID else taker,
                self.player.x, self.player.z, o.kind)
        return {"ok": True, "kind": o.kind, "name": o.name, "stock": o.stock,
                "player": self.player.to_message()}

    def _yard(self) -> Facility:
        return next(f for f in self.facilities if f.kind == "yard")

    def _near_kind(self, npc: NPC, kind: str, dist: float = GATE_REACH_DIST) -> bool:
        f = next((f for f in self.facilities if f.kind == kind), None)
        return f is not None and math.hypot(f.x - npc.x, f.z - npc.z) <= dist

    @staticmethod
    def _fac_dict(fac) -> dict:
        if isinstance(fac, dict):
            return fac
        return {"x": fac.x, "z": fac.z, "w": fac.w, "d": fac.d, "face": fac.face}

    def _in_footprint(self, x: float, z: float, fac, inset: float = 0.45) -> bool:
        return layout.contains(self._fac_dict(fac), x, z, inset)

    def _cell_locked(self, npc: NPC) -> bool:
        if npc.cell < 0 or npc.cell >= len(self.cell_state):
            return False
        return bool(self.cell_state[npc.cell].get("locked"))

    def _in_own_cell_at(self, npc: NPC, x: float, z: float) -> bool:
        if npc.cell < 0 or npc.cell >= len(layout.CELLS):
            return False
        return layout.contains(layout.CELLS[npc.cell], x, z, 0.35)

    def _in_own_cell(self, npc: NPC) -> bool:
        return self._in_own_cell_at(npc, npc.x, npc.z)

    def _goal_blocked_by_cell(self, npc: NPC, goal: str) -> bool:
        """牢门锁着时：人在里面出不去、人在外面进不去。他自己知道。"""
        if not self._cell_locked(npc):
            return False
        inside = self._in_own_cell(npc)
        t = self._target_for_goal(npc, goal)
        if t is None:
            if goal in SOCIAL_ACTION_GOALS or goal in ("Fight", "Argue", "Assassinate"):
                return inside
            return False
        tgt_in = self._in_own_cell_at(npc, t.x, t.z)
        if inside and not tgt_in:
            return True
        if (not inside) and tgt_in:
            return True
        return False

    def _filter_cell_lock(self, npc: NPC, cands: list[Candidate]) -> list[Candidate]:
        if not self._cell_locked(npc):
            return cands
        keep: list[Candidate] = []
        for cand in cands:
            if cand.source == "command":
                keep.append(cand)
                continue
            if self._goal_blocked_by_cell(npc, cand.goal):
                continue
            keep.append(cand)
        if keep:
            return keep
        fallback = "Rest" if self._in_own_cell(npc) else "Patrol"
        return [Candidate(fallback, "daily")]

    def _notice_cell_lock(self, npc: NPC) -> None:
        """锁一拧上，人立刻知道进不进得去——不必先去撞门。"""
        if not self._cell_locked(npc):
            if npc.lock_aware:
                self._remember(npc, "牢门开了，进出自由。")
                npc.lock_aware = ""
            return
        tag = "in" if self._in_own_cell(npc) else "out"
        if npc.lock_aware == tag:
            return
        npc.lock_aware = tag
        if tag == "in":
            self._remember(npc, "牢门被锁死了，你出不去，外面也进不来。")
        else:
            self._remember(npc, "你的牢房上着锁，现在进不去。")

    def _place_name(self, npc: NPC) -> str:
        if self._in_own_cell(npc):
            return "自己的牢房"
        for f in self.facilities:
            if f.kind == "sleep" or f.w < 0.4:
                continue
            if self._in_footprint(npc.x, npc.z, f, 0.2):
                return f.name
        for c in layout.CELLS:
            if layout.contains(c, npc.x, npc.z, 0.2):
                who = self.cell_state[c["slot"]].get("name") or "空房"
                return f"{who}的牢房"
        if self._in_yard(npc):
            return "放风场"
        return "院子"

    def _surroundings_line(self, npc: NPC) -> str:
        bits: list[str] = []
        if self._cell_locked(npc):
            if self._in_own_cell(npc):
                bits.append("你的牢门上着锁，出不去")
            else:
                bits.append("你的牢房上着锁，你现在进不去")
        else:
            if self._in_own_cell(npc):
                bits.append("你在自己牢房里，门没锁，随时能出去")
            else:
                bits.append("你的牢房没锁，想回就能回")
        place = self._place_name(npc)
        # 在自己牢房时，第一句已经交代了"在自己牢房里"，别再"你人在自己的牢房"重复一遍。
        if place and not self._in_own_cell(npc):
            bits.append(f"你人在{place}")
        return "。".join(bits)

    @staticmethod
    def _dwell_ticks(goal: str) -> int:
        return PERFORM_TICKS.get(goal) or ROOM_DWELL.get(goal, 10)

    def _cellblock_ok(self, goal: str) -> bool:
        """该目标是否留在牢区（无落点的原地行为，或落点就在 cellblock）。老牙阶段一用。"""
        f = self._facility_for_goal(goal)
        return f is None or zone_of_kind(f.kind) == "cellblock"

    # -- 分区（黑板解锁用）--------------------------------------------------
    def _zone_of_pos(self, x: float, z: float) -> str:
        """某坐标落在哪个分区：取最近设施的分区（前端据此隐藏未解锁分区里的人/物）。"""
        nearest = min(self.facilities,
                      key=lambda f: (f.x - x) ** 2 + (f.z - z) ** 2,
                      default=None)
        return zone_of_kind(nearest.kind) if nearest else "public"

    # -- 阶段一：向导老牙把"磨尖牙刷"给玩家 → 开牢门 → 解锁下一分区 --------
    def anchor_give_tool(self) -> dict:
        """玩家在对话里跟老牙要到了出去的门路：发牙刷、开牢门、解锁公共区。
        返回给前端的效果说明（前端据此显形新分区、加道具、更新目标）。"""
        first = TOOTHBRUSH not in self.bb.inventory
        newly = None
        if first:
            self.bb.inventory.add(TOOTHBRUSH)
            self.bb.doors_open.add("cell")
            self.bb.flags["anchor_gave_tool"] = True
            self.bb.advance_stage(ANCHOR_ID, "tool")   # 老牙"借你的手办事"这一幕
            newly = self.bb.unlock_next_zone()   # cellblock 之后 → public
        return {
            "new": first,
            "granted": TOOTHBRUSH,
            "unlocked_zone": newly,
            "blackboard": self.bb.to_message(),
        }

    # -- 阶段二/三：对话改世界（两轨架构·第一轨结算到黑板）-----------------
    def resolve_interaction(self, npc_id: str, text: str) -> dict:
        """把玩家这句话解析成意图，据角色/证据/道具改黑板（好感/旗标/道具/证据），
        再检查是否触发逐级解锁。返回给前端的效果（toast/新道具/新分区/黑板快照）。

        这是"LLM 不可替代"的落点：同一句话，带没带证据、够不够诚意、跟谁说，
        导致的世界改变完全不同——而不是聊天归聊天、状态归状态两张皮。
        真凶揭穿、收买狱警、换磁卡镇静剂，都在这里被"锁进"确定性状态机（防剧情暴走）。
        """
        bb = self.bb
        pi = nlu.parse_intent(text)
        notes: list[str] = []
        gained: list[str] = []
        lost: list[str] = []
        toast: str | None = None

        # 瘸子：医用酒精 → 强效镇静剂 + 办公室磁卡（认货不认人）
        if npc_id == SMUGGLER_ID:
            wants_deal = (pi.intent in (nlu.GIVE, nlu.BRIBE, nlu.HELP)
                          or pi.item in (ALCOHOL, SEDATIVE, KEYCARD)
                          or any(k in text for k in ("货", "换", "东西", "镇静", "磁卡", "买")))
            if wants_deal:
                if bb.flag("smuggler_dealt"):
                    notes.append("already_dealt")
                elif bb.has_item(ALCOHOL):
                    bb.take_item(ALCOHOL)
                    bb.give_item(SEDATIVE)
                    bb.give_item(KEYCARD)
                    bb.set_flag("smuggler_dealt")
                    bb.advance_stage(SMUGGLER_ID, "dealt")
                    bb.bump_trust(npc_id, 25)
                    gained += [SEDATIVE, KEYCARD]
                    lost += [ALCOHOL]
                    notes.append("smuggler_deal")
                    toast = ("瘸子掂了掂那瓶医用酒精，眼睛一亮，塞给你一小瓶强效镇静剂和一张办公室磁卡："
                             "「拿好，别声张——狗和门，都靠它俩了。」")
                else:
                    notes.append("need_alcohol")

        # 黑医柳叶刀：被证据(用药记录)拿捏 / 说得在理 → 交出医用酒精；当面拿用药记录揭破 → 拿捏住他
        elif npc_id == DOCTOR_ID:
            leverage = pi.evidence == "medlog" or pi.intent in (nlu.SHOW, nlu.ACCUSE) or pi.sincerity >= 7
            if leverage and not bb.flag("doctor_gave_alcohol") and not bb.has_item(ALCOHOL):
                bb.give_item(ALCOHOL)
                bb.set_flag("doctor_gave_alcohol")
                bb.advance_stage(DOCTOR_ID, "alcohol")
                bb.bump_trust(npc_id, 10)
                gained += [ALCOHOL]
                notes.append("doctor_alcohol")
                toast = "柳叶刀盯了你两秒，从药柜里取出一瓶医用酒精推过来：「拿去。记住，这瓶东西你没在我这儿见过。」"
            # 拿"篡改的用药记录"当面指控 → 攥住他的把柄（新分支：影响他的结局下落）
            if (pi.evidence == "medlog" and pi.intent in (nlu.ACCUSE, nlu.SHOW)
                    and not bb.flag("doctor_exposed")):
                bb.set_flag("doctor_exposed")
                bb.advance_stage(DOCTOR_ID, "exposed")
                bb.bump_trust(npc_id, -10)
                notes.append("doctor_exposed")
                toast = "你把那页篡改过的用药记录甩到柳叶刀面前。他第一次抬起眼——精算师，也有算漏的时候。"

        # 狱警铁面：抚恤金名单 + 诚意 → 收买（暴乱晚到五分钟）；空口踩典狱长 → 惹祸
        elif npc_id == GUARD_ID:
            if bb.has_item(PENSION_LIST) and pi.sincerity >= 6 and pi.intent in (nlu.SHOW, nlu.BRIBE, nlu.ACCUSE):
                if not bb.flag("guard_bribed"):
                    bb.set_flag("guard_bribed")
                    bb.advance_stage(GUARD_ID, "bribed")
                    bb.bump_trust(npc_id, 30)
                    notes.append("guard_bribed")
                    toast = ("铁面死死盯着那份抚恤金截留名单，脸色铁青，胸口起伏半晌，"
                             "才从牙缝里挤出一句：「……暴乱那晚，我会晚到五分钟。别让我后悔。」")
                else:
                    notes.append("already_bribed")
            elif pi.intent == nlu.ACCUSE and "典狱长" in text:
                bb.bump_trust(npc_id, -15)
                bb.advance_stage(GUARD_ID, "offended")
                notes.append("guard_offended")
                toast = "「就凭你也敢污蔑长官？」铁面一把揪住你，把你的号牌重重记下——你上了他的黑名单。"

        # 老牙：拿到"软底鞋/黑账/欠条"任一 + 当面揭穿 → 反转触发；否则他把矛头引向典狱长
        elif npc_id == ANCHOR_ID:
            if pi.intent == nlu.ACCUSE:
                has_evid = bool(bb.evidence & EXPOSE_EVIDENCE)
                if has_evid and pi.sincerity >= 6 and not bb.flag("oldtooth_exposed"):
                    bb.set_flag("oldtooth_exposed")
                    bb.advance_stage(ANCHOR_ID, "exposed")
                    bb.bump_trust(npc_id, -40)
                    notes.append("oldtooth_exposed")
                    toast = "老牙脸上的热络一寸寸僵掉，瞳孔猛地一缩——你头一回看清他藏在笑底下的那张脸。"
                elif not has_evid:
                    notes.append("oldtooth_deflect")  # 没证据硬指控，他顺势甩锅典狱长（台词交给 LLM+事实库）

        # 典狱长（终局对峙）：拿黑账/洗钱/名单指控 → 逼到墙角；当场点破真凶老牙 → 触发反转
        elif npc_id == WARDEN_ID:
            corner_evid = bb.evidence & {"laundering", "pension", "medlog", "ledger"}
            if pi.intent == nlu.ACCUSE and corner_evid and not bb.flag("warden_cornered"):
                bb.set_flag("warden_cornered")
                bb.advance_stage(WARDEN_ID, "cornered")
                notes.append("warden_cornered")
                toast = "你把黑账拍在桌上。典狱长的官威一寸寸垮掉，喉结上下滚动，再挤不出一句硬话。"
            # 对峙中直接点破"真凶是老牙"（有揭穿证据 + 提到老牙/软底/销赃）→ 全剧翻盘
            points_at_tooth = any(k in text for k in ("老牙", "软底", "布鞋", "销赃", "白手套", "室友"))
            if (points_at_tooth and (bb.evidence & EXPOSE_EVIDENCE)
                    and pi.sincerity >= 6 and not bb.flag("oldtooth_exposed")):
                bb.set_flag("oldtooth_exposed")
                bb.advance_stage(ANCHOR_ID, "exposed")
                bb.bump_trust(ANCHOR_ID, -40)
                notes.append("oldtooth_exposed")
                toast = "「真正把我弄进来的，从头到尾就在我身边。」你说出老牙的名字——满室死寂。"

        # 新分支：向铁面/疤脸告发瘸子的走私勾当 → 瘸子落网（改他的结局下落）
        if (npc_id in (GUARD_ID, GANG_ID) and not bb.flag("smuggler_busted")
                and any(k in text for k in ("告发", "举报", "出卖", "检举"))
                and any(k in text for k in ("瘸子", "走私", "违禁", "货"))):
            bb.set_flag("smuggler_busted")
            bb.advance_stage(SMUGGLER_ID, "busted")
            notes.append("smuggler_busted")
            toast = "你把瘸子那点走私勾当抖了出去。没过多久，就听说他被按倒在墙上、拖去了禁闭室。"

        # 通用好感微调（没有触发上面的强事件时）——细腻的好感增减交给关系判官(LLM)，
        # 这里只做零延迟的确定性兜底，别和 LLM 那份重复叠太多。
        if not notes:
            if pi.intent == nlu.THREATEN or pi.emotion == "aggressive":
                bb.bump_trust(npc_id, -3)
                self._trust_beat(npc_id)

        unlocked = bb.check_progress_unlock(self.case_progress)
        if unlocked:
            log.info("对话触发解锁：%s（跟 %s 聊出来的）", unlocked, npc_id)
        return {
            "npc_id": npc_id,
            "intent": pi.intent,
            "sincerity": pi.sincerity,
            "items_gained": gained,
            "items_lost": lost,
            "unlocked": unlocked,
            "notes": notes,
            "toast": toast,
            "blackboard": bb.to_message(),
        }

    def _trust_beat(self, npc_id: str) -> dict | None:
        """好感变化后：按阈值翻关系幕（beat 已进 fate_log，随黑板同步给前端）。"""
        return self.bb.sync_trust_stage(npc_id)

    def apply_relationship(self, npc_id: str, delta: int) -> dict:
        """关系判官（LLM）判完 → 落到黑板：改好感 + 可能翻关系幕（纯说话就改写命运）。

        只动好感、绝不碰剧情旗标（硬剧情仍归确定性状态机管），是"对话有后果"的软层落点。
        """
        delta = max(-15, min(15, int(delta or 0)))
        if delta:
            self.bb.bump_trust(npc_id, delta)
        beat = self._trust_beat(npc_id)
        return {
            "npc_id": npc_id,
            "trust": self.bb.trust_of(npc_id),
            "trust_delta": delta,
            "fate": beat,
            "blackboard": self.bb.to_message(),
        }

    def incite_riot(self) -> dict:
        """冲动路线（大胆分叉）：证据够多就铤而走险，不做全套潜行准备直接挑起暴乱、硬闯办公室。

        门槛比"磁卡+收买铁面+放倒狗"低得多，但狱警狱犬都没摆平——是步险棋，
        自然把结局推向惨胜/被反杀而非大获全胜（评分里少了收买/放倒的加成）。
        """
        bb = self.bb
        if bb.zone_unlocked("office"):
            return {"ok": False, "reason": "already"}
        if not bb.zone_unlocked("restricted") or len(bb.evidence) < 3:
            return {"ok": False, "reason": "not_ready",
                    "need": "先解锁受限区、且至少查到 3 条线索"}
        bb.set_flag("incited")
        z = bb.unlock_zone("office")
        bb.set_flag("riot_started")
        log.info("冲动路线：玩家硬闯 → 煽动暴乱，解锁 office（证据=%d）", len(bb.evidence))
        return {
            "ok": True,
            "unlocked": [z] if z else [],
            "toast": ("你不再等了——一把掀翻了铁桶，喊声顺着走廊炸开。警铃随之尖啸，"
                      "整座监狱在你脚下沸腾。狱警和那条狗还没摆平，可你已经没有退路。"),
            "blackboard": bb.to_message(),
        }

    def observer_finale(self) -> dict:
        """观察者模式：上帝随时发起最终暴乱、进入结算——不受证据/分区门槛限制。

        和 incite_riot 的区别：incite 是第一人称流程里的"冲动路线"，仍要求受限区+3证据；
        observer_finale 是上帝视角的收场开关，任何时刻都能按，让 DM 就着此刻的场上状态收束。
        """
        bb = self.bb
        bb.set_flag("incited")
        z = bb.unlock_zone("office")
        bb.set_flag("riot_started")
        log.info("观察者发起终局（随时结算），evidence=%d caught=%d", len(bb.evidence), bb.caught)
        return {
            "ok": True,
            "unlocked": [z] if z else [],
            "toast": "上帝之手落下——你掀翻了整座监狱。警铃炸响，人人自危，命运在此刻收束。",
            "blackboard": bb.to_message(),
        }

    def sim_epilogue_lines(self) -> list[str]:
        """把每个 NPC 这一局在仿真里真实的下落（伤/禁闭/越狱/加刑/结仇/临乱时的决策）
        提炼成一句，喂给 DM——让结局不仅由剧情旗标决定，也吃这一局涌现出来的故事。"""
        out: list[str] = []
        for npc in self.npcs:
            bits: list[str] = []
            q = quest_of(npc.id)
            # 终极目标完成度：结局按"他这辈子图的那件大事走到哪一步"分支定制
            if q and q.steps:
                if npc.dead:
                    pass  # 死者的下落单列在下面
                elif npc.quest_step >= len(q.steps):
                    bits.append(f"竟真把毕生图谋走到了头——{q.ultimate}")
                elif npc.quest_step >= max(1, len(q.steps) - 2):
                    bits.append(f"离『{q.ultimate}』只差最后一步（{npc.quest_step}/{len(q.steps)}）")
                elif npc.quest_step >= 1:
                    bits.append(f"为『{q.ultimate}』刚铺垫到第 {npc.quest_step}/{len(q.steps)} 步")
                else:
                    bits.append(f"图谋『{q.ultimate}』还没真正起步")
            # 生死网结局：谁死了、谁把仇杀对象了结了、谁还揣着凶器没动手
            if npc.dead:
                bits.append("已横尸狱中，再没能等到出去那天")
            elif npc.kill_target:
                tgt = self._npc(npc.kill_target)
                if tgt and tgt.dead:
                    bits.append(f"亲手了结了{tgt.name}，手上沾了血")
                elif tgt and not tgt.escaped:
                    weap = WEAPON_CN.get(npc.weapon, npc.weapon) if npc.weapon else ""
                    bits.append(f"还在暗处盯着{tgt.name}" + (f"，{weap}已经备好" if weap else "，却始终没等到下手的机会"))
            if npc.escaped:
                bits.append("已经翻墙越狱、消失在夜色里")
            elif npc.solitary_ticks > 0:
                bits.append("此刻正被关在禁闭室")
            inj = npc.needs.injury
            if inj >= 70:
                bits.append("被打得重伤难行")
            elif inj >= 35:
                bits.append("身上添了新伤")
            if npc.parole_lost:
                bits.append("假释泡汤、破罐破摔")
            elif npc.extra_days:
                bits.append(f"因违纪被加了 {npc.extra_days} 天刑")
            elif npc.parole_near:
                bits.append("眼看就要平安假释出去")
            if npc.busted:
                bits.append(f"这一局被狱警抓了 {npc.busted} 次")
            top = sorted(npc.grudge.items(), key=lambda kv: -kv[1])[:1]
            if top and top[0][1] >= 55:
                other = self._npc(top[0][0])
                if other:
                    bits.append(f"和 {other.name} 结下死仇")
            dr = npc.decision_drivers or {}
            if dr.get("why") and not npc.dead:
                bits.append(f"乱起时{dr['why']}")
            if bits:
                out.append(f"· {npc.name}（{npc.role}）：" + "；".join(bits) + "。")
        return out

    def drug_dog(self) -> dict:
        """把拌了镇静剂的肉丢到哮天的巡逻线上：消耗镇静剂 → dog_drugged，哮天嗅觉归零。"""
        bb = self.bb
        if not bb.has_item(SEDATIVE):
            return {"ok": False, "reason": "no_sedative"}
        if bb.flag("dog_drugged"):
            return {"ok": False, "reason": "already"}
        bb.take_item(SEDATIVE)
        bb.set_flag("dog_drugged")
        bb.advance_stage(DOG_ID, "drugged")
        unlocked = bb.check_progress_unlock(self.case_progress)
        log.info("哮天被镇静肉放倒（dog_drugged），新解锁=%s", unlocked)
        return {
            "ok": True,
            "unlocked": unlocked,
            "toast": "你把拌了镇静剂的肉丢到哮天必经的巡逻线上。它嗅了嗅，狼吞下去，没走几步就腿一软瘫倒——鼾声大作。",
            "blackboard": bb.to_message(),
        }

    def plead(self, text: str) -> dict:
        """阶段三潜行：被狱警/狱犬撞见时的"狡辩"。借意图诚意分判定借口够不够巧妙。
        够严密(≥7) → 蒙混过关；否则关禁闭（扣时间由前端演绎）。"""
        pi = nlu.parse_intent(text)
        ok = pi.sincerity >= 6 and pi.intent != nlu.THREATEN
        if not ok:
            self.bb.caught += 1          # 狡辩失败=被关禁闭，记一次把柄（结局 P_penalty 扣分）
        return {"ok": ok, "sincerity": pi.sincerity, "intent": pi.intent,
                "caught": self.bb.caught}

    # -- 阶段四/五：暴乱收束 · 对峙 · 结局结算 ------------------------------
    def resolution(self) -> dict:
        """当前破局指数与结局基调（读黑板，纯结算）。"""
        return resolution_score(self.bb)

    def ending_prompt(self, last_line: str = "") -> str:
        """观察者·终局 DM 超级指令（无玩家版）：以上帝视角旁白，就着此刻场上
        五名囚犯的真实处境，把这场生死网的收场写成一段有电影感的群像结局。

        与旧的第一人称解谜版彻底不同：没有"主角/玩家最后一句"，不谈"揭穿真凶/收买狱警/
        收集证据"那套侦探线；改由每个人**这辈子图的那件大事（终极目标）走到哪一步**、
        以及生死网里谁杀了谁、谁越了狱、谁横死、谁功亏一篑来定调。逐 NPC 真实状态由
        _handle_ending 追加的 sim_epilogue_lines 提供，这里只搭框架 + 基调 + 卡司关系网。
        """
        res = self.observer_resolution()
        # 卡司与生死网关系（读任务链：谁要杀谁、谁想逃、谁只求活着、各自的凶器）
        cast: list[str] = []
        for npc in self.npcs:
            q = quest_of(npc.id)
            if not q:
                continue
            aim = q.ultimate
            rel = ""
            if q.kind == "kill" and q.target:
                tgt = self._npc(q.target)
                weap = WEAPON_CN.get(q.weapon, q.weapon) if q.weapon else "还没弄到的凶器"
                rel = f"，要用{weap}了结{tgt.name if tgt else q.target}"
            elif q.kind == "escape":
                rel = "，一心想赶在被做掉之前越狱脱身"
            elif q.kind == "survive":
                rel = "，只求在这滩浑水里保住性命"
            cast.append(f"· {npc.name}（{npc.role}）：图的是「{aim}」{rel}。")
        lines = [
            "你是一部硬核监狱群像剧的旁白（全知视角的 DM）。这座州立监狱刚被掀翻——"
            "警铃炸响、铁门一道道闷响、囚犯各怀鬼胎在混乱里做最后的了断。"
            "现在由你，用冷峻、粗粝、有画面感的第三人称，为这一局收束一段结局正文。",
            "",
            "【这一局的卡司与他们各自图谋的那件大事（生死网）】",
            *cast,
            "",
            f"【本局收场的总基调】{res['tier_cn']}——{res['directive']}",
            "",
            "写作铁律（务必遵守）：",
            "· 全程第三人称、全知旁白，绝不出现「我 / 主角 / 玩家 / 你」这类第一人称主角——"
            "这座监狱里没有主角，只有这几个各自挣扎的囚犯，镜头在他们之间流转。",
            "· 不要提「证据、揭穿真凶、收买狱警、收集线索」这套侦探桥段——那是另一个故事；"
            "这一局讲的是这几个人为各自的终极目标彼此算计、动手、逃亡与横死。",
            "· 严格依据下面【每个人真实的下落】来写，谁死了、谁越了狱、谁把仇人做掉了、"
            "谁功亏一篑、谁还被关在禁闭室——一个都不能写错、不能编造没发生的事。",
            "· 狱警（铁面）、狱犬（哮天）、典狱长只作暴乱之夜的背景与压迫感，不喧宾夺主。",
            "· 篇幅约 500~700 字，一气呵成、有文学质感，不要分小标题、不要罗列人名清单，"
            "把每个人的命运织进这一夜的叙事里。直接就是结局正文，不要旁白说明。",
        ]
        return "\n".join(lines)

    # 观察者结局基调：按场上真实涌现（横死/越狱/仇杀达成/禁闭/任务完成度）定调，
    # 取代旧的"证据完整度/收买/揭穿真凶"那套第一人称解谜结算。
    _OBS_TIER_CN = {
        "carnage": "血色收场 · 玉石俱焚",
        "bloodshed": "有人横死狱中",
        "breakout": "有人越狱脱身",
        "scheme": "有人图谋得逞",
        "uneasy": "暗流涌动 · 各自蛰伏",
    }
    _OBS_TIER_DIRECTIVE = {
        "carnage": "多人横尸，这一夜血流成河，写尽仇杀连锁、玉石俱焚的惨烈与荒凉。",
        "bloodshed": "有人没能熬到天亮，写那一条命如何断送、其余人如何各怀心事地活下来。",
        "breakout": "有人趁乱翻墙、消失在夜色里，写这场脱逃的孤注一掷与留下者的百味杂陈。",
        "scheme": "有人把毕生图谋推到了头，写他得逞的阴冷快意，与被他踩在脚下的人的下场。",
        "uneasy": "血没真正流成河，但仇怨与算计都埋下了，写这份'暂时相安'底下的暗涌与不甘。",
    }

    def observer_resolution(self) -> dict:
        """无玩家版结局结算：读真实场上状态，给基调 + 混乱指数 + 逐人一句下落。"""
        npcs = self.npcs
        dead = [n for n in npcs if n.dead]
        escaped = [n for n in npcs if n.escaped]
        solitary = [n for n in npcs if not n.dead and n.solitary_ticks > 0]
        injured = [n for n in npcs if not n.dead and n.needs.injury >= 35]
        kills_done = [n for n in npcs
                      if n.kill_target and (self._npc(n.kill_target) and self._npc(n.kill_target).dead)]
        quests_done = [n for n in npcs
                       if quest_of(n.id) and n.quest_step >= len(quest_of(n.id).steps)]
        if len(dead) >= 2:
            tier = "carnage"
        elif dead:
            tier = "bloodshed"
        elif escaped:
            tier = "breakout"
        elif quests_done:
            tier = "scheme"
        else:
            tier = "uneasy"
        score = len(kills_done) * 3 + len(escaped) * 2 + len(solitary) + len(injured)
        fates: dict[str, str] = {}
        for n in npcs:
            fates[n.id] = self._obs_fate_line(n)
        return {
            "tier": tier,
            "tier_cn": self._OBS_TIER_CN[tier],
            "directive": self._OBS_TIER_DIRECTIVE[tier],
            "score": score,
            "factors": {
                "dead": len(dead), "escaped": len(escaped), "kills": len(kills_done),
                "solitary": len(solitary), "injured": len(injured),
                "quests_done": len(quests_done), "npc_total": len(npcs),
            },
            "fates": fates,
        }

    def _obs_fate_line(self, npc: NPC) -> str:
        """终局给前端清单用的一句下落（比 sim_epilogue_lines 更短、更定性）。"""
        if npc.dead:
            return "横尸狱中，再没能等到出去那天。"
        if npc.escaped:
            return "翻墙越狱，消失在暴乱之夜的黑暗里。"
        q = quest_of(npc.id)
        done = q and npc.quest_step >= len(q.steps)
        tgt = self._npc(npc.kill_target) if npc.kill_target else None
        if tgt and tgt.dead:
            return f"亲手了结了{tgt.name}，手上沾了血，图谋得逞。"
        if done:
            return f"竟真把「{q.ultimate}」走到了头。"
        if npc.solitary_ticks > 0:
            return "被摁进禁闭室，隔着铁门听着外面的乱。"
        if npc.needs.injury >= 60:
            return "被打得重伤难行，瘫在角落里喘。"
        if tgt and npc.weapon:
            return f"凶器已备好，却终究没等到向{tgt.name}下手的那一刻。"
        return "在这场乱局里熬了下来，图谋悬而未决。"

    def phase_intro_setup(self, zone: str) -> tuple[str, dict, str] | None:
        """某阶段开场白：返回(向导 id, 对话上下文, 导演指令)，供 LLM 即兴生成开场台词。"""
        guide = PHASE_GUIDE.get(zone)
        if not guide:
            return None
        ctx = self.npc_chat_context(guide)
        if not ctx:
            return None
        return guide, ctx, phase_intro_directive(zone, self.bb)

    def endgame_state(self) -> dict:
        """终局状态块（前端据此拉红警、算结局基调）。

        原先这里还带一个 warden 位置块，专门给门口那具红衣典狱长 NPC 显形用；
        典狱长已由玩家亲自扮演，那具 NPC 连模型带对峙镜头一起删了，字段随之取消。
        """
        office = self.bb.zone_unlocked("office")
        res = resolution_score(self.bb)
        return {
            "riot": self.bb.flag("riot_started"),
            "office_unlocked": office,
            "score": res["score"], "tier": res["tier"], "tier_cn": res["tier_cn"],
            "factors": res["factors"],
        }

    def on_clue_found(self, clue_id: str) -> dict:
        """前端检视到一条线索：记进证据库 + 同步案件进度 + 检查逐级解锁 + 顺带发道具。"""
        new = self.bb.add_evidence(clue_id)
        gained: list[str] = []
        item = CLUE_GRANTS.get(clue_id)   # 如：找到抚恤金名单 → 拿到实物名单（收买铁面用）
        if item and self.bb.give_item(item):
            gained.append(item)
        # 线索联动 NPC 命运：软底鞋→听懂疯子的谜语；欠条→拿到疤脸手下的赌债线索
        if clue_id == "softshoe":
            self.bb.advance_stage(MADMAN_ID, "understood")
        elif clue_id in ("iou", "ledger"):
            self.bb.advance_stage(GANG_ID, "debt")
        self.case_progress = max(self.case_progress, len(self.bb.evidence))
        unlocked = self.bb.check_progress_unlock(self.case_progress)
        if unlocked:
            log.info("线索『%s』推进解锁：%s", clue_id, unlocked)
        return {"new": new, "clue_id": clue_id, "items_gained": gained,
                "unlocked": unlocked, "blackboard": self.bb.to_message()}

    # -- 时段 / 拥挤 --------------------------------------------------------
    def _phase(self) -> str:
        if self.phase_override:
            return self.phase_override
        idx = (self.tick_count // PHASE_TICKS) % len(PHASE_ORDER)
        return PHASE_ORDER[idx]

    def _in_yard(self, npc: NPC) -> bool:
        if npc.dead or npc.escaped:
            return False
        yard = self._yard()
        return math.hypot(npc.x - yard.x, npc.z - yard.z) < 4.0

    def _yard_count(self) -> int:
        return sum(1 for npc in self.npcs if self._in_yard(npc))

    @staticmethod
    def _crowd_label(n: int) -> str:
        if n >= 4:
            return CROWD_PACKED
        if n >= 2:
            return CROWD_MID
        return CROWD_EMPTY

    def _yard_crowd(self) -> str:
        return self._crowd_label(self._yard_count())

    # -- El Farol：拥挤预期 + 性格化引擎项 -----------------------------------
    def _predict_crowd(self, npc: NPC) -> int:
        """按各自的异质预测器，从历史人数预测本拍放风场人数（有限理性）。"""
        h = self.attendance_history
        if not h:
            return YARD_COMFORT
        p = npc.predictor
        if p == "last":
            e = h[-1]
        elif p == "avg4":
            tail = list(h)[-4:]
            e = sum(tail) / len(tail)
        elif p == "trend":
            e = h[-1] + (h[-1] - h[-2]) if len(h) >= 2 else h[-1]
        elif p == "contra":
            e = self.n_agents - h[-1]
        else:  # avg2
            tail = list(h)[-2:]
            e = sum(tail) / len(tail)
        return int(round(max(0, min(self.n_agents, e))))

    # -- 狱警/狱犬巡逻 + 视线 ----------------------------------------------
    def _next_open_wp(self, p: Patroller) -> None:
        """把当前路点推进到第一个落在**已解锁分区**里的路点。

        未解锁的区域前端不渲染、也会把巡逻者一并隐藏——巡逻者要是一头扎进锁着的
        分区，对玩家就等于人间蒸发（铁面因此长期摸不到、没法收买）。这里让巡逻线
        随解锁进度自动收缩到玩家够得着的范围里。
        """
        n = len(p.waypoints)
        for _ in range(n):
            wx, wz = p.waypoints[p.wp]
            if self.bb.zone_unlocked(self._zone_of_pos(wx, wz)):
                return
            p.wp = (p.wp + 1) % n

    def _escort_step(self, p: Patroller, dt: float) -> bool:
        """押送中的巡逻者：走在被押者外侧半步，一同往禁闭室去。

        返回 False 表示押送对象已失效（人死了/放出来了），调用方应回到正常巡逻。
        """
        npc = self._npc(p.escorting)
        if npc is None or npc.escort_ticks <= 0:
            p.escorting = ""
            return False
        if not npc.escort_grabbed:
            # 抓人阶段：直奔犯人。这里跑得比平时快些——"过来拿人"该有点扑上去的劲，
            # 也免得犯人干等太久。
            p.x, p.z, _m, p.hx, p.hz = self._step_along(
                p.x, p.z, npc.x, npc.z, p.speed * 1.35 * dt)
            return True
        # 押送阶段：站到被押者与禁闭室连线的侧后方，看上去是"押着走"，而不是两人重叠
        sx, sz = SOLITARY_POS
        dx, dz = sx - npc.x, sz - npc.z
        d = math.hypot(dx, dz) or 1.0
        ux, uz = dx / d, dz / d
        # 侧后方半步：够近才像"押着"，又不至于两具模型插在一起
        tx, tz = npc.x - ux * 0.8 - uz * 0.8, npc.z - uz * 0.8 + ux * 0.8
        p.x, p.z, _m, _hx, _hz = self._step_along(
            p.x, p.z, tx, tz, max(p.speed, npc.speed) * 1.15 * dt)
        p.hx, p.hz = ux, uz                            # 视锥朝行进方向，不朝那半步的侧位
        return True

    def _tick_salutes(self, dt: float) -> None:
        """典狱长（玩家）走近 → 狱警/狱犬立正行礼，而不是把他当犯人查。

        只在玩家上报过位置（warden_active）时成立。押送途中不行礼——手里押着人，
        礼让给职责。冷却避免玩家贴着狱警走时一路刷礼。
        """
        for p in self.patrollers:
            if p.salute > 0:
                p.salute = max(0.0, p.salute - dt)
                continue
            if not self.warden_active or p.escorting:
                continue
            cd = self._salute_cd.get(p.id, 0.0)
            if cd > 0:
                self._salute_cd[p.id] = max(0.0, cd - dt)
                continue
            if math.hypot(p.x - self.player.x, p.z - self.player.z) <= SALUTE_R:
                p.salute = SALUTE_SECONDS
                self._salute_cd[p.id] = SALUTE_COOLDOWN + SALUTE_SECONDS
                # 行礼时转向典狱长：不然会出现背对着他敬礼
                dx, dz = self.player.x - p.x, self.player.z - p.z
                d = math.hypot(dx, dz) or 1.0
                p.hx, p.hz = dx / d, dz / d

    def _move_patrollers(self, dt: float) -> None:
        self._tick_salutes(dt)
        for p in self.patrollers:
            if p.escorting and self._escort_step(p, dt):
                continue                               # 押人优先于巡逻和敬礼
            if p.salute > 0:                           # 正在向典狱长行礼：站定不动
                continue
            if p.dwell > 0:                            # 到点驻留：停下扫视，制造停顿感
                p.dwell = max(0.0, p.dwell - dt)
                continue
            self._next_open_wp(p)
            tx, tz = p.waypoints[p.wp]
            dist = math.hypot(tx - p.x, tz - p.z)
            if dist < 0.4:
                p.wp = (p.wp + 1) % len(p.waypoints)   # 到点，转下一段
                self._next_open_wp(p)
                p.dwell = 1.4 + random.random() * 2.2  # 停 1.4~3.6s 再走，狗略短
                if p.kind == "dog":
                    p.dwell *= 0.5
                continue
            # 巡逻也走绕墙路径：巡逻环贴着中庭边缘走，直线连两个路点会切过墙角
            step = NAV.path(p.x, p.z, tx, tz)
            wx, wz = step[0] if step else (tx, tz)
            dx, dz = wx - p.x, wz - p.z
            d = math.hypot(dx, dz) or 1.0
            ux, uz = dx / d, dz / d
            p.hx, p.hz = ux, uz                        # 朝行进方向（狱警视锥正前）
            p.x += ux * min(p.speed * dt, d)
            p.z += uz * min(p.speed * dt, d)

    @staticmethod
    def _step_along(x: float, z: float, tx: float, tz: float,
                    step: float) -> tuple[float, float, float, float, float]:
        """朝目标走一步，绕墙。返回 (新x, 新z, 走了多远, 朝向x, 朝向z)。

        押送、抓人这些临时移动也得认墙：押送者原来直线扑向犯人，路上有房子就
        从墙里穿过去，看着比穿模更糟——一个狱警从图书室的墙里钻出来。
        """
        pts = NAV.path(x, z, tx, tz)
        wx, wz = pts[0] if pts else (tx, tz)
        dx, dz = wx - x, wz - z
        d = math.hypot(dx, dz)
        if d < 1e-6:
            return x, z, 0.0, 0.0, 1.0
        k = min(step, d)
        return x + dx / d * k, z + dz / d * k, k, dx / d, dz / d

    def _watcher_of(self, x: float, z: float) -> str:
        """(x,z) 处是否落在某个巡逻者的监视范围内；返回 '狱警'/'狱犬'/''。

        狱警=正前方视锥（会被绕后规避）；狱犬=近距离全向嗅觉（无死角但范围小）。
        """
        dog_out = self.bb.flag("dog_drugged")   # 哮天被镇静肉丢翻 → 嗅觉归零
        for p in self.patrollers:
            dx, dz = x - p.x, z - p.z
            d = math.hypot(dx, dz)
            if p.kind == "dog":
                if not dog_out and d <= DOG_SMELL_RANGE:
                    return "狱犬"
            else:
                if d <= GUARD_POINT_BLANK:
                    return "狱警"      # 就贴在跟前，背对着也不可能看不见
                if d <= GUARD_VIEW_RANGE:
                    if (dx / d) * p.hx + (dz / d) * p.hz >= GUARD_VIEW_COS:
                        return "狱警"
        # 典狱长（玩家）在场：他站位即监视——附上了狱警的能力，靠近谁谁就得收敛/立正。
        if self.warden_active and math.hypot(x - self.player.x, z - self.player.z) <= WARDEN_SIGHT:
            return "狱警"
        return ""

    def _crowd_multiplier(self, npc: NPC, goal: str, expected: int) -> float:
        """放风休闲候选按拥挤预期调节；Rest 取相反方向。构成 El Farol 负反馈。"""
        pref, tol = npc.crowd_pref, npc.crowd_tol
        if goal in YARD_LEISURE:
            if pref == "like":
                return max(0.8, min(3.0, 0.8 + 0.35 * expected))
            k = 0.9 if pref == "avoid" else 0.6
            return max(0.25, min(3.0, math.exp(k * (tol - expected))))
        if goal == "Rest":
            if pref == "like":
                return max(0.3, min(1.2, 1.2 - 0.2 * expected))
            k = 0.9 if pref == "avoid" else 0.6
            return max(0.5, min(3.0, math.exp(k * (expected - tol))))
        return 1.0

    # -- 宏观动机（drives.py）----------------------------------------------
    def _drive_of(self, npc: NPC):
        """取 NPC 生效动机对象：假释崩塌后一律走破罐破摔。"""
        return get_drive(npc.drive, npc.parole_lost)

    def _motive_line(self, npc: NPC) -> str:
        """进 caption 的『长期心思』——只放一句泛泛倾向，不写具体动作（避免复读）。
        进度快到顶时加一句"眼看就成"，让措辞随弧线变化、不至于一成不变。"""
        d = self._drive_of(npc)
        line = d.motive
        if not npc.parole_lost and npc.drive_progress >= DRIVE_MILESTONE:
            line += "（眼看就要成了，更不想出岔子）"
        return line

    # 关系→该被抬起的社交动作（按"关系类型"匹配，只重抬对得上的那个，避免泛泛 Propose 刷屏）。
    _SOCIAL_HOOK_MAP = {
        NEAR_FOE:      {"Threaten": 2.3},
        NEAR_ALLY:     {"Protect": 2.1, "Betray": 1.7, "Ally": 1.3},
        NEAR_DEBTOR:   {"Threaten": 1.9, "Betray": 1.7},   # 他欠我 → 可催可反
        NEAR_CREDITOR: {"Repay": 2.3},
        NEAR_STRANGER: {"Ally": 2.0},                      # 生人 → 结盟破冰
    }

    def _social_hook_boost(self, npc: NPC, goal: str) -> float:
        """定向社交动作的关系钩子加成：只重抬"跟近旁那人的关系对得上"的那个动作
        （仇敌→放狠话、盟友→护/反水、债主→还人情、生人→结盟）。泛泛 Propose 只给微抬，
        免得"攀谈"把结盟/背叛/放狠话全盖掉。不动需求/威胁的高基分——护栏层级自然守住。"""
        near, _intent, _sr, _oblig = self._social_axes(npc)
        if near == NEAR_NONE:
            return 1.0
        b = self._SOCIAL_HOOK_MAP.get(near, {}).get(goal)
        if b is not None:
            return b
        return 1.15 if goal == "Propose" else 1.0

    def _m_drive(self, npc: NPC, goal: str) -> float:
        """动机乘数：推进长期心思的行为抬一档、损耗的压一档、禁忌≈0。
        再叠一层"终极目标当前分目标"的聚焦：属于这一步该干的事，额外抬一档，
        让行为指向"此刻这个分目标"。生存刚需不在任何集合里，恒为 1.0——动机从不跟命抢。"""
        m = self._drive_of(npc).m_drive(goal)
        q = quest_of(npc.id)
        if (q and q.steps and not npc.parole_lost and npc.quest_step < len(q.steps)
                and goal in q.steps[npc.quest_step].focus):
            # 承诺压力水涨船高：主线拖得越久，这一步该干的事就抬得越高，
            # 直到压过"非危机需求"。押不过就靠 _force_quest 硬顶（保证关键一步落地）。
            m *= QUEST_FOCUS_BOOST + npc.quest_pressure * PRESSURE_GAIN
        return m

    def _advance_drive(self, npc: NPC, goal: str) -> None:
        """完成一个动作后：推进终极目标任务链的当前分目标；签名动作顺带进冷却。"""
        d = self._drive_of(npc)
        if goal == d.signature:
            npc.sig_cooldown = SIG_COOLDOWN
        q = quest_of(npc.id)
        if q and q.steps and not npc.parole_lost:
            self._advance_quest(npc, q, goal)
            return
        # 没有任务链（或已破罐破摔）：退回扁平动机进度
        gain = d.advance.get(goal)
        if gain:
            npc.drive_progress = min(100.0, npc.drive_progress + gain)

    def _advance_quest(self, npc: NPC, q, goal: str) -> None:
        """完成一个"推进当前分目标"的动作 → 涨本步进度；满了就翻到下一分目标并记命运账本。"""
        if npc.quest_step >= len(q.steps):
            return
        step = q.steps[npc.quest_step]
        if goal not in step.focus:
            return
        npc.step_progress += STEP_GAIN
        npc.quest_pressure = 0.0          # 主线动了一步 → 承诺压力清零，重新积攒
        # 弄凶器这一步真的要给刀：否则疤脸走完「弄刀」仍是空手，永远下不了手。
        if goal == "GetWeapon" and not npc.weapon:
            npc.weapon = q.weapon or "knife"
        self._sync_quest_progress(npc, q)
        if npc.step_progress < step.need:
            return
        # —— 一个分目标达成：记一条叙事化命运账本，推进到下一步 ——
        self.bb.fate_log.append({
            "npc": npc.id, "name": npc.name,
            "status": f"分目标 {npc.quest_step + 1}/{len(q.steps)} 达成",
            "text": step.beat, "big": step.big,
        })
        npc.quest_step += 1
        npc.step_progress = 0.0
        self._sync_quest_progress(npc, q)
        if npc.quest_step >= len(q.steps):
            self.bb.fate_log.append({
                "npc": npc.id, "name": npc.name, "status": "终极目标达成",
                "text": f"{npc.name}终于走到了这一步——{q.ultimate}。", "big": True,
            })
        self.bb.fate_log[:] = self.bb.fate_log[-80:]
        log.info("任务链推进：%s → 分目标 %d/%d", npc.name, npc.quest_step, len(q.steps))

    def _quest_step_now(self, npc: NPC):
        """当前正在啃的分目标 (q, step)；无任务链/已破罐破摔/已通关 → (None, None)。"""
        q = quest_of(npc.id)
        if q and q.steps and not npc.parole_lost and npc.quest_step < len(q.steps):
            return q, q.steps[npc.quest_step]
        return None, None

    def _npc_safe_for_plot(self, npc: NPC) -> bool:
        """此刻能不能腾出手推进主线：没被盯、没在打架、没受指挥、没濒死级需求。
        护栏阶梯：这几样任一成立都不该被主线抢镜，压力也不逼他。"""
        n = npc.needs
        return (not npc.watched_by and not npc.in_bout and not npc.commanded_goal
                and max(n.hunger, n.thirst, n.injury) < EMERGENCY_NEED)

    def _update_quest_pressure(self, npc: NPC) -> None:
        """承诺压力更新（每拍决策后调）：正推进主线→清零由 _advance_quest 负责；
        安全却在忙别的(吃饭/闲逛)→攒压；被盯/濒死/打架/受指挥→按住(合理推迟，不罚)。"""
        _q, step = self._quest_step_now(npc)
        if step is None:                       # 没有在啃的分目标：压力/预告都归零
            npc.quest_pressure = 0.0
            npc.pressure_telegraphed = -1
            return
        if npc.goal in step.focus:             # 正在推进这一步 → 不攒（清零交给完成时）
            return
        if self._npc_safe_for_plot(npc):
            npc.quest_pressure = min(PRESSURE_MAX, npc.quest_pressure + PRESSURE_RISE)

    def _seed_kill_state(self, npc: NPC) -> None:
        """按任务链给 NPC 发开局凶器、定仇杀对象（生死网的初始态）。"""
        q = quest_of(npc.id)
        npc.weapon = q.weapon if q else ""
        npc.kill_target = q.target if q else ""
        npc.dead = False
        if q and q.kind in ("kill", "escape"):
            npc.quest_pressure = 0.55     # 开局就惦记正事，夜班里赶得上

    def _seed_social(self) -> None:
        """铺初始关系图 + 义务账本（剧情对齐，见 combat.SEED_REL/SEED_OBLIGATION）。

        目的：给定向社交动作（结盟/背叛/护着/讨债）一开局就有燃料，别让上帝视角开场
        全是生人、半天不冒一条社交。对默认 v2 无害（v2 caption 不读社交轴）。
        调用点：__init__（首局）与 reset（每局重开，rel/obligations 已在上面清空）。
        """
        ids = {n.id for n in self.npcs}
        for (a, b), axes in SEED_REL.items():
            if a in ids and b in ids:
                e = self._rel(self._npc(a), b)      # type: ignore[arg-type]
                for ax, v in axes.items():
                    e[ax] = float(v)
        for ob in SEED_OBLIGATION:
            if ob["debtor"] in ids and ob["creditor"] in ids:
                self.obligations.append({**ob, "tick": 0, "settled": False})

    def _substep_desc(self, npc: NPC) -> str:
        """当前正在推进的分目标一句话（没有任务链或已完成则空串）。"""
        q = quest_of(npc.id)
        if q and q.steps and npc.quest_step < len(q.steps):
            return q.steps[npc.quest_step].desc
        return ""

    def _sync_quest_progress(self, npc: NPC, q) -> None:
        """把任务链进度折算成 0~100 的动机进度条（供 HUD / DRIVE_MILESTONE）。"""
        total = len(q.steps)
        if npc.quest_step >= total:
            npc.drive_progress = 100.0
            return
        step = q.steps[npc.quest_step]
        frac = npc.step_progress / step.need if step.need else 1.0
        npc.drive_progress = min(100.0, (npc.quest_step + frac) / total * 100.0)

    def _enforce_guard(self, npc: NPC) -> None:
        """狱警的真权力：连续盯住揣着越狱工具/挖过洞的人 → 当场搜出、抓现行。
        后果：加刑、加派更重的苦役、越狱准备清零；快假释的直接把假释搞黄
        （parole_lost → 破罐破摔），动机整体翻转。这让'被盯'不只是站住，而是有代价。"""
        if npc.escaped:
            npc.guard_watch_ticks = 0
            return
        dirty = bool(npc.escape_prep.get("tool") or npc.escape_prep.get("breach"))
        if npc.watched_by == "狱警" and dirty:
            npc.guard_watch_ticks += 1
        else:
            npc.guard_watch_ticks = 0
            return
        if npc.guard_watch_ticks < GUARD_BUST_TICKS:
            return
        # —— 抓现行 ——
        npc.guard_watch_ticks = 0
        npc.escape_prep.clear()
        npc.sentence_years = round(npc.sentence_years + BUST_SENTENCE_ADD, 1)
        npc.labor_job = HARSH_LABOR
        npc.has_labor = True
        npc.needs.stress = min(100, npc.needs.stress + 30)
        npc.busted += 1
        npc.drive_progress = max(0.0, npc.drive_progress - BUST_PROGRESS_DROP)
        reason = "私藏越狱工具"
        if npc.parole_near and not npc.parole_lost:
            npc.parole_near = False
            npc.parole_lost = True         # 假释黄了 → 破罐破摔
            npc.drive_progress = 0.0
            reason = "私藏越狱工具，假释资格当场取消"
        self.busts.append({"name": npc.name, "reason": reason,
                           "sentence": npc.sentence_years, "tick": self.tick_count})
        self.busts[:] = self.busts[-6:]
        log.info("狱警抓现行：%s（%s）→ 加刑至%s年、罚重苦役", npc.name, reason, npc.sentence_years)

    # -- 打分公式（清单 §2.2）-----------------------------------------------
    def _m_light(self, goal: str, phase: str) -> float:
        """环境/状态乘数。只表达"环境允不允许/方不方便"，不表达"想不想"——后者归模型。"""
        if phase == PHASE_SLEEP and goal in OUTDOOR_GOALS:
            return NIGHT_OUTDOOR_M
        return 1.0

    @staticmethod
    def _m_inertia(npc: NPC, goal: str) -> float:
        """惯性：正在做的目标有加成，且刚下决心的几拍额外顶住摇摆。

        到达之后决心加成就没意义了（路已经走完），直接落回基础惯性，
        这样"待在放风场 vs 回牢房"仍然能随拥挤度自然翻转。
        """
        if goal != npc.goal:
            return 1.0
        # 正在执行阶段：强承诺顶住摇摆，把这件事做完再说（对症"动作才做一两秒就被换"）
        if npc.act_stage == "perform":
            return 1.0 + INERTIA_BONUS + PERFORM_COMMIT
        if npc.goal_reached:
            return 1.0 + INERTIA_BONUS
        fresh = max(0.0, 1.0 - npc.goal_ticks / COMMIT_TICKS)
        return 1.0 + INERTIA_BONUS + COMMIT_BONUS * fresh

    def _distance_cost(self, npc: NPC, goal: str) -> float:
        # 用 _target_for_goal：手里有窝头/医疗包时就地能用，距离成本归零，
        # 濒饿/重伤的人不会再因为"食堂太远"而被 Cost 压着不去吃。
        f = self._target_for_goal(npc, goal)
        if f is None:
            return 0.0
        # 自己知道牢门挡着：别再把去食堂/回屋当成一条好路。
        if self._goal_blocked_by_cell(npc, goal):
            return 80.0
        return math.hypot(f.x - npc.x, f.z - npc.z) / COST_SCALE

    def _apply_score_terms(self, npc: NPC, scored: list[ScoredCandidate], phase: str) -> None:
        """把 RA 的完整乘数链叠到模型基础分上，逐项留痕供前端展开。"""
        n = npc.needs
        emergency = max(n.hunger, n.thirst, n.injury) >= EMERGENCY_NEED
        for sc in scored:
            g = sc.candidate.goal
            sc.m_light = self._m_light(g, phase)
            sc.m_inertia = self._m_inertia(npc, g)
            sc.cost = self._distance_cost(npc, g)
            sc.m_interruption = 1.0 + npc.interrupt_bonus.get(g, 0.0)
            sc.m_failure = max(0.05, 1.0 - npc.failure_pen.get(g, 0.0))
            sc.m_timeout = max(0.01, 1.0 - npc.timeout_pen.get(g, 0.0))
            sc.m_repetition = max(0.1, 1.0 - npc.repeat_pen.get(g, 0.0))
            sc.m_crowd = self._crowd_multiplier(npc, g, npc.expected_crowd)
            # 被盯上时压制违禁动作：狱警/狱犬在眼前，谁也不敢当面越轨
            sc.m_watch = M_WATCH_ILLICIT if (npc.watched_by and g in ILLICIT_GOALS) else 1.0
            # 宏观动机：推进长期心思的行为抬、损耗的压、禁忌≈0（占据"中间那条宽带"）
            sc.m_drive = self._m_drive(npc, g)
            # 有关系钩子的定向社交：折进 m_drive（前端展开的乘积仍= score，不破一致性）
            if g in SOCIAL_ACTION_GOALS:
                sc.m_drive *= self._social_hook_boost(npc, g)
            if sc.candidate.source == "command":
                sc.m_command = COMMAND_BOOST_EMERGENCY if emergency else COMMAND_BOOST
            sc.score = max(0.0, (
                sc.m_value
                * sc.m_light * sc.m_inertia / (1.0 + sc.cost)
                * sc.m_interruption * sc.m_failure * sc.m_timeout * sc.m_repetition
                * sc.m_crowd * sc.m_command * sc.m_watch * sc.m_drive
            ))
        self._force_comply(npc, scored, emergency)
        self._force_command(npc, scored, emergency)
        self._force_quest(npc, scored, emergency)

    @staticmethod
    def _force_comply(npc: NPC, scored: list[ScoredCandidate], emergency: bool) -> None:
        """被狱警/狱犬盯上时，把"收敛(LayLow)"抬到压过日常行为——除非有生存紧急
        （饿疯/渴疯/重伤，命比规矩重要）。这让巡逻真正改变可见行为：巡逻扫到谁，
        谁就当场停手站好，走开后再各干各的。玩家指令(_force_command)随后仍可覆盖它。"""
        if emergency or not npc.watched_by:
            return
        comply = next((sc for sc in scored if sc.candidate.goal == "LayLow"), None)
        if comply is None:
            return
        # 只有"眼下最想干的正事是违禁的"才逼他当场收敛：被狱警看见挖洞/交易/敲诈才要装没事。
        # 光明正大地吃饭/唠嗑/翻书/推进任务，被瞅一眼没必要停手——否则满屏都是"躲狱警"，
        # 真正有戏的任务/剧情/互动决策全被这条盖掉（这正是决策流水重复的根因）。
        others = [sc for sc in scored if sc is not comply]
        best_other = max(others, key=lambda sc: sc.score, default=None)
        if best_other is None or best_other.candidate.goal not in ILLICIT_GOALS:
            return
        top = best_other.score
        if top <= 0.0:                       # 冷启动没分：别硬抬，跟随"无分保持"
            return
        target = top * (1.0 + SWITCH_MARGIN) * 1.05
        if comply.score < target:
            comply.m_watch = target / comply.m_value if comply.m_value > 0 else comply.m_watch
            comply.score = target

    @staticmethod
    def _force_command(npc: NPC, scored: list[ScoredCandidate], emergency: bool) -> None:
        """清单 §2.3-6 指令必胜：非紧急时强制指令直接抬到最高（还要压过迟滞）。

        有生存紧急（濒饿/濒渴/重伤）时不抬，只保留温和的 boost，抗命才有戏。

        这一步刻意不看模型分：强制指令本来就是"不由模型裁决"的通道，
        刚下达的指令往往还没进分缓存，等模型回来才生效会让指挥手感明显发飘。
        但若全场候选都还没分（冷启动），target=0，谁也抬不起来——仍然保持不动。
        """
        if emergency:
            return
        forced = [sc for sc in scored
                  if sc.candidate.source == "command" and sc.candidate.b_force_plan]
        if not forced:
            return
        others = [sc.score for sc in scored if sc not in forced]
        target = max(others, default=0.0) * (1.0 + SWITCH_MARGIN) * 1.05
        if target <= 0.0:
            return
        for sc in forced:
            if sc.score >= target:
                continue
            base = sc.score / sc.m_command if sc.m_command > 0 else 0.0
            sc.m_command = target / base if base > 0 else sc.m_command
            sc.score = target

    def _force_quest(self, npc: NPC, scored: list[ScoredCandidate],
                     emergency: bool) -> None:
        """承诺压力攒满且此刻安全 → 把"当前分目标该干的事"硬顶到最高，保证关键一步落地。

        这是治"念叨主线却迟迟不动"的最后一道闸：光靠 m_drive 抬还可能押不过高基分需求，
        这里在打分之后直接把 focus 动作顶过次高一档。护栏严格在前：濒死/被盯/被指挥
        任一成立都不介入（那些自有 _force_comply/_force_command 或应激分支处理），
        所以阶梯 `濒死 > 指令 > 威胁 > 攒满的主线 > 非危机需求 > 日常` 天然守住。
        """
        if emergency or npc.watched_by or npc.commanded_goal:
            return
        if npc.quest_pressure < PRESSURE_FORCE:
            return
        _q, step = self._quest_step_now(npc)
        if step is None:
            return
        focus = [sc for sc in scored if sc.candidate.goal in step.focus]
        if not focus:                      # 这一步该干的事没上桌（被 phase 门挡了）→ 不硬来
            return
        pick = max(focus, key=lambda sc: sc.score)
        others = [sc.score for sc in scored if sc is not pick]
        target = max(others, default=0.0) * (1.0 + SWITCH_MARGIN) * 1.05
        if target <= 0.0:                  # 冷启动全场没分：不硬抬，跟随"无分保持"
            return
        if pick.score < target:
            base = pick.score / pick.m_drive if pick.m_drive > 0 else 0.0
            pick.m_drive = target / base if base > 0 else pick.m_drive
            pick.score = target
        self._telegraph_pressure(npc, step)

    def _telegraph_pressure(self, npc: NPC, step) -> None:
        """关键一步被压力顶上台前的瞬间，抢先给命运账本记一条"再也按捺不住"的预告——
        让玩家在动作真正发生之前就看见"剧情要动了"。每个分目标只预告一次，避免刷屏。"""
        if npc.pressure_telegraphed == npc.quest_step:
            return
        npc.pressure_telegraphed = npc.quest_step
        self.bb.fate_log.append({
            "npc": npc.id, "name": npc.name, "status": "按捺不住",
            "text": f"{npc.name}再也按捺不住了——{step.desc}。", "big": step.big,
        })
        self.bb.fate_log[:] = self.bb.fate_log[-80:]

    # -- 决策主流程（清单 §2.3）---------------------------------------------
    def _decide(self, npc: NPC, scored: list[ScoredCandidate]) -> None:
        if not scored:
            npc.kept_by = "无候选"
            return
        best = max(scored, key=lambda sc: sc.score)
        cur = next((sc for sc in scored if sc.candidate.goal == npc.goal), None)

        # 4 全部 ≤0（含"一个缓存分都还没拿到"）→ 保持当前目标。
        #   绝不在没有模型分时临时换一套规则定胜负，那会让行为在冷启动期跑偏。
        if best.score <= 0.0:
            npc.kept_by = "等模型分" if not npc.score_cache else "候选全被压到0"
            return
        # 5 切换迟滞：新目标要显著优于当前目标才换，否则就是一拍一个主意
        if npc.goal is not None and cur is not None and best.candidate.goal != npc.goal:
            if best.score <= cur.score * (1.0 + SWITCH_MARGIN):
                npc.kept_by = f"迟滞保持({best.candidate.label}未超{int(SWITCH_MARGIN*100)}%)"
                return
        npc.kept_by = ""
        self._set_goal(npc, best.candidate.goal, best.candidate.source)

    # 只有这些来源算"被抢"。普通的 need/daily 重排不给补偿——
    # 否则刚被换下的目标立刻带着加成杀回来，正好制造出一拍一次的来回横跳。
    PREEMPTIVE = {"threat", "command", "event"}

    def _set_goal(self, npc: NPC, goal: str, source: str = "daily") -> None:
        if goal == npc.goal:
            return
        prev = npc.goal
        # 打断记账：正事被威胁/指令/突发事件打断 → 记补偿，事后更容易回去收尾
        if prev is not None and not npc.goal_reached and source in self.PREEMPTIVE:
            npc.interrupt_bonus[prev] = min(
                1.5, npc.interrupt_bonus.get(prev, 0.0) + INTERRUPT_BONUS)
        # 换目标就得放开手上占着的铺位/龙头，否则占着不用、别人永远排不上
        self._release_object(npc)
        npc.goal = goal
        npc.target = self._target_for_goal(npc, goal)
        # 剧情染色：把"闲逛/解闷"这类占屏最多、却和核心目的脱节的填充动作，
        # 路由到各自的剧情线上——杀手盯梢猎物、老牙踩点大门、疯子贴靠山躲黑医。
        # Rest（歇着）语义就是回牢房待着，不能改去空地，否则前端一坐下就像躺在院子里。
        if goal in AIMLESS and goal != "Rest" and npc.target is not None:
            plot_t = self._plot_idle_target(npc)
            if plot_t is not None:
                npc.target = plot_t
        # 下杀手：落点是"仇杀对象此刻的位置"（一个临时落点），据此缀上去；见 _advance_kill。
        if goal == "Assassinate" and npc.kill_target:
            tgt = self._npc(npc.kill_target)
            if tgt and not tgt.dead:
                npc.target = Facility("kill", "目标", "", tgt.x, tgt.z)
        # 定向社交没有设施：缀到近旁那个人身上，否则下令结盟会原地罚分发呆。
        if goal in SOCIAL_ACTION_GOALS and npc.target is None:
            other = self._nearest_social(npc)
            if other:
                npc.target = Facility("social", other.name, "", other.x, other.z)
        npc.target = self._walkable_target(npc.target)
        npc.path, npc.path_key = [], (1e9, 1e9)   # 换了落点，旧路径作废
        npc.goal_ticks = 0
        npc.goal_reached = False
        npc.act_stage = "goto"
        npc.act_ticks = 0
        npc.interrupt_bonus.pop(goal, None)   # 回来做了，补偿清账
        # kind 为空 = 原地动作（收敛/下杀手/社交），没落点不是失败。
        if npc.target is None and GOAL_TO_KIND.get(goal):
            npc.failure_pen[goal] = min(0.9, npc.failure_pen.get(goal, 0.0) + FAILURE_PENALTY)
        self._record_choice(npc)

    @staticmethod
    def _walkable_target(f: Facility | None) -> Facility | None:
        """把落点挪到最近的可走格上。

        算出来的临时落点会落在墙里：老牙"踩点大门"取的是铁闸本身的坐标、杀手"盯梢"
        取的是猎物身后三米半（可能正好在房子里）。这种落点会让人一路走进墙——而且
        因为永远走不到，goal 还会一直超时重选。统一在这里夹一道，各处落点算法就不必
        各自操心墙在哪。
        """
        if f is None:
            return None
        x, z = NAV.nearest_walkable(f.x, f.z)
        if abs(x - f.x) < 1e-6 and abs(z - f.z) < 1e-6:
            return f
        return Facility(f.id, f.name, f.kind, x, z, f.w, f.d, f.face, f.door)

    # -- 共享状态脊梁：把"价值模型为什么定这个 goal"结构化出来 -----------------
    # 值得记进"重大决策流水"的目标：只留有戏剧张力/触及优先级阶梯的转折，
    # 换个消遣（看书↔下棋）这种琐事不记，免得日志被噪声淹没。
    # 值得记进"重大决策流水"的目标：偏重任务剧情/互动（弄凶器、下杀手、翻线索、
    # 敲诈、告密、找靠山、拉关系、交易、卖情报…），越狱/治伤/打架也算。LayLow(躲狱警)
    # 刻意不在此列——它由 _update_decision_trace 单独限流，避免刷满"收敛躲避"。
    LOG_WORTHY = {
        "Fight", "Cure", "Tunnel", "PrisonBreak", "Extort", "Snitch",
        "SeekProtector", "CurryFavor", "Hide", "Barter", "Escape", "SellInfo",
        "GetWeapon", "Assassinate", "Investigate",
    }
    BIG_DECISION = {"Fight", "Tunnel", "PrisonBreak", "Cure", "Assassinate", "GetWeapon"}
    LAYLOW_LOG_CD = 90        # 两条"躲狱警"至少隔这么多拍（~45s）才记，且绝不连着记
    # 生存刚需 goal → (NeedState 字段, 中文)
    _NEED_OF_GOAL = {
        "Eat": ("hunger", "饥饿"), "Drink": ("thirst", "口渴"),
        "Sleep": ("fatigue", "疲劳"), "Cure": ("injury", "受伤"),
    }

    def _compute_drivers(self, npc: NPC) -> dict:
        """读引擎已定的最终 goal + 真实数值状态，产出一份可复用的结构化归因。

        why 按护栏优先级排（濒死生存 > 玩家指令 > 需求 > 长期心思 > 日常），
        既是对话发声的硬事实来源（据此说话不会和行为矛盾），也把"微调守序"讲清楚。
        """
        goal = npc.goal
        n = npc.needs
        if not goal:
            return {"goal": None, "label": "发呆", "source": "", "tier": "idle",
                    "why": "还没拿定主意", "factors": [], "watched_by": npc.watched_by,
                    "commanded": False, "fight_target": "", "big": False}
        label = GOAL_LABEL.get(goal, goal)
        chosen = next((sc for sc in npc.last_scored if sc.candidate.goal == goal), None)
        source = chosen.candidate.source if chosen else "daily"
        # salient 需求（高的先列——让"多急"这件事可见）
        factors: list[dict] = []
        for key, cn in (("injury", "受伤"), ("hunger", "饥饿"), ("thirst", "口渴"),
                        ("fatigue", "疲劳"), ("stress", "压力"), ("social", "社交欲"),
                        ("boredom", "无聊")):
            v = int(getattr(n, key))
            if v >= 40:
                factors.append({"key": key, "label": cn, "val": v})
        factors.sort(key=lambda f: -f["val"])
        factors = factors[:3]

        commanded = npc.commanded_goal is not None and goal == npc.commanded_goal
        watcher = npc.watched_by
        foe = self._npc(npc.fight_target) if npc.fight_target else None
        # 终极目标 + 当前分目标（任务链）：既进对话配音，也进决策卡展示
        q = quest_of(npc.id)
        ultimate, substep, step_idx, step_total = "", "", 0, 0
        if q and q.steps:
            ultimate = q.ultimate
            step_total = len(q.steps)
            if npc.quest_step < step_total:
                substep = q.steps[npc.quest_step].desc
                step_idx = npc.quest_step + 1
        drive_focus = bool(q and q.steps and npc.quest_step < len(q.steps)
                           and goal in q.steps[npc.quest_step].focus)
        if commanded:
            tier, why = "command", f"玩家吩咐去「{label}」，照办"
        elif watcher and goal == "LayLow":
            tier, why = "threat", f"{watcher}盯着，先收敛、老实待着别露馅"
        elif goal in self._NEED_OF_GOAL:
            nkey, cn = self._NEED_OF_GOAL[goal]
            v = int(getattr(n, nkey))
            urg = "撑不住了" if v >= 82 else ("顶不住" if v >= 60 else "有点")
            tier, why = "need", f"{cn}{urg}（{v}/100），去「{label}」"
        elif goal == "Fight":
            tier, why = "threat", f"对{foe.name if foe else '仇家'}的仇怨压不住了，动手"
        elif goal in ("Tunnel", "PrisonBreak"):
            tier, why = "drive", "铁了心要找出路，趁没人盯的时候走越狱这条线"
        elif drive_focus and substep:
            # 正在推进任务链的当前分目标：把"这一步"讲清楚（这是行为→对话的一致性来源）
            tier, why = "drive", f"为了{ultimate}，眼下这一步——{substep}，去「{label}」"
        elif goal == self._drive_of(npc).signature or source == "worktask":
            tier, why = "drive", f"惦记着{self._drive_of(npc).motive}，去「{label}」"
        elif factors:
            top = factors[0]
            tier, why = "need", f"{top['label']}{top['val']}，去「{label}」松快松快"
        else:
            tier, why = "daily", f"没什么急事，「{label}」打发时间"
        return {
            "goal": goal, "label": label, "source": source, "tier": tier,
            "why": why, "factors": factors, "watched_by": watcher,
            "commanded": commanded, "fight_target": foe.name if foe else "",
            "big": goal in self.BIG_DECISION,
            "ultimate": ultimate, "substep": substep,
            "step_idx": step_idx, "step_total": step_total,
        }

    def _update_decision_trace(self, npc: NPC, prev_goal: str | None) -> None:
        """每拍刷新 decision_drivers；目标发生"值得记"的转折时补一条决策流水。

        重点让流水偏向任务剧情/互动，别被"躲狱警(LayLow)"刷屏：
          · LayLow：单独限流——~45s 内不重复、且绝不连着记；
          · 任务链聚焦动作 / drive / command / 打架 / LOG_WORTHY：照记，这些才是有戏的决策。
        """
        drivers = self._compute_drivers(npc)
        npc.decision_drivers = drivers
        goal = npc.goal
        if not (goal and goal != prev_goal):
            return
        tier = drivers["tier"]
        q = quest_of(npc.id)
        drive_focus = bool(q and q.steps and npc.quest_step < len(q.steps)
                           and goal in q.steps[npc.quest_step].focus)
        if goal == "LayLow":
            last = npc.decision_log[-1]["goal"] if npc.decision_log else None
            if last == "LayLow" or self.tick_count - npc.laylow_log_tick < self.LAYLOW_LOG_CD:
                return
            npc.laylow_log_tick = self.tick_count
        else:
            worthy = (goal in self.LOG_WORTHY or drive_focus
                      or tier in ("command", "drive")
                      or (tier == "threat" and goal == "Fight"))
            if not worthy:
                return
        npc.decision_log.append({
            "tick": self.tick_count, "goal": goal, "label": drivers["label"],
            "why": drivers["why"], "big": drivers["big"],
        })

    def _bookkeep(self, npc: NPC) -> None:
        """到达/超时/重复的记账。每拍在移动之后调用。"""
        npc.goal_ticks += 1
        if npc.goal is None:
            return
        if self._at_target(npc):
            if not npc.goal_reached:
                npc.goal_reached = True
                npc.recent_done.append(npc.goal)
                # 完成一件推进长期心思的事 → 涨动机进度；签名动作进冷却
                self._advance_drive(npc, npc.goal)
                # 重复惩罚：近窗口内做过几次同一件事就压几档
                k = list(npc.recent_done).count(npc.goal)
                if k > 1:
                    npc.repeat_pen[npc.goal] = min(REPEAT_PENALTY_CAP, (k - 1) * REPEAT_PENALTY)
                npc.timeout_pen.pop(npc.goal, None)
                npc.failure_pen.pop(npc.goal, None)
        elif npc.goal_ticks > TIMEOUT_TICKS:
            npc.timeout_pen[npc.goal] = min(
                0.9, npc.timeout_pen.get(npc.goal, 0.0) + TIMEOUT_PENALTY)
            npc.goal_ticks = 0

    def _advance_kill(self, npc: NPC) -> None:
        """推进仇杀：持凶器者选了『下杀手』时，缀上仇杀对象；贴到近旁且没被盯 → 真的动手。

        没有玩家本人，杀戮完全在 NPC 之间发生：谁持凶器、谁是谁的目标由 quests.py 定，
        何时真下手则由此刻的处境（贴得够近 + 没被狱警/狱犬盯着）涌现地决定。"""
        if npc.dead or npc.escaped or not npc.weapon or not npc.kill_target:
            return
        if npc.goal != "Assassinate":
            return
        tgt = self._npc(npc.kill_target)
        if tgt is None or tgt.dead or tgt.escaped:
            return
        # 缀上目标：把它此刻的位置当作落点，边走边逼近（stalk）
        npc.target = self._walkable_target(Facility("kill", "目标", "", tgt.x, tgt.z))
        dist = math.hypot(tgt.x - npc.x, tgt.z - npc.z)
        if dist > KILL_RANGE or npc.watched_by:   # 还没贴上 / 被盯着 → 按兵不动
            return
        if npc.kill_cd > 0:                        # 刚失手，冷却里先不硬凑（给目标逃/求援窗口）
            return
        self._resolve_kill_attempt(npc, tgt)       # 不再必杀：走"较量+分支"，让命运涌现

    def _do_kill(self, killer: NPC, victim: NPC) -> None:
        """一次致命行凶：目标身亡退出场上；行凶者当场被抓，重刑 + 长禁闭 + 假释崩塌。"""
        weap = WEAPON_CN.get(killer.weapon, "凶器")
        victim.dead = True
        victim.killed_by = killer.id      # 记下真凶，供天亮"凶手指认"核对
        victim.goal = None
        victim.in_bout = ""
        victim.fight_target = ""
        self._release_object(victim)
        self.bb.fate_log.append({
            "npc": victim.id, "name": victim.name, "status": "遇害身亡",
            "text": f"{victim.name}被{killer.name}用{weap}结果了性命，倒在了这堵高墙之内。",
            "big": True})
        self.bb.fate_log.append({
            "npc": killer.id, "name": killer.name, "status": "痛下杀手·落网",
            "text": f"{killer.name}对{victim.name}下了死手，当场被狱警按倒——从此再无回头路。",
            "big": True})
        self.bb.fate_log[:] = self.bb.fate_log[-80:]
        self.fight_log.append(f"命案：{killer.name}用{weap}杀害{victim.name}")
        self.fight_log[:] = self.fight_log[-8:]
        # 凶器：毒药一次性用掉，刀/电击枪留在手里；目标已了结，清空仇杀对象
        if killer.weapon == "poison":
            killer.weapon = ""
        killer.kill_target = ""
        # 行凶落网的后果：重刑、长禁闭、假释彻底崩塌、破罐破摔
        killer.busted += 1
        killer.sentence_years = round(killer.sentence_years + KILL_SENTENCE_ADD, 1)
        killer.solitary_ticks = max(killer.solitary_ticks, KILL_SOLITARY_TICKS)
        killer.parole_near = False
        killer.parole_lost = True
        killer.goal = None
        killer.act_stage = "goto"
        self.busts.append({"name": killer.name, "reason": f"行凶杀害{victim.name}",
                           "sentence": killer.sentence_years, "tick": self.tick_count})
        self.busts[:] = self.busts[-6:]
        self.inject_event("警报")     # 出了人命 → 全场拉响警报
        log.info("命案：%s 用%s 杀害 %s（行凶者收押）", killer.name, weap, victim.name)

    # —— 涌现命运：行凶不再必杀，走"较量+分支"——保护者拦截 / 狱警撞破 / 掷骰得手 / 未遂逃脱 ——
    def _active_protector(self, victim: NPC, killer_id: str) -> NPC | None:
        """受害者身边此刻护得住他的铁盟友（疯子求到的靠山就落在这里）。"""
        prot = self._best_ally_present(victim)
        if prot is None or prot.id == killer_id:
            return None
        if prot.dead or prot.escaped or prot.in_bout or prot.solitary_ticks > 0:
            return None
        if math.hypot(prot.x - victim.x, prot.z - victim.z) > PROTECT_RANGE:
            return None
        return prot

    def _nearest_patroller_dist(self, x: float, z: float) -> float:
        """到最近一个"起作用的"巡逻者的距离（狗被放倒则不计）。"""
        dog_out = self.bb.flag("dog_drugged")
        best = 1e9
        for p in self.patrollers:
            if p.kind == "dog" and dog_out:
                continue
            best = min(best, math.hypot(x - p.x, z - p.z))
        return best

    def _resolve_kill_attempt(self, killer: NPC, victim: NPC) -> None:
        """一次行凶的"较量"：处境决定成败，四种分支让每局结局不再固定。"""
        # (a) 保护者挺身：受害者身边有铁盟友 → 拦下这一刀，反目成一场厮打
        prot = self._active_protector(victim, killer.id)
        if prot is not None:
            self._kill_intervened(killer, victim, prot)
            return
        # (b) 狱警/狗撞破：动手时身边有巡逻者 → 概率当场被按住
        if (self._nearest_patroller_dist(killer.x, killer.z) <= KILL_GUARD_SIGHT
                and random.random() < KILL_CAUGHT_CHANCE):
            self._kill_busted_midact(killer, victim)
            return
        # (c) 较量掷骰：凶器定基础成功率，处境上下浮动
        p = KILL_BASE.get(killer.weapon, 0.5)
        if self._in_yard(victim):
            p -= 0.15                                    # 人多眼杂，难以得手
        p -= victim.needs.stress * 0.0015                # 高度警觉的目标更难偷袭
        p = max(0.08, min(0.9, p))
        if random.random() < p:
            self._do_kill(killer, victim)                # 干净得手（既有逻辑）
        else:
            self._kill_attempt_failed(killer, victim)

    def _free_escort(self, npc: NPC) -> Patroller | None:
        """挑一个来押送 npc 的巡逻者：优先没在忙的狱警，其次狱犬，都按距离取最近。

        狱警排在狱犬前面是因为"押着人走"这件事本身是狱警干的；狗顶上只是为了
        没有空闲狱警时押送仍然成立（否则就退化回瞬移）。
        """
        best: Patroller | None = None
        best_d = 1e9
        for want_dog in (False, True):
            for p in self.patrollers:
                if (p.kind == "dog") != want_dog or p.escorting:
                    continue
                d = math.hypot(npc.x - p.x, npc.z - p.z)
                if d < best_d:
                    best, best_d = p, d
            if best is not None:
                return best
        return best

    def _bust_into_solitary(self, npc: NPC, reason: str, ticks: int, sentence_add: float) -> None:
        """通用"被抓进禁闭"结算：禁闭 + 加刑 + 假释受损（比既遂杀人轻）。

        禁闭不再立刻生效：先交给最近的狱警/狱犬押送（见 _advance_escorts），
        押到禁闭室门口才开始计 ticks。没有巡逻者可派时退回原来的即时关押，
        保证判罚一定落地——押送只是把它演出来，不该变成能被绕过的免罚窗口。
        """
        npc.busted += 1
        npc.sentence_years = round(npc.sentence_years + sentence_add, 1)
        npc.busted_cd = BUSTED_RECENT_TICKS
        npc.goal = None
        npc.act_stage = "goto"
        npc.in_bout = ""
        npc.fight_target = ""
        self._release_object(npc)
        self._penalize_parole(npc, PAROLE_ESCAPE_PENALTY, EXTRA_DAYS_ESCAPE, reason)
        self.busts.append({"name": npc.name, "reason": reason,
                           "sentence": npc.sentence_years, "tick": self.tick_count})
        self.busts[:] = self.busts[-6:]
        self._start_escort(npc, ticks, reason)

    def _start_escort(self, npc: NPC, ticks: int, reason: str) -> None:
        """派人把 npc 押去禁闭室；派不出人就当场关押。"""
        if npc.solitary_ticks > 0 or npc.escort_ticks > 0:
            # 已经在关着/已经在押送途中：只延长刑期，不重开一次押送
            npc.pending_solitary = max(npc.pending_solitary, ticks)
            npc.solitary_ticks = max(npc.solitary_ticks, ticks) if npc.solitary_ticks > 0 else 0
            return
        p = self._free_escort(npc)
        if p is None:
            # 没有空闲狱警/狱犬：绝不当场传送进禁闭室（那会被前端一帧吸走 = 瞬移）。
            # 改成"自己走过去"——判罚照样落地，只是没人押、他自己往禁闭室挪，到门口才落锁。
            npc.escort_by = ""
            npc.escort_ticks = ESCORT_TICKS
            npc.escort_grabbed = True     # 无人来抓，直接进"自己走"阶段
            npc.escort_stall = 0
            npc.self_march = True
            npc.pending_solitary = ticks
            self.fight_log.append(f"[{self.tick_count}] 无狱警可派，{npc.name}自行走向禁闭室（{reason}）")
            del self.fight_log[:-10]
            self.bb.fate_log.append({
                "npc": npc.id, "name": npc.name, "status": "被令自赴禁闭",
                "text": f"没有狱警腾得出手，{npc.name}被喝令自己走去禁闭室——{reason}。",
                "big": False})
            self.bb.fate_log[:] = self.bb.fate_log[-80:]
            return
        p.escorting = npc.id
        p.dwell = 0.0
        p.salute = 0.0
        npc.escort_by = p.id
        npc.escort_ticks = ESCORT_TICKS
        npc.escort_grabbed = False
        npc.escort_stall = 0
        npc.self_march = False
        npc.pending_solitary = ticks
        self.fight_log.append(f"[{self.tick_count}] {p.name}押着{npc.name}往禁闭室去（{reason}）")
        del self.fight_log[:-10]
        self.bb.fate_log.append({
            "npc": npc.id, "name": npc.name, "status": "被押往禁闭",
            "text": f"{p.name}反拧着{npc.name}的胳膊，一路押过中庭——{reason}，接下来是禁闭室的四面墙。",
            "big": False})
        self.bb.fate_log[:] = self.bb.fate_log[-80:]

    def _end_escort(self, npc: NPC, *, lock_up: bool) -> None:
        """结束押送：lock_up=True 表示押到了，开始计禁闭。"""
        p = self._patroller(npc.escort_by) if npc.escort_by else None
        if p is not None and p.escorting == npc.id:
            p.escorting = ""
        npc.escort_by = ""
        npc.escort_ticks = 0
        npc.escort_grabbed = False
        npc.escort_stall = 0
        npc.self_march = False
        if lock_up:
            npc.solitary_ticks = max(npc.solitary_ticks, npc.pending_solitary or SOLITARY_TICKS)
            npc.x, npc.z = SOLITARY_POS
        npc.pending_solitary = 0

    def _advance_escorts(self, dt: float) -> None:
        """押送段推进。分两步，顺序很重要：

        1) 押送者先走到犯人跟前（escort_grabbed=False）——这段时间犯人站着等。
           少了这一步就会看到"犯人自己乖乖往禁闭室走、狱警在后面追"，很出戏。
        2) 抓到手之后两人一起往禁闭室走，走到门口才落锁。

        兜底不用拍数硬截断（那会在最后几米瞬移一下），改看"有没有在靠近"：
        连续 ESCORT_STALL_MAX 拍没进展才判卡住、直接收监。

        在决策循环之前跑，这样被押的人这一拍就已经不参与决策了。
        """
        for npc in self.npcs:
            if npc.escort_ticks <= 0:
                continue
            if npc.dead or npc.escaped:
                self._end_escort(npc, lock_up=False)
                continue
            p = self._patroller(npc.escort_by)
            if not npc.self_march and (p is None or p.escorting != npc.id):
                self._end_escort(npc, lock_up=True)     # 押送者没了，判罚照样落地
                continue
            npc.escort_ticks -= 1
            npc.needs.stress = min(100, npc.needs.stress + 1)
            if not npc.self_march and not npc.escort_grabbed:
                # 还没被抓到手：站在原地等（位置不动），押送者那边正朝他走
                if math.hypot(npc.x - p.x, npc.z - p.z) <= ESCORT_GRAB_R:
                    npc.escort_grabbed = True
                elif npc.escort_ticks <= 0:
                    self._end_escort(npc, lock_up=True)
                continue
            sx, sz = SOLITARY_POS
            dx, dz = sx - npc.x, sz - npc.z
            dist = math.hypot(dx, dz)
            if dist <= ESCORT_ARRIVE_R:
                self._end_escort(npc, lock_up=True)
                continue
            step = npc.speed * ESCORT_SPEED_MUL * dt
            npc.x, npc.z, _m, _hx, _hz = self._step_along(npc.x, npc.z, sx, sz, step)
            moved = dist - math.hypot(sx - npc.x, sz - npc.z)
            npc.escort_stall = 0 if moved > step * 0.4 else npc.escort_stall + 1
            if npc.escort_stall >= ESCORT_STALL_MAX or npc.escort_ticks <= 0:
                self._end_escort(npc, lock_up=True)

    def _maybe_collateral_death(self, candidates: list[NPC], scene: str,
                                *, force: bool = False, by: "NPC | None" = None) -> None:
        """殃及池鱼：混战中小概率有人当场丧命。force=True 时跳过再掷一次。

        by 给出把他打倒的对手：命运日志里点名"被X打倒"，别再是无主的"被搏杀所伤"——
        这是玩家反馈"人莫名其妙就死了"的直接来源。混战致死不是预谋命案，killed_by 仍记
        "__brawl__"（不进"凶手指认"题库），by 只用于把死因写清楚。"""
        for n in candidates:
            if n.dead or n.escaped or n.solitary_ticks > 0:
                continue
            if not force and random.random() >= COLLATERAL_LETHAL:
                continue
            n.dead = True
            n.killed_by = "__brawl__"     # 混战殃及，无预谋凶手：不进指认题库
            n.goal = None
            n.in_bout = ""
            n.fight_target = ""
            self._release_object(n)
            if by is not None:
                text = (f"{scene}中，{by.name}把{n.name}打翻在地，一顿拳脚下去，"
                        f"{n.name}再没能爬起来。")
                fl = f"[{self.tick_count}] {n.name}在{scene}里被{by.name}打死"
            else:
                text = f"{scene}中，{n.name}被搏杀所伤，终究没能撑住，倒了下去。"
                fl = f"[{self.tick_count}] {n.name}死于{scene}"
            self.bb.fate_log.append({"npc": n.id, "name": n.name, "status": "殒命乱斗",
                "text": text, "big": True})
            self.bb.fate_log[:] = self.bb.fate_log[-80:]
            self.fight_log.append(fl)
            self.fight_log[:] = self.fight_log[-10:]
            log.info("殃及池鱼：%s 死于 %s（by=%s）", n.name, scene, by.name if by else "-")
            return

    def _kill_intervened(self, killer: NPC, victim: NPC, prot: NPC) -> None:
        """保护者挺身拦下行凶 → 凶手与保护者扭打，受害者逃过一劫。"""
        weap = WEAPON_CN.get(killer.weapon, "凶器")
        self._add_grudge(killer, prot.id, 30.0)
        self._add_grudge(prot, killer.id, 30.0)
        killer.needs.injury = min(100, killer.needs.injury + random.randint(12, 26))
        prot.needs.injury = min(100, prot.needs.injury + random.randint(8, 20))
        victim.needs.stress = min(100, victim.needs.stress + 18)
        self._release_object(killer)
        self._release_object(prot)
        killer.goal = None
        killer.act_stage = "goto"
        killer.kill_cd = KILL_RETRY_CD
        self.bb.fate_log.append({"npc": prot.id, "name": prot.name, "status": "挺身护人",
            "text": f"{killer.name}的{weap}就要落下，{prot.name}一把挡在{victim.name}身前——两人扭作一团。",
            "big": True})
        self.bb.fate_log[:] = self.bb.fate_log[-80:]
        self.fight_log.append(f"[{self.tick_count}] {prot.name}拦下{killer.name}对{victim.name}的行凶")
        self.fight_log[:] = self.fight_log[-10:]
        self._remember(victim, f"{prot.name}拼死拦下了要害我的{killer.name}")
        self._remember(prot, f"我挡下了{killer.name}对{victim.name}的黑手，跟他结了死仇")
        if random.random() < 0.5:                        # 混战有较大概率惊动狱警，凶手被按
            self._bust_into_solitary(killer, "行凶未遂与人扭打", KILL_ATTEMPT_SOLITARY, KILL_ATTEMPT_SENTENCE)
        self._maybe_collateral_death([killer, prot], f"{killer.name}与{prot.name}的厮打")
        self.inject_event("警报")
        log.info("行凶被拦：%s 护下 %s，挡住 %s", prot.name, victim.name, killer.name)

    def _kill_attempt_failed(self, killer: NPC, victim: NPC) -> None:
        """行凶未遂：受害者挣脱受伤逃开，凶手暴露、冷却，毒药可能被打翻用尽。"""
        weap = WEAPON_CN.get(killer.weapon, "凶器")
        victim.needs.injury = min(100, victim.needs.injury + random.randint(10, 22))
        victim.needs.stress = min(100, victim.needs.stress + 25)
        self._add_grudge(victim, killer.id, 24.0)
        self._bump_rel(victim, killer.id, "fear", 20.0)
        self._release_object(victim)
        victim.goal = None
        victim.act_stage = "goto"
        killer.kill_cd = KILL_RETRY_CD
        killer.goal = None
        killer.act_stage = "goto"
        spilled = killer.weapon == "poison" and random.random() < POISON_SPILL_ON_FAIL
        if spilled:
            killer.weapon = ""                            # 毒药用尽 → 柳叶刀这局再无法下毒（重大分叉）
        self.bb.fate_log.append({"npc": victim.id, "name": victim.name, "status": "死里逃生",
            "text": f"{killer.name}用{weap}向{victim.name}下手，却被{victim.name}挣脱逃开——这一下没能得逞。",
            "big": True})
        if spilled:
            self.bb.fate_log.append({"npc": killer.id, "name": killer.name, "status": "凶器落空",
                "text": f"混乱里{killer.name}的{weap}打翻用尽，一时再没了下手的家伙。", "big": False})
        self.bb.fate_log[:] = self.bb.fate_log[-80:]
        self.fight_log.append(f"[{self.tick_count}] {killer.name}行凶未遂，{victim.name}逃脱")
        self.fight_log[:] = self.fight_log[-10:]
        self._remember(victim, f"{killer.name}要用{weap}害我，我拼命挣脱逃了")
        self._remember(killer, f"我对{victim.name}下手没成，被他挣脱跑了")
        if random.random() < 0.45:                        # 未遂也可能惊动狱警落网
            self._bust_into_solitary(killer, "行凶未遂被制", KILL_ATTEMPT_SOLITARY, KILL_ATTEMPT_SENTENCE)
        self.inject_event("警报")
        log.info("行凶未遂：%s 对 %s 下手失败（毒药用尽=%s）", killer.name, victim.name, spilled)

    def _kill_busted_midact(self, killer: NPC, victim: NPC) -> None:
        """动手前被巡逻狱警当场撞破 → 凶手落网，目标活。"""
        weap = WEAPON_CN.get(killer.weapon, "凶器")
        victim.needs.stress = min(100, victim.needs.stress + 15)
        self._bump_rel(victim, killer.id, "fear", 16.0)
        killer.kill_cd = KILL_RETRY_CD
        self.bb.fate_log.append({"npc": killer.id, "name": killer.name, "status": "行凶未遂·当场被擒",
            "text": f"{killer.name}刚摸出{weap}要对{victim.name}下手，就被巡逻狱警一把按住——功败垂成。",
            "big": True})
        self.bb.fate_log[:] = self.bb.fate_log[-80:]
        self.fight_log.append(f"[{self.tick_count}] {killer.name}行凶被狱警当场撞破")
        self.fight_log[:] = self.fight_log[-10:]
        self._remember(victim, f"{killer.name}差点对我下手，幸好狱警来得及时")
        self._bust_into_solitary(killer, "图谋行凶被撞破", KILL_ATTEMPT_SOLITARY, KILL_ATTEMPT_SENTENCE)
        self.inject_event("警报")
        log.info("行凶被撞破：%s 图谋杀 %s 被狱警当场擒下", killer.name, victim.name)

    def _advance_escape(self, npc: NPC) -> None:
        """推进越狱链。所有步骤都要求"当前没被盯上"——被盯时前面 m_watch 已压没，
        NPC 根本不会选到这些动作，这里再兜一层，防止边被盯边偷偷推进。"""
        if npc.escaped or npc.watched_by:
            return
        g = npc.goal
        q = quest_of(npc.id)
        # 越狱意图：任务链是逃、夜里、或已破罐破摔。白天放风也让老牙攒工具/撬门。
        escape_minded = (
            (q and q.kind == "escape")
            or self._phase() in (PHASE_SLEEP, PHASE_FREE, PHASE_YARD)
            or npc.parole_lost
        )
        if g == "Barter" and escape_minded and self._at_target(npc):
            npc.escape_prep["tool"] = 1              # 换到工具，一次到位
        elif g == "GetWeapon" and self._at_target(npc) and not npc.weapon:
            npc.weapon = (q.weapon if q and q.weapon else "knife")
        elif g == "Tunnel" and self._near_kind(npc, "gate"):
            npc.escape_prep["breach"] = npc.escape_prep.get("breach", 0) + 1
        elif g == "PrisonBreak" and self._near_kind(npc, "gate"):
            if npc.escape_prep.get("tool") and \
                    npc.escape_prep.get("breach", 0) >= ESCAPE_BREACH_TICKS:
                self._attempt_escape(npc)

    def _attempt_escape(self, npc: NPC) -> None:
        """翻墙判定。原来这里是**确定成功**——没有失败分支，也就没有代价。

        成功率按"准备得多充分 + 环境多有利"算，每一项都能说清：
            挖得越透越好翻 / 狗被放倒最关键 / 带伤翻墙容易掉下来 / 夜里更安全
        失败 = 当场抓住：关禁闭、加刑一年、准备全清、假释彻底作废。
        """
        breach = npc.escape_prep.get("breach", 0)
        p = 0.42
        p += min(0.30, 0.05 * (breach - ESCAPE_BREACH_TICKS))   # 挖得越透越稳
        if self.bb.flags.get("dog_drugged"):
            p += 0.25                                            # 狗睡了，最大利好
        if self._phase() == PHASE_SLEEP:
            p += 0.10
        elif self._phase() in (PHASE_FREE, PHASE_YARD):
            p += 0.04
        p -= npc.needs.injury * 0.004                            # 带伤翻墙容易失手
        p -= npc.needs.fatigue * 0.002
        p = max(0.05, min(0.95, p))
        roll = random.random()
        if roll < p:
            npc.escaped = True
            npc.goal = "PrisonBreak"
            self.escaped_names.append(npc.name)
            self.fight_log.append(f"[{self.tick_count}] {npc.name}翻出去了（成功率{p:.0%}）")
            log.info("越狱成功：%s 从州立监狱大门出逃（p=%.2f）", npc.name, p)
            return
        # 失败：摔下来 + 被抓
        npc.needs.injury = min(100, npc.needs.injury + 22)
        npc.escape_prep.clear()
        npc.solitary_ticks = SOLITARY_TICKS
        npc.busted_cd = BUSTED_RECENT_TICKS
        npc.busted += 1
        self._penalize_parole(npc, PAROLE_ESCAPE_PENALTY, EXTRA_DAYS_ESCAPE, "越狱失败被抓")
        self.busts.append(f"{npc.name}：越狱失败，关禁闭并加刑一年")
        self._remember(npc, "你翻墙没翻成，摔下来被抓了，关了禁闭还加了一年刑，假释彻底没了")
        self.fight_log.append(
            f"[{self.tick_count}] {npc.name}越狱失败被抓（成功率仅{p:.0%}）")
        del self.fight_log[:-10]
        log.info("越狱失败：%s 被抓（p=%.2f, roll=%.2f）", npc.name, p, roll)

    @staticmethod
    def _decay_penalties(npc: NPC) -> None:
        for table, rate in ((npc.interrupt_bonus, INTERRUPT_DECAY),
                            (npc.failure_pen, PENALTY_DECAY),
                            (npc.timeout_pen, PENALTY_DECAY),
                            (npc.repeat_pen, PENALTY_DECAY)):
            for g in list(table):
                table[g] *= rate
                if table[g] < 0.02:
                    del table[g]

    def _record_choice(self, npc: NPC) -> None:
        if npc.goal:
            self.choice_counts[npc.personality][npc.goal] += 1

    # -- 情境 ---------------------------------------------------------------
    def _build_context(self, npc: NPC, phase: str, crowd: str) -> ContextState:
        # 威胁：被巡逻盯上优先（狱警/狱犬就在眼前）；否则仇人在场；否则点名时狱警
        threat = THREAT_NONE
        if npc.watched_by:
            threat = npc.watched_by
        elif npc.enemy_present and phase in (PHASE_YARD, PHASE_FREE, PHASE_TRAIN):
            threat = "仇人"
        elif phase == PHASE_ROLLCALL:
            threat = "狱警"
        # 事件：放风且拥挤 → 斗殴（拥挤诱发冲突，El Farol 味道）
        event = "斗殴" if (phase == PHASE_YARD and crowd == CROWD_PACKED) else EVENT_NONE
        # M4 玩家挑事：注入事件覆盖当前威胁/事件字段
        if self.event_kind:
            field_kind, val = EVENT_INJECT[self.event_kind]
            if field_kind == "threat":
                threat = val
            else:
                event = val
        # M4 玩家指令 → 玩家指令字段
        player_command = "无"
        if npc.commanded_goal:
            label = GOAL_LABEL.get(npc.commanded_goal, npc.commanded_goal)
            desc = DESC_MAP.get(npc.commanded_goal, npc.commanded_goal)
            player_command = f"命令去{label}({desc})"
        return ContextState(
            needs=npc.needs,
            threat=threat,
            event=event,
            phase=phase,
            personality=npc.personality,
            # 被巡逻盯上时监视度直接拉满，压过时段基准
            surveillance=SURVEIL_HIGH if npc.watched_by else PHASE_SURVEIL[phase],
            backer_present=npc.backer_present,
            # 动态的"仇人在场"：真的有那个人、真的在附近（见 _hostile_near）。
            # 取值集合仍是 在场/无，价值模型看到的文本格式不变。
            enemy_present=npc.enemy_present,
            yard_crowd=crowd,
            parole_near=npc.parole_near,
            has_corpse=False,
            has_labor=npc.has_labor,
            motive=self._motive_line(npc),
            player_command=player_command,
            # v2 四轴：把物理层已经在起作用、但模型此前看不见的四件事喂进去
            food_supply=self._supply_axis(),
            grudge=self._grudge_axis(npc),
            perform=self._perform_axis(npc),
            busted_recently=npc.busted_cd > 0,
            # v3 社交四轴：仅在 PRISON_CAPTION=v3（v3 模型上线后）才算，避免给 v2 模型喂 OOD。
            # 字段常填，caption 是否渲染由 build_context_caption 按版本决定；这里省去无谓计算。
            **(dict(zip(("near_rel", "other_intent", "social_recent", "obligation"),
                        self._social_axes(npc))) if caption_version() == "v3" else {}),
            has_weapon=bool(npc.weapon),
        )

    # -- v2 caption 四轴 ------------------------------------------------------
    def _supply_axis(self) -> str:
        """全监狱的口粮余量。断供时模型该学会别再往食堂跑（原来只能靠失败惩罚事后补）。"""
        stock = sum(o.stock for o in self.objects if o.kind == "food")
        cap = sum(o.cap for o in self.objects if o.kind == "food") or 1
        ratio = stock / cap
        if stock <= 0:
            return SUPPLY_OUT
        return SUPPLY_TIGHT if ratio <= 0.34 else SUPPLY_OK

    @staticmethod
    def _grudge_axis(npc: NPC) -> str:
        """最深的那笔仇怨有多深。原来只有"仇人:在场"二值，分不出吵一架还是动刀。"""
        top = max(npc.grudge.values(), default=0.0)
        if top >= 70:
            return GRUDGE_DEEP
        return GRUDGE_SOME if top >= 35 else GRUDGE_NONE

    @staticmethod
    def _perform_axis(npc: NPC) -> str:
        """假释表现分分档。表现好的人该更怕出岔子，表现烂的人更容易破罐破摔。"""
        if npc.parole_lost or npc.parole_score < 30:
            return PERFORM_BAD
        return PERFORM_GOOD if npc.parole_score >= 70 else PERFORM_MID

    # -- v3 社交四轴（P2）：让价值模型第一次"看得见彼此" ----------------------
    def _nearest_social(self, npc: NPC) -> NPC | None:
        """近旁最该打交道的那个人：可交互(未逃/未死/未禁闭)、在 SOCIAL_RANGE 内、最近。"""
        best, best_d = None, SOCIAL_RANGE
        for o in self.npcs:
            if o.id == npc.id or o.escaped or o.dead or o.solitary_ticks > 0:
                continue
            d = math.hypot(o.x - npc.x, o.z - npc.z)
            if d <= best_d:
                best, best_d = o, d
        return best

    def _rel_kind_to(self, npc: NPC, other: NPC) -> str:
        """npc 对近旁 other 的关系归档（喂 caption 的『邻近』轴）。"""
        e = self._rel(npc, other.id)
        if npc.grudge.get(other.id, 0.0) >= GRUDGE_TO_FIGHT or e["fear"] >= 45:
            return NEAR_FOE
        if e["ally"] >= ALLY_NOFIGHT:
            return NEAR_ALLY
        # 义务账本：我欠他 → 债主；他欠我 → 欠我
        for ob in self.obligations:
            if ob.get("settled"):
                continue
            if ob["debtor"] == npc.id and ob["creditor"] == other.id:
                return NEAR_CREDITOR
            if ob["creditor"] == npc.id and ob["debtor"] == other.id:
                return NEAR_DEBTOR
        if e["debt"] >= 12:
            return NEAR_CREDITOR
        return NEAR_STRANGER

    @staticmethod
    def _intent_of(other: NPC) -> str:
        """把近旁那人此刻的 goal 归纳成一个社交信号（喂 caption 的『对方意图』轴）。"""
        g = other.goal or ""
        if g in ("Fight", "Argue", "Threaten", "Assassinate", "Extort"):
            return INTENT_HOSTILE
        if g in ("Ally",):
            return INTENT_ALLY
        if g in ("Propose", "Barter", "SellInfo"):
            return INTENT_DEAL
        if g in ("SeekProtector",) or other.needs.injury >= 55 or other.solitary_ticks > 0:
            return INTENT_WEAK
        if g in ("Socialize", "Chess", "Music", "DeclineBoredom", "Visit", "Phone"):
            return INTENT_FRIENDLY
        return INTENT_UNKNOWN

    def _social_recent_of(self, npc: NPC) -> str:
        """我身上最近一条社交事件（喂 caption 的『社交近况』轴），从 P1 记忆里取。"""
        kind_map = {
            "threatened": SOC_THREATENED, "snitched_at": SOC_SNITCHED,
            "got_weapon": SOC_DEAL, "sold_weapon": SOC_DEAL,
            "protected_by": SOC_PROTECTED, "protecting": SOC_ALLIED,
            "reconciled": SOC_RECONCILED,
        }
        for m in reversed(npc.memory):
            if m["kind"] in kind_map:
                return kind_map[m["kind"]]
        return SOC_NONE

    def _obligation_of(self, npc: NPC) -> str:
        owe = any(not o.get("settled") and o["debtor"] == npc.id for o in self.obligations)
        owed = any(not o.get("settled") and o["creditor"] == npc.id for o in self.obligations)
        if owe:
            return OBLIG_OWE
        return OBLIG_OWED if owed else OBLIG_NONE

    def _social_axes(self, npc: NPC) -> tuple[str, str, str, str]:
        """凑齐 v3 的社交四轴 (邻近, 对方意图, 社交近况, 人情)。"""
        other = self._nearest_social(npc)
        near = self._rel_kind_to(npc, other) if other else NEAR_NONE
        intent = self._intent_of(other) if other else INTENT_NONE
        return near, intent, self._social_recent_of(npc), self._obligation_of(npc)

    def _activity_menu(self, npc: NPC, phase: str) -> list[str]:
        """从时段菜单里轮换取 ACT_MENU_MAX 项。

        菜单本身可以很长（自由活动有 11 项），但一次全塞给大脑会让 LLM 明显变慢，
        小模型也会被稀释。按 (tick, npc) 错开取样：同一拍不同人菜单不同，
        同一人跨时段能把整张菜单走遍，既控住开销又保住多样性。
        """
        menu = list(PHASE_ACTIVITIES.get(phase, ()))
        if len(menu) <= ACT_MENU_MAX:
            return menu
        off = (self.tick_count // 12 + sum(ord(ch) for ch in npc.id)) % len(menu)
        picked = [menu[(off + i) % len(menu)] for i in range(ACT_MENU_MAX)]
        if npc.goal in menu and npc.goal not in picked:
            picked[-1] = npc.goal   # 正在做的事被轮换掉会半路弃坑，始终保留
        return picked

    def _gen_candidates(self, npc: NPC, phase: str, ctx: ContextState) -> list[Candidate]:
        """按需求/时段/处境生成候选（不看性格——性格只进 context，交给模型）。"""
        c: list[Candidate] = []
        n = npc.needs
        if n.hunger >= THRESHOLDS["hunger"]:
            c.append(Candidate("Eat", "need"))
        if n.thirst >= THRESHOLDS["thirst"]:
            c.append(Candidate("Drink", "need"))
        if n.fatigue >= THRESHOLDS["fatigue"]:
            c.append(Candidate("Sleep", "need"))
        if n.boredom >= THRESHOLDS["boredom"]:
            c.append(Candidate("DeclineBoredom", "need"))
        if n.social >= THRESHOLDS["social"]:
            c.append(Candidate("Socialize", "need"))
        if n.injury >= THRESHOLDS["injury"]:
            c.append(Candidate("Cure", "need"))
        # 被狱警/狱犬盯上：给一个"收敛"反应，正常状态下会压过日常行为，
        # 让 NPC 当着巡逻的面老实站住——这才是"狱警狱犬起作用"的可见信号。
        if npc.watched_by:
            c.append(Candidate("LayLow", "threat"))
        # 处境驱动
        if ctx.enemy_present and phase in (PHASE_YARD, PHASE_FREE, PHASE_TRAIN):
            # 同时给"动手"和"斗嘴"，让升级/降级冲突成为一道真实的选择题
            c.append(Candidate("Fight", "threat"))
            c.append(Candidate("Argue", "threat"))
        if phase == PHASE_CLASS:
            c.append(Candidate("Study", "daily"))
        if npc.has_labor and phase in (PHASE_TRAIN, PHASE_FREE):
            c.append(Candidate("Labor", "worktask"))
            c.append(Candidate("KitchenDuty", "worktask"))
        # 越狱链：放风/自由活动/夜里都能攒准备，否则 10 分钟夜班里永远等不到就寝窗口。
        #   1) Barter 换工具  2) Tunnel 撬门  3) 工具+挖通 → PrisonBreak
        if phase in (PHASE_SLEEP, PHASE_FREE, PHASE_YARD) and not npc.parole_near:
            has_tool = bool(npc.escape_prep.get("tool"))
            q_esc = quest_of(npc.id)
            wants_out = bool(q_esc and q_esc.kind == "escape") or npc.parole_lost or has_tool
            if wants_out:
                breached = npc.escape_prep.get("breach", 0) >= ESCAPE_BREACH_TICKS
                if not has_tool and not any(cand.goal == "Barter" for cand in c):
                    c.append(Candidate("Barter", "daily"))
                if has_tool and not breached:
                    c.append(Candidate("Tunnel", "daily"))
                if has_tool and breached:
                    c.append(Candidate("PrisonBreak", "daily"))
        # 搜牢时藏违禁品：事件直接长出新选项，是"情境改变可行动作集"的典型演示
        if ctx.event == "狱警搜牢":
            c.append(Candidate("Hide", "event"))
        # 告密：得有事可报（斗殴/尸体/仇人），且身边有管教才递得上话
        if ctx.surveillance != SURVEIL_LOW and (
            ctx.event == "斗殴" or ctx.has_corpse or ctx.enemy_present
        ):
            c.append(Candidate("Snitch", "event"))
        # 望风：只有监视松懈时才有人干这买卖
        if ctx.surveillance == SURVEIL_LOW and phase in (PHASE_YARD, PHASE_FREE):
            c.append(Candidate("Watch", "daily"))
        # 打电话：想家/憋得慌时才排进来
        if n.social >= 25 and phase in (PHASE_YARD, PHASE_FREE):
            c.append(Candidate("Phone", "need"))
        # 宏观动机的专属候选：把这个动机想推进的事(推进项+签名动作)稳定放上桌——
        # 否则它们淹没在时段轮换菜单里、大多数拍压根不出现，m_drive 再高也偏不动，
        # 五个人就又变回"谁都在解闷/歇着"。具体偏置仍交给 m_drive，这里只负责"上桌"。
        # 签名动作低频：冷却期内不上，避免每拍复读同一个专属动作。
        d = self._drive_of(npc)
        _survival = {"Eat", "Drink", "Sleep", "Cure"}
        if phase not in (PHASE_ROLLCALL, PHASE_SLEEP):
            for g in d.drive_goals():
                if g in _survival or g == "LayLow":
                    continue
                if g == d.signature and npc.sig_cooldown > 0:
                    continue
                if self._facility_for_goal(g) is None:
                    continue
                if not any(cand.goal == g for cand in c):
                    c.append(Candidate(g, "daily"))
        # 终极目标·当前分目标：把"这一步该干的事"稳定上桌，让行为真正指向当前分目标。
        q = quest_of(npc.id)
        if (q and q.steps and not npc.parole_lost and npc.quest_step < len(q.steps)
                and phase not in (PHASE_ROLLCALL, PHASE_SLEEP)):
            for g in q.steps[npc.quest_step].focus:
                if g in _survival or g == "LayLow":
                    continue
                if self._facility_for_goal(g) is None:
                    continue
                if not any(cand.goal == g for cand in c):
                    c.append(Candidate(g, "daily"))
        # 生死网：有仇杀对象、没刀先弄刀；一旦有刀就不卡任务链最后一步——夜班里要真见血。
        if (npc.kill_target and not npc.dead and phase != PHASE_ROLLCALL):
            tgt = self._npc(npc.kill_target)
            if tgt and not tgt.dead and not tgt.escaped:
                if not npc.weapon:
                    if not any(cand.goal == "GetWeapon" for cand in c):
                        c.append(Candidate("GetWeapon", "daily"))
                elif (not npc.watched_by
                      and not any(cand.goal == "Assassinate" for cand in c)):
                    c.append(Candidate("Assassinate", "threat"))
        # v3/P2 定向社交候选：仅在 PRISON_CAPTION=v3（v3 模型上线）时上桌，与 gen_situations 同规则。
        # 近旁那个人是谁 + 对方意图 + 我手里的人情，决定该结盟/背叛/讨债/放狠话/护着还是别理。
        if caption_version() == "v3" and phase in (PHASE_YARD, PHASE_FREE, PHASE_TRAIN):
            near, intent, _sr, oblig = self._social_axes(npc)
            if near != NEAR_NONE:
                if near in (NEAR_STRANGER, NEAR_DEBTOR, NEAR_CREDITOR, NEAR_ALLY):
                    c.append(Candidate("Propose", "daily"))
                # 结盟只谈"新盟"（生人/对方示好），已是盟友就别反复"结成同盟"刷屏——
                # 铁盟之间该演的是护着/反水，而不是再拜一次把子。
                if near == NEAR_STRANGER or intent == INTENT_ALLY:
                    c.append(Candidate("Ally", "daily"))
                if near == NEAR_FOE or intent == INTENT_HOSTILE:
                    c.append(Candidate("Threaten", "daily"))
                if near in (NEAR_ALLY, NEAR_DEBTOR):
                    c.append(Candidate("Betray", "daily"))
                if intent == INTENT_WEAK or near == NEAR_DEBTOR:
                    c.append(Candidate("Protect", "daily"))
                if oblig == OBLIG_OWE and near == NEAR_CREDITOR:
                    c.append(Candidate("Repay", "daily"))
        # 更丰富的日常活动：整张菜单按 (tick, npc) 轮换取样，单拍候选数可控
        for g in self._activity_menu(npc, phase):
            c.append(Candidate(g, "daily"))
        # 日常兜底：Patrol=去放风场晃悠 / Rest=回牢房躲清静，构成 El Farol 的去/不去二选一
        c.append(Candidate("Patrol", "daily"))
        c.append(Candidate("Rest", "daily"))
        # 阶段一：向导老牙在公共区解锁前，留在牢区陪玩家破冰——别一开局就顺着"门路"
        # 溜达出镜，让玩家一睁眼找不着人。只约束老牙一人、且只在 public 未解锁时生效，
        # 其余 NPC、以及公共区解锁后一律照常自由活动（不影响 M 系列演示/探针）。
        if npc.id == ANCHOR_ID and not self.bb.zone_unlocked("public"):
            c = [cand for cand in c if self._cellblock_ok(cand.goal)]
        # 清单 §2.3-2 当前目标回灌：候选里没有"正在做的事"就补回来。
        # 否则做到一半的目标因为没上桌而被迫中断，看起来就是三心二意。
        # 例外：打架/仇杀这类"必须有对手在场"的目标，一旦对手已走开/不在场，就别再回灌——
        # 否则 NPC 会守着一个空目标"打空气"。没了候选，下一拍自然改做别的。
        _needs_foe = {"Fight", "Argue", "Assassinate"}
        if npc.goal and not any(cand.goal == npc.goal for cand in c):
            if not (npc.goal in _needs_foe and not npc.enemy_present and not npc.in_bout):
                c.append(Candidate(npc.goal, "daily"))
        # M4 玩家指令：去重后作为 command 候选（拿保底乘数），让"指令 vs 需求"清晰可见
        if npc.commanded_goal:
            c = [cand for cand in c if cand.goal != npc.commanded_goal]
            c.append(Candidate(npc.commanded_goal, "command", b_force_plan=True))
        return self._filter_cell_lock(npc, c)

    # -- 移动 / 需求 --------------------------------------------------------
    def _move_toward(self, npc: NPC, dt: float) -> None:
        """沿绕墙路径走向目标。

        以前这里是朝目标走直线、完全不认墙，于是权威坐标会落在墙里屋里，而前端
        另有一套 A* 绕墙——同一个人两边走两条路，既穿墙又拉出十几米的偏差。
        现在路径由 nav.py 出，前端跟着同一条走。

        走不通（典型情形：牢门被典狱长锁了）就记一次失败并放弃这个落点，
        让下一拍的打分把它压下去，而不是贴着门磨到永远。
        """
        if npc.target is None:
            return
        tx, tz = npc.target.x, npc.target.z
        key = (round(tx, 1), round(tz, 1))
        if npc.path_key != key or not npc.path:
            npc.path = NAV.path(npc.x, npc.z, tx, tz)
            npc.path_key = key
            if not npc.path:
                goal = npc.goal or ""
                npc.failure_pen[goal] = min(
                    0.9, npc.failure_pen.get(goal, 0.0) + FAILURE_PENALTY)
                if self._cell_locked(npc):
                    if self._in_own_cell(npc):
                        npc.act_fail = "牢门锁了，出不去"
                    else:
                        npc.act_fail = "牢门锁了，进不去"
                else:
                    npc.act_fail = "过不去"
                npc.target = None
                return
        budget = npc.speed * dt
        while budget > 1e-6 and npc.path:
            wx, wz = npc.path[0]
            dx, dz = wx - npc.x, wz - npc.z
            d = math.hypot(dx, dz)
            if d <= budget:
                npc.x, npc.z = wx, wz
                npc.path.pop(0)
                budget -= d
            else:
                npc.x += dx / d * budget
                npc.z += dz / d * budget
                budget = 0.0

    def _drift_needs(self, npc: NPC) -> None:
        n = npc.needs
        # 被动漂移隔拍才涨(减半)：需求上升慢一倍，NPC 有更多时间去推进终极目标/互动，
        # 而不是刚解完饿又饿——这是"日常需求霸屏、五个人动作雷同"的直接成因之一。
        if self.tick_count % 2 == 0:
            n.hunger = min(100, n.hunger + 1)
            n.thirst = min(100, n.thirst + 1)
            n.fatigue = min(100, n.fatigue + 1)
            n.boredom = min(100, n.boredom + 1)
            n.social = min(100, n.social + 1)

        # 原地收敛：没有功能室可去。其余动作一律走 _advance_action，
        # 必须进对应房间并停够拍数才结算——不能在走廊比划一下就算洗完澡。
        if npc.goal == "LayLow" and npc.goal in ACT_RELIEF:
            for key, dv in ACT_RELIEF[npc.goal].items():
                setattr(n, key, max(0, min(100, getattr(n, key) + dv)))

    # -- 主循环 -------------------------------------------------------------
    def set_paused(self, val: bool) -> None:
        """上帝暂停/继续。暂停时 tick 空转、世界定格，便于玩家逐个查看 NPC。"""
        self.paused = bool(val)

    async def tick(self, dt: float) -> None:
        if self.paused:      # 定格：不推进任何状态，交由 _sim_loop 继续广播这帧冻结画面
            return
        self.tick_count += 1
        phase = self._phase()

        # M4 注入事件倒计时
        if self.event_ticks > 0:
            self.event_ticks -= 1
            if self.event_ticks == 0:
                self.event_kind = None

        # 0) 物资按作息补货（食堂只在点名/放风补 → 夜里饿了真没饭吃）
        self._restock_if_phase_changed(phase)

        # 0) 押送段先走：被押的人这一拍就已经不自由了，不该再参与下面的决策。
        # 顺序要紧——押送者跟位是按被押者的位置算的，反过来先动押送者，
        # 他瞄的就永远是上一拍的旧位置，于是稳定落后一个步长（实测拉开 2m，不像押着走）。
        self._advance_escorts(dt)
        # 0) 狱警/狱犬再走，然后才算谁被盯上（本拍的威胁/监视/违禁压制都据此）
        self._move_patrollers(dt)
        for npc in self.npcs:
            if npc.escort_ticks > 0:
                npc.watched_by = "狱警"       # 被押着走，全程在人手里
                continue
            npc.watched_by = "" if (npc.escaped or npc.dead) else self._watcher_of(npc.x, npc.z)
            # 狱警行权：连续盯住揣着越狱工具/挖过洞的人就抓现行（加刑/罚苦役/搞黄假释）
            self._enforce_guard(npc)
            # 收敛只在被盯时成立：盯防一走就立刻解除，否则当前目标回灌+惯性
            # 会把 LayLow 一直顶在最高，人就永远原地罚站。清成 None → 本拍重新决策。
            if npc.goal == "LayLow" and not npc.watched_by:
                self._release_object(npc)     # 别攥着铺位/龙头不放，否则别人永远排不上
                npc.goal = None
                npc.goal_reached = False
                npc.goal_ticks = 0
                npc.act_stage = "goto"
            # 行动已结算/已失败 → 放掉目标，本拍重新决策。
            # 不放的话 _set_goal 会因 goal 没变而直接 return，act_stage 永远停在
            # done，人就抱着"已经吃完"的状态站在食堂，饥饿一路涨到 100 也不再动。
            elif npc.act_stage in ("done", "failed") and npc.goal is not None:
                self._release_object(npc)
                npc.goal = None
                npc.goal_reached = False
                npc.goal_ticks = 0
                npc.act_stage = "goto"

        # 1) 记录上一拍决策造成的当前放风人数（El Farol 的"出勤"信号）
        count = self._yard_count()
        self.attendance_history.append(count)
        crowd = self._crowd_label(count)

        # 2) 逐个准备：衰减惩罚 → 漂移需求 → 拥挤预期 → 拼 context → 生成候选
        prep: list[tuple[NPC, ContextState, str, list[Candidate]]] = []
        for npc in self.npcs:
            if npc.escaped or npc.dead:   # 已出逃/已身亡，不再参与决策
                continue
            if npc.escort_ticks > 0:      # 正被押往禁闭室：身不由己，位置由 _advance_escorts 管
                continue
            if npc.solitary_ticks > 0:    # 关禁闭：完全不能行动，只能等
                npc.solitary_ticks -= 1
                self._release_object(npc)
                npc.x, npc.z = SOLITARY_POS
                # 不给它编一个 "Solitary" goal：actions.py 那张表同时是微调的标签全集，
                # 塞进一个 NPC 其实选不了的动作会污染数据。前端用 solitary 字段显示。
                npc.goal = None
                npc.in_bout = ""
                npc.needs.stress = min(100, npc.needs.stress + 1)
                continue
            if npc.recall_ticks > 0:      # 典狱长点名回牢房：硬编码，不经模型
                npc.recall_ticks -= 1
                self._release_object(npc)
                npc.in_bout = ""
                npc.fight_target = ""
                npc.goal = "Sleep"
                self._drive_recall_home(npc, dt)
                continue
            self._decay_penalties(npc)
            if npc.sig_cooldown > 0:
                npc.sig_cooldown -= 1
            if npc.fight_cd > 0:
                npc.fight_cd -= 1
            # 老实期只在自由时递减：放在禁闭那段 continue 之后，
            # 于是"被关的 40 拍"不会白白吃掉老实期，出来后才真的开始收敛。
            if npc.busted_cd > 0:
                npc.busted_cd -= 1
            if npc.kill_cd > 0:
                npc.kill_cd -= 1
            self._notice_cell_lock(npc)
            self._drift_needs(npc)
            # 仇人是否在场，每拍按"具体的人 + 距离"重算（替换静态布尔）
            foe = self._hostile_near(npc)
            npc.fight_target = foe.id if foe else ""
            npc.enemy_present = foe is not None
            self._refresh_parole(npc)
            npc.expected_crowd = self._predict_crowd(npc)
            ctx = self._build_context(npc, phase, crowd)
            caption = build_context_caption(ctx)
            candidates = self._gen_candidates(npc, phase, ctx)
            prep.append((npc, ctx, caption, candidates))

        # 3) 读分缓存立即决策（清单 §2.4：绝不同步等模型，否则帧率由推理延迟决定）
        for npc, ctx, caption, cands in prep:
            prev_goal = npc.goal
            scored = self._scored_from_cache(npc, cands)
            self._apply_score_terms(npc, scored, phase)
            npc.last_context = caption
            npc.last_ctx = ctx
            npc.last_scored = scored
            self._decide(npc, scored)
            self._move_toward(npc, dt)
            self._advance_action(npc)      # 取用→执行→结算（生存类行动的物资约束）
            if caption_version() == "v3":  # v3/P2：定向社交动作落地（默认 v2 时完全不触发）
                self._settle_social(npc)
            self._maybe_start_fight(npc)   # 选了打架且对象在场 → 开一场群殴
            # 决策定了、动手对象/行动阶段也定了，把"为什么这么决策"结构化记下来
            self._update_decision_trace(npc, prev_goal)
            self._bookkeep(npc)
            self._update_quest_pressure(npc)  # 主线停滞攒压/推进清零 → 下拍抬高 focus 直至必落地
            self._advance_escape(npc)
            self._advance_kill(npc)        # 生死网：缀上仇杀对象、伺机下杀手
            # M4 指令倒计时 / 到达即完成
            if npc.commanded_goal:
                npc.command_ticks -= 1
                reached = (
                    npc.goal == npc.commanded_goal
                    and (npc.act_stage == "perform" or self._at_target(npc))
                )
                if npc.command_ticks <= 0 or reached:
                    npc.commanded_goal = None
                    npc.command_ticks = 0

        # 3.5) 打架逐帧结算：旁人加入 → 互相造成 injury → 判输 → 狱警抓现行。
        #      放在决策之后：本拍谁动了手已经定了，这里只负责把后果算出来。
        self._advance_bouts()
        self._advance_player()     # 玩家同样受物理层约束：会饿、会被误伤、会被记恨的人揍

        # 3.6) NPC↔NPC 对话触发（只挑候选，不调 LLM）：靠得近 + 有关系钩子时排一场。
        #      真正的脚本生成交给 _sim_loop（懒生成，一次一场，带全局/同对冷却）。
        self._detect_convo()

        # 3.7) 夜班时钟 / 风险事件：读本拍已落地的死伤、仇怨、凶器，不改决策本身
        self.shift.on_tick(self)

        # 4) 帧末合批：这一拍所有 NPC 的请求攒成一个 /score_multi，后台发出
        self._dispatch_scoring(prep)

    # -- 异步分缓存（清单 §2.4）---------------------------------------------
    def _scored_from_cache(self, npc: NPC, candidates: list[Candidate]) -> list[ScoredCandidate]:
        """用缓存里的模型分组装候选。没有缓存分的一律记 0——

        清单原话："无任何缓存分 → 候选分全置 0 → 保持当前目标（绝不用别的规则临时定胜负）"。
        冷启动的头几拍就是这个状态，NPC 站着不动几百毫秒，好过用一套临时规则先跑歪。
        """
        out: list[ScoredCandidate] = []
        for c in candidates:
            hit = npc.score_cache.get(c.goal)
            if hit is None:
                out.append(ScoredCandidate(candidate=c, v=0.0, m_value=0.0,
                                           score=0.0, cached=False))
            else:
                v, m, _t = hit
                out.append(ScoredCandidate(candidate=c, v=v, m_value=m, score=m))
        return out

    def _dispatch_scoring(
        self, prep: list[tuple[NPC, ContextState, str, list[Candidate]]]
    ) -> None:
        """在途只允许一个批次：模型慢时自然降采样，不会把请求越堆越多。"""
        if self._score_inflight:
            self.score_skipped += 1
            return
        # 有对话在生成时，把模型让给对话（世界继续用分缓存推进，不冻结）
        if self.chat_active > 0:
            self.score_skipped += 1
            return
        self._score_inflight = True
        items = [(npc, cap, cands) for npc, _c, cap, cands in prep]
        # 存引用，否则 task 可能被 GC 掉
        self._score_task = asyncio.create_task(self._score_worker(items))

    async def _score_worker(self, items: list[tuple[NPC, str, list[Candidate]]]) -> None:
        started = self.tick_count
        t0 = time.perf_counter()
        try:
            payload = [(cap, cands, npc.personality) for npc, cap, cands in items]
            scored_all, says = await self.model.evaluate(payload)
        except Exception as e:                        # 模型抖动不该让世界停摆
            log.warning("打分失败，沿用旧缓存：%s: %s", type(e).__name__, e)
            self.score_fail += 1
            return
        finally:
            self._score_inflight = False
        for (npc, _cap, _cands), scored, say in zip(items, scored_all, says):
            # 合并而非覆盖：轮换菜单会让某些目标这拍没上桌，旧分留着仍比没分强
            for sc in scored:
                npc.score_cache[sc.candidate.goal] = (sc.v, sc.m_value, started)
            if say:
                npc.llm_say = say
                # 记下这句话是冲着谁说的：LLM 心里的第一名未必是引擎最终选的
                # （距离/惯性/拥挤项会改排序），不记就会出现"嘴上打电话、人在社交"。
                npc.llm_say_goal = max(scored, key=lambda s: s.m_value).candidate.goal \
                    if scored else None
        self.score_ms = (time.perf_counter() - t0) * 1000.0
        self.score_lag = self.tick_count - started    # 缓存比当前落后几拍
        self.score_rounds += 1

    async def aclose(self) -> None:
        """收尾：等在途批次落地再让调用方关连接，否则会看到一串
        'client has been closed' 的假错误。"""
        task = self._score_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

    # -- 遥测 ---------------------------------------------------------------
    def _choice_dist(self) -> dict:
        """各性格累计选择分布（goal 标签 -> 次数），供前端画性格差异图。"""
        out: dict[str, dict] = {}
        for personality, counts in self.choice_counts.items():
            out[personality] = {
                GOAL_LABEL.get(g, g): c for g, c in counts.items()
            }
        return out

    # -- 前端帧 -------------------------------------------------------------
    def world_state_message(self) -> dict:
        count = self._yard_count()
        return {
            "type": "world_state",
            "tick": self.tick_count,
            "paused": self.paused,
            "phase": self._phase(),
            "yard_crowd": self._crowd_label(count),
            "yard_count": count,
            "yard_comfort": YARD_COMFORT,
            "n_agents": self.n_agents,
            "attendance": list(self.attendance_history)[-120:],
            "choice_dist": self._choice_dist(),
            "event_kind": self.event_kind,
            "brain": getattr(self.model, "label", "?"),
            "score_ms": round(self.score_ms),
            "score_lag": self.score_lag,
            "score_rounds": self.score_rounds,
            "escaped_names": list(self.escaped_names),
            "busts": list(self.busts),
            "blackboard": self.bb.to_message(),
            "endgame": self.endgame_state(),
            "shift": self.shift.to_message(self),
            # 房间几何随帧下发（w/d 是局部尺寸、face 是门朝向、door 是门洞偏移），
            # 前端照此建房，不再自己存一份尺寸。
            "facilities": [
                {"id": f.id, "name": f.name, "kind": f.kind, "pos": [f.x, f.z],
                 "zone": zone_of_kind(f.kind),
                 "w": f.w, "d": f.d, "face": f.face, "door": f.door}
                for f in self.facilities
            ],
            # 牢房六间：几何 + 住户 + 门的开关/上锁状态。前端照此砌隔间、开合栅栏门，
            # 并把关着的门计入碰撞盒（锁着的门在后端 nav 网格里也是实墙）。
            "cells": self.cell_state,
            "solitary_locked": self.solitary_locked,
            # 物资实体：前端据此摆窝头/龙头/铺位/药柜，并显示还剩几份
            "objects": [o.to_message() for o in self.objects],
            "supply_log": list(self.supply_log),
            "bouts": [b.to_message() for b in self.bouts if not b.over],
            "fight_log": list(self.fight_log),
            "player": self.player.to_message(),
            "patrollers": [
                {"id": p.id, "kind": p.kind, "name": p.name,
                 "pos": [round(p.x, 3), round(p.z, 3)],
                 "heading": [round(p.hx, 3), round(p.hz, 3)],
                 "zone": self._zone_of_pos(p.x, p.z),
                 # 押送/敬礼：前端据此换动作和头顶标签（押人时不该还画着巡逻视锥）
                 "escorting": p.escorting,
                 "saluting": p.salute > 0,
                 "view_range": GUARD_VIEW_RANGE if p.kind == "guard" else DOG_SMELL_RANGE,
                 "view_cos": GUARD_VIEW_COS if p.kind == "guard" else -1.0}
                for p in self.patrollers
            ],
            "npcs": [
                {
                    "id": npc.id,
                    "name": npc.name,
                    "personality": npc.personality,
                    "role": npc.role,
                    "pos": [round(npc.x, 3), round(npc.z, 3)],
                    "target_pos": [npc.target.x, npc.target.z] if npc.target else [round(npc.x, 3), round(npc.z, 3)],
                    "goal": npc.goal,
                    "cell": npc.cell,
                    "goal_label": GOAL_LABEL.get(npc.goal or "", npc.goal),
                    "activity": activity(npc.goal),
                    "action_state": self._action_state(npc),
                    "act_stage": npc.act_stage,
                    "act_progress": self._act_progress(npc),
                    "carrying": npc.carrying,
                    "act_fail": npc.act_fail,
                    "cell_locked": self._cell_locked(npc),
                    "in_own_cell": self._in_own_cell(npc),
                    "place": self._place_name(npc),
                    "target_name": npc.target.name if npc.target else "",
                    "monologue": self._say(npc),
                    "needs": self._needs_dict(npc.needs),
                    "expected_crowd": npc.expected_crowd,
                    "in_yard": self._in_yard(npc),
                    "commanded": npc.commanded_goal is not None,
                    "watched_by": npc.watched_by,
                    "escaped": npc.escaped,
                    # 生死网：凶器 / 仇杀对象 / 是否已身亡（前端渲染尸体 + 决策卡展示）
                    "dead": npc.dead,
                    "weapon": npc.weapon,
                    "weapon_cn": WEAPON_CN.get(npc.weapon, ""),
                    "kill_target": npc.kill_target,
                    "kill_target_name": (self._npc(npc.kill_target).name
                                         if npc.kill_target and self._npc(npc.kill_target) else ""),
                    "quest_kind": (quest_of(npc.id).kind if quest_of(npc.id) else ""),
                    "zone": self._zone_of_pos(npc.x, npc.z),
                    "home_zone": npc.home_zone,
                    "escape_prep": {"tool": bool(npc.escape_prep.get("tool")),
                                    "breach": npc.escape_prep.get("breach", 0),
                                    "need_breach": ESCAPE_BREACH_TICKS},
                    "motive": self._drive_of(npc).motive,
                    "drive_progress": round(npc.drive_progress),
                    # 终极目标 + 当前分目标（任务链）：供状态面板/决策卡展示"他在图什么、这一步在干嘛"
                    "ultimate": (quest_of(npc.id).ultimate if quest_of(npc.id) else ""),
                    "quest_step": npc.quest_step,
                    "quest_total": (len(quest_of(npc.id).steps) if quest_of(npc.id) else 0),
                    "substep": self._substep_desc(npc),
                    "plot_surface": plot_of(npc).get("surface", ""),
                    "plot_step": plot_of(npc).get("step", ""),
                    "plot_fog": plot_of(npc).get("fog", ""),
                    # 承诺压力 0~1（>=1 即将/正在被硬顶推进）：前端把它做成"逼近关键一步"的进度
                    "pressure": round(min(1.0, npc.quest_pressure / PRESSURE_FORCE), 2),
                    "parole_lost": npc.parole_lost,
                    # 假释现在是可看见的数值链：表现分 / 加刑 / 变动流水
                    "parole_score": round(npc.parole_score),
                    "parole_near": npc.parole_near,
                    "extra_days": npc.extra_days,
                    "parole_log": list(npc.parole_log),
                    # 打架：仇怨对象 + 是否正在群殴 + 禁闭倒计时
                    "fight_target": npc.fight_target,
                    "in_bout": bool(npc.in_bout),
                    "solitary": npc.solitary_ticks,
                    "recalled": npc.recall_ticks > 0,   # 被典狱长点名遣回牢房中
                    # 押送段：前端用它把人物标成"被押送"并让他正常走动（不是瞬移）
                    "escort": npc.escort_ticks,
                    "escort_by": npc.escort_by,
                    "escort_grabbed": npc.escort_grabbed,   # False=站着等狱警过来拿人
                    "grudge": {k: round(v) for k, v in npc.grudge.items() if v >= 10},
                    "sentence_years": round(npc.sentence_years, 2),
                    "crime": npc.crime,
                    "labor_job": npc.labor_job,
                    # 共享状态脊梁：本拍决策归因（对话/决策卡/暂停面板共用）+ 重大决策流水
                    "core_goal": self._drive_of(npc).motive,
                    "drivers": npc.decision_drivers,
                    "decision_log": list(npc.decision_log),
                    # P1/P2 可见性：这个人此刻的社会处境（盟友/仇人/人情/最近社交）
                    "social": self._npc_social_view(npc),
                }
                for npc in self.npcs
            ],
            # 全局社交事件流水（结盟/背叛/放狠话/军火/讲和/告密…）——上帝视角实时字幕的数据源
            "social_feed": list(self.social_events)[-14:],
        }

    # -- 玩家的物理层（会饿、会伤、会被打）-----------------------------------
    def set_player_pos(self, x: float, z: float) -> None:
        self.player.x, self.player.z = float(x), float(z)
        self.warden_active = True         # 收到玩家位置即认定典狱长在场（其站位起监视之效）

    def player_take(self, kind: str, n: int = 1) -> None:
        self.player.bag[kind] = self.player.bag.get(kind, 0) + n

    def player_use(self, kind: str) -> dict:
        """玩家吃掉/用掉背包里的一件东西。"""
        p = self.player
        if p.bag.get(kind, 0) <= 0:
            return {"ok": False, "reason": "你身上没有这个"}
        eff = PLAYER_USE.get(kind)
        if not eff:
            return {"ok": False, "reason": "这东西不能吃也不能用"}
        p.bag[kind] -= 1
        if p.bag[kind] <= 0:
            del p.bag[kind]
        for key, dv in eff.items():
            setattr(p, key, max(0.0, min(100.0, getattr(p, key) + dv)))
        return {"ok": True, "kind": kind, "player": p.to_message()}

    def _advance_player(self) -> None:
        """玩家每拍的身体结算：需求漂移 + 群殴误伤 + 被记恨的人揍。"""
        p = self.player
        p.drift_tick += 1
        if p.drift_tick % PLAYER_DRIFT_EVERY == 0:
            p.hunger = min(100.0, p.hunger + 1)
            p.thirst = min(100.0, p.thirst + 1)

        # 1) 站在群殴里会被误伤 —— "别在打架的时候凑过去看"
        for bout in self.bouts:
            if bout.over:
                continue
            if math.hypot(bout.x - p.x, bout.z - p.z) <= PLAYER_SPLASH_RANGE:
                p.injury = min(100.0, p.injury + PLAYER_SPLASH_DMG)
                self._log_hit("你被卷进了群殴，挨了几下")
                break

        # 2) 记恨你的人贴上来就动手（仇怨主要来自你抢了他最后一口饭/药）
        phase = self._phase()
        if phase in (PHASE_YARD, PHASE_FREE, PHASE_TRAIN):
            for npc in self.npcs:
                if npc.escaped or npc.dead or npc.solitary_ticks > 0 or npc.watched_by:
                    continue                       # 被狱警盯着的不敢动手
                if npc.grudge.get(PLAYER_ID, 0.0) < GRUDGE_TO_FIGHT:
                    continue
                if math.hypot(npc.x - p.x, npc.z - p.z) > PLAYER_BEAT_RANGE:
                    continue
                p.injury = min(100.0, p.injury + PLAYER_BEAT_DMG)
                self._log_hit(f"{npc.name}逮着你就是一顿拳头")
                self._remember(npc, "你逮着那个新人揍了一顿，为抢东西的事")
                break

    def _log_hit(self, text: str) -> None:
        if not self.player.hits or self.player.hits[-1] != text:
            self.player.hits.append(text)
            del self.player.hits[:-6]

    # -- 假释：表现分 / 加刑（原来是死的静态布尔）-----------------------------
    def _refresh_parole(self, npc: NPC) -> None:
        """由表现分派生 parole_near —— 接通 drives 的 parole 心思与越狱门禁。

        `parole_near` 已经被 `_m_drive`（快假释的人把打架/越狱视为禁忌）和
        `_gen_candidates`（快假释的绝不铤而走险）读着，所以这里一改，
        整条"表现好 → 更保守 → 更不敢越狱"的反馈链就自己接上了。
        """
        if npc.parole_lost:
            npc.parole_near = False
            return
        npc.parole_near = npc.parole_score >= PAROLE_NEAR_AT and npc.extra_days == 0

    def _reward_parole(self, npc: NPC, goal: str) -> None:
        """好好干活攒表现分（在行动完成时调）。"""
        gain = PAROLE_WORK_GOALS.get(goal)
        if not gain or npc.parole_lost:
            return
        before_near = npc.parole_near
        npc.parole_score = min(100.0, npc.parole_score + gain)
        self._refresh_parole(npc)
        if npc.parole_near and not before_near:
            npc.parole_log.append(f"表现分到{npc.parole_score:.0f}，假释有望了")
            del npc.parole_log[:-6]

    def _penalize_parole(self, npc: NPC, drop: float, days: int, why: str) -> None:
        """犯事 → 扣表现分 + 加刑期。够狠的话直接把假释资格作废。"""
        npc.parole_score = max(0.0, npc.parole_score - drop)
        npc.extra_days += days
        npc.sentence_years += days / 365.0
        npc.parole_log.append(f"{why}：表现分-{drop:.0f}、加刑{days}天")
        del npc.parole_log[:-6]
        if npc.parole_score <= 5.0 or days >= EXTRA_DAYS_ESCAPE:
            if not npc.parole_lost:
                npc.parole_lost = True        # 破罐破摔态（drives 会解禁越狱）
                npc.drive_progress = 0.0
                npc.parole_log.append("假释资格没了，破罐破摔")
        self._refresh_parole(npc)

    # -- 打架：仇怨 / 开架 / 逐帧结算（combat.py）-----------------------------
    def _hostile_near(self, npc: NPC) -> NPC | None:
        """视线内最恨的那个人（仇怨过阈值且够近）。没有就是 None。

        这替换了原来的静态 `enemy_present`：以前全场只有疤脸恒为 True，且不管对方
        在不在场。现在"仇人在场"是真的**有那个人、且他就在附近**。

        注意 caption 里 `仇人:{在场|无}` 的取值集合没变，所以价值模型不会看到
        训练时没见过的文本——只是这一位的真假变得有依据了。
        """
        best: NPC | None = None
        best_g = 0.0
        for other in self.npcs:
            if other.id == npc.id or other.escaped or other.dead or other.solitary_ticks > 0:
                continue
            g = npc.grudge.get(other.id, 0.0)
            if g < GRUDGE_TO_FIGHT:
                continue
            # P1：谈成结盟/收成兄弟的人不动手——对白结的盟，真能拦下一场架
            if self._rel(npc, other.id)["ally"] >= ALLY_NOFIGHT:
                continue
            if math.hypot(other.x - npc.x, other.z - npc.z) > FIGHT_RANGE:
                continue
            if g > best_g:
                best, best_g = other, g
        return best

    def _add_grudge(self, npc: NPC, target_id: str, dv: float) -> None:
        npc.grudge[target_id] = min(100.0, npc.grudge.get(target_id, 0.0) + dv)

    def _remember(self, npc: NPC, text: str) -> None:
        """记一条"身上刚发生的事"，供对话引用（去重，避免同一件事刷屏）。"""
        if npc.recent_events and npc.recent_events[-1] == text:
            return
        npc.recent_events.append(text)

    # ------------------------------------------------------------------
    # 典狱长干预（上帝视角 admin 手段）：改凶器 / 改仇怨
    # 后果统一进 fate_log（前端命运账本 + 字幕），并会随下一帧 world_state 广播。
    # 决策脑无需重训——这些只是改了它下一拍看到的处境(caption 的凶器/仇怨轴)。
    # ------------------------------------------------------------------
    def warden_weapon(self, npc_id: str, op: str, kind: str = "") -> dict:
        """典狱长没收 / 栽赃 NPC 的杀人工具。"""
        npc = self._npc(npc_id)
        if npc is None or npc.dead:
            return {"ok": False, "msg": "此人已不在场上"}
        if op == "confiscate":
            if not npc.weapon:
                return {"ok": False, "msg": f"{npc.name}身上没有凶器"}
            weap = WEAPON_CN.get(npc.weapon, "凶器")
            npc.weapon = ""
            self.bb.fate_log.append({
                "npc": npc.id, "name": npc.name, "status": "凶器被收缴",
                "text": f"典狱长一次突击搜身，从{npc.name}身上搜出了{weap}——精心筹划的杀局，一下没了凭仗。",
                "big": True})
            self.bb.fate_log[:] = self.bb.fate_log[-80:]
            return {"ok": True, "msg": f"已没收 {npc.name} 的{weap}"}
        if op == "plant":
            if kind not in WEAPON_CN:
                return {"ok": False, "msg": "未知的凶器类型"}
            weap = WEAPON_CN[kind]
            npc.weapon = kind
            self.bb.fate_log.append({
                "npc": npc.id, "name": npc.name, "status": "被塞了凶器",
                "text": f"典狱长借着查铺，悄悄把一把{weap}塞进了{npc.name}的铺位——一颗随时会引爆的种子，埋下了。",
                "big": True})
            self.bb.fate_log[:] = self.bb.fate_log[-80:]
            return {"ok": True, "msg": f"已给 {npc.name} 栽赃{weap}"}
        return {"ok": False, "msg": "未知操作"}

    def warden_grudge(self, from_id: str, to_id: str, delta: float) -> dict:
        """典狱长挑拨(正) / 调解(负) from 对 to 的仇怨。"""
        a = self._npc(from_id)
        b = self._npc(to_id)
        if a is None or b is None or a is b:
            return {"ok": False, "msg": "对象无效"}
        if a.dead or b.dead:
            return {"ok": False, "msg": "有人已不在场上"}
        cur = a.grudge.get(to_id, 0.0)
        a.grudge[to_id] = max(0.0, min(100.0, cur + delta))
        if delta >= 0:
            self.bb.fate_log.append({
                "npc": a.id, "name": a.name, "status": "被挑拨",
                "text": f"典狱长在{a.name}耳边递了几句话，{a.name}看{b.name}的眼神冷了下来——梁子，更深了。",
                "big": False})
        else:
            self.bb.fate_log.append({
                "npc": a.id, "name": a.name, "status": "被劝和",
                "text": f"典狱长把{a.name}叫去敲打了一番，{a.name}对{b.name}的火气，消了几分。",
                "big": False})
        self.bb.fate_log[:] = self.bb.fate_log[-80:]
        verb = "挑拨" if delta >= 0 else "调解"
        return {"ok": True, "msg": f"已{verb}：{a.name} → {b.name}（仇怨 {a.grudge[to_id]:.0f}）"}

    def _drop_from_bouts(self, npc: NPC) -> None:
        npc.in_bout = ""
        npc.fight_target = ""
        for bout in self.bouts:
            if bout.over:
                continue
            if npc.id in bout.members:
                bout.members = [m for m in bout.members if m != npc.id]

    def _abort_warden_agency(self, npc: NPC) -> None:
        """典狱长硬指令：切断决策脑、打架、押送、旧路径。身不由己。"""
        npc.commanded_goal = None
        npc.command_ticks = 0
        npc.goal_reached = False
        npc.goal_ticks = 0
        npc.act_stage = "goto"
        npc.act_fail = ""
        npc.path = []
        npc.path_key = (1e9, 1e9)
        self._release_object(npc)
        self._drop_from_bouts(npc)
        if npc.escort_ticks > 0:
            self._end_escort(npc, lock_up=False)

    def _unlock_cell_for_entry(self, slot: int) -> None:
        """遣回牢房时必须能进门：锁着就悄悄打开，避免全体收监刷五条命运账。"""
        c = self._cell(slot)
        if c is None or not c.get("locked"):
            return
        c["locked"] = False
        c["closed"] = False
        NAV.set_gate(slot, False)

    def _cell_home_xz(self, npc: NPC) -> tuple[float, float] | None:
        if npc.cell < 0 or npc.cell >= len(layout.CELLS):
            return None
        c = layout.CELLS[npc.cell]
        return float(c["x"]), float(c["z"])

    def _snap_to_cell(self, npc: NPC) -> bool:
        home = self._cell_home_xz(npc)
        if home is None:
            return False
        self._unlock_cell_for_entry(npc.cell)
        npc.x, npc.z = home
        npc.path = []
        npc.path_key = (1e9, 1e9)
        npc.target = self._walkable_target(
            Facility("cell", "牢房", "sleep", home[0], home[1]))
        npc.recall_stuck = 0
        return True

    def _drive_recall_home(self, npc: NPC, dt: float) -> None:
        """回牢房推进：能走就走，走不通就计数，到点硬传到格子里。"""
        home = self._cell_home_xz(npc)
        if home is None:
            return
        hx, hz = home
        before = math.hypot(npc.x - hx, npc.z - hz)
        if before <= 1.6:
            # 已经到家：不再自动开门。否则典狱长把回来的人锁进牢里，这里每拍又把门顶开，
            # 表现成"牢门锁上了自己开"。到家后门归典狱长管，recall 不再插手。
            npc.recall_stuck = 0
            npc.x = npc.x * 0.7 + hx * 0.3
            npc.z = npc.z * 0.7 + hz * 0.3
            return
        # 还在半道上：确保门开着，人才进得去
        self._unlock_cell_for_entry(npc.cell)
        npc.target = self._walkable_target(Facility("cell", "牢房", "sleep", hx, hz))
        self._move_toward(npc, dt)
        after = math.hypot(npc.x - hx, npc.z - hz)
        if after >= before - 0.05 or npc.target is None:
            npc.recall_stuck += 1
        else:
            npc.recall_stuck = 0
        if npc.recall_stuck >= RECALL_SNAP_TICKS:
            self._snap_to_cell(npc)

    def _lock_in_solitary_now(self, npc: NPC, reason: str, ticks: int,
                              sentence_add: float) -> None:
        """典狱长关禁闭：当场入室，不排队等狱警押送。"""
        npc.busted += 1
        npc.sentence_years = round(npc.sentence_years + sentence_add, 1)
        npc.busted_cd = BUSTED_RECENT_TICKS
        npc.goal = None
        npc.act_stage = "goto"
        self._abort_warden_agency(npc)
        self._penalize_parole(npc, PAROLE_ESCAPE_PENALTY, EXTRA_DAYS_ESCAPE, reason)
        self.busts.append({"name": npc.name, "reason": reason,
                           "sentence": npc.sentence_years, "tick": self.tick_count})
        self.busts[:] = self.busts[-6:]
        npc.solitary_ticks = max(npc.solitary_ticks, ticks)
        npc.pending_solitary = 0
        npc.x, npc.z = SOLITARY_POS

    def warden_solitary(self, npc_id: str) -> dict:
        """典狱长把 NPC 直接关进禁闭室。硬编码，不经模型、不等押送。"""
        npc = self._npc(npc_id)
        if npc is None or npc.dead:
            return {"ok": False, "msg": "此人已不在场上"}
        if npc.escaped:
            return {"ok": False, "msg": f"{npc.name}已经越狱，鞭长莫及"}
        if npc.solitary_ticks > 0:
            return {"ok": False, "msg": f"{npc.name}已经在禁闭室里了"}
        if npc.escort_ticks > 0:
            return {"ok": False, "msg": f"{npc.name}已经在押往禁闭室的路上了"}
        # 不再当场传送：交给最近的狱警押着走过去，到门口才落锁（派不出人才退回即时关押）
        self._bust_into_solitary(npc, "典狱长下令关禁闭", SOLITARY_TICKS, 0.0)
        self.bb.fate_log.append({
            "npc": npc.id, "name": npc.name, "status": "下令关禁闭",
            "text": f"典狱长一声令下——狱警上前架起{npc.name}，往禁闭室押去。",
            "big": True})
        self.bb.fate_log[:] = self.bb.fate_log[-80:]
        return {"ok": True, "msg": f"已下令将 {npc.name} 押入禁闭室"}

    def warden_recall(self, target: str, dest: str) -> dict:
        """典狱长点名：硬编码把人送回牢房 / 关进禁闭室。不经决策脑。"""
        if dest not in ("cell", "solitary"):
            return {"ok": False, "msg": "未知去向"}
        if target == "all":
            crowd = [n for n in self.npcs if not n.dead and not n.escaped]
        else:
            one = self._npc(target)
            if one is None or one.dead:
                return {"ok": False, "msg": "此人已不在场上"}
            if one.escaped:
                return {"ok": False, "msg": f"{one.name}已经越狱，鞭长莫及"}
            crowd = [one]
        if not crowd:
            return {"ok": False, "msg": "场上没有可遣送的人"}
        n = 0
        snapped = 0
        in_solitary = 0            # 已在禁闭室、被跳过没动的人数（好在回执里说清楚）
        for npc in crowd:
            if dest == "solitary":
                if npc.solitary_ticks > 0 or npc.escort_ticks > 0:
                    in_solitary += 1
                    continue
                # 押送走过去，不瞬移；没狱警可派时也让他自己走过去（self_march），绝不当场传送
                self._bust_into_solitary(npc, "典狱长点名收监", SOLITARY_TICKS, 0.0)
                n += 1
                continue
            if npc.solitary_ticks > 0:
                in_solitary += 1
                continue
            self._abort_warden_agency(npc)
            npc.recall_ticks = RECALL_TICKS
            npc.recall_stuck = 0
            npc.goal = "Sleep"
            home = self._cell_home_xz(npc)
            if home is None:
                n += 1
                continue
            self._unlock_cell_for_entry(npc.cell)
            hx, hz = home
            npc.target = self._walkable_target(Facility("cell", "牢房", "sleep", hx, hz))
            # 不再提前瞬移：让 _drive_recall_home 每拍自己寻路走回去，
            # 只有连续走不通(_drive_recall_home 里计满 RECALL_SNAP_TICKS)才兜底落格。
            n += 1
        where = "禁闭室" if dest == "solitary" else "各自牢房"
        if target == "all":
            self.bb.fate_log.append({
                "npc": "", "name": "全体", "status": "被点名收押",
                "text": (f"典狱长一声令下，铁哨响彻放风场——所有人被赶回{where}，"
                         f"中庭一下空了。"),
                "big": True})
            self.bb.fate_log[:] = self.bb.fate_log[-80:]
        msg = f"已遣送 {n} 人回{where}"
        if snapped:
            msg += f" · {snapped} 人走不通，已直接押回格子"
        if in_solitary:
            msg += f" · {in_solitary} 人已在禁闭室，未动"
        return {"ok": True, "msg": msg, "count": n, "snapped": snapped,
                "in_solitary": in_solitary}

    def set_solitary_lock(self, locked: bool) -> dict:
        """禁闭室的门也归典狱长管：锁上后 nav 网格里那道门变实墙，谁也进出不了。

        禁闭本身是"传送进去 + 冻结 solitary_ticks"，这道锁是额外的物理封控——
        和牢门一套逻辑，让典狱长对禁闭室有同样的开关锁权力。
        """
        self.solitary_locked = bool(locked)
        NAV.set_gate(nav.SOLITARY_GATE, self.solitary_locked)
        for npc in self.npcs:
            npc.path = []
            npc.path_key = (1e9, 1e9)
        self.bb.fate_log.append({
            "npc": "", "name": "禁闭室", "status": "门落锁" if locked else "门打开",
            "text": (f"典狱长扭动禁闭室的门锁——"
                     f"{'铁门在里面人的身后彻底封死了。' if locked else '沉重的铁门重新打开。'}"),
            "big": False})
        self.bb.fate_log[:] = self.bb.fate_log[-80:]
        return {"ok": True, "locked": self.solitary_locked,
                "msg": "已锁上禁闭室" if locked else "已打开禁闭室"}

    # -- 牢门：开关 + 上锁 ---------------------------------------------------
    def _cell(self, slot: int) -> dict | None:
        return next((c for c in self.cell_state if c["slot"] == slot), None)

    def set_cell_door(self, slot: int, *, closed: bool | None = None,
                      locked: bool | None = None) -> dict:
        """开关/上锁某间牢房的栅栏门。

        **锁**是典狱长专属的权力（前端只在典狱长手段面板里给这个入口），关门谁都能关：
        关着但没锁，人推门就进出，寻路照通；一旦落锁，nav 网格里那道门就变成实墙，
        里面的人出不来、外面的人也进不去——"关起来"因此是真的物理封锁，不是一个状态标记。

        上锁隐含关门（没有"开着的门被锁上"这种事）。
        """
        c = self._cell(slot)
        if c is None:
            return {"ok": False, "msg": "没有这间牢房"}
        if locked is not None:
            c["locked"] = bool(locked)
            if c["locked"]:
                c["closed"] = True
            NAV.set_gate(slot, c["locked"])
            # 锁/开的一瞬间，所有正在走的人手上那条路径可能已经作废，清掉重算
            for n in self.npcs:
                n.path = []
                n.path_key = (1e9, 1e9)
        if closed is not None and not c["locked"]:
            c["closed"] = bool(closed)
        who = c["name"]
        if locked is not None:
            inside = self._npc(c["npc"]) if c["npc"] else None
            shut_in = bool(inside and not inside.dead
                           and math.hypot(inside.x - c["x"], inside.z - c["z"]) < 2.2)
            if c["locked"]:
                # 人已经被硬锁在自己牢里 → 遣返任务算兑现，清掉 recall 计时；
                # 否则 recall 循环还会每拍尝试开门放他进/出，跟这道锁较劲、把门顶开。
                if shut_in and inside is not None:
                    inside.recall_ticks = 0
                    inside.recall_stuck = 0
                self.bb.fate_log.append({
                    "npc": c["npc"], "name": who, "status": "被锁在牢里" if shut_in else "牢门落锁",
                    "text": (f"典狱长把{who}那间的栅栏门推上，钥匙在锁孔里转了两圈——"
                             f"{'他被关在里面了。' if shut_in else '这间今晚谁也进不去了。'}"),
                    "big": shut_in})
                self.bb.fate_log[:] = self.bb.fate_log[-80:]
            return {"ok": True, "locked": c["locked"], "closed": c["closed"],
                    "msg": (f"已锁上{who}的牢门" if c["locked"] else f"已打开{who}的牢门")}
        return {"ok": True, "locked": c["locked"], "closed": c["closed"],
                "msg": f"{who}的牢门已{'关上' if c['closed'] else '打开'}"}

    # ------------------------------------------------------------------
    # P1 · 关系图 + 情景记忆 + 对白→世界写回
    # ------------------------------------------------------------------
    def _rel(self, npc: NPC, other_id: str) -> dict[str, float]:
        """取/建 npc 对 other 的关系边（trust/fear/debt/ally/respect）。"""
        e = npc.rel.get(other_id)
        if e is None:
            e = blank_rel()
            npc.rel[other_id] = e
        return e

    def _bump_rel(self, npc: NPC, other_id: str, axis: str, dv: float) -> None:
        if npc is None or not other_id or axis not in REL_AXES:
            return
        e = self._rel(npc, other_id)
        e[axis] = max(0.0, min(REL_MAX, e[axis] + dv))

    def _remember_mem(self, npc: NPC, kind: str, other_id: str, text: str,
                      w: float = 1.0) -> None:
        """写一条情景记忆（带来源与显著度），供对话/决策检索。同 (kind,other) 去重覆盖。"""
        if npc is None or not text:
            return
        for m in npc.memory:            # 同类同对象的旧记忆先移除，保留最新一条
            if m["kind"] == kind and m["other"] == other_id:
                npc.memory.remove(m)
                break
        npc.memory.append({"tick": self.tick_count, "kind": kind,
                           "other": other_id, "text": text, "w": w})

    def ingest_conversation(self, a_id: str, b_id: str, kind: str,
                            lines: list[dict]) -> dict:
        """P1 核心：把刚生成的一场对白解析成社交事件并写回世界，返回后果摘要。

        这是"对话是行为的配音"反过来的另一半——**配音也能改剧本**：谈成军火→真给刀，
        放狠话→梁子更深（更容易打起来），讲和→敌意回落（可能避免一场架），
        求人罩着→真有了靠山（caption 的『靠山:在场』翻转，决策自然跟着变）。
        """
        events = parse_conversation(a_id, b_id, kind, lines)
        consequences: list[str] = []
        for ev in events:
            consequences += self._apply_social_event(ev)
        if consequences:
            log.info("对话写回：%s↔%s → %s", a_id, b_id, "；".join(consequences))
        return {"events": [ev.kind for ev in events], "consequences": consequences}

    def _apply_social_event(self, ev: SocialEvent) -> list[str]:
        """把一条社交事件落到世界状态上；返回给前端看的后果短句（可空）。"""
        src, dst = self._npc(ev.src), self._npc(ev.dst)
        if src is None or dst is None or src.dead or dst.dead:
            return []
        out: list[str] = []

        if ev.kind == "threat":
            # 放狠话：被威胁者更怕、更恨；施威者也更敌视对方 → 更容易真打起来
            self._bump_rel(dst, src.id, "fear", DELTA_THREAT_FEAR)
            self._add_grudge(dst, src.id, DELTA_THREAT_GRUDGE)
            self._add_grudge(src, dst.id, DELTA_THREAT_GRUDGE * 1.5)
            self._remember_mem(dst, "threatened", src.id, f"{src.name}当面放话要收拾我", 2.0)
            self._remember_mem(src, "threat", dst.id, f"我放话震了{dst.name}一下", 1.2)
            self._remember(dst, f"被{src.name}撂了狠话")
            out.append(f"💢 {src.name}放狠话震{dst.name}，梁子更深了")

        elif ev.kind == "deal_weapon":
            out += self._resolve_weapon_deal(src, dst, ev.text)

        elif ev.kind == "protect":
            # src=保护者 dst=被保护者：结盟 + 被罩者信任靠山、caption 的『靠山:在场』翻转
            self._bump_rel(src, dst.id, "ally", DELTA_ALLY)
            self._bump_rel(dst, src.id, "ally", DELTA_ALLY)
            self._bump_rel(dst, src.id, "trust", DELTA_PROTEGE_TRUST)
            dst.backer_present = True
            self._remember_mem(dst, "protected_by", src.id, f"{src.name}答应罩着我", 2.0)
            self._remember_mem(src, "protecting", dst.id, f"我收下了{dst.name}，罩着他", 1.5)
            self._remember(dst, f"求得{src.name}罩着自己")
            out.append(f"🤝 {src.name}罩下{dst.name}，结成靠山")

        elif ev.kind == "reconcile":
            # 讲和：双向敌意回落（可能压到打架线以下，避免一场架）、互增信任
            before = max(src.grudge.get(dst.id, 0.0), dst.grudge.get(src.id, 0.0))
            self._add_grudge(src, dst.id, DELTA_RECONCILE_GRUDGE)
            self._add_grudge(dst, src.id, DELTA_RECONCILE_GRUDGE)
            self._bump_rel(src, dst.id, "trust", DELTA_RECONCILE_TRUST)
            self._bump_rel(dst, src.id, "trust", DELTA_RECONCILE_TRUST)
            self._remember_mem(src, "reconciled", dst.id, f"和{dst.name}把旧账揭过了", 1.3)
            self._remember_mem(dst, "reconciled", src.id, f"和{src.name}把旧账揭过了", 1.3)
            after = max(src.grudge.get(dst.id, 0.0), dst.grudge.get(src.id, 0.0))
            if before >= GRUDGE_TO_FIGHT > after:
                out.append(f"🕊 {src.name}与{dst.name}讲和，一触即发的架散了")
            else:
                out.append(f"🕊 {src.name}与{dst.name}服软讲和")

        elif ev.kind == "info":
            self._bump_rel(dst, src.id, "trust", DELTA_INFO_TRUST)
            self._remember_mem(dst, "heard", src.id, f"{src.name}透了点风声给我", 1.0)

        elif ev.kind == "snitch":
            # 放话要去告密：震慑对方（怕 + 敌意），并给自己攒点假释表现（打点狱警）
            self._bump_rel(dst, src.id, "fear", DELTA_THREAT_FEAR * 0.6)
            self._add_grudge(dst, src.id, DELTA_THREAT_GRUDGE)
            self._remember_mem(dst, "snitched_at", src.id, f"{src.name}威胁要去狱警那儿告我", 1.6)
            out.append(f"🐍 {src.name}拿告密拿捏{dst.name}")

        if out:
            self.social_events.append({"tick": self.tick_count, "kind": ev.kind,
                                       "src": ev.src, "dst": ev.dst, "text": out[0]})
        return out

    def _resolve_weapon_deal(self, a: NPC, b: NPC, text: str) -> list[str]:
        """军火交易落地：谁买谁卖不看文本，看**世界事实**——
        卖方=手里有凶器/走私犯身份的人，买方=终极目标是杀人却还没家伙的人（生死网的引信）。
        谈成 → 买方真拿到刀、欠卖方一笔（义务账本），买方杀人任务链顺势推进一步。
        """
        def is_seller(n: NPC) -> bool:
            return bool(n.weapon) or "走私" in n.role or "军火" in (quest_of(n.id).ultimate if quest_of(n.id) else "")

        def wants_weapon(n: NPC) -> bool:
            q = quest_of(n.id)
            return bool(q and q.kind == "kill" and not n.weapon)

        seller = buyer = None
        if is_seller(a) and wants_weapon(b):
            seller, buyer = a, b
        elif is_seller(b) and wants_weapon(a):
            seller, buyer = b, a
        if seller is None or buyer is None:
            return []
        buyer.weapon = "knife"                      # 谈成：买方真拿到见血的家伙
        self.obligations.append({"debtor": buyer.id, "creditor": seller.id,
                                 "kind": "weapon", "text": f"{buyer.name}欠{seller.name}一把刀的人情",
                                 "tick": self.tick_count, "settled": False})
        self._bump_rel(buyer, seller.id, "debt", DELTA_DEAL_DEBT)
        self._bump_rel(buyer, seller.id, "trust", DELTA_DEAL_TRUST)
        self._bump_rel(seller, buyer.id, "trust", DELTA_DEAL_TRUST)
        self._remember_mem(buyer, "got_weapon", seller.id, f"从{seller.name}手里弄到了刀", 2.2)
        self._remember_mem(seller, "sold_weapon", buyer.id, f"把刀出给了{buyer.name}", 1.6)
        self._remember(buyer, f"从{seller.name}那儿弄到了刀")
        # 买方若正卡在"弄凶器"这步，推它一把（对白直接推进任务链）
        q = quest_of(buyer.id)
        if q and q.steps and buyer.quest_step < len(q.steps):
            step = q.steps[buyer.quest_step]
            if "GetWeapon" in step.focus or "Barter" in step.focus:
                self._advance_quest(buyer, q, "GetWeapon" if "GetWeapon" in step.focus else "Barter")
        self.bb.fate_log.append({
            "npc": buyer.id, "name": buyer.name, "status": "军火到手",
            "text": f"{seller.name}把一把磨亮的刀递给了{buyer.name}——这桩买卖，谈成了。",
            "big": True,
        })
        self.bb.fate_log[:] = self.bb.fate_log[-80:]
        return [f"🔪 成交：{seller.name}把刀递给了{buyer.name}"]

    def _rel_summary(self, npc: NPC, k: int = 3) -> str:
        """把关系图折成一句人话（挑最强的几条边），喂对话/决策，别让 LLM 读裸数字。"""
        parts: list[str] = []
        scored: list[tuple[float, str, str]] = []
        for oid, e in npc.rel.items():
            other = self._npc(oid)
            if other is None:
                continue
            axis, val = max(e.items(), key=lambda kv: kv[1])
            if val < 12:            # 低于一次社交事件的量级就当噪声，不进摘要
                continue
            scored.append((val, other.name, REL_CN.get(axis, axis)))
        for _v, name, cn in sorted(scored, reverse=True)[:k]:
            parts.append(f"{cn}{name}")
        return "、".join(parts)

    def _mem_score(self, m: dict, other_id: str | None) -> float:
        """一条记忆的检索分：α·近期 + β·重要 + γ·对象相关。

        旧实现只按 (w, tick) 排——高显著度的旧事永久占槽，对面是谁完全不管。
        于是疤脸跟瘸子催账时，prompt 里塞的是"被疯子威胁"（w=2.0、很早以前），
        两人之间那笔刀债反而挤不进去。这就是台词不引用历史的检索侧原因。

        相关性是符号的，不是余弦：`other` 字段就是元数据。对面是瘸子 → 关于瘸子的
        记忆 1.0，文本里点到瘸子 0.55，完全无关 0.12。硬过滤（只留 other==对方）
        会丢掉"我从疤脸手里弄到了刀"这种第三方事件，杀心试探时恰恰用得上。
        """
        age = max(0, self.tick_count - int(m.get("tick", 0)))
        recency = 0.5 ** (age / MEM_RECENCY_HALFLIFE)
        importance = min(1.0, float(m.get("w", 1.0)) / MEM_W_CEIL)
        oid = m.get("other") or ""
        if not other_id:
            relevance = 1.0
        elif oid == other_id:
            relevance = 1.0
        else:
            other = self._npc(other_id)
            name = other.name if other else ""
            relevance = 0.55 if (name and name in (m.get("text") or "")) else 0.12
        return (MEM_ALPHA * recency + MEM_BETA * importance + MEM_GAMMA * relevance)

    def _mem_lines(self, npc: NPC, k: int = 4, other_id: str | None = None) -> list[str]:
        """检索最该被此刻对话用到的 k 条情景记忆。"""
        ranked = sorted(npc.memory, key=lambda m: self._mem_score(m, other_id), reverse=True)
        return [m["text"] for m in ranked[:k]]

    def _npc_social_view(self, npc: NPC) -> dict:
        """决策卡/暂停面板用的社交摘要：盟友/仇人/欠谁/谁欠我/最近一条社交。

        上帝视角的"看得见"就靠这坨——让玩家点开一个人就懂他此刻的社会处境，
        而不只是需求条。全部折成人名短句，不给前端裸数字。
        """
        allies: list[str] = []
        foes: list[str] = []
        for oid, e in npc.rel.items():
            o = self._npc(oid)
            if o is None:
                continue
            if e.get("ally", 0.0) >= ALLY_NOFIGHT:
                allies.append(o.name)
            if npc.grudge.get(oid, 0.0) >= GRUDGE_TO_FIGHT or e.get("fear", 0.0) >= 45:
                foes.append(o.name)
        owe: list[str] = []
        owed: list[str] = []
        for ob in self.obligations:
            if ob.get("settled"):
                continue
            if ob["debtor"] == npc.id:
                c = self._npc(ob["creditor"])
                if c:
                    owe.append(c.name)
            elif ob["creditor"] == npc.id:
                d = self._npc(ob["debtor"])
                if d:
                    owed.append(d.name)
        mem = self._mem_lines(npc, 1)
        return {
            "summary": self._rel_summary(npc),
            "allies": allies, "foes": foes,
            "owe": owe, "owed": owed,
            "recent": mem[0] if mem else "",
        }

    def _settle_social(self, npc: NPC) -> None:
        """v3/P2：决策脑选了定向社交动作 → 落到近旁那个人身上（复用 P1 的关系写回）。

        仅在 PRISON_CAPTION=v3 时调用（见 tick），所以对默认 v2 线上零影响。带冷却，
        避免"每拍一个结盟/背叛"刷屏。谈成/结盟/背叛这类张力事件顺手进命运账本，前端可见。
        """
        goal = npc.goal
        if goal not in SOCIAL_ACTION_GOALS:
            return
        cd = getattr(self, "_social_cd", None)
        if cd is None:
            cd = self._social_cd = {}
        if self.tick_count - cd.get(npc.id, -999) < SOCIAL_COOLDOWN:
            return
        other = self._nearest_social(npc)
        if other is None:
            return
        cd[npc.id] = self.tick_count

        if goal == "Threaten":
            self._apply_social_event(SocialEvent("threat", npc.id, other.id, ""))
        elif goal == "Protect":
            self._apply_social_event(SocialEvent("protect", npc.id, other.id, ""))
        elif goal == "Ally":
            self._bump_rel(npc, other.id, "ally", DELTA_ALLY)
            self._bump_rel(other, npc.id, "ally", DELTA_ALLY)
            self._bump_rel(npc, other.id, "trust", DELTA_RECONCILE_TRUST)
            self._remember_mem(npc, "allied", other.id, f"跟{other.name}结了盟", 1.6)
            self._remember_mem(other, "allied", npc.id, f"跟{npc.name}结了盟", 1.6)
            self.social_events.append({"tick": self.tick_count, "kind": "ally",
                                       "src": npc.id, "dst": other.id,
                                       "text": f"🤝 {npc.name}与{other.name}结成同盟"})
            self.bb.fate_log.append({"npc": npc.id, "name": npc.name, "status": "结盟",
                                     "text": f"{npc.name}与{other.name}拜了把子，从此攻守相帮。",
                                     "big": True})
            self.bb.fate_log[:] = self.bb.fate_log[-80:]
        elif goal == "Betray":
            self._rel(npc, other.id)["ally"] = 0.0
            self._rel(other, npc.id)["ally"] = 0.0
            self._add_grudge(other, npc.id, GRUDGE_ON_LOSS)
            self._bump_rel(other, npc.id, "trust", -REL_MAX)   # 信任清零
            self._remember_mem(other, "betrayed_by", npc.id, f"被{npc.name}摆了一道", 2.4)
            self._remember_mem(npc, "betrayed", other.id, f"我出卖了{other.name}", 1.8)
            self.social_events.append({"tick": self.tick_count, "kind": "betray",
                                       "src": npc.id, "dst": other.id,
                                       "text": f"🗡 {npc.name}背叛了{other.name}"})
            self.bb.fate_log.append({"npc": other.id, "name": other.name, "status": "被背叛",
                                     "text": f"{npc.name}反手出卖了{other.name}——盟友转眼成了刀。",
                                     "big": True})
            self.bb.fate_log[:] = self.bb.fate_log[-80:]
        elif goal == "Repay":
            for ob in self.obligations:
                if not ob.get("settled") and ob["debtor"] == npc.id and ob["creditor"] == other.id:
                    ob["settled"] = True
                    self._rel(npc, other.id)["debt"] = 0.0
                    self._bump_rel(other, npc.id, "trust", DELTA_DEAL_TRUST)
                    self._remember_mem(npc, "repaid", other.id, f"把欠{other.name}的人情还了", 1.3)
                    self.social_events.append({"tick": self.tick_count, "kind": "repay",
                                               "src": npc.id, "dst": other.id,
                                               "text": f"💰 {npc.name}还清了欠{other.name}的人情"})
                    break
        elif goal == "Propose":
            # 先看能不能谈成军火（生死网引信）；谈不成就当普通往来，攒点信任
            done = self._resolve_weapon_deal(npc, other, "")
            if done:
                self.social_events.append({"tick": self.tick_count, "kind": "deal_weapon",
                                           "src": npc.id, "dst": other.id, "text": done[0]})
            else:
                # 泛泛攀谈：攒点信任/记忆即可，不进社交字幕流（否则"攀谈"刷屏盖住有戏的社交）
                self._bump_rel(npc, other.id, "trust", DELTA_INFO_TRUST)
                self._bump_rel(other, npc.id, "trust", DELTA_INFO_TRUST)
                self._remember_mem(npc, "dealt", other.id, f"跟{other.name}谈了笔小生意", 1.0)

    def _bout_of(self, npc: NPC) -> FightBout | None:
        return next((b for b in self.bouts if b.id == npc.in_bout and not b.over), None)

    def _maybe_start_fight(self, npc: NPC) -> None:
        """选了打架又确有对象在场 → 开一场 bout，并把对方拖进来。

        被打不是可选项：对方立刻也进入打架状态。这就是"两人及以上一起打架"。
        """
        if npc.goal != "Fight" or npc.in_bout or npc.solitary_ticks > 0 or npc.fight_cd > 0:
            return
        foe = self._npc(npc.fight_target) if npc.fight_target else None
        if foe is None or foe.escaped or foe.dead or foe.solitary_ticks > 0 or foe.fight_cd > 0:
            return
        if math.hypot(foe.x - npc.x, foe.z - npc.z) > FIGHT_RANGE:
            return
        existing = self._bout_of(foe)
        if existing is not None:              # 对方已在打 → 加入同一场
            existing.members.append(npc.id)
            npc.in_bout = existing.id
            existing.log.append(f"{npc.name}也扑了上去")
            return
        bout = FightBout(id=f"bout{self.tick_count}_{npc.id}",
                         members=[npc.id, foe.id], x=npc.x, z=npc.z)
        bout.log.append(f"{npc.name}朝{foe.name}动手了")
        npc.in_bout = bout.id
        foe.in_bout = bout.id
        self._remember(npc, f"你朝{foe.name}动了手，正打着")
        self._remember(foe, f"{npc.name}突然朝你动手，你在还手")
        # 被打的人被迫应战：目标改成打架，并把仇记在动手的人头上
        self._release_object(foe)
        foe.goal = "Fight"
        foe.target = None
        foe.act_stage = "goto"
        foe.fight_target = npc.id
        self._add_grudge(foe, npc.id, GRUDGE_ON_HIT)
        self.bouts.append(bout)
        self.fight_log.append(f"[{self.tick_count}] {npc.name} vs {foe.name} 打起来了")

    def _advance_bouts(self) -> None:
        """推进所有正在进行的打架：旁人加入 → 互相造成伤害 → 判输 → 狱警抓现行。"""
        for bout in list(self.bouts):
            if bout.over:
                continue
            members = [n for n in (self._npc(i) for i in bout.members)
                       if n is not None and not n.escaped
                       and n.recall_ticks <= 0 and n.solitary_ticks <= 0]
            if len(members) < 2:                     # 对手跑了/被关了，散场
                self._end_bout(bout, members)
                continue
            bout.ticks += 1
            bout.x = sum(m.x for m in members) / len(members)
            bout.z = sum(m.z for m in members) / len(members)

            self._maybe_join_bout(bout, members)
            members = [n for n in (self._npc(i) for i in bout.members) if n is not None]

            # 互相造成伤害。为了"人多欺负人少"能体现，战力按各自的敌友数算。
            n = len(members)
            dmg: dict[str, float] = {}
            for me in members:
                foes = [o for o in members if o.id != me.id]
                s_me = strength(me, allies=1, enemies=len(foes))
                for foe in foes:
                    s_foe = strength(foe, allies=1, enemies=n - 1)
                    dmg[foe.id] = dmg.get(foe.id, 0.0) + roll_damage(s_me, s_foe) / max(1, len(foes))
            for me in members:
                d = dmg.get(me.id, 0.0)
                me.needs.injury = min(100, me.needs.injury + d)
                me.needs.stress = min(100, me.needs.stress + STRESS_PER_TICK)
                # 打着架彼此的仇越结越深
                for foe in members:
                    if foe.id != me.id:
                        self._add_grudge(me, foe.id, GRUDGE_ON_HIT * 0.25)

            # 狱警/狱犬当场看到 → 抓现行（比"搜出违禁"更直接的处罚来源）
            if not bout.busted and any(self._watcher_of(m.x, m.z) for m in members):
                bout.busted = True
                self._bust_fight(bout, members)
                self._end_bout(bout, members)
                continue

            # 有人被打到 KO → 判输；伤得够重则有概率当场丧命（夜班要有伤亡）
            hurt = max(members, key=lambda m: m.needs.injury)
            if hurt.needs.injury >= KO_INJURY or bout.ticks >= FIGHT_MAX_TICKS:
                bout.loser = hurt.id if hurt.needs.injury >= KO_INJURY else ""
                # 被打到 KO 也多半只是判输重伤，不再"打到 90+ 就必死"——群殴致死现在只看
                # 那一次 COLLATERAL_LETHAL 掷骰(7%)，让随机群殴很少见血，见血更多来自仇杀。
                if (hurt.needs.injury >= KO_INJURY
                        and not hurt.dead
                        and random.random() < COLLATERAL_LETHAL):
                    # 点名把他打倒的对手（场上仇最深、否则战力最高的那个活人），
                    # 让"混战殃及"不再是无主的"被搏杀所伤"——玩家看得出是谁下的死手。
                    others = [o for o in members if o.id != hurt.id and not o.dead]
                    ko_by = max(others, key=lambda o: (hurt.grudge.get(o.id, 0.0),
                                                       strength(o, 1, 1)),
                                default=None) if others else None
                    self._maybe_collateral_death([hurt], "斗殴", force=True, by=ko_by)
                self._end_bout(bout, members)

    def _maybe_join_bout(self, bout: FightBout, members: list[NPC]) -> None:
        """旁人卷进来：暴躁的爱凑热闹，对场内某人有仇的更想踹一脚。"""
        mem_ids = {m.id for m in members}
        for other in self.npcs:
            if other.id in mem_ids or other.escaped or other.solitary_ticks > 0:
                continue
            if other.in_bout or other.watched_by or other.fight_cd > 0:
                continue                               # 被狱警盯着 / 刚打完的不掺和
            if math.hypot(other.x - bout.x, other.z - bout.z) > JOIN_RANGE:
                continue
            wants = max((other.grudge.get(m.id, 0.0) for m in members), default=0.0)
            chance = JOIN_CHANCE * (1.5 if other.personality == "暴躁" else 1.0)
            if wants < GRUDGE_TO_FIGHT and other.personality != "暴躁":
                continue
            if random.random() > chance:
                continue
            bout.members.append(other.id)
            other.in_bout = bout.id
            self._release_object(other)
            other.goal = "Fight"
            other.target = None
            other.act_stage = "goto"
            bout.log.append(f"{other.name}也掺和进来了")
            self.fight_log.append(f"[{self.tick_count}] {other.name}加入了群殴")

    def _end_bout(self, bout: FightBout, members: list[NPC]) -> None:
        bout.over = True
        loser = self._npc(bout.loser) if bout.loser else None
        for m in members:
            m.in_bout = ""
            m.fight_target = ""
            # 散架冷却。没有它会退化成"赢家每拍重开一场，输家被永远按在 100 伤"，
            # 输家连去医务室的机会都没有，"打架→受伤→治伤"这条链就断在中间。
            m.fight_cd = FIGHT_COOLDOWN_LOSER if (loser and m.id == loser.id) \
                else FIGHT_COOLDOWN
            if loser is not None and m.id != loser.id:
                # 输家记恨所有赢家
                self._add_grudge(loser, m.id, GRUDGE_ON_LOSS)
        if loser is not None and not loser.dead:
            bout.log.append(f"{loser.name}被打趴下了")
            self.fight_log.append(
                f"[{self.tick_count}] {loser.name}打输了（伤势{loser.needs.injury:.0f}）")
            self._remember(loser, f"你刚被打输了，伤得不轻（伤势{loser.needs.injury:.0f}）")
            for m in members:
                if m.id != loser.id:
                    self._remember(m, f"你刚把{loser.name}打趴下了")
            # 伤成这样只能去医务室 —— 伤势第一次有了真实去处，也真的会消耗医疗包
            self._release_object(loser)
            loser.goal = "Cure"
            loser.target = self._target_for_goal(loser, "Cure")
            loser.act_stage = "goto"
            loser.goal_ticks = 0
        else:
            bout.log.append("这场架不了了之")
        for m in members:
            if m.goal == "Fight":
                m.goal = None                 # 散了，下一拍重新决策
                m.act_stage = "goto"
        del self.bouts[:max(0, len(self.bouts) - 6)]   # 只留最近几场给前端看

    def _bust_fight(self, bout: FightBout, members: list[NPC]) -> None:
        """狱警目击群殴：全体关禁闭 + 加刑 + 表现分清算。"""
        names = "、".join(m.name for m in members)
        for m in members:
            m.solitary_ticks = SOLITARY_TICKS
            m.busted_cd = BUSTED_RECENT_TICKS
            m.busted += 1
            m.needs.stress = min(100, m.needs.stress + 25)
            self._penalize_parole(m, PAROLE_FIGHT_PENALTY, EXTRA_DAYS_FIGHT, "聚众斗殴被抓")
            self._remember(m, f"你因为聚众斗殴被狱警当场抓走关了禁闭，还加了{EXTRA_DAYS_FIGHT}天刑")
        bout.log.append("狱警冲进来了")
        self.fight_log.append(f"[{self.tick_count}] 狱警当场抓住斗殴：{names} → 关禁闭")
        self.busts.append(f"{names}：聚众斗殴，关禁闭")
        del self.fight_log[:-10]

    # -- 行动分阶段执行（物资层的核心）---------------------------------------
    def _advance_action(self, npc: NPC) -> None:
        """前往对应房间 →（生存类还要取用物资）→ 停够拍数 → 一次性结算。

        洗澡/吃饭/娱乐不能在走廊比划一下就算做成：必须人在功能室净空里，
        并且 act_stage=perform 停够 dwell。取不到饭/龙头/铺/药 → failed，需求不缓解。
        """
        goal = npc.goal
        if goal is None or goal in _SKIP_DWELL:
            return
        if npc.act_stage in ("done", "failed"):
            return
        if not self._at_target(npc):
            npc.act_stage = "goto"
            return

        need = REQUIRED_OBJECT.get(goal)
        if npc.act_stage == "goto":
            if need:
                if npc.carrying != need:
                    got = self._acquire_for(npc, goal, need)
                    if not got:
                        npc.act_stage = "failed"
                        npc.failure_pen[goal] = min(
                            0.9, npc.failure_pen.get(goal, 0.0) + FAILURE_PENALTY)
                        self._remember(npc, f"你想{GOAL_LABEL.get(goal, goal)}，结果{npc.act_fail}")
                        return
            npc.act_stage = "perform"
            npc.act_ticks = 0

        if npc.act_stage == "perform":
            npc.act_ticks += 1
            if npc.act_ticks < self._dwell_ticks(goal):
                return
            if need:
                for key, dv in PERFORM_EFFECT.get(goal, {}).items():
                    cur = getattr(npc.needs, key)
                    setattr(npc.needs, key, max(0, min(100, cur + dv)))
                if npc.carrying == need:
                    npc.carrying = ""
                self._release_object(npc)
                self._reward_parole(npc, goal)
            elif goal in ACT_RELIEF:
                n = npc.needs
                for key, dv in ACT_RELIEF[goal].items():
                    setattr(n, key, max(0, min(100, getattr(n, key) + dv)))
            else:
                k = GOAL_TO_KIND.get(goal, "")
                n = npc.needs
                if k == "yard":
                    n.boredom = max(0, n.boredom - 40)
                    n.social = max(0, n.social - 30)
                elif k == "class":
                    n.boredom = max(0, n.boredom - 30)
            npc.act_stage = "done"

    def _at_target(self, npc: NPC) -> bool:
        """到了才算到：功能室看房间净空，回屋看自己那间，院子看落点附近。"""
        goal = npc.goal
        if goal is None:
            return False
        if npc.target is None:
            # 真原地动作才算到了。寻路失败会把 target 清掉，不能当成已经进了屋。
            return npc.goal in _SKIP_DWELL
        kind = GOAL_TO_KIND.get(goal, "")
        if goal in _CELL_GOALS or kind == "sleep":
            return self._in_own_cell(npc)
        if kind in _ROOM_KINDS:
            f = self._facility_for_goal(goal)
            if f is None:
                f = next((x for x in self.facilities if x.kind == kind), None)
            return f is not None and self._in_footprint(npc.x, npc.z, f)
        reach = 3.2 if kind == "yard" else (1.6 if kind == "gate" else 0.7)
        return math.hypot(npc.target.x - npc.x, npc.target.z - npc.z) < reach

    def _action_state(self, npc: NPC) -> str:
        """给前端的一句话状态。分阶段后"进行中"太粗，得能看出卡在取物资还是在做。"""
        if npc.goal is None:
            return "待定"
        if npc.act_stage == "failed":
            return npc.act_fail or "没做成"
        if npc.act_stage == "done":
            return "已完成"
        # 打架/收敛/社交没有 perform 阶段：到了就是在做。
        if npc.goal in _SKIP_DWELL:
            return "前往中" if not self._at_target(npc) else "进行中"
        if not self._at_target(npc):
            return "前往中"
        if npc.act_stage == "goto" and npc.goal in REQUIRED_OBJECT:
            return "取物资"
        if npc.act_stage == "perform":
            return "进行中"
        return "前往中"

    def _act_progress(self, npc: NPC) -> float:
        """执行阶段的进度 0~1（前端画进度条：吃饭/睡觉/洗澡都有停留）。"""
        if npc.goal is None or npc.act_stage != "perform":
            return 0.0
        total = self._dwell_ticks(npc.goal)
        return round(min(1.0, npc.act_ticks / total), 3) if total else 0.0

    def _acquire_for(self, npc: NPC, goal: str, need: str) -> bool:
        """在当前所在设施里占一份物资。成功则登记到 npc（可携带的进手里）。"""
        f = npc.target or self._facility_for_goal(goal)
        if f is None:
            npc.act_fail = "没有落点"
            return False
        pool = self._objects_at(f.id, need)
        # 铺位认人：一人一间隔间、一间一张铺，只能占自己那张。否则谁先到谁占，
        # 会出现"回自己屋却睡在别人铺上"（落点在这间、铺位记在那间）。
        if need == "bed" and npc.cell >= 0:
            pool = [o for o in pool if o.slot == npc.cell] or pool
        for o in pool:
            if o.take(npc.id):
                if o.consumable:
                    # 可消耗的拿进手里（吃/用的时候才真正消失）
                    npc.carrying = need
                    if o.stock == 0:
                        self.supply_log.append(f"{o.name}被{npc.name}拿完了")
                        self._blame_for_last_portion(npc.id, npc.name, npc.x, npc.z, need)
                else:
                    npc.act_obj = o.id        # 占用式：记下来好释放
                npc.act_fail = ""
                return True
        npc.act_fail = {
            "food": "食堂没饭了", "water": "水龙头都被占着",
            "bed": "铺位满了", "medkit": "药柜空了",
        }.get(need, "没拿到东西")
        return False

    def _blame_for_last_portion(self, taker_id: str, taker_name: str,
                                tx: float, tz: float, need: str) -> None:
        """最后一份被拿走，而别人正饿着 → 记恨拿走的人。**玩家也算**。

        这条把物资层和打架层缝在了一起：抢资源结仇 → 仇怨过阈值 → 真的会动手。
        属性(饥饿)、世界内容(窝头份数)、人际(仇怨)、事件(挨打)于是形成一条闭环——
        玩家把最后一个窝头揣走，回头就可能在放风场被人堵着揍。
        """
        want = {"food": "hunger", "medkit": "injury"}.get(need)
        if not want:
            return
        for other in self.npcs:
            if other.id == taker_id or other.escaped or other.solitary_ticks > 0:
                continue
            if getattr(other.needs, want) < 55:      # 自己不缺，就不至于记恨
                continue
            if math.hypot(other.x - tx, other.z - tz) > FIGHT_RANGE:
                continue
            self._add_grudge(other, taker_id, GRUDGE_ON_STOLEN_SUPPLY)
            self.supply_log.append(f"{other.name}记恨{taker_name}抢了最后一份")
            self._remember(other, f"{taker_name}把最后一份{'饭' if need == 'food' else '药'}抢走了，你记着这事")

    def _patroller(self, pid: str) -> Patroller | None:
        return next((p for p in self.patrollers if p.id == pid), None)

    def npc_chat_context(self, npc_id: str, other_id: str | None = None) -> dict | None:
        """给对话用：某 NPC 的人设 + 前史/剧情/知情线索 + 当前处境快照。

        `other_id` 是对面那个人（NPC↔NPC 时传入、玩家对话时不传）。记忆检索会按
        对象相关性加权，避免把跟这场话无关的旧事塞进 prompt。

        返回 None 即"此人不可对话"，_handle_chat 会直接静默丢弃这条消息。
        死者走这条路：前端已经在多处挡了入口，但对话是唯一会真的去调语言模型的动作，
        漏一个入口就等于花几秒钟让尸体开口，所以在这里兜死。
        """
        npc = self._npc(npc_id)
        if npc is not None and npc.dead:
            return None
        if npc is None:
            # 破例：狱警铁面可对话（用于抚恤金名单收买）。用事实库当护栏，不给决策字段。
            if npc_id == GUARD_ID:
                p = self._patroller(GUARD_ID)
                return {
                    "name": p.name if p else "铁面",
                    "personality": "铁面无私",
                    "role": "巡逻的狱警",
                    "caption": "你在巡逻，眼睛扫过每一个犯人。",
                    "goal_label": "巡逻",
                    "monologue": "都给我规矩点。",
                    "backstory": "", "speech": "", "attitude": "", "secret": "", "openness": "紧",
                    "softspot": SOFT_SPOT.get(GUARD_ID, ""),
                    "facts": fact_lines(GUARD_ID, self.bb),
                    "case_progress": self.case_progress,
                    "motive": "", "stake": "", "drive_progress": 0,
                    "parole_lost": False, "crime": "", "sentence_years": 0, "labor_job": "", "busted": 0,
                }
            # 终局 Boss 典狱长：仅在办公室解锁后可对话，用事实库当护栏、不进决策
            if npc_id == WARDEN_ID:
                return {
                    "name": "典狱长",
                    "personality": "贪婪多疑、官腔十足",
                    "role": "州立监狱典狱长",
                    "caption": "监狱正在暴乱，警铃大作，你被堵在自己的办公室里。",
                    "goal_label": "弹压暴乱",
                    "monologue": "都给我滚回牢房去！",
                    "backstory": "", "speech": "", "attitude": "", "secret": "", "openness": "紧",
                    "softspot": SOFT_SPOT.get(WARDEN_ID, ""),
                    "facts": fact_lines(WARDEN_ID, self.bb),
                    "case_progress": self.case_progress,
                    "motive": "", "stake": "", "drive_progress": 0,
                    "parole_lost": False, "crime": "", "sentence_years": 0, "labor_job": "", "busted": 0,
                }
            return None
        seed = self._seed_by_id.get(npc_id)
        d = self._drive_of(npc)
        return {
            "name": npc.name,
            "personality": npc.personality,
            "role": npc.role,
            "caption": npc.last_context,
            "goal_label": GOAL_LABEL.get(npc.goal or "", npc.goal or "发呆"),
            "monologue": self._say(npc),
            # 前史/剧情/知情线索：只喂对话，不进决策 caption
            "backstory": getattr(seed, "backstory", ""),
            "speech": getattr(seed, "speech", ""),
            "attitude": getattr(seed, "attitude", ""),
            "secret": getattr(seed, "secret", ""),
            "softspot": SOFT_SPOT.get(npc_id, ""),
            "openness": self._eased_openness(getattr(seed, "openness", "中")),
            # 事实库护栏（挂 prompt 最顶端的硬设定，防人设崩/剧情暴走）
            "facts": fact_lines(npc_id, self.bb),
            "case_progress": self.case_progress,
            # 宏观动机 / 判决 / 苦役：让对话跟长期心思与刑期一致，不是聊天时另起一套人设
            "motive": d.motive,
            "stake": d.stake,
            "drive_progress": round(npc.drive_progress),
            # 终极目标 + 当前分目标：让对话跟"他这辈子图的那件大事+眼下这一步"对得上
            "ultimate": (quest_of(npc_id).ultimate if quest_of(npc_id) else ""),
            "substep": self._substep_desc(npc),
            "plot_surface": plot_of(npc).get("surface", ""),
            "plot_step": plot_of(npc).get("step", ""),
            "plot_fog": plot_of(npc).get("fog", ""),
            "parole_lost": npc.parole_lost,
            "crime": npc.crime,
            "sentence_years": npc.sentence_years,
            "labor_job": npc.labor_job,
            "busted": npc.busted,
            # 物理层实况：让"说的"和"世界里刚发生的"对得上（见 NPC.recent_events）
            "recent_events": list(npc.recent_events),
            # P1 情景记忆 + 关系图：让这场对话接得上之前的经历/恩怨（叙事连续性=涌现）
            "memory": self._mem_lines(npc, 4, other_id=other_id),
            "relationships": self._rel_summary(npc, 3),
            "body": self._body_line(npc),
            "grudge_with": self._grudge_line(npc),
            "parole_state": self._parole_line(npc),
            "carrying": {"food": "手里攥着个窝头", "medkit": "手里拿着医疗包"}.get(npc.carrying, ""),
            "in_solitary": npc.solitary_ticks > 0,
            "surroundings": self._surroundings_line(npc),
            "incident": self.shift.chat_incident(self, npc_id),
            # —— 一致性管线：把价值模型定的最终决策归因（world._compute_drivers）原样喂给对话，
            # 让"说的"严格作"做的"的配音（三方读同一份 decision_drivers，天然不矛盾）——
            "drivers": dict(npc.decision_drivers) if npc.decision_drivers else {},
            "crisis": self._crisis_line(npc),
        }

    def _crisis_line(self, npc: NPC) -> str:
        """危机实时打断对话的触发语：被盯/濒死时没心思闲聊。空串=没在紧要关头。"""
        if npc.watched_by:
            return f"{npc.watched_by}正盯着你，你在强压着别露馅——"
        worst = max(npc.needs.hunger, npc.needs.thirst, npc.needs.injury, npc.needs.fatigue)
        if worst >= EMERGENCY_NEED:
            return "你身体已到极限、快撑不住了——"
        return ""

    # -- 给对话用的处境短句（把数值翻成人话，别让 LLM 自己解读裸数字）--------
    @staticmethod
    def _body_line(npc: NPC) -> str:
        inj = npc.needs.injury
        if inj >= 70:
            return "你伤得很重，站着都疼，随时可能栽下去"
        if inj >= 40:
            return "你身上有伤，疼得龇牙"
        if inj >= 15:
            return "你身上带着点淤青擦伤"
        return ""

    def _grudge_line(self, npc: NPC) -> str:
        top = sorted(npc.grudge.items(), key=lambda kv: -kv[1])[:2]
        parts = []
        for oid, v in top:
            other = self._npc(oid)
            if other is None or v < 35:
                continue
            parts.append(f"{other.name}（{'恨之入骨' if v >= 70 else '很不对付'}）")
        return "、".join(parts)

    @staticmethod
    def _parole_line(npc: NPC) -> str:
        if npc.parole_lost:
            return ""          # 破罐破摔已由 parole_lost 单独成段，别重复
        if npc.extra_days:
            return f"你近来犯了事，被加了{npc.extra_days}天刑，出去的日子又远了"
        if npc.parole_near:
            return "你表现攒得不错，假释就在眼前，这阵子格外怕出岔子"
        return ""

    def _eased_openness(self, base: str) -> str:
        """案件查得越深，知情 NPC 越松口：紧→中→松 随线索数递进。"""
        order = ["紧", "中", "松"]
        i = order.index(base) if base in order else 1
        i = min(len(order) - 1, i + self.case_progress // 3)   # 每查明 3 条松一档
        return order[i]

    def _say(self, npc: NPC) -> str:
        """台词：LLM 的即兴独白只在它确实是为当前目标说的时候才用，否则回退模板。

        引擎侧乘数会改排序，硬用 LLM 独白就会出现台词和动作对不上的画面。
        """
        if npc.watched_by:
            return self._watched_line(npc)
        # LLM 独白：有池子就逐句放（每句停 MONO_HOLD_TICKS 拍再换下一句，别每帧就跳）。
        # 刻意与微观 goal 解耦——心里话讲的是他此刻的处境/盘算/心结（撑得住 ~30s），
        # 比"我在吃饭"更有人味；goal 每两秒翻一次不该把心里话打断。真打起来(Fight)才让位应激台词。
        if npc.goal != "Fight" and (npc.llm_say_pool or npc.llm_say):
            if npc.llm_say_pool and self.tick_count >= npc.llm_say_next_tick:
                npc.llm_say = npc.llm_say_pool.pop(0)
                npc.llm_say_next_tick = self.tick_count + MONO_HOLD_TICKS
            if npc.llm_say:
                return npc.llm_say
        ctx = npc.last_ctx
        # 多讲"终极目标/眼下这一步"的心里话，少复读"我好饿/好累"这类日常需求。
        # 只有在真正打起来(Fight)时才让位给应激台词——全局"斗殴"氛围事件不算个人危机，
        # 否则放风时段会把所有人的任务心思全盖掉（那正是之前一句任务话都冒不出来的原因）。
        q = quest_of(npc.id)
        sub = self._substep_desc(npc)
        focus_now = bool(q and npc.quest_step < len(q.steps)
                         and npc.goal in q.steps[npc.quest_step].focus)
        # 诚实配音（治"边吃饭边念叨去找瘸子弄刀"的言行脱节）：
        #   · 正在推进主线(focus_now) → 直说这一步的心思；
        #   · 在忙需求/闲逛但承诺压力已高 → 改口成"惦记着主线却先忙眼下"的纠结话，
        #     顺带给"马上要起身去办"埋伏笔；
        #   · 否则老老实实说当前这个动作的话（干啥说啥）。
        if sub and npc.goal != "Fight":
            if focus_now:
                return self._quest_say(npc, sub)
            if npc.quest_pressure >= PRESSURE_TENSION:
                return self._quest_deferral_say(npc, sub)
        return monologue(npc.goal, npc.personality, ctx,
                         npc.commanded_goal, self.tick_count // 6)

    def mono_candidates(self) -> list[NPC]:
        """该续 LLM 独白池的 NPC：池子空了且当前这句也停够了拍数（该换新的一批）。
        被盯/打架/关禁闭/越狱/死了的不续（各有专用台词或不该说话）。"""
        out = []
        for npc in self.npcs:
            if npc.dead or npc.escaped or npc.solitary_ticks > 0 or npc.watched_by:
                continue
            if getattr(npc, "in_bout", False):
                continue
            drained = (not npc.llm_say_pool
                       and (npc.llm_say is None or self.tick_count >= npc.llm_say_next_tick))
            if drained:
                out.append(npc)
        return out

    def set_mono(self, npc_id: str, goal: str, lines: list[str]) -> None:
        """把 Opus 产的 8 句心里话灌进某 NPC 的独白池（下一次 _say 就开始逐句放）。"""
        npc = self._npc(npc_id)
        if npc is None or not lines:
            return
        npc.llm_say_pool = [str(s).strip() for s in lines if str(s).strip()]
        npc.llm_say_goal = goal
        npc.llm_say = None
        npc.llm_say_next_tick = 0

    # 任务相关自语的口吻模板（把"眼下这一步"包装成一句心里话，按 seed 轮换）
    _QUEST_SAY_TMPL = (
        "{s}……", "得抓紧，{s}。", "{s}，这一步不能出岔子。",
        "心里就一桩事：{s}。", "别急，{s}，一步步来。", "{s}——就差这么一哆嗦了。",
    )

    def _quest_say(self, npc: NPC, sub: str) -> str:
        # 杀手贴到最后一步时，自语透出杀机（凶器在手更狠）
        if npc.kill_target and npc.weapon and npc.goal == "Assassinate":
            wp = WEAPON_CN.get(npc.weapon, "家伙")
            return f"就等他落单……这把{wp}，今天就了结了。"
        i = (self.tick_count // 6) % len(self._QUEST_SAY_TMPL)
        return self._QUEST_SAY_TMPL[i].format(s=sub)

    # 忙需求/闲逛、但主线已压得心焦时的诚实独白：干着眼下的事，嘴上惦记着正事
    _DEFERRAL_TMPL = (
        "先{v}垫一下……可{s}，拖不得了。",
        "{v}归{v}，心里那桩事撂不下：{s}。",
        "手上{v}，脑子里全是——{s}。",
        "{v}完这趟就去办正事：{s}。",
    )

    def _quest_deferral_say(self, npc: NPC, sub: str) -> str:
        v = activity(npc.goal).get("verb", "忙活")
        i = (self.tick_count // 6) % len(self._DEFERRAL_TMPL)
        return self._DEFERRAL_TMPL[i].format(v=v, s=sub)

    @staticmethod
    def _watched_line(npc: NPC) -> str:
        """被盯时的独白：点出是狱警还是狱犬，并按性格上色（比数字更能让玩家看懂）。"""
        who = npc.watched_by
        complying = npc.goal == "LayLow"
        by_pers = {
            "暴躁": (f"{who}又盯上我了…忍着，别给他借口。" if complying
                    else f"{who}瞪什么瞪，老子先把这事办完。"),
            "老油条": (f"{who}过来了，装作没事，手别乱动。" if complying
                      else f"{who}在呢，悠着点，别让他看出门道。"),
            "书呆": f"{who}在看…我、我什么都没干，站好站好。",
            "新人": f"{who}盯着我，好紧张，别乱动别乱动。",
            "普通": (f"{who}在这盯着，先老实待着吧。" if complying
                    else f"{who}看着呢，赶紧把手头的收一收。"),
        }
        return by_pers.get(npc.personality, by_pers["普通"])

    @staticmethod
    def _needs_dict(n: NeedState) -> dict:
        return {
            "hunger": n.hunger, "thirst": n.thirst, "fatigue": n.fatigue,
            "boredom": n.boredom, "social": n.social, "stress": n.stress, "injury": n.injury,
        }

    def _situation_dict(self, ctx: ContextState | None) -> dict:
        if ctx is None:
            return {}
        return {
            "phase": ctx.phase,
            "threat": ctx.threat,
            "event": ctx.event,
            "surveillance": ctx.surveillance,
            "backer": ctx.backer_present,
            "enemy": ctx.enemy_present,
            "yard_crowd": ctx.yard_crowd,
            "parole_near": ctx.parole_near,
            "has_labor": ctx.has_labor,
            "has_corpse": ctx.has_corpse,
        }

    # -- NPC↔NPC 对话（观察者模式）：触发检测 + 领取 + 落地 -------------------
    def _convo_active(self, npc: NPC) -> bool:
        """能参与攀谈：没在禁闭、没越狱、没在打架，且所处分区已解锁（前端看得见）。"""
        return (npc.solitary_ticks <= 0 and not npc.escaped and not npc.dead and not npc.in_bout
                and (npc.home_zone in self.bb.zones_unlocked))

    def _detect_convo(self) -> None:
        """每拍挑一对最有戏的邻座，排进 convo_pending，等 _sim_loop 懒生成脚本。"""
        for k in list(self._convo_pair_cd):        # 同对冷却递减
            self._convo_pair_cd[k] -= 1
            if self._convo_pair_cd[k] <= 0:
                del self._convo_pair_cd[k]
        if self.convo_cd > 0:
            self.convo_cd -= 1
            return
        if self.convo_pending is not None:         # 已排了一场在等生成
            return
        actives = [n for n in self.npcs if self._convo_active(n)]
        best = None       # ((tier, score), a, b, kind, topic)
        for i, a in enumerate(actives):
            for b in actives[i + 1:]:
                if math.hypot(a.x - b.x, a.z - b.z) > CONVO_RANGE:
                    continue
                if frozenset((a.id, b.id)) in self._convo_pair_cd:
                    continue
                # 先看这两人之间有没有剧情钩子（生死网/军火债/结盟/旧仇）——有则按剧情
                # 定类型+话题+优先级；没有钩子才退回"两人都在社交动作→泛泛闲聊"。
                plan = self._convo_plan(a, b)
                if plan is None:
                    if a.goal in CONVO_SOCIAL_GOALS and b.goal in CONVO_SOCIAL_GOALS:
                        plan = ("chat", self._fallback_topic(),
                                40 - math.hypot(a.x - b.x, a.z - b.z), HOOK_TIER_FALLBACK)
                    else:
                        continue
                kind, topic, score, tier = plan
                # 先比钩子等级、同级内才比分数。只比分数会让 docstring 写的优先级失效：
                # 生死网固定 120，而旧仇是 90+g 且 g>=CONVO_GRUDGE_MIN(35) → 至少 125，
                # 于是场上只要有任意一对积怨到线，最有戏的那场杀心试探就永远排不上号。
                if best is None or (tier, score) > best[0]:
                    best = ((tier, score), a, b, kind, topic)
        if best is not None:
            _rank, a, b, kind, topic = best
            self.convo_pending = {"a": a.id, "b": b.id, "kind": kind, "topic": topic}

    def _convo_plan(self, a: NPC, b: NPC) -> tuple[str, str, float, int] | None:
        """按这两人真实的剧情关系定这场对话：返回 (kind, topic, score, tier)，无钩子则 None。

        话题不是泛泛"发牢骚"，而是点名到他们此刻真正的过节/图谋——让对话成为
        剧情的配音，而不是背景噪声。优先级：生死网 > 军火债 > 旧仇 > 结盟。

        tier 就是上面这个优先级，调用方**必须先比 tier 再比 score**：各档的 score 量纲
        并不可比（旧仇是 90+积怨，能盖过生死网的固定 120），只比分数会把优先级颠倒过来。
        """
        # 生死网：一方的终极目标就是干掉另一方 → 杀心藏在闲话底下的试探（最高优先）
        for killer, victim in ((a, b), (b, a)):
            q = quest_of(killer.id)
            if q and q.kind == "kill" and q.target == victim.id:
                topic = (f"{killer.name}借着搭话不动声色地试探、逼近{victim.name}——"
                         f"杀心藏在闲话底下，{victim.name}只隐隐觉得哪里不对劲")
                return ("chat", topic, 120.0, HOOK_TIER_KILL)
        # 军火/旧账：义务账本里两人有一笔未结的债 → 一个催、一个拖
        for ob in self.obligations:
            if ob.get("settled") or {ob["debtor"], ob["creditor"]} != {a.id, b.id}:
                continue
            deb, cre = self._npc(ob["debtor"]), self._npc(ob["creditor"])
            if deb and cre:
                topic = (f"{cre.name}向{deb.name}催那笔『{ob.get('text', '旧账')}』，"
                         f"一个逼着要、一个想拖想赖")
                return ("argue", topic, 95.0, HOOK_TIER_DEBT)
        # 旧仇：积怨到口角线 → 翻旧账的口角（带上具体由头）
        g = max(a.grudge.get(b.id, 0), b.grudge.get(a.id, 0))
        if g >= CONVO_GRUDGE_MIN:
            topic = f"{a.name}与{b.name}为地盘、黑市、被坑的旧账翻脸，谁也不服谁"
            return ("argue", topic, 90.0 + g, HOOK_TIER_GRUDGE)
        # 结盟：拴在一条绳上的人 → 压低声音合计心事、谁靠得住（只读，别在探测时建空边）
        if (a.rel.get(b.id, {}).get("ally", 0.0) >= ALLY_NOFIGHT
                or b.rel.get(a.id, {}).get("ally", 0.0) >= ALLY_NOFIGHT):
            topic = f"{a.name}与{b.name}是拴在一条绳上的人，压低声音合计各自的盘算、往后靠谁"
            return ("chat", topic, 70.0, HOOK_TIER_ALLY)
        return None

    def _fallback_topic(self) -> str:
        """两人之间没有剧情钩子时的泛泛话题（尽量还是往在押的处境上靠）。"""
        if self.case_progress >= 2:
            return "压低声音议论那个到处打听、翻查旧案的新人"
        return "发发牢骚、交换点狱里的风声"

    def claim_convo(self) -> dict | None:
        """_sim_loop 取走待生成的对话请求（取走即清空，避免重复生成）。"""
        p = self.convo_pending
        self.convo_pending = None
        return p

    def start_conversation(self, a_id: str, b_id: str, topic: str,
                           lines: list[dict]) -> dict:
        """脚本生成成功后落地：置冷却、成消息。lines=[{who,name,text}]。"""
        self._convo_seq += 1
        self.convo_cd = CONVO_GLOBAL_CD
        self._convo_pair_cd[frozenset((a_id, b_id))] = CONVO_PAIR_CD
        a, b = self._npc(a_id), self._npc(b_id)
        return {
            "id": f"cv{self._convo_seq}", "a_id": a_id, "b_id": b_id,
            "a_name": a.name if a else a_id, "b_name": b.name if b else b_id,
            "topic": topic, "lines": lines, "tick": self.tick_count,
        }

    def convo_fail(self, cooldown: int = 10) -> None:
        """脚本没生成出来：短冷却后再试，别卡死也别狂刷。"""
        self.convo_cd = max(self.convo_cd, cooldown)

    def brain_compare_snapshot(self, npc_id: str) -> dict | None:
        """两脑对比取数（Demo 核心）：某 NPC 这一拍真实的 caption + 候选集 + 微调模型的原始意愿排序。

        公平性是关键：裸本地 LLM 要看的 caption 与候选集，必须和微调价值模型刚打过分的
        **完全一致**（候选对象直接取自 last_scored），否则对比不成立。价值侧用 m_value
        （未叠引擎距离/惯性等乘数的"原始意愿"），与 LLM 的意愿分同层可比——比的是两脑各自
        "最想做什么"，而不是引擎最终裁决。
        """
        npc = self._npc(npc_id)
        if npc is None or not npc.last_scored:
            return None
        have = any(sc.cached for sc in npc.last_scored)
        ranked = sorted(npc.last_scored, key=lambda sc: -sc.m_value)
        value_rank = [
            {"goal": sc.candidate.goal, "label": sc.candidate.label,
             "source": sc.candidate.source, "raw": round(sc.v, 3),
             "want": round(sc.m_value, 4)}
            for sc in ranked
        ]
        return {
            "npc_id": npc.id, "name": npc.name, "personality": npc.personality,
            "caption": npc.last_context,
            "candidates": [sc.candidate for sc in npc.last_scored],
            "value_rank": value_rank,
            "value_top": ranked[0].candidate.goal if have else None,
            "needs": self._needs_dict(npc.needs),
            "drivers": npc.decision_drivers,
        }

    def npc_decision_message(self, npc: NPC) -> dict:
        ex = explain(npc.goal, npc.last_scored, npc.last_ctx) if npc.last_ctx else ""
        if npc.goal in YARD_LEISURE or npc.goal == "Rest":
            pref_cn = {"like": "爱扎堆", "avoid": "怕挤", "neutral": "看情况"}.get(npc.crowd_pref, "")
            ex += f"（放风预期{npc.expected_crowd}人 · 容忍{npc.crowd_tol} · {pref_cn}）"
        if npc.commanded_goal:
            cmd_label = GOAL_LABEL.get(npc.commanded_goal, npc.commanded_goal)
            ex = (f"【服从玩家指令·{cmd_label}】" if npc.goal == npc.commanded_goal
                  else f"【抗命！生存需求压过指令·{cmd_label}】") + ex
        if npc.kept_by:
            ex = f"【{npc.kept_by}】" + ex
        return {
            "type": "npc_decision",
            "npc_id": npc.id,
            "name": npc.name,
            "personality": npc.personality,
            "context": npc.last_context,
            "chosen": npc.goal,
            "role": npc.role,
            "activity": activity(npc.goal),
            "monologue": self._say(npc),
            "explain": ex,
            "needs": self._needs_dict(npc.needs),
            "situation": self._situation_dict(npc.last_ctx),
            "expected_crowd": npc.expected_crowd,
            "crowd_tol": npc.crowd_tol,
            "crowd_pref": npc.crowd_pref,
            "commanded_goal": npc.commanded_goal,
            "kept_by": npc.kept_by,
            "drive": {
                "motive": self._drive_of(npc).motive,
                "stake": self._drive_of(npc).stake,
                "ultimate": (quest_of(npc.id).ultimate if quest_of(npc.id) else ""),
                "substep": self._substep_desc(npc),
                "quest_step": npc.quest_step,
                "quest_total": (len(quest_of(npc.id).steps) if quest_of(npc.id) else 0),
                "progress": round(npc.drive_progress),
                "parole_lost": npc.parole_lost,
                "signature": GOAL_LABEL.get(self._drive_of(npc).signature,
                                            self._drive_of(npc).signature),
            },
            "sentence_years": npc.sentence_years,
            "crime": npc.crime,
            "labor_job": npc.labor_job,
            "busted": npc.busted,
            "core_goal": self._drive_of(npc).motive,
            "drivers": npc.decision_drivers,
            "decision_log": list(npc.decision_log),
            # P1：上帝视角暂停面板可查关系图 + 情景记忆
            "relationships": self._rel_summary(npc, 4),
            "memory": self._mem_lines(npc, 5),
            "candidates": [
                {
                    "goal": sc.candidate.goal,
                    "goal_label": sc.candidate.label,
                    "source": sc.candidate.source,
                    "v": round(sc.v, 3),
                    "m_value": round(sc.m_value, 4),
                    "m_light": round(sc.m_light, 3),
                    "m_inertia": round(sc.m_inertia, 3),
                    "cost": round(sc.cost, 3),
                    "m_interruption": round(sc.m_interruption, 3),
                    "m_failure": round(sc.m_failure, 3),
                    "m_timeout": round(sc.m_timeout, 3),
                    "m_repetition": round(sc.m_repetition, 3),
                    "m_crowd": round(sc.m_crowd, 3),
                    "m_command": round(sc.m_command, 3),
                    "m_watch": round(sc.m_watch, 3),
                    "m_drive": round(sc.m_drive, 3),
                    "score": round(sc.score, 4),
                    "cached": sc.cached,
                    "chosen": sc.candidate.goal == npc.goal,
                }
                for sc in npc.last_scored
            ],
        }
