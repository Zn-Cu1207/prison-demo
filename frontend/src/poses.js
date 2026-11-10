// 过程化姿态叠加层：把后端选出的 ~49 种动作演成看得出区别的身体语言。
//
// 为什么不做动画片段：写实角色的 GLB（laochen/balian/hero/doctor/madman/police）每具只带
// idle / walk / dying，个别多一个 kneeling 或 fight。旧的 GOAL_ANIM 表是给占位素模 Xbot
// 写的（Sitting/Wave/Punch/ThumbsUp/Dance…），换成写实模型后这些名字全部落到
// clip._fallback = "idle"：40 多个决策在画面上一律是"走过去，然后站着"。价值模型选得再
// 不一样，观众也看不见。为每个动作走一遍 Mixamo→Blender→压缩的流程代价太大，
// 所以把 main.js 里 applySalutePose 那一手推广开来：
//
//   让 AnimationMixer 照常把 idle 写进骨骼（上身仍在呼吸，不是定格木偶），
//   之后再伸手进骨架，把几根骨头按声明式的目标方向压过去，权重 0.25s 渐入渐出。
//
// 关键约束（和 applySalutePose 相同）：必须在 entry.mixer.update(dt) **之后**调用，
// 否则 mixer 会把这一帧的骨骼原样覆盖回去。
//
// —— 骨架体检结论（frontend/tools/_probe_bones.py）——
// 六具角色 GLB 全是同一套 Mixamo 骨架：各只有 1 具 skin，去掉前缀后 45 根骨骼名完全一致
// （Hips/Spine/Spine1/Spine2/Neck/Head/{Left,Right}{Shoulder,Arm,ForeArm,Hand}/
//   {Left,Right}{UpLeg,Leg,Foot}… 姿态要用的 20 根一个不缺）。差异只有三处，都不影响姿态：
//   1. 前缀：madman.glb 是 mixamorig5:，其余是 mixamorig:（GLTFLoader 会把冒号删掉，
//      所以运行时名字长成 mixamorig5Spine / mixamorigSpine）——canon() 统一剥掉。
//   2. hero.glb 只有 47 根骨骼：没有右手的手指骨（也没有右手指尖）；balian 多了头发骨、
//      眼球骨和一根 Weapon。取不到的骨骼一律静默跳过，不抛异常。
//   3. 六具全都是 Z-up：骨架根节点带 rotation=X+90°、scale=0.01，Hips 静置平移的竖直分量
//      在 Z 上（laochen -90.16、balian -121.86…）。所以 main.js 注释里说的"laochen 是 Z-up"
//      其实是全体如此。
// 因此姿态表只需要一份，不必按角色分表。
//
// —— 为什么不手填欧拉角 ——
// tools/_probe_solve_salute.js 的注释已经踩过这个坑：Mixamo 骨骼的静置朝向是沿肢体轴的，
// 局部空间里哪个轴是"外展"根本不直观，手填的敬礼角度让手停在了胸口，最后是靠浏览器里
// 撒点+爬山解出来的。所以这里换一种写法：姿态不写"骨骼的局部旋转"，而写**这根骨头要指向
// 哪儿**（角色自身的前/上/右三向量的线性组合），运行时反解出局部四元数。
// 好处是：
//   * 可读可推：[f,u,r]=[1,-0.2,0] 就是"大腿朝前、略微朝下"，能拿骨长算出膝/踝落在哪。
//   * 与骨骼滚转约定无关：用的是"当前朝向→目标朝向"的最小旋转，静置姿势怎么定都不影响。
//   * 与竖直轴约定无关：父骨的世界朝向是实时累乘出来的，Armature 的 X+90° 自动被抵消。
//
// 敬礼没有搬进来：_SALUTE_* 那三组角度是解出来并对着截图校过的绝对局部欧拉角，
// 换成本文件的"指向"写法等于重解一次，得不到逐像素相同的结果。既然它已经好了，
// 就留在 main.js 的 applySalutePose 里不动（巡逻狱警的调用点也没改）。

import * as THREE from "three";

// ---------------------------------------------------------------------------
// 骨骼名归一 / 骨架解析
// ---------------------------------------------------------------------------
// mixamorig:Spine（原始）/ mixamorigSpine（GLTFLoader 删了冒号）/ mixamorig5Spine（madman）
// / Spine（手搓骨架）都归到 Spine。
const BONE_PREFIX = /^mixamorig\d*[:_]?/i;
const canon = (name) => (name || "").replace(BONE_PREFIX, "");

// 每根骨头的"自身轴"由它指向哪个子骨决定。Spine2/Hips 这种有多个子骨（颈+两肩、脊+两腿）
// 的必须点名，否则 traverse 顺序一变，"脊椎朝上"就可能被解成"朝左肩"。
const CHILD_BONE = {
  Hips: "Spine", Spine: "Spine1", Spine1: "Spine2", Spine2: "Neck",
  Neck: "Head", Head: "HeadTop_End",
  LeftShoulder: "LeftArm", LeftArm: "LeftForeArm", LeftForeArm: "LeftHand",
  LeftHand: "LeftHandMiddle1",
  RightShoulder: "RightArm", RightArm: "RightForeArm", RightForeArm: "RightHand",
  RightHand: "RightHandMiddle1",
  LeftUpLeg: "LeftLeg", LeftLeg: "LeftFoot", LeftFoot: "LeftToeBase",
  RightUpLeg: "RightLeg", RightLeg: "RightFoot", RightFoot: "RightToeBase",
};

// ---------------------------------------------------------------------------
// 姿态表
// ---------------------------------------------------------------------------
// 一条 aim 的字段：
//   bone     骨骼名（去前缀）。写成 "*Arm" 表示左右各来一份，右侧原样、左侧把 r 分量与
//            twist 取反（镜像），省得同一组数字抄两遍。
//   dir      目标朝向 [f, u, r]：f=角色正前、u=世界上、r=角色右手侧，不必归一化。
//   to       第二个目标朝向；有它就在 dir↔to 之间按正弦往复（呼吸/抬举/挥击）。
//   twist    绕骨骼自身轴扭转的弧度（转头、扭腰这类"指向解不出来"的自转）；
//   twistTo  同理的第二个值，与 dir/to 共用同一个正弦相位。
//   hz       往复频率（Hz），默认 0.5；phase 给同一姿态里的不同骨骼错开相位。
//
// 调这张表最容易踩的一个坑：dir 是**绝对朝向**，不是相对父骨的角度，所以脊椎的弯曲不叠加。
// 只给 Spine 写 [0.62,1,0]（约 32°）、上面的 Spine1/Spine2 不写或写得更小，
// 上半身反而会被"扶正"回来，实测哈腰只弯出 22°、头顶才下来 5cm。要弯得深，
// 整条脊椎（Spine/Spine1/Spine2）都得写上相当的前倾。
// 姿态字段：
//   drop     髋部相对站立下沉多少米（世界向下），坐/跪/躺/蹲用。
//   speed    整体节奏倍率（乘到所有 hz 上）。
//   extends  继承另一个姿态：drop/speed 被覆盖，aims 按 bone 同名覆盖、异名累加。
//
// 数字不是随手填的：这批模型统一归一化到 1.85m 高，站立时髋高约 0.87~0.95m、
// 大腿与小腿各约 0.45m。先按骨长算一版，再用 tools/_probe_pose_geom.js 量脚踝/膝盖的
// 实际高度收敛——算的那一版普遍偏深，坐姿 0.42 会把整只脚穿进地板，最后落在
// 坐 0.32（髋 ~0.55m、踝 ~0.10m 贴地）、跪 0.40（膝 ~0.05m 触地）、蹲/刨 0.30。
// 躺姿 drop=0.72 让身体贴地展开（不取床面高度 0.72m：睡觉的落点是"床这个设施"而不是
// 床板锚点，抬到床面高度一旦没对准就会整个人浮在空中，贴地至少永远说得通）。
export const POSES = {
  // —— 基础十式 ——
  // 坐：髋下沉、大腿前伸、小腿垂下、上身略后靠，双手搭在腿上。
  // drop 是量出来定的：0.42 时探针报脚踝落到 y=0.00（比待机的 0.11 低了一整个脚掌，
  // 也就是脚穿进地板），收到 0.32 正好把脚放回地面。
  sit: {
    drop: 0.32,
    aims: [
      { bone: "Spine", dir: [-0.12, 1, 0] },
      { bone: "Spine1", dir: [-0.05, 1, 0] },
      { bone: "*UpLeg", dir: [1, -0.2, 0.12] },
      { bone: "*Leg", dir: [0.14, -1, 0] },
      { bone: "*Foot", dir: [0.9, -0.35, 0] },
      { bone: "*Arm", dir: [0.2, -1, 0.2] },
      { bone: "*ForeArm", dir: [0.85, -0.45, -0.1] },
      // 坐着也别定格：躯干极轻微前后晃，配合 idle 的呼吸就有活气
      { bone: "Spine2", dir: [0.02, 1, 0], to: [-0.04, 1, 0], hz: 0.22 },
    ],
  },
  // 躺：只要把 Hips 自身轴（髋→脊）从"朝上"扳成"朝后"，整具身体就绕髋倒下去——
  // 腿是 Hips 的子骨，跟着从朝下变成朝前，正好是仰卧展开的样子。
  lie: {
    drop: 0.72,
    aims: [
      { bone: "Hips", dir: [-1, 0.06, 0] },
      { bone: "Spine", dir: [-1, 0.1, 0] },
      { bone: "Spine1", dir: [-1, 0.08, 0] },
      { bone: "Head", dir: [-0.9, 0.35, 0] },        // 后脑垫在枕头上，脸朝上
      { bone: "*UpLeg", dir: [1, 0.02, 0.1] },
      { bone: "*Leg", dir: [1, -0.08, 0] },
      { bone: "*Arm", dir: [0.55, -0.15, 0.55] },    // 手臂顺着身体摆在两侧
      { bone: "*ForeArm", dir: [0.9, -0.1, 0.2] },
      { bone: "Spine2", dir: [-1, 0.06, 0], to: [-1, 0.12, 0], hz: 0.18 },  // 呼吸起伏
    ],
  },
  // 跪：大腿垂直向下（膝触地）、小腿朝后、上身略前倾。
  // 同样是量出来的：0.50 会让膝盖沉到 y=-0.05、脚背 -0.09（跪进地板），0.40 刚好触地。
  kneel: {
    drop: 0.36,
    aims: [
      { bone: "Spine", dir: [0.1, 1, 0] },
      { bone: "*UpLeg", dir: [0.05, -1, 0.1] },
      { bone: "*Leg", dir: [-1, -0.12, 0] },
      // 脚背贴地：朝向压得太"下"会让脚尖扎进地板（探针量到脚踝 0.01、比站立低一整掌）
      { bone: "*Foot", dir: [-1, -0.2, 0] },
      { bone: "*Arm", dir: [0.3, -1, 0.15] },
      { bone: "*ForeArm", dir: [0.8, -0.5, -0.15] },
      { bone: "Head", dir: [0.2, 1, 0] },
    ],
  },
  // 举铁：双臂过顶推举，往复。两个端点拉得够开，手的行程才看得出是在推举
  // （第一版 [0,0.75,0.66]→[0,1,0.22] 只走 7cm，小臂把上臂的转动抵掉了）。
  liftWeight: {
    aims: [
      { bone: "*Arm", dir: [0, 0.5, 0.85], to: [0, 1, 0.12], hz: 0.7 },
      { bone: "*ForeArm", dir: [0, 0.9, 0.4], to: [0, 1, 0.02], hz: 0.7 },
      { bone: "Spine", dir: [-0.08, 1, 0], to: [0.02, 1, 0], hz: 0.7 },
      { bone: "*UpLeg", dir: [0.12, -1, 0.14] },      // 微屈膝的发力站姿
      { bone: "*Leg", dir: [-0.08, -1, 0] },
    ],
  },
  // 送到嘴边：头略低，一只手反复抬到嘴边再放下（吃/喝）。
  // 低点要真放下去（小臂朝前偏下），不然手一直端在胸口，看不出"送一口"这个动作。
  eatHand: {
    aims: [
      { bone: "Head", dir: [0.35, 1, 0] },
      { bone: "Spine", dir: [0.1, 1, 0] },
      { bone: "RightArm", dir: [0.3, -1, 0.28] },
      { bone: "RightForeArm", dir: [0.9, -0.3, -0.05], to: [0.2, 0.9, -0.4], hz: 0.45 },
    ],
  },
  // 案前：躯干前倾、双手前下方（桌面高度），头看着桌面。
  writeDesk: {
    aims: [
      { bone: "Spine", dir: [0.35, 1, 0] },
      { bone: "Spine1", dir: [0.18, 1, 0] },
      { bone: "Head", dir: [0.5, 1, 0] },
      { bone: "*Arm", dir: [0.45, -0.9, 0.25] },
      { bone: "*ForeArm", dir: [0.9, -0.28, -0.2] },
    ],
  },
  // 向下刨：屈膝半蹲 + 躯干深弯 + 双臂反复往下够（挖洞/擦地/搓衣/翻找）。
  // 光弯腰不够：第一版只沉 0.12、腿不动，手最低只到 0.77（胯那么高），
  // 看着像在扶着什么而不是在地上刨。加上半蹲把肩膀整体压下去，手才够得到地面附近。
  digDown: {
    drop: 0.30,
    aims: [
      // 整条脊椎一起前倾（见文件头那条"dir 是绝对朝向、不叠加"的坑）
      { bone: "Spine", dir: [0.62, 1, 0], to: [0.45, 1, 0], hz: 0.85 },
      { bone: "Spine1", dir: [0.55, 1, 0], to: [0.4, 1, 0], hz: 0.85 },
      { bone: "Spine2", dir: [0.45, 1, 0] },
      { bone: "Head", dir: [0.6, 1, 0] },
      // 肘要真的伸屈：两个端点从"小臂朝前收着"到"整条手臂垂直向下"，手才够得到地面
      { bone: "*Arm", dir: [0.55, -0.85, 0.2], to: [0.15, -1, 0.15], hz: 0.85 },
      { bone: "*ForeArm", dir: [0.85, -0.35, -0.1], to: [0.15, -1, -0.05], hz: 0.85 },
      // 腿用和 crouch 同一组角度：小腿再直一点（[-0.5,-0.87]）就会把脚踝压到 0.03、
      // 整只脚埋进地板，探针的"脚踩进地板"那一条就是这么抓出来的
      { bone: "*UpLeg", dir: [0.9, -0.3, 0.16] },
      { bone: "*Leg", dir: [-0.55, -0.85, 0] },
      { bone: "*Foot", dir: [0.85, -0.45, 0] },
    ],
  },
  // 张望：脖子左右扫（转头是绕骨骼自身轴的自转，"指向"解不出来，只能用 twist），
  // 腰跟着轻轻带一点。频率压得很低，才像在盯梢而不是摇头。
  lookAround: {
    aims: [
      { bone: "Neck", twist: -0.55, twistTo: 0.55, hz: 0.16 },
      { bone: "Head", twist: -0.18, twistTo: 0.18, hz: 0.16 },
      { bone: "Spine1", twist: -0.16, twistTo: 0.16, hz: 0.16, phase: 0.6 },
      { bone: "*Arm", dir: [0.05, -1, 0.14] },
    ],
  },
  // 说话的手势：一只小臂来回摆，幅度压小——这是背景演出，不是打人。
  talkGesture: {
    aims: [
      { bone: "RightArm", dir: [0.35, -0.9, 0.3] },
      { bone: "RightForeArm", dir: [0.85, 0.3, -0.15], to: [0.7, 0.0, -0.3], hz: 0.8 },
      { bone: "Head", twist: -0.12, twistTo: 0.12, hz: 0.35 },
      { bone: "Spine1", dir: [0.06, 1, 0] },
    ],
  },
  // 挥击：手臂后撤蓄力再打出去，带腰的扭转（打架/立威/行凶）。
  strike: {
    aims: [
      { bone: "RightArm", dir: [-0.35, -0.1, 0.9], to: [0.95, -0.05, 0.25], hz: 1.1 },
      { bone: "RightForeArm", dir: [-0.2, 0.5, -0.8], to: [1, 0.05, 0], hz: 1.1 },
      { bone: "LeftArm", dir: [0.35, -0.7, -0.5] },        // 另一只手抬起来护着
      { bone: "LeftForeArm", dir: [0.85, 0.35, 0.3] },
      { bone: "Spine1", twist: -0.22, twistTo: 0.22, hz: 1.1 },
      { bone: "*UpLeg", dir: [0.2, -1, 0.16] },
      { bone: "*Leg", dir: [-0.12, -1, 0] },
    ],
  },

  // —— 组合式：基础十式加几行就派生出来的变体 ——
  // 坐着吃饭（食堂有桌子，比站着啃更像样）
  eatSit: {
    extends: "sit",
    aims: [
      { bone: "Head", dir: [0.3, 1, 0] },
      { bone: "RightArm", dir: [0.3, -1, 0.25] },
      { bone: "RightForeArm", dir: [0.85, 0.05, -0.1], to: [0.2, 0.9, -0.4], hz: 0.45 },
    ],
  },
  // 坐着面对桌面：牌/棋子/纹身针都在手前方一点点
  sitTable: {
    extends: "sit",
    aims: [
      { bone: "Spine", dir: [0.16, 1, 0] },
      { bone: "Head", dir: [0.45, 1, 0] },
      { bone: "*Arm", dir: [0.45, -0.85, 0.25] },
      { bone: "*ForeArm", dir: [0.9, -0.15, -0.2], to: [0.9, -0.05, -0.15], hz: 0.5 },
    ],
  },
  // 坐着看书：书举在胸前，头微低
  readSit: {
    extends: "sit",
    aims: [
      { bone: "Head", dir: [0.35, 1, 0] },
      { bone: "*Arm", dir: [0.3, -0.9, 0.3] },
      { bone: "*ForeArm", dir: [0.75, 0.5, -0.35], to: [0.7, 0.45, -0.3], hz: 0.25 },
    ],
  },
  // 坐着聊（探视室）
  sitTalk: {
    extends: "sit",
    aims: [
      { bone: "RightArm", dir: [0.35, -0.9, 0.3] },
      { bone: "RightForeArm", dir: [0.85, 0.3, -0.15], to: [0.7, 0.0, -0.3], hz: 0.8 },
      { bone: "Head", twist: -0.1, twistTo: 0.1, hz: 0.35 },
    ],
  },
  // 自己处理伤口：坐着低头，一只手在另一条小臂上按着
  tendWound: {
    extends: "sit",
    aims: [
      { bone: "Head", dir: [0.5, 1, 0] },
      { bone: "LeftArm", dir: [0.5, -0.7, -0.35] },
      { bone: "LeftForeArm", dir: [0.65, 0.25, 0.55] },
      { bone: "RightArm", dir: [0.45, -0.8, 0.3] },
      { bone: "RightForeArm", dir: [0.5, 0.15, -0.75], to: [0.55, 0.05, -0.7], hz: 0.6 },
    ],
  },
  // 台面上手上有活（帮厨切菜）：站着前倾，双手小幅上下
  workTable: {
    extends: "writeDesk",
    aims: [
      { bone: "*ForeArm", dir: [0.9, -0.28, -0.2], to: [0.8, -0.05, -0.2], hz: 1.3 },
    ],
  },
  // 打电话：一只手贴耳，只有极轻微的晃动（别让它像在扇自己）
  phone: {
    aims: [
      { bone: "RightArm", dir: [0.2, -0.75, 0.5] },
      { bone: "RightForeArm", dir: [0.08, 1, -0.42], to: [0.04, 1, -0.36], hz: 0.3 },
      { bone: "Head", twist: -0.1, twistTo: 0.1, hz: 0.25 },
      { bone: "LeftArm", dir: [0.1, -1, 0.12] },
    ],
  },
  // 抽烟：就是把"送到嘴边"放慢——一口烟的节奏比吃饭慢一半还多
  smoke: { extends: "eatHand", speed: 0.4 },
  // 洗澡：双手在头顶附近搓
  washSelf: {
    aims: [
      { bone: "Head", dir: [0.15, 1, 0] },
      { bone: "*Arm", dir: [0.1, 0.35, 0.85] },
      { bone: "*ForeArm", dir: [0.2, 0.9, -0.5], to: [0.35, 0.8, -0.35], hz: 1.0 },
    ],
  },
  // 唱：头微仰、一只手随节奏摆，比说话慢
  sing: {
    extends: "talkGesture",
    speed: 0.6,
    aims: [
      { bone: "Head", dir: [-0.25, 1, 0] },
      { bone: "Spine1", twist: -0.12, twistTo: 0.12, hz: 0.5 },
    ],
  },
  // 蹲着藏：髋下沉屈膝 + 继续左右张望（藏东西/躲着）
  crouch: {
    extends: "lookAround",
    drop: 0.30,
    aims: [
      { bone: "Spine", dir: [0.28, 1, 0] },
      { bone: "*UpLeg", dir: [0.9, -0.3, 0.16] },
      { bone: "*Leg", dir: [-0.55, -0.85, 0] },
      { bone: "*Foot", dir: [0.85, -0.45, 0] },
      { bone: "*Arm", dir: [0.35, -0.9, 0.2] },
      { bone: "*ForeArm", dir: [0.85, -0.3, -0.2] },
    ],
  },
  // 撬门：双手在胸前抓着往回拽，身体后仰发力
  pryDoor: {
    aims: [
      { bone: "Spine", dir: [-0.25, 1, 0], to: [-0.1, 1, 0], hz: 0.9 },
      { bone: "*Arm", dir: [0.6, -0.55, 0.4], to: [0.35, -0.75, 0.35], hz: 0.9 },
      { bone: "*ForeArm", dir: [0.95, 0.15, -0.2], to: [0.7, 0.35, -0.35], hz: 0.9 },
      { bone: "Head", dir: [0.25, 1, 0] },
      { bone: "*UpLeg", dir: [0.25, -1, 0.16] },
      { bone: "*Leg", dir: [-0.18, -1, 0] },
    ],
  },
  // 哈腰：讨好/致哀共用——上身前倾、头低、双手在身前收着。
  // 弯到 0.62/0.3（约 35°）才看得出是"哈着腰"：0.42 那版头只下来 1.6cm，跟站着差不多。
  bow: {
    aims: [
      // 角度要给足：这套骨架静置时脊椎本身就带一点前倾，写 0.6（31°）跟站着差不多，
      // 探针量到头顶只下来 5cm。0.9（42°）才是"哈着腰"。颈也得跟着弯，
      // 否则脖子会把脑袋重新扶正。
      { bone: "Spine", dir: [1.1, 1, 0], to: [0.75, 1, 0], hz: 0.2 },
      { bone: "Spine1", dir: [1.05, 1, 0], to: [0.72, 1, 0], hz: 0.2 },
      { bone: "Spine2", dir: [0.85, 1, 0] },
      { bone: "Neck", dir: [0.7, 1, 0] },
      { bone: "Head", dir: [0.75, 1, 0] },
      { bone: "*Arm", dir: [0.3, -0.95, 0.2] },
      { bone: "*ForeArm", dir: [0.85, 0.15, -0.35] },
    ],
  },
  // 指着对方：手臂前伸定住，只有很小的抖动（威胁/放狠话）
  point: {
    aims: [
      { bone: "RightArm", dir: [1, -0.12, 0.24], to: [1, -0.04, 0.2], hz: 0.7 },
      { bone: "RightForeArm", dir: [1, 0.05, -0.05] },
      { bone: "Spine1", dir: [0.12, 1, 0] },
      { bone: "LeftArm", dir: [0.15, -1, 0.12] },
    ],
  },
  // 吵架：说话的手势加大加快，再带上腰的扭转
  argue: {
    extends: "talkGesture",
    speed: 1.6,
    aims: [
      { bone: "RightArm", dir: [0.5, -0.7, 0.45] },
      { bone: "RightForeArm", dir: [0.95, 0.45, -0.2], to: [0.8, -0.15, -0.4], hz: 0.8 },
      { bone: "Spine1", twist: -0.14, twistTo: 0.14, hz: 0.8 },
      { bone: "Head", dir: [-0.12, 1, 0] },
    ],
  },
};

// ---------------------------------------------------------------------------
// goal → 姿态 / goal → 基础片段
// ---------------------------------------------------------------------------
// goal 全表抄自 backend/app/actions.py 的 ACTIONS（49 条，见 tools/_probe_goals.py），
// 不是凭记忆列的。没有合适姿态的显式写 null，落到干净的 idle——
// 写 null 是"想过了，就该站着"，漏写是"忘了"，两者要能分开。
export const GOAL_POSE = {
  // 生存刚需
  Eat: "eatSit", Drink: "eatHand", Sleep: "lie", Cure: "tendWound",
  // 情绪 / 社交
  DeclineBoredom: "talkGesture", Socialize: "talkGesture", Phone: "phone", Visit: "sitTalk",
  // 威胁应对（Fight 只在 e.inBout 为真时才叠 strike，调用点把关：没真开打就不打空气）
  Fight: "strike", Argue: "argue", LayLow: "lookAround",
  // 日常作息
  Patrol: null,                 // 在放风场溜达：站定的那几秒就该是待机
  Rest: "sit", Study: "writeDesk", Labor: "digDown", KitchenDuty: "workTable",
  Laundry: "digDown", Clean: "digDown",
  // 消遣
  Workout: "liftWeight", Shower: "washSelf", Read: "readSit", Gamble: "sitTable",
  Smoke: "smoke", Chess: "sitTable", Music: "sing", Pray: "kneel",
  Barter: "talkGesture", Tattoo: "sitTable",
  // 越狱链 / 阴谋
  Watch: "lookAround", Investigate: "writeDesk", GetWeapon: "digDown",
  Assassinate: "strike", Snitch: "talkGesture", Hide: "crouch", Tunnel: "digDown",
  PrisonBreak: "pryDoor",
  // 宏观动机
  CurryFavor: "bow", Extort: "strike", SellInfo: "talkGesture",
  ReviewFiles: "writeDesk", SeekProtector: "kneel",
  // 剧情 / 关系
  Bury: "digDown", Mourn: "bow", Propose: "talkGesture", Ally: "talkGesture",
  Protect: "lookAround", Repay: "talkGesture", Threaten: "point", Betray: "talkGesture",
};

// goal → 到位后播哪个"逻辑片段名"（给 main.js 的 GOAL_ANIM 用）。
// 现在几乎所有动作的身体语言都由过程化姿态负责，这张表只留"某具模型真有对应片段"的例外：
// 老牙的 laochen.glb 带 kneeling，跪着求人/祷告用真片段比手搓的好。没有该片段的角色会由
// resolveClip 回退到 idle，姿态层再补一个过程化的 kneel（applyGoalPose 靠 entry.current
// 是不是 idle 来判断该不该叠，所以两边不会同时生效）。
export const GOAL_CLIP = {
  Pray: "Kneel", SeekProtector: "Kneel",
};

// ---------------------------------------------------------------------------
// 姿态展开：extends 继承 + 左右镜像
// ---------------------------------------------------------------------------
const _flat = new Map();
function flatten(name) {
  if (_flat.has(name)) return _flat.get(name);
  const def = POSES[name];
  if (!def) return null;
  const parent = def.extends ? flatten(def.extends) : null;
  const aims = new Map(parent ? parent.aims : null);      // 同名骨骼覆盖，异名累加
  for (const a of def.aims || []) aims.set(a.bone, a);
  const out = {
    drop: def.drop != null ? def.drop : (parent ? parent.drop : 0),
    speed: def.speed != null ? def.speed : (parent ? parent.speed : 1),
    aims,
  };
  _flat.set(name, out);
  return out;
}

// ---------------------------------------------------------------------------
// 每个实体一份的骨架索引
// ---------------------------------------------------------------------------
const _tmpQ = new THREE.Quaternion();
const _tmpV = new THREE.Vector3();

// 世界朝向靠逐级累乘局部四元数算，而不是读 matrixWorld：姿态是在 mixer 之后、渲染之前跑的，
// 这时 matrixWorld 还是上一帧的。（体型缩放 girth 只有 ±8% 的非等比，忽略它带来的方向偏斜。）
function worldQuat(obj, out) {
  out.copy(obj.quaternion);
  for (let p = obj.parent; p; p = p.parent) out.premultiply(p.quaternion);
  return out;
}

function buildRig(root) {
  const bones = new Map();
  root.traverse((o) => {
    if (!o.isBone) return;
    const key = canon(o.name);
    if (!bones.has(key)) bones.set(key, o);      // 同名取第一具（清理过的 GLB 只有一具骨架）
  });
  const rig = { bones, depth: new Map(), compiled: new Map(), rightSign: 0, hips: bones.get("Hips") || null };
  for (const [key, bone] of bones) {
    let d = 0;
    for (let p = bone.parent; p; p = p.parent) d += 1;
    rig.depth.set(key, d);
  }
  return rig;
}

// 哪边是"角色的右手侧"不做推导，实测：肩→上臂这段偏移几乎纯横向，把它的世界方向投到
// (上 × 前) 上看符号即可。tools/_probe_solve_salute.js 的注释解释了为什么不能靠猜——
// 这类骨架的轴向约定不直观，猜错就左右手互换。
function measureRightSign(rig, fwd, up) {
  const cand = _tmpV.copy(up).cross(fwd);
  for (const [shoulder, arm, sign] of [["RightShoulder", "RightArm", 1], ["LeftShoulder", "LeftArm", -1]]) {
    const sb = rig.bones.get(shoulder), ab = rig.bones.get(arm);
    if (!sb || !ab) continue;
    const d = new THREE.Vector3().copy(ab.position).applyQuaternion(worldQuat(sb, _tmpQ));
    if (d.lengthSq() < 1e-9) continue;
    const dot = d.dot(cand) * sign;
    if (Math.abs(dot) > 1e-4) return dot > 0 ? 1 : -1;
  }
  return -1;   // 兜底：正面 +Z、上 +Y 的右手系里，人物的右手侧是 -X
}

function compile(rig, name) {
  if (rig.compiled.has(name)) return rig.compiled.get(name);
  const flat = flatten(name);
  let out = null;
  if (flat) {
    const items = [];
    for (const aim of flat.aims.values()) {
      const names = aim.bone.startsWith("*")
        ? [["Right" + aim.bone.slice(1), 1], ["Left" + aim.bone.slice(1), -1]]
        : [[aim.bone, 1]];
      for (const [boneName, sign] of names) {
        const bone = rig.bones.get(boneName);
        if (!bone) continue;                        // 这具骨架没有这根骨头：静默跳过
        const childName = CHILD_BONE[boneName];
        let child = childName ? rig.bones.get(childName) : null;
        if (!child) child = (bone.children || []).find((c) => c.isBone) || null;
        if (!child && aim.dir) continue;            // 没有子骨就定不出"自身轴"，指向类跳过
        items.push({
          bone, child, sign,
          dir: aim.dir || null, to: aim.to || null,
          twist: aim.twist != null ? aim.twist : null,
          twistTo: aim.twistTo != null ? aim.twistTo : aim.twist,
          hz: (aim.hz != null ? aim.hz : 0.5) * flat.speed,
          phase: aim.phase || 0,
          depth: rig.depth.get(boneName) || 0,
        });
      }
    }
    // 父骨先算：叠加会改动局部四元数，子骨的父级世界朝向必须已经是改过之后的值
    items.sort((a, b) => a.depth - b.depth);
    out = { drop: flat.drop, items };
  }
  rig.compiled.set(name, out);
  return out;
}

// ---------------------------------------------------------------------------
// 应用
// ---------------------------------------------------------------------------
const RAMP = 0.25;                 // 渐入/渐出秒数
const TAU = Math.PI * 2;
const _fwd = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _right = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _axis = new THREE.Vector3();
const _cur = new THREE.Vector3();
const _qp = new THREE.Quaternion();
const _qi = new THREE.Quaternion();
const _qd = new THREE.Quaternion();
const _qt = new THREE.Quaternion();
const _m3 = new THREE.Matrix3();
const _hips = new THREE.Vector3();
let _phaseSeed = 0;

function poseState(entry) {
  if (!entry._pose) {
    entry._pose = {
      rig: null, name: null, want: null, w: 0, t: 0,
      // 每人一个固定相位：同一时刻两个人做同一件事，别摆成整齐的广播操
      phase: (_phaseSeed = (_phaseSeed + 2.399963) % TAU),
      hipsBase: null, hipsWrote: null,
    };
  }
  return entry._pose;
}

// 髋部下沉：Hips 的平移在父骨（Armature 根，带 X+90° 与 0.01 缩放）的局部空间里，
// 单位既不是米、竖直轴也不是 Y。所以不硬编码轴向，拿父骨世界矩阵的 3×3 求逆，
// 把"世界向下 d 米"换算成局部平移——轴向约定怎么变都不用改这里。
function applyHipsDrop(rig, st, drop) {
  const hips = rig.hips;
  if (!hips || !hips.parent) return;
  // mixer 每帧都会重写 Hips 的平移（片段里有根骨通道），但不能假定一定会：
  // 若这一帧的值和我们上一帧写进去的一模一样，说明 mixer 没覆盖，得拿自己记的基准值。
  if (!st.hipsBase) st.hipsBase = new THREE.Vector3();
  const keep = st.hipsWrote && hips.position.equals(st.hipsWrote);
  const base = keep ? st.hipsBase : st.hipsBase.copy(hips.position);
  if (!(drop > 0)) { hips.position.copy(base); st.hipsWrote = null; return; }
  _m3.setFromMatrix4(hips.parent.matrixWorld).invert();
  _hips.set(0, -drop, 0).applyMatrix3(_m3);
  if (!Number.isFinite(_hips.x) || _hips.lengthSq() < 1e-12) return;   // matrixWorld 还没算过
  hips.position.copy(base).add(_hips);
  st.hipsWrote = (st.hipsWrote || new THREE.Vector3()).copy(hips.position);
}

function releaseHips(rig, st) {
  if (!st.hipsWrote || !rig || !rig.hips) { st.hipsWrote = null; return; }
  if (rig.hips.position.equals(st.hipsWrote) && st.hipsBase) rig.hips.position.copy(st.hipsBase);
  st.hipsWrote = null;
}

function poseBone(item, s, w) {
  const bone = item.bone;
  if (!bone.parent) return;
  _qt.copy(bone.quaternion);
  if (item.dir) {
    const d = item.dir, t = item.to;
    let f = d[0], u = d[1], r = d[2];
    if (t) { f += (t[0] - f) * s; u += (t[1] - u) * s; r += (t[2] - r) * s; }
    _dir.set(0, 0, 0).addScaledVector(_fwd, f).addScaledVector(_up, u)
      .addScaledVector(_right, r * item.sign);
    if (_dir.lengthSq() < 1e-9) return;
    _dir.normalize();
    _cur.copy(item.child.position);
    if (_cur.lengthSq() < 1e-12) return;
    _cur.normalize();                                   // 骨骼自身轴（骨骼局部空间）
    worldQuat(bone.parent, _qp);
    _qi.copy(_qp).invert();
    _dir.applyQuaternion(_qi).normalize();              // 目标方向搬进父骨空间
    _axis.copy(_cur).applyQuaternion(bone.quaternion).normalize();   // mixer 刚写出的当前朝向
    _qd.setFromUnitVectors(_axis, _dir);                // 当前→目标的最小旋转
    _qt.premultiply(_qd);
  }
  if (item.twist != null) {
    const ang = (item.twist + (item.twistTo - item.twist) * s) * item.sign;
    if (item.child) {
      _axis.copy(item.child.position);
      if (_axis.lengthSq() < 1e-12) return;
      _axis.normalize().applyQuaternion(_qt).normalize();  // 绕"叠加之后"的骨轴自转
    } else {
      _axis.set(0, 1, 0).applyQuaternion(_qt).normalize();
    }
    _qd.setFromAxisAngle(_axis, ang);
    _qt.premultiply(_qd);
  }
  bone.quaternion.slerp(_qt, w);
}

/**
 * 在 mixer 写完骨骼之后，按 goal 叠一层过程化姿态。
 *
 * 必须在 entry.mixer.update(dt) 之后调用（同 applySalutePose），否则会被 mixer 覆盖掉。
 *
 * @param entry  NPC/巡逻者条目，需要有 .root（含骨架）和 .mixer；没有 mixer 的方块角色直接跳过
 * @param goal   当前动作名；传 null / undefined 表示"不要姿态"（走路、立正、没真开打）
 * @param dt     本帧秒数
 * @param opts   { pose } 指定姿态名，绕过 GOAL_POSE（探针/调试用）
 * @returns 实际生效的姿态名，未生效返回 null
 */
export function applyGoalPose(entry, goal, dt, opts) {
  if (!entry || !entry.mixer || !entry.root) return null;      // 方块狱警/狱犬：程序化四肢，不管
  const st = poseState(entry);
  let want = null;
  if (opts && opts.pose !== undefined) want = opts.pose;
  else if (goal) want = GOAL_POSE[goal] || null;
  if (want && !POSES[want]) want = null;

  // 正在播真片段（走/跑/打架/跪/倒地）就别叠：那些片段自己已经把全身写好了，
  // 再压一层只会互相打架。判据是"当前片段是不是这具模型的 idle"。
  // 注意只是把 want 清掉走渐出，而不是直接 return——否则从坐着起身去走路会瞬间弹起来。
  const cur = entry.current || "";
  const idleName = entry.clipMap ? (entry.clipMap.Idle || entry.clipMap._fallback || "") : "";
  const onIdle = idleName ? cur === idleName : /idle/i.test(cur);
  if (!onIdle) want = null;

  const rig = st.rig || (st.rig = buildRig(entry.root));
  if (want !== st.name) {                       // 换姿态：先把旧的收干净再起新的，免得穿插
    st.w -= dt / RAMP;
    if (st.w <= 0) { st.w = 0; st.name = want; st.t = 0; }
  } else if (st.name) {
    st.w = Math.min(1, st.w + dt / RAMP);
  }
  if (!st.name || st.w <= 0.001) { releaseHips(rig, st); return null; }

  const pose = compile(rig, st.name);
  if (!pose) { releaseHips(rig, st); return null; }
  st.t += dt;

  if (!rig.rightSign) {
    entry.root.updateMatrixWorld(true);         // 只在首帧：后面的算法都不依赖 matrixWorld
  }
  // 角色自身坐标系：前=本地 +Z 经朝向旋转，上=世界上，右=实测符号的 (上 × 前)
  _fwd.set(0, 0, 1).applyQuaternion(worldQuat(entry.root, _qp));
  _fwd.y = 0;
  if (_fwd.lengthSq() < 1e-9) _fwd.set(0, 0, 1);
  _fwd.normalize();
  if (!rig.rightSign) rig.rightSign = measureRightSign(rig, _fwd, _up);
  _right.copy(_up).cross(_fwd).multiplyScalar(rig.rightSign);

  const w = st.w * st.w * (3 - 2 * st.w);       // smoothstep：起收更顺，不是线性楔子
  applyHipsDrop(rig, st, pose.drop * w);
  for (const item of pose.items) {
    const s = item.hz > 0
      ? 0.5 + 0.5 * Math.sin(TAU * item.hz * st.t + item.phase + st.phase)
      : 0;
    poseBone(item, s, w);
  }
  return st.name;
}

/** 这个 goal 会不会被姿态层演出来（给探针/文档用，主循环不需要）。 */
export function poseForGoal(goal) {
  return (goal && GOAL_POSE[goal]) || null;
}

// 调试挂钩，和 main.js 的 window.__dbg 一个路子：无头浏览器探针要能强扳姿态、
// 核对骨骼真的动到了不同地方。只读，不参与任何游戏逻辑。
if (typeof window !== "undefined") {
  window.__poses = { applyGoalPose, poseForGoal, POSES, GOAL_POSE, GOAL_CLIP, canon };
}
