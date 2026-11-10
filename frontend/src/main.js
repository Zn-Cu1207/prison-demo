import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { EXRLoader } from "three/addons/loaders/EXRLoader.js";
import { clone as cloneSkeleton } from "three/addons/utils/SkeletonUtils.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutlinePass } from "three/addons/postprocessing/OutlinePass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { SMAAPass } from "three/addons/postprocessing/SMAAPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { applyGoalPose, GOAL_CLIP } from "./poses.js";

// ---------------------------------------------------------------------------
// 表现层：带动画的低模人物（RobotExpressive, CC0）+ 监狱场景 + 切视角决策面板。
// 无构建，three.js 走 CDN importmap；角色模型走本地 assets。
// ---------------------------------------------------------------------------

// 后端地址从当前页面主机名推导，端口固定 8000。
// 之前写死 127.0.0.1：本机自己玩没问题，但托管给别人时，对方浏览器里的 127.0.0.1
// 指向的是他自己的机器，连不上真正的后端。用 window.location.hostname 后，
// 无论页面是本机打开还是通过内网 IP 打开，WS/接口都指向同一台服务器。
const BACKEND_HOST = `${window.location.hostname}:8000`;
const WS_URL = `ws://${BACKEND_HOST}/ws`;
// 场地尺寸必须与后端 layout.WALL 对齐（后端按它建寻路网格、派生巡逻环）。
// 从 60 收到 39.6：原来单程五十多米，后端 3.5 m/s 直线赶路、前端按各人步幅只有
// 1.2~1.7 m/s，差距被行程放大成"前端画的人落后后端十几米"。收小之后最长单程约 35 米。
const YARD = 39.6;        // 场地边长 = 2 × layout.WALL
const WALL = YARD / 2;    // 围墙半径 = layout.WALL
const GATE_X = 5.1;       // 大门在北墙上的位置 = layout.FREE 里 gate 的 x

// 夜景切面 diorama：整座监狱像一件浮在纯黑里的模型，所有亮处都来自实体灯具。
// 这一个开关同时决定天空/雾/后处理/昼夜三处的取值；改成 false 就回到原来的白天随作息变化。
const NIGHT_DIORAMA = true;

const PERSONALITY_COLOR = {
  暴躁: 0xff5c6c, 书呆: 0x4aa3e0, 老油条: 0xe0a44a,
  新人: 0x5bbf6a, 普通: 0x9aa0b0,
};
const KIND_COLOR = {
  eat: 0xe0a44a, drink: 0x4aa3e0, sleep: 0x9b6bd6, cure: 0xde6f9c,
  yard: 0x5bbf6a, class: 0x6bc5c5, labor: 0xb59a6b, gate: 0xd0d0d0,
  gym: 0xd08a4a, shower: 0x4ac0d0, visit: 0xc06be0, library: 0x8a9a5a,
  laundry: 0x6aa8b8, phone: 0x3f7fbf, market: 0x8a6bd0,
  guardpost: 0x5a6b8a, commons: 0xd0a860,
};

// goal -> 到达目标后播放的动作。原来这张表填的是 Sitting/Wave/Punch/ThumbsUp/Dance 之类
// RobotExpressive、Xbot 自带的情绪片段名；换成写实角色 GLB 之后那些片段一个都不存在，
// 全部落到 clip._fallback = "idle"，四十来种决策在画面上一律是"走过去，然后站着"。
// 现在身体语言交给 src/poses.js 的过程化姿态叠加层（在 mixer 之后压骨骼），
// 这里只留"某具模型真有对应片段"的例外——目前就是老牙的 kneeling。
const GOAL_ANIM = GOAL_CLIP;
// Kneel 是为"求人"新增的逻辑动作名（贴靠山 / 祷告）。不能直接复用 Wave——
// Wave 还被唠嗑/探视/换货/告密/卖消息共用，那样老牙会一边聊天一边跪下去。
// 只有老牙有真正的跪姿片段；其余角色回退到 idle，再由姿态层补一个过程化的跪姿。
// 需要面朝最近同伴的行动（打架/斗嘴/唠嗑/对弈/交易），更有互动感
const FACE_PEER = new Set(["Fight", "Argue", "Socialize", "Chess", "Barter",
  "Extort", "SellInfo", "SeekProtector"]);

// goal 标签 -> 颜色（性格选择分布图用）
const GOAL_COLOR = {
  吃饭: "#e0a44a", 喝水: "#4aa3e0", 睡觉: "#9b6bd6", 治伤: "#de5c6c",
  解闷: "#6bc5c5", 社交: "#5bbf6a", 打架: "#ff5c6c", 巡逻: "#8a93a8",
  学习: "#49c0d6", 越狱: "#f0f0f0", 苦役: "#b59a6b", 歇着: "#5566aa",
  健身: "#d08a4a", 洗澡: "#4ac0d0", 探视: "#c06be0", 看书: "#8a9a5a",
  赌钱: "#c9a24a", 抽烟: "#9098a4", 打扫: "#7ba86b", 祷告: "#c0a0d0",
  打电话: "#4a90e0", 洗衣: "#6aa8b8", 帮厨: "#e0b06a", 换货: "#8a6bd0",
  告密: "#a0a8b8", 下棋: "#c8c0a0", 唱歌: "#e07ab0", 纹身: "#b06a5a",
  望风: "#7a8a5a", 吵架: "#ff9a5c", 藏东西: "#6a6a8a", 收敛: "#8a8f9a", 撬门: "#c0785a",
  刷表现: "#6bbf9a", 立威: "#e05c5c", 卖消息: "#c9a24a", 核卷宗: "#5a9ad6", 贴靠山: "#9a7ad0",
};

// ---------------------------------------------------------------------------
// 渲染器 / 场景 / 相机
// ---------------------------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ antialias: false }); // 抗锯齿交给后处理 SMAA，MSAA 再开一层等于白烧 GPU
const MAX_DPR = 1.25;   // 2 在 2K/4K 上像素量翻 4 倍，GTAO+Bloom 会直接卡成幻灯
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DPR));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;   // Soft 采样更多，夜景里看不出差别
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.getElementById("app").appendChild(renderer.domElement);

const scene = new THREE.Scene();
function skyTexture() {
  const c = document.createElement("canvas");
  c.width = 8; c.height = 256;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0.0, "#243b57");   // 天顶
  grad.addColorStop(0.5, "#3c5064");
  grad.addColorStop(0.78, "#6b6f74");  // 地平线雾
  grad.addColorStop(1.0, "#20242c");
  g.fillStyle = grad; g.fillRect(0, 0, 8, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
// 夜景下先铺纯黑垫底；EveningSky HDRI 载入后换成晚霞，围墙外不再是死黑。
// 雾仍按上帝档机位距离（约 150）收边，让台基远处沉进暮色。
scene.background = NIGHT_DIORAMA ? new THREE.Color(0x04050a) : skyTexture();
scene.fog = NIGHT_DIORAMA ? new THREE.Fog(0x12182a, 90, 220) : new THREE.Fog(0x565f6a, 60, 130);

const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 400);
camera.position.set(2, 30, 40);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 2, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 12;
controls.maxDistance = 90;
controls.update();

// 调试挂钩：截图工具和浏览器控制台靠它摆机位看资产、临时调灯光数值。
// 只读引用，不参与任何游戏逻辑。
window.__dbg = {
  THREE, scene, camera, controls, renderer,
  look(cx, cz, dist = 22, pitch = 0.62, yaw = 0.78, fov = 34) {
    camera.fov = fov; camera.updateProjectionMatrix();
    camera.position.set(cx - Math.sin(yaw) * Math.cos(pitch) * dist,
                        Math.sin(pitch) * dist,
                        cz - Math.cos(yaw) * Math.cos(pitch) * dist);
    controls.target.set(cx, 1, cz);
    controls.update();
    camera.lookAt(cx, 1, cz);
    return `look ${cx},${cz} d=${dist}`;
  },
  // 这两个 Map 声明在文件后面，用取值器惰性拿（直接引用会撞上 TDZ）。
  // 校对敬礼/押送姿态时可以手动扳状态，不用真的等世界里发生一次。
  get npcs() { return npcs; },
  get patrollers() { return patrollers; },
  // 接新角色时用来读归一化缩放，反推 pace（步频对不齐就会脚底贴地平移）
  get npcTemplates() { return npcTemplates; },
  // 改布局后用来体检导航：查有没有人卡在墙里、视觉位置与后端位置差多远
  //（见 tools/_probe_nav.js）
  get blockers() { return blockers; },
  get latestWorld() { return latestWorld; },
};

// ---------------------------------------------------------------------------
// 后处理管线（Bloom 泛光 + 描边高亮 + 调色/暗角 + SMAA 抗锯齿）
// 任何构建失败都回退到直接渲染，绝不黑屏。
// ---------------------------------------------------------------------------
const GradeShader = {
  uniforms: {
    tDiffuse: { value: null }, uBrightness: { value: 1.0 },
    uContrast: { value: NIGHT_DIORAMA ? 1.12 : 1.06 },
    uSaturation: { value: NIGHT_DIORAMA ? 1.2 : 1.14 },
    // 暗角只用来收边，不参与"压暗整体"。之前给到 1.45，衰减从画面中段就开始吃，
    // 边上的房间跟着一起沉黑，是"太暗"的一部分成因。
    uVignette: { value: NIGHT_DIORAMA ? 1.12 : 1.1 },
  },
  vertexShader: "varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }",
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uBrightness,uContrast,uSaturation,uVignette;
    varying vec2 vUv;
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      vec3 col = c.rgb * uBrightness;
      col = (col - 0.5) * uContrast + 0.5;
      float l = dot(col, vec3(0.2126,0.7152,0.0722));
      col = mix(vec3(l), col, uSaturation);
      float d = length(vUv - 0.5) * uVignette;
      col *= smoothstep(0.9, 0.35, d) * 0.45 + 0.55;
      gl_FragColor = vec4(clamp(col,0.0,1.0), c.a);
    }`,
};

let composer = null, outlinePass = null, gradePass = null, gtaoPass = null;
function setupPostFx() {
  try {
    const size = new THREE.Vector2(window.innerWidth, window.innerHeight);
    composer = new EffectComposer(renderer);
    composer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DPR));
    composer.setSize(size.x, size.y);
    composer.addPass(new RenderPass(scene, camera));
    // GTAO 当初是给无贴图石墙补接触阴影的。现在墙地都有 PBR，再跑全屏 16 采样
    // 等于每帧多渲一遍场景，网页端操作会明显掉帧。需要时把下面这块取消注释。
    // try {
    //   gtaoPass = new GTAOPass(scene, camera, size.x, size.y);
    //   ...
    // } catch (e) { gtaoPass = null; }
    outlinePass = new OutlinePass(size, scene, camera);
    outlinePass.edgeStrength = 3.2; outlinePass.edgeGlow = 0.4; outlinePass.edgeThickness = 1.4;
    outlinePass.visibleEdgeColor.set(0xffd23f); outlinePass.hiddenEdgeColor.set(0x5a4a10);
    composer.addPass(outlinePass);
    // 夜景：泛光调强、阈值压低，火盆和探照灯才会"晕"开来（参考图的暖光池全靠这个）
    composer.addPass(NIGHT_DIORAMA
      ? new UnrealBloomPass(size, 0.42, 0.62, 0.76)
      : new UnrealBloomPass(size, 0.38, 0.5, 0.85));
    composer.addPass(new OutputPass());
    gradePass = new ShaderPass(GradeShader);
    composer.addPass(gradePass);
    composer.addPass(new SMAAPass(size.x, size.y));
    window.__dbg.passes = composer.passes;      // 调试用：逐个 pass 开关，定位画面异常
  } catch (e) {
    console.warn("后处理管线初始化失败，回退直接渲染：", e);
    composer = null;
  }
}
setupPostFx();

// ---------------------------------------------------------------------------
// 灯光（含阴影）
// ---------------------------------------------------------------------------
const hemi = new THREE.HemisphereLight(0xbcd0ff, 0x2a2620, 0.55);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff4e0, 1.15);
sun.position.set(24, 40, 18);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = -YARD; sun.shadow.camera.right = YARD;
sun.shadow.camera.top = YARD; sun.shadow.camera.bottom = -YARD;
sun.shadow.camera.near = 1; sun.shadow.camera.far = 120;
sun.shadow.bias = -0.0004;
scene.add(sun);
const ambient = new THREE.AmbientLight(0xffffff, 0.18);
scene.add(ambient);

// 夜间可点亮的灯（岗楼灯 / 泛光灯柱），white=0 时熄灭
const nightLamps = [];    // [{light, baseIntensity, bulb?}]
const nightGlow = [];     // 夜间发光的材质（窗户玻璃）

// ---------------------------------------------------------------------------
// 地面（程序化混凝土贴图）
// ---------------------------------------------------------------------------
function concreteTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#3a3f4b"; g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 5000; i++) {
    const v = 40 + Math.random() * 40;
    g.fillStyle = `rgba(${v},${v + 4},${v + 10},0.06)`;
    g.fillRect(Math.random() * 256, Math.random() * 256, 2, 2);
  }
  g.strokeStyle = "rgba(20,22,28,0.55)"; g.lineWidth = 2;
  for (let i = 0; i <= 256; i += 64) {
    g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 256); g.stroke();
    g.beginPath(); g.moveTo(0, i); g.lineTo(256, i); g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(YARD / 4, YARD / 4);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function wallTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#7c828d"; g.fillRect(0, 0, 256, 256);
  // 混凝土砌块 + 接缝
  for (let y = 0; y < 256; y += 32) {
    const off = (y / 32) % 2 ? 32 : 0;
    for (let x = -32; x < 256; x += 64) {
      const v = 118 + Math.random() * 22;
      g.fillStyle = `rgb(${v},${v + 3},${v + 8})`;
      g.fillRect(x + off + 1, y + 1, 62, 30);
    }
  }
  g.strokeStyle = "rgba(30,32,38,0.5)"; g.lineWidth = 1;
  for (let i = 0; i < 900; i++) { // 斑驳污渍
    g.fillStyle = `rgba(40,42,48,${Math.random() * 0.05})`;
    g.fillRect(Math.random() * 256, Math.random() * 256, 3, 3);
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(YARD, YARD),
  new THREE.MeshStandardMaterial({ map: concreteTexture(), roughness: 0.95, metalness: 0.0 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// 场地外的暗色地基。夜景 diorama 不要它：一张 180m 的平面会接光、接阴影、再淡进雾里，
// 于是监狱看着是"摊在一块大地板上"，而参考图里围墙之外什么都没有，只有黑。
// 岩石台基（buildTerrainPlinth）接手了收边的活。
if (!NIGHT_DIORAMA) {
  const apron = new THREE.Mesh(
    new THREE.PlaneGeometry(YARD * 3, YARD * 3),
    new THREE.MeshStandardMaterial({ color: 0x14171f, roughness: 1 })
  );
  apron.rotation.x = -Math.PI / 2; apron.position.y = -0.02;
  apron.receiveShadow = true;
  scene.add(apron);
}

// ---------------------------------------------------------------------------
// 围墙 + 铁丝网 + 岗楼
// ---------------------------------------------------------------------------
const wallMat = new THREE.MeshStandardMaterial({ color: 0x9298a2, roughness: 0.92, map: wallTexture() });
const railMat = new THREE.MeshStandardMaterial({ color: 0x9aa0ad, roughness: 0.4, metalness: 0.6 });
const wallH = 4;
// 碰撞代理不渲染，但 mesh.visible 必须保持 true：Box3.setFromObject 在部分 three
// 版本会跳过 hidden 物体，一跳过墙体 AABB 就是空的，人就从墙和家具里穿过去。
const hiddenBlockerMat = new THREE.MeshBasicMaterial({ visible: false });
// 围墙在 blockers 数组声明之前就要砌出来，先记下尺寸，声明后再登记碰撞盒。
const _perimeterWalls = [];

function wallSegment(len, x, z, rotY) {
  const w = new THREE.Mesh(new THREE.BoxGeometry(len, wallH, 0.5), wallMat);
  w.position.set(x, wallH / 2, z); w.rotation.y = rotY;
  w.castShadow = true; w.receiveShadow = true;
  scene.add(w);
  _perimeterWalls.push({ len, x, z, rotY, thick: 0.5 });
  // 顶部三道铁丝网横杆
  for (let i = 0; i < 3; i++) {
    const r = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, len, 6), railMat);
    r.rotation.z = Math.PI / 2; r.rotation.y = rotY;
    const off = new THREE.Vector3(0, wallH + 0.25 + i * 0.35, 0);
    r.position.set(x, off.y, z);
    scene.add(r);
  }
}
// 北墙留出大门缺口。缺口不在正中：北面那排房子（苦役区/探视室/图书室/管教办公室）
// 是连着排的，大门只能落在特意留出的那段空墙上（layout.RUNS 里 kind="gap" 那一项）。
const gateW = 3.4;
const leftLen = GATE_X - gateW / 2 + WALL;
const rightLen = WALL - GATE_X - gateW / 2;
wallSegment(leftLen, (GATE_X - gateW / 2 - WALL) / 2, WALL, 0);
wallSegment(rightLen, (GATE_X + gateW / 2 + WALL) / 2, WALL, 0);
wallSegment(YARD, 0, -WALL, 0);                        // 后墙
wallSegment(YARD, -WALL, 0, Math.PI / 2);              // 左墙
wallSegment(YARD, WALL, 0, Math.PI / 2);               // 右墙

// 两点之间连一根圆柱（桁架梁/斜撑/栏杆共用）。给岗楼搭骨架用，省去手算角度。
const _bmA = new THREE.Vector3(), _bmB = new THREE.Vector3(), _bmUp = new THREE.Vector3(0, 1, 0);
function beamBetween(x1, y1, z1, x2, y2, z2, r, mat) {
  _bmA.set(x1, y1, z1); _bmB.set(x2, y2, z2);
  const len = _bmA.distanceTo(_bmB);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 6), mat);
  m.position.copy(_bmA).add(_bmB).multiplyScalar(0.5);
  m.quaternion.setFromUnitVectors(_bmUp, _bmB.sub(_bmA).normalize());
  m.castShadow = true;
  return m;
}

// 岗楼：一座有桁架腿、交叉斜撑、带栏杆的观察台和玻璃岗亭的钢木哨塔。
// 旧版是"四根筷子 + 一个方盒子 + 一顶圆锥"，正对镜头时一眼就是 low-poly 占位物；
// 这一版按真实瞭望塔的结构层次搭：混凝土墩 → 桁架腿 → 观察平台 → 岗亭 → 挑檐顶，
// 探照灯逻辑（spot/aim/beam/lamp 与 updateSearchlights）原样保留，只是抬到新平台高度。
function guardTower(x, z) {
  const g = new THREE.Group();
  const legMat = new THREE.MeshStandardMaterial({ color: 0x53585f, roughness: 0.7, metalness: 0.35 });
  const braceMat = new THREE.MeshStandardMaterial({ color: 0x3f444c, roughness: 0.7, metalness: 0.5 });
  const deckMat = new THREE.MeshStandardMaterial({ color: 0x6b5334, roughness: 0.92 });
  const cabinMat = new THREE.MeshStandardMaterial({ color: 0x8a8f98, roughness: 0.9, map: wallTexture() });
  const roofMat = new THREE.MeshStandardMaterial({ color: 0x33373f, roughness: 0.82, metalness: 0.2 });
  // 岗亭玻璃单独一份材质（不能用后面才声明的 glassMat——这函数在模块加载时就调用了，
  // 引用还没初始化的 const 会触发 TDZ）。照样登记进 nightGlow，夜里跟着亮起暖光。
  const towerGlass = new THREE.MeshStandardMaterial({
    color: 0x14222f, roughness: 0.3, metalness: 0.4, emissive: 0xffcc66, emissiveIntensity: 0,
  });
  nightGlow.push(towerGlass);

  const S = 1.15;            // 腿间半跨
  const DECK = 5.4;          // 观察平台台面高
  // 混凝土墩 + 四条桁架腿
  for (const [dx, dz] of [[-S, -S], [S, -S], [-S, S], [S, S]]) {
    g.add(boxMesh(0.5, 0.6, 0.5, cabinMat, dx, 0.3, dz));
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.22, DECK, 0.22), legMat);
    leg.position.set(dx, DECK / 2, dz); leg.castShadow = true; g.add(leg);
  }
  // 四个侧面：两道水平横梁 + 一组交叉斜撑，看着是桁架而不是四根光腿
  const faces = [[-S, -S, S, -S], [S, -S, S, S], [S, S, -S, S], [-S, S, -S, -S]];
  for (const [x1, z1, x2, z2] of faces) {
    for (const hy of [2.0, 4.0]) g.add(beamBetween(x1, hy, z1, x2, hy, z2, 0.07, braceMat));
    g.add(beamBetween(x1, 0.6, z1, x2, DECK - 0.2, z2, 0.06, braceMat));
    g.add(beamBetween(x2, 0.6, z2, x1, DECK - 0.2, z1, 0.06, braceMat));
  }
  // 观察平台
  const deckSize = 2 * S + 0.7;
  g.add(boxMesh(deckSize, 0.18, deckSize, deckMat, 0, DECK + 0.09, 0));
  const rTop = DECK + 0.18, railH = 1.0, dh = deckSize / 2 - 0.06;
  const corners = [[-dh, -dh], [dh, -dh], [dh, dh], [-dh, dh]];
  for (const [px, pz] of corners) g.add(boxMesh(0.08, railH, 0.08, railMat, px, rTop + railH / 2, pz));
  for (let i = 0; i < 4; i++) {
    const [ax, az] = corners[i], [bx, bz] = corners[(i + 1) % 4];
    for (const hy of [railH * 0.5, railH]) g.add(beamBetween(ax, rTop + hy, az, bx, rTop + hy, bz, 0.028, railMat));
  }
  // 爬梯：贴在朝院子那面（+z）两条腿之间
  const lz = S + 0.16;
  for (const rx of [-0.32, 0.32]) g.add(boxMesh(0.05, DECK + 0.1, 0.05, railMat, rx, (DECK + 0.1) / 2, lz));
  for (let ry = 0.5; ry < DECK; ry += 0.42) g.add(boxMesh(0.7, 0.04, 0.04, railMat, 0, ry, lz));
  // 岗亭：角柱 + 下裙墙 + 四面玻璃观察窗 + 顶圈梁
  const cw = deckSize - 1.0, cH = 1.9, cy = rTop, half = cw / 2, sill = 0.65;
  for (const [px, pz] of [[-half, -half], [half, -half], [half, half], [-half, half]]) {
    g.add(boxMesh(0.14, cH, 0.14, cabinMat, px, cy + cH / 2, pz));
  }
  for (const s of [{ x: 0, z: -half, ax: 1 }, { x: half, z: 0, ax: 0 },
                   { x: 0, z: half, ax: 1 }, { x: -half, z: 0, ax: 0 }]) {
    const ww = s.ax ? cw : 0.1, dd = s.ax ? 0.1 : cw;
    g.add(boxMesh(ww, sill, dd, cabinMat, s.x, cy + sill / 2, s.z));
    g.add(boxMesh(ww, cH - sill - 0.18, dd, towerGlass, s.x, cy + sill + (cH - sill - 0.18) / 2, s.z));
  }
  g.add(boxMesh(cw + 0.24, 0.14, cw + 0.24, roofMat, 0, cy + cH + 0.02, 0));
  // 挑檐四坡顶 + 屋脊帽
  const roof = new THREE.Mesh(new THREE.ConeGeometry((half + 0.5) * 1.42, 1.25, 4), roofMat);
  roof.position.y = cy + cH + 0.72; roof.rotation.y = Math.PI / 4; roof.castShadow = true; g.add(roof);
  g.add(boxMesh(0.14, 0.45, 0.14, roofMat, 0, cy + cH + 1.4, 0));

  const lamp = new THREE.PointLight(0xffe08a, 0, 30, 2);
  lamp.position.set(0, DECK + 1.0, 0); g.add(lamp);
  nightLamps.push({ light: lamp, baseIntensity: 8 });
  // 探照灯：一束会来回扫的冷白光 + 看得见的光柱。参考图里最抓眼的一笔就是这个。
  // 光柱是个加法混合的空心锥，不写深度——纯粹骗眼睛，比体积光便宜得多。
  const ANG = 0.075, LEN = 26;
  // 探照灯装在朝院子那面（-z）的平台边沿上，不再埋在岗亭正中——埋在玻璃盒里光柱会穿墙。
  const sx0 = 0, sy0 = rTop + 0.35, sz0 = -dh;
  g.add(boxMesh(0.42, 0.34, 0.34, braceMat, sx0, sy0, sz0));         // 灯箱外壳
  g.add(boxMesh(0.06, 0.28, 0.28, towerGlass, sx0, sy0, sz0 - 0.2)); // 灯面（朝院子）
  const spot = new THREE.SpotLight(0xeaf1ff, 0, LEN + 8, ANG, 0.5, 1.0);
  spot.position.set(sx0, sy0, sz0);
  const aim = new THREE.Object3D();
  aim.position.set(0, -6.4, 14);
  g.add(aim); spot.target = aim; g.add(spot);
  nightLamps.push({ light: spot, baseIntensity: 8 });
  const beamGeo = new THREE.ConeGeometry(Math.tan(ANG) * LEN, LEN, 20, 1, true);
  beamGeo.translate(0, -LEN / 2, 0);      // 锥尖挪到原点
  // 顶点色做纵向渐隐：灯口白、远端归零。少了这一步，加法混合的锥体就是一片灰白板子。
  const bp = beamGeo.attributes.position;
  const bc = new Float32Array(bp.count * 3);
  for (let i = 0; i < bp.count; i++) {
    const k = Math.pow(1 + bp.getY(i) / LEN, 1.8);   // 锥尖 1 → 锥底 0
    bc[i * 3] = bc[i * 3 + 1] = bc[i * 3 + 2] = k;
  }
  beamGeo.setAttribute("color", new THREE.BufferAttribute(bc, 3));
  beamGeo.rotateX(-Math.PI / 2);          // 让光柱沿 +Z 射出，好用 lookAt 对准目标
  const beam = new THREE.Mesh(beamGeo, new THREE.MeshBasicMaterial({
    color: 0xdce8ff, transparent: true, opacity: 0, side: THREE.DoubleSide,
    depthWrite: false, blending: THREE.AdditiveBlending, vertexColors: true,
  }));
  beam.position.copy(spot.position);
  beam.frustumCulled = false;
  beam.name = "searchbeam";
  g.add(beam);
  // 记下岗楼的世界落点：扫射目标要按"从这座塔看向中庭"来算。
  // 原来写的是固定的局部偏移（+z 方向 19 米），塔一换位置光柱就扫到墙外的黑地上去了。
  searchlights.push({ spot, aim, beam, x, z, phase: searchlights.length * 2.1 });
  g.position.set(x, 0, z);
  scene.add(g);
}
// 岗楼探照灯：{spot, aim, beam, x, z, phase}，在主循环里来回扫（见 updateSearchlights）
const searchlights = [];
const _beamAim = new THREE.Vector3();
function updateSearchlights(t) {
  for (const s of searchlights) {
    const a = s.phase + t * 0.22;
    // 扫的是院子，不是天：目标点绕着"塔与场地中心之间"那个点慢慢摆，
    // 半径 7 米。这样不论塔在哪个角，光柱都落在中庭里。
    s.aim.position.set(-s.x * 0.62 + Math.sin(a) * 7, -6.4,
                       -s.z * 0.62 + Math.cos(a * 0.7) * 7);
    s.aim.getWorldPosition(_beamAim);
    s.beam.lookAt(_beamAim);
    s.beam.material.opacity = lightState.night * 0.05;
  }
}
// 岗楼摆在北面两个角：房子贴墙成排之后，南面四角全被洗衣房/管教办占住了，
// 塔立在那儿就是一根穿过屋顶剖面的柱子。西北/东北这两片是布局故意留空的。
guardTower(-WALL + 2.5, WALL - 3.0);
guardTower(WALL - 2.5, WALL - 3.0);

// 建筑/道具共用材质。必须在下方任何构建调用之前声明（否则触发 TDZ）。
const glassMat = new THREE.MeshStandardMaterial({
  color: 0x11202e, roughness: 0.3, metalness: 0.4,
  emissive: 0xffcc66, emissiveIntensity: 0,   // 夜间点亮成暖黄窗
});
nightGlow.push(glassMat);
const barMat = new THREE.MeshStandardMaterial({ color: 0x2a2e36, roughness: 0.5, metalness: 0.7 });
const doorMat = new THREE.MeshStandardMaterial({ color: 0x14171d, roughness: 0.8 });

const npcRoots = [];              // 用于射线拾取
const clueMeshes = [];            // 可交互线索物（侦探系统）
const interactables = [];         // 可互动家具/物品（坐/躺/捡/开关灯）
const stations = [];              // NPC 坐/躺工位：从床/凳锚点登记，claim 后才播坐躺，避免空地上做动作
window.__dbg.stations = stations;
const roomLamps = [];             // 室内灯：{light, bulb, on, manual}
const blockers = [];              // 墙体碰撞盒（世界系 AABB），主角不能穿墙
// 家具碰撞盒（桌椅床柜等）。单列一份，只挡玩家、并按 nav 标志决定是否让 NPC 绕行：
//   · 玩家（第一/三人称走化身）撞上去被弹开——修"人从桌子里穿过去"。
//   · NPC 位置是后端权威、且要走到床/凳上休息，硬挡会和后端拉橡皮筋；所以 NPC 不做
//     家具碰撞，只在寻路网格里绕开 nav=true 的大件（院子/房中央的桌台），牢房里的
//     床/马桶/柜(nav=false)不进网格，免得把本就狭小的隔间封死、人卡在门口。
const propBlockers = [];
window.__dbg.props = propBlockers;   // 探针据此核对家具碰撞盒
(function registerPerimeterBlockers() {
  for (const { len, x, z, rotY, thick } of _perimeterWalls) {
    const c = Math.abs(Math.cos(rotY)), s = Math.abs(Math.sin(rotY));
    const hx = c * len / 2 + s * thick / 2;
    const hz = s * len / 2 + c * thick / 2;
    blockers.push({ minX: x - hx, maxX: x + hx, minZ: z - hz, maxZ: z + hz });
  }
})();

// ---------------------------------------------------------------------------
// 全局黑板镜像（后端 world_state.blackboard 的前端副本，唯一真相源）
// 观察者（上帝视角）：整座监狱一开局就全部建出、全部活着（不再逐区展开）。
// ---------------------------------------------------------------------------
let bb = { zones_unlocked: ["cellblock", "public", "restricted", "office"], inventory: [], flags: {} };
// 失效即放行：拿不到 zone（后端还没重启/旧版）时按"可见"处理，绝不把全场藏没了。
function zoneUnlocked(z) { return !z || bb.zones_unlocked.includes(z); }
// 玩家可活动范围：已解锁设施的包围盒 + 余量。未解锁分区走不过去（那边还没长出来）。
let playerBounds = null;          // {minX,maxX,minZ,maxZ}
let boundsKey = "";               // zones_unlocked 变了才重算
function recomputeBounds(facilities) {
  const key = bb.zones_unlocked.slice().sort().join(",");
  if (key === boundsKey || !facilities) return;
  boundsKey = key;
  const M = 9;                    // 房间四周的活动余量
  let minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9;
  for (const f of facilities) {
    if (!zoneUnlocked(f.zone)) continue;
    minX = Math.min(minX, f.pos[0]); maxX = Math.max(maxX, f.pos[0]);
    minZ = Math.min(minZ, f.pos[1]); maxZ = Math.max(maxZ, f.pos[1]);
  }
  if (minX > maxX) { playerBounds = null; return; }   // 还没有已解锁设施
  // 上限贴着围墙内表面。房子现在紧靠围墙（外墙面 ±29.2），上限要是还按 WALL-2.2=27.8 收，
  // 下面那圈"分区边界墙"就会横穿一排房子。真正把人挡在里头的是房子本身 + 围墙。
  const lim = WALL - 0.25;
  playerBounds = {
    minX: Math.max(-lim, minX - M), maxX: Math.min(lim, maxX + M),
    minZ: Math.max(-lim, minZ - M), maxZ: Math.min(lim, maxZ + M),
  };
  rebuildBoundaryWalls();             // 已解锁范围四周用真墙围起来（没解锁的空地走不过去）
}

// 沿"已解锁范围"四周砌墙：玩家被实墙挡在里头，看得见摸得着；分区一解锁，墙就往外扩。
let boundaryGroup = null;
function rebuildBoundaryWalls() {
  if (boundaryGroup) {
    scene.remove(boundaryGroup);
    boundaryGroup.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    boundaryGroup = null;
  }
  if (!playerBounds) return;
  const { minX, maxX, minZ, maxZ } = playerBounds;
  const H = 3.2, T = 0.5;                         // 墙高、墙厚
  const midX = (minX + maxX) / 2, midZ = (minZ + maxZ) / 2;
  const w = maxX - minX + T, d = maxZ - minZ + T;
  const g = new THREE.Group();
  const wall = (cx, cz, sx, sz) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(sx, H, sz), wallMat);
    m.position.set(cx, H / 2, cz); m.castShadow = true; m.receiveShadow = true;
    g.add(m);
  };
  // 已经顶到围墙的那一面不再另砌：那儿本来就有围墙，重复一道纯属浪费，
  // 还会和贴墙那排房子的外墙面打架。
  const at = (v) => Math.abs(Math.abs(v) - (WALL - 0.25)) < 0.01;
  if (!at(minZ)) wall(midX, minZ, w, T);            // 南
  if (!at(maxZ)) wall(midX, maxZ, w, T);            // 北
  if (!at(minX)) wall(minX, midZ, T, d);            // 西
  if (!at(maxX)) wall(maxX, midZ, T, d);            // 东
  scene.add(g);
  boundaryGroup = g;
}
// 某坐标属于哪个分区（取最近设施的 zone）——线索按此决定是否显形。
function zoneOfPos(x, z, facilities) {
  let best = null, bd = 1e9;
  for (const f of facilities) {
    const dx = f.pos[0] - x, dz = f.pos[1] - z, d = dx * dx + dz * dz;
    if (d < bd) { bd = d; best = f; }
  }
  return best ? best.zone : "public";
}
// 线索物只在其所在分区解锁后才显形（阶段一牢房里不该有发光线索漂在黑暗里）。
function syncClueVisibility(facilities) {
  if (!facilities) return;
  for (const c of clueEntries) {
    if (c._zone === undefined) c._zone = zoneOfPos(c.data.pos[0], c.data.pos[1], facilities);
    c.group.visible = zoneUnlocked(c._zone);
  }
}

// 房间摆好位置后再收集墙体包围盒：建墙时 group 还没定位，此时算才是世界坐标。
// 不用 Box3.setFromObject：它在部分 three 版本会跳过 visible=false 的代理墙，
// 算出空盒（min=+Inf），碰撞等于没登记，人就穿墙穿家具。
const _box = new THREE.Box3();
const _boxMesh = new THREE.Box3();
function geometryWorldBox(o, target) {
  const geo = o.geometry;
  if (!geo) { target.makeEmpty(); return target; }
  if (geo.boundingBox == null) geo.computeBoundingBox();
  if (!geo.boundingBox || geo.boundingBox.isEmpty()) { target.makeEmpty(); return target; }
  return target.copy(geo.boundingBox).applyMatrix4(o.matrixWorld);
}
function collectBlockers(root) {
  root.updateWorldMatrix(true, true);
  root.traverse((o) => {
    if (o.userData.blocker && o.isMesh) {
      geometryWorldBox(o, _box);
      if (_box.isEmpty()) return;
      blockers.push({ minX: _box.min.x, maxX: _box.max.x, minZ: _box.min.z, maxZ: _box.max.z });
    } else if (o.userData.prop) {
      // 家具组：整组算一个 AABB（子网格没有 prop 标记，不会重复登记）。
      // 稍微内缩 0.06，免得 body 半径叠上盒子后连门口都过不去。
      _box.makeEmpty();
      o.traverse((c) => {
        if (!c.isMesh) return;
        geometryWorldBox(c, _boxMesh);
        if (!_boxMesh.isEmpty()) _box.union(_boxMesh);
      });
      if (_box.isEmpty()) return;
      const minX = _box.min.x + 0.06, maxX = _box.max.x - 0.06;
      const minZ = _box.min.z + 0.06, maxZ = _box.max.z - 0.06;
      if (maxX <= minX || maxZ <= minZ) return;
      propBlockers.push({ minX, maxX, minZ, maxZ, nav: !!o.userData.propNav });
      NAV.dirty = true;
    }
  });
}
// 家具直接摆进场景（buildStaticDecor）时用它：定位后收一次碰撞盒。
function addProp(obj, { nav = false } = {}) {
  obj.userData.prop = true;
  obj.userData.propNav = nav;
  scene.add(obj);
  collectBlockers(obj);
  return obj;
}
// 烧进静态批次的环境模块（envAdd 出来的桶/箱/工作台）没有独立对象可取 AABB，
// 用它按摆放点手工登记一块碰撞盒。默认 nav:false（只挡玩家，不扰后端权威的 NPC）。
function propBox(x, z, w, d, { nav = false } = {}) {
  propBlockers.push({ minX: x - w / 2, maxX: x + w / 2, minZ: z - d / 2, maxZ: z + d / 2, nav });
  NAV.dirty = true;
}

// 登记一件可互动物。obj 会被标成 hoverRoot，供描边高亮与射线拾取统一识别。
function addInteractable(obj, spec) {
  obj.userData.hoverRoot = true;
  obj.userData.act = spec;
  const it = { obj, ...spec };
  obj.userData.entry = it;
  interactables.push(it);
  if ((spec.type === "sit" || spec.type === "lie") && spec.anchor) {
    registerStation(spec.anchor, spec.type, spec.kind || "", spec.slot ?? -1, it);
  }
  return it;
}

function registerStation(anchor, type, kind, slot = -1, it = null) {
  const rec = { type, kind: kind || "", slot, claimed: "", anchor, it };
  stations.push(rec);
  if (it) it.station = rec;
  return rec;
}

function releaseStation(e) {
  if (e?.station) {
    if (e.station.claimed === e.id) e.station.claimed = "";
    e.station = null;
  }
}

function claimStation(e, spec) {
  if (e.station && e.station.type === spec.type && e.station.kind === spec.kind
      && (spec.own ? e.station.slot === e.cell : true)) {
    return e.station;
  }
  releaseStation(e);
  const pool = stations.filter((s) => s.type === spec.type && s.kind === spec.kind
    && (!s.claimed || s.claimed === e.id)
    && !(s.it && s.it.taken));
  let pick = null;
  if (spec.own && e.cell >= 0) {
    pick = pool.find((s) => s.slot === e.cell) || null;
  }
  if (!pick) {
    let bd = 1e9;
    const px = e.dest.x, pz = e.dest.z;
    for (const s of pool) {
      s.anchor.getWorldPosition(_wp);
      const d = (_wp.x - px) ** 2 + (_wp.z - pz) ** 2;
      if (d < bd) { bd = d; pick = s; }
    }
  }
  if (pick) {
    pick.claimed = e.id;
    e.station = pick;
  }
  return pick;
}
// 取世界坐标（家具通常挂在房间 group 下，本地坐标不能直接用）
const _wp = new THREE.Vector3();
function worldPos(obj) { return obj.getWorldPosition(_wp).clone(); }

// ---------------------------------------------------------------------------
// 场景道具：泛光灯柱 / 水塔 / 大门招牌 / 铁丝网栅栏
// ---------------------------------------------------------------------------
const poleMat = new THREE.MeshStandardMaterial({ color: 0x3a3f49, metalness: 0.5, roughness: 0.6 });

function floodPole(x, z) {
  const g = new THREE.Group();
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.16, 7.5, 10), poleMat);
  pole.position.y = 3.75; pole.castShadow = true; g.add(pole);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.4, 0.6),
    new THREE.MeshStandardMaterial({ color: 0x2b2f3a }));
  head.position.set(0, 7.5, 0.35); g.add(head);
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.2, 10, 10),
    new THREE.MeshStandardMaterial({ color: 0xfff2c0, emissive: 0xffe08a, emissiveIntensity: 0 }));
  bulb.position.set(0, 7.42, 0.62); g.add(bulb);
  const light = new THREE.PointLight(0xffe6b0, 0, 34, NIGHT_DIORAMA ? 1.5 : 1.8);
  light.position.set(0, 7.4, 0.6); g.add(light);
  nightLamps.push({ light, baseIntensity: NIGHT_DIORAMA ? 9 : 16, bulb });
  g.position.set(x, 0, z); scene.add(g);
}
// 灯柱只能立在布局留空的墙脚。以前四根按老落点钉在 (±27.5, -2/3) 和北面两角，
// 房子贴墙成排之后有两根正插在牢房和娱乐室里——从斜上方看是一根黑柱穿过牢房剖面。
// 现在两根守大门（缺口两侧），两根立在西/东墙未被占用的那一段。
floodPole(-4.2, WALL - 2.4); floodPole(4.2, WALL - 2.4);
floodPole(-WALL + 2.5, 13); floodPole(WALL - 2.5, 15);

function waterTower(x, z) {
  const g = new THREE.Group();
  const legMat = new THREE.MeshStandardMaterial({ color: 0x545a66, roughness: 0.85 });
  for (const [dx, dz] of [[-1.1, -1.1], [1.1, -1.1], [-1.1, 1.1], [1.1, 1.1]]) {
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 6.5, 8), legMat);
    leg.position.set(dx, 3.25, dz); leg.castShadow = true; g.add(leg);
    for (const yy of [2, 4]) {
      const brace = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.08, 0.08), legMat);
      brace.position.set(0, yy, dz); g.add(brace);
    }
  }
  const tank = new THREE.Mesh(new THREE.CylinderGeometry(2.1, 2.1, 2.6, 18),
    new THREE.MeshStandardMaterial({ color: 0x8a9098, roughness: 0.7, metalness: 0.3 }));
  tank.position.y = 7.8; tank.castShadow = true; g.add(tank);
  const cone = new THREE.Mesh(new THREE.ConeGeometry(2.15, 1.1, 18),
    new THREE.MeshStandardMaterial({ color: 0x3a4150 }));
  cone.position.y = 9.6; g.add(cone);
  const base = new THREE.Mesh(new THREE.ConeGeometry(2.1, 0.9, 18),
    new THREE.MeshStandardMaterial({ color: 0x6b7078 }));
  base.position.y = 6.1; base.rotation.x = Math.PI; g.add(base);
  g.position.set(x, 0, z); scene.add(g);
}
// 水塔原在西南角 (-26,-26)，那儿现在是洗衣房主间，塔腿正好落在屋里。
// 挪到西墙中段这片空地：禁闭室北端 z=7.6 以上、苦役区西端 x=-21 以西都没占。
waterTower(-WALL + 4, 20);

function textTexture(text, sub) {
  const c = document.createElement("canvas");
  c.width = 512; c.height = 160;
  const g = c.getContext("2d");
  g.fillStyle = "#1a1d24"; g.fillRect(0, 0, 512, 160);
  g.strokeStyle = "#c9a24a"; g.lineWidth = 6; g.strokeRect(6, 6, 500, 148);
  g.fillStyle = "#e8d9a8"; g.textAlign = "center"; g.textBaseline = "middle";
  g.font = "bold 58px system-ui"; g.fillText(text, 256, 62);
  g.font = "24px system-ui"; g.fillStyle = "#b9c0cc"; g.fillText(sub, 256, 118);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const sign = new THREE.Mesh(new THREE.BoxGeometry(6, 1.8, 0.2),
  new THREE.MeshStandardMaterial({ map: textTexture("州 立 监 狱", "STATE PENITENTIARY · BLOCK D") }));
sign.position.set(0, 5.4, WALL); sign.castShadow = true; scene.add(sign);
for (const sx of [-2.6, 2.6]) {
  const p = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 6.4, 8), poleMat);
  p.position.set(sx, 3.2, WALL); scene.add(p);
}

// ---------------------------------------------------------------------------
// 模块化室内道具：开放牢房(床/柜/马桶) + 桌椅/板条箱等散景
// ---------------------------------------------------------------------------
const woodMat = new THREE.MeshStandardMaterial({ color: 0x6b5334, roughness: 0.9 });
const metalMat = new THREE.MeshStandardMaterial({ color: 0x6a7078, roughness: 0.45, metalness: 0.6 });
const mattressMat = new THREE.MeshStandardMaterial({ color: 0x9aa0aa, roughness: 1 });
const ceramicMat = new THREE.MeshStandardMaterial({ color: 0xd8dde4, roughness: 0.35 });

function boxMesh(w, h, d, mat, x, y, z) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z); m.castShadow = true; m.receiveShadow = true;
  return m;
}
function makeLocker() {
  const g = new THREE.Group();
  g.add(boxMesh(0.9, 2.0, 0.5, metalMat, 0, 1.0, 0));
  const door = boxMesh(0.4, 1.8, 0.02, new THREE.MeshStandardMaterial({ color: 0x565c65, roughness: 0.5, metalness: 0.5 }), -0.2, 1.0, 0.26);
  g.add(door);
  g.add(boxMesh(0.4, 1.8, 0.02, door.material, 0.2, 1.0, 0.26));
  g.add(boxMesh(0.05, 0.12, 0.04, metalMat, -0.02, 1.05, 0.29));  // 把手
  g.userData.prop = true;
  return g;
}
function makeBunk() {
  const g = new THREE.Group();
  for (const dx of [-0.9, 0.9]) for (const dz of [-0.35, 0.35]) g.add(boxMesh(0.1, 1.5, 0.1, metalMat, dx, 0.75, dz));
  for (const y of [0.5, 1.35]) {
    g.add(boxMesh(2.0, 0.12, 0.9, metalMat, 0, y, 0));
    g.add(boxMesh(1.9, 0.16, 0.82, mattressMat, 0, y + 0.13, 0));
    g.add(boxMesh(0.5, 0.14, 0.34, new THREE.MeshStandardMaterial({ color: 0xe6e8ee, roughness: 1 }), -0.65, y + 0.15, 0));  // 枕
  }
  g.userData.prop = true;
  return g;
}
function makeToilet() {
  const g = new THREE.Group();
  g.add(boxMesh(0.5, 0.45, 0.6, ceramicMat, 0, 0.22, 0));
  const bowl = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.2, 0.2, 16), ceramicMat);
  bowl.position.y = 0.52; g.add(bowl);
  g.add(boxMesh(0.5, 0.5, 0.12, ceramicMat, 0, 0.7, -0.28));  // 水箱
  g.userData.prop = true;
  return g;
}
function makeCrate(s = 0.9) {
  // 优先用 Jampot 木箱：程序化那版是"方盒子 + 三道箍"，而模块自带板条、铁角和翘起的箱盖，
  // 是同一个位置能拿到的最大一笔细节增益。拿不到模块（加载失败）再回退。
  const jam = envObject("crate", s);
  if (jam) { jam.userData.prop = true; return jam; }
  const g = new THREE.Group();
  g.add(boxMesh(s, s, s, woodMat, 0, s / 2, 0));
  const edge = new THREE.MeshStandardMaterial({ color: 0x4f3d26, roughness: 0.9 });
  for (const yy of [0.15, 0.5, 0.85]) g.add(boxMesh(s + 0.02, 0.05, s + 0.02, edge, 0, s * yy, 0));
  g.userData.prop = true;
  return g;
}
function makeTable() {
  const g = new THREE.Group();
  g.add(boxMesh(1.8, 0.1, 0.9, woodMat, 0, 0.75, 0));
  for (const dx of [-0.8, 0.8]) for (const dz of [-0.35, 0.35]) g.add(boxMesh(0.1, 0.75, 0.1, woodMat, dx, 0.37, dz));
  g.userData.prop = true;
  return g;
}
// —— 后端物资实体的可视化（world_objects.py）——
// 份数是后端权威的：这里只负责"还剩几份就显几个"，让"食堂没饭了"看得见。
function makeFoodTray(portions = 6) {
  const g = new THREE.Group();
  const tray = new THREE.MeshStandardMaterial({ color: 0x8d939c, roughness: 0.6, metalness: 0.35 });
  g.add(boxMesh(1.0, 0.06, 0.62, tray, 0, 0.83, 0));
  const bun = new THREE.MeshStandardMaterial({ color: 0xd9c08a, roughness: 0.85 });
  const buns = [];
  for (let i = 0; i < portions; i++) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.11, 12, 10), bun);
    m.position.set(-0.36 + (i % 3) * 0.36, 0.92, -0.13 + Math.floor(i / 3) * 0.26);
    m.scale.y = 0.72;
    g.add(m); buns.push(m);
  }
  g.userData.buns = buns;
  return g;
}
function makeTap() {
  const g = new THREE.Group();
  g.add(boxMesh(0.24, 0.7, 0.24, metalMat, 0, 0.35, 0));
  g.add(boxMesh(0.1, 0.1, 0.34, metalMat, 0, 0.68, 0.14));
  const basin = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.5, metalness: 0.5 });
  g.add(boxMesh(0.6, 0.12, 0.44, basin, 0, 0.3, 0.22));
  return g;
}
// 铺位占用标记：牢房里已经有床的模型了，这里只加一层"这张有没有人"的指示，
// 免得再摆一张床跟原有家具打架。
function makeBedMarker() {
  const g = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color: 0x4a4f5a, transparent: true,
                                            opacity: 0.35, side: THREE.DoubleSide });
  const pad = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 0.72), mat);
  pad.rotation.x = -Math.PI / 2; pad.position.y = 0.62;
  g.add(pad);
  const zzz = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 8, 7),
    new THREE.MeshBasicMaterial({ color: 0x9fd0ff, transparent: true, opacity: 0.8 }));
  zzz.position.set(0, 1.15, 0); zzz.visible = false;
  g.add(zzz);
  g.userData.mark = pad;
  g.userData.zzz = zzz;
  return g;
}
function makeMedCabinet() {
  const g = new THREE.Group();
  const white = new THREE.MeshStandardMaterial({ color: 0xe8eef2, roughness: 0.55 });
  g.add(boxMesh(0.7, 1.1, 0.34, white, 0, 0.55, 0));
  const red = new THREE.MeshStandardMaterial({ color: 0xd9433f, roughness: 0.5 });
  g.add(boxMesh(0.34, 0.09, 0.02, red, 0, 0.78, 0.18));
  g.add(boxMesh(0.09, 0.34, 0.02, red, 0, 0.78, 0.18));
  g.userData.prop = true;
  return g;
}
// 食堂的长条餐桌：一块长台面 + 沿长边分布的桌腿。makeTable 是固定 1.8m 的小方桌，
// 撑不起"横贯食堂的一排餐桌"，所以单独出一个按长度生成的。
function makeMessTable(len = 6.0) {
  const g = new THREE.Group();
  g.add(boxMesh(len, 0.1, 0.9, woodMat, 0, 0.75, 0));
  const n = Math.max(2, Math.round(len / 2.2));
  for (let i = 0; i < n; i++) {
    const x = -len / 2 + 0.35 + (len - 0.7) * (i / (n - 1));
    for (const dz of [-0.35, 0.35]) g.add(boxMesh(0.1, 0.75, 0.1, woodMat, x, 0.37, dz));
  }
  g.userData.prop = true;
  return g;
}
function makeStool(x, z) {
  const g = new THREE.Group();
  g.add(boxMesh(0.5, 0.08, 0.5, woodMat, 0, 0.45, 0));
  for (const dx of [-0.2, 0.2]) for (const dz of [-0.2, 0.2]) g.add(boxMesh(0.06, 0.45, 0.06, woodMat, dx, 0.22, dz));
  g.position.set(x, 0, z); g.userData.prop = true; return g;
}
function makeChair() {                       // 带靠背的椅子（办公/教室）
  const g = new THREE.Group();
  g.add(boxMesh(0.52, 0.08, 0.52, woodMat, 0, 0.46, 0));
  g.add(boxMesh(0.52, 0.6, 0.07, woodMat, 0, 0.76, -0.23));
  for (const dx of [-0.21, 0.21]) for (const dz of [-0.21, 0.21]) g.add(boxMesh(0.06, 0.46, 0.06, metalMat, dx, 0.23, dz));
  g.userData.prop = true;
  return g;
}
function makeBench(len = 2.0) {              // 长凳（澡堂/健身/放风）
  // Jampot 石板长凳：按长度缩放去顶替程序化的"木板 + 两条铁腿"
  const jam = envObject("bench", len, "x");
  if (jam) { jam.userData.prop = true; return jam; }
  const g = new THREE.Group();
  g.add(boxMesh(len, 0.12, 0.46, woodMat, 0, 0.44, 0));
  for (const dx of [-len / 2 + 0.22, len / 2 - 0.22]) g.add(boxMesh(0.1, 0.44, 0.42, metalMat, dx, 0.22, 0));
  g.userData.prop = true;
  return g;
}
function makeDesk() {                        // 书桌/办公桌（带抽屉）
  const g = new THREE.Group();
  g.add(boxMesh(1.5, 0.09, 0.75, woodMat, 0, 0.76, 0));
  g.add(boxMesh(0.5, 0.5, 0.66, new THREE.MeshStandardMaterial({ color: 0x5b4529, roughness: 0.9 }), 0.44, 0.47, 0));
  g.add(boxMesh(0.44, 0.05, 0.03, metalMat, 0.44, 0.6, 0.34));
  for (const dx of [-0.68, 0.68]) g.add(boxMesh(0.08, 0.72, 0.68, woodMat, dx, 0.36, 0));
  g.userData.prop = true;
  return g;
}
function makeShelf() {                       // 书架（图书室/办公室）
  const g = new THREE.Group();
  const side = new THREE.MeshStandardMaterial({ color: 0x5b4529, roughness: 0.95 });
  for (const dx of [-0.62, 0.62]) g.add(boxMesh(0.08, 2.1, 0.36, side, dx, 1.05, 0));
  const bookCols = ["#8a3b3b", "#3b5a8a", "#4a7a4a", "#8a7a3b", "#6a3b7a", "#3b7a7a"];
  for (let s = 0; s < 4; s++) {
    const y = 0.35 + s * 0.52;
    g.add(boxMesh(1.32, 0.06, 0.36, side, 0, y, 0));
    for (let b = 0; b < 9; b++) {
      const bh = 0.3 + (b % 3) * 0.05;
      const bk = boxMesh(0.1, bh, 0.26,
        new THREE.MeshStandardMaterial({ color: bookCols[(b + s) % bookCols.length], roughness: 0.9 }),
        -0.56 + b * 0.13, y + 0.03 + bh / 2, 0);
      g.add(bk);
    }
  }
  g.userData.prop = true;
  return g;
}
function makeCounter(len = 2.6) {            // 打饭窗口/柜台
  const g = new THREE.Group();
  g.add(boxMesh(len, 1.0, 0.6, new THREE.MeshStandardMaterial({ color: 0x7a8088, roughness: 0.6, metalness: 0.3 }), 0, 0.5, 0));
  g.add(boxMesh(len + 0.14, 0.08, 0.76, metalMat, 0, 1.04, 0));
  for (let i = 0; i < 3; i++) {              // 餐盘保温槽
    const pan = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.06, 0.4), metalMat);
    pan.position.set(-len / 3 + i * (len / 3), 1.1, 0); g.add(pan);
  }
  g.userData.prop = true;
  return g;
}
function makeSink() {                        // 水槽 + 龙头（水房/洗衣房）
  const g = new THREE.Group();
  g.add(boxMesh(1.0, 0.25, 0.55, ceramicMat, 0, 0.85, 0));
  g.add(boxMesh(0.9, 0.06, 0.45, new THREE.MeshStandardMaterial({ color: 0x2a3038, roughness: 0.4 }), 0, 0.95, 0));
  for (const dx of [-0.32, 0.32]) g.add(boxMesh(0.1, 0.72, 0.1, metalMat, dx, 0.36, 0));
  const tap = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.42, 8), metalMat);
  tap.position.set(0, 1.14, -0.22); g.add(tap);
  const spout = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.22, 8), metalMat);
  spout.rotation.x = Math.PI / 2; spout.position.set(0, 1.32, -0.13); g.add(spout);
  return g;
}
function makeShowerHead() {                  // 花洒（澡堂）
  const g = new THREE.Group();
  const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.9, 8), metalMat);
  pipe.position.y = 1.65; g.add(pipe);
  const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.3, 8), metalMat);
  arm.rotation.z = Math.PI / 2; arm.position.set(0.15, 2.5, 0); g.add(arm);
  const head = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.07, 0.1, 12), metalMat);
  head.position.set(0.3, 2.42, 0); g.add(head);
  return g;
}
function makeMedBed() {                      // 病床（医务室）
  const g = new THREE.Group();
  g.add(boxMesh(2.0, 0.12, 0.9, metalMat, 0, 0.58, 0));
  g.add(boxMesh(1.92, 0.18, 0.84, new THREE.MeshStandardMaterial({ color: 0xdfe6ec, roughness: 1 }), 0, 0.73, 0));
  g.add(boxMesh(0.5, 0.14, 0.34, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1 }), -0.66, 0.85, 0));
  for (const dx of [-0.9, 0.9]) for (const dz of [-0.38, 0.38]) g.add(boxMesh(0.07, 0.52, 0.07, metalMat, dx, 0.26, dz));
  g.add(boxMesh(0.08, 0.5, 0.86, metalMat, -1.02, 0.9, 0));   // 床头栏
  return g;
}
function makeWorkbench() {                   // 工作台（苦役区）
  const g = new THREE.Group();
  g.add(boxMesh(2.2, 0.14, 0.9, new THREE.MeshStandardMaterial({ color: 0x59452b, roughness: 1 }), 0, 0.86, 0));
  for (const dx of [-1.0, 1.0]) g.add(boxMesh(0.12, 0.86, 0.86, metalMat, dx, 0.43, 0));
  const vise = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.24, 0.22), metalMat);
  vise.position.set(-0.8, 1.05, 0.2); g.add(vise);
  return g;
}
function makeWeightRack() {                  // 杠铃架（健身区）
  const g = new THREE.Group();
  for (const dx of [-0.8, 0.8]) g.add(boxMesh(0.12, 1.3, 0.3, metalMat, dx, 0.65, 0));
  const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 2.2, 10), metalMat);
  bar.rotation.z = Math.PI / 2; bar.position.y = 1.32; g.add(bar);
  for (const dx of [-0.95, 0.95]) {
    const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.09, 18),
      new THREE.MeshStandardMaterial({ color: 0x22262e, roughness: 0.7 }));
    plate.rotation.z = Math.PI / 2; plate.position.set(dx, 1.32, 0); g.add(plate);
  }
  return g;
}
function makeWashTub() {                     // 洗衣槽（洗衣房）
  const g = new THREE.Group();
  const tub = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.42, 0.66, 18, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x8d949f, roughness: 0.6, metalness: 0.35, side: THREE.DoubleSide }));
  tub.position.y = 0.55; g.add(tub);
  const water = new THREE.Mesh(new THREE.CircleGeometry(0.46, 20),
    new THREE.MeshStandardMaterial({ color: 0x4a6b78, roughness: 0.2, metalness: 0.1 }));
  water.rotation.x = -Math.PI / 2; water.position.y = 0.74; g.add(water);
  for (const a of [0, 2.1, 4.2]) g.add(boxMesh(0.08, 0.24, 0.08, metalMat, Math.sin(a) * 0.4, 0.12, Math.cos(a) * 0.4));
  return g;
}
function makeBlackboard() {                  // 黑板（教室）
  const g = new THREE.Group();
  g.add(boxMesh(2.6, 1.3, 0.08, new THREE.MeshStandardMaterial({ color: 0x1f2a24, roughness: 0.95 }), 0, 1.5, 0));
  g.add(boxMesh(2.76, 0.1, 0.16, woodMat, 0, 0.82, 0.04));
  return g;
}

// 房间顶灯：一根吊杆 + 灯泡 + 点光源。默认关，玩家可开关；没被碰过的会跟随昼夜自动亮。
// 功能室一律冷白、亮度约四成；牢房更暗（dim=0.2）。
function makeCeilingLamp(x, y, z, cold = false, dim = 1) {
  const g = new THREE.Group();
  const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.4, 6), metalMat);
  rod.position.y = 0.2; g.add(rod);
  const shade = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.2, 14, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x3c424c, roughness: 0.7, side: THREE.DoubleSide }));
  shade.position.y = -0.06; g.add(shade);
  const bulbCol = cold ? 0xe8f2ff : 0xfff0c8;
  const emitCol = cold ? 0xb8d4ff : 0xffc86a;
  const lightCol = cold ? 0xc5dcff : 0xffb774;
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 10),
    new THREE.MeshStandardMaterial({ color: bulbCol, emissive: emitCol, emissiveIntensity: 0 }));
  bulb.position.y = -0.16; g.add(bulb);
  // 同理把衰减做扁：屋顶灯要照亮整间屋，不是在正下方烧个白点
  const light = new THREE.PointLight(lightCol, 0, 14, NIGHT_DIORAMA ? 1.45 : 2);
  light.position.y = -0.2; g.add(light);
  g.position.set(x, y, z);
  const rec = {
    light, bulb, on: false, manual: false,
    base: (NIGHT_DIORAMA ? 11 : 7) * dim,
    glow: dim,
  };
  roomLamps.push(rec);
  return { group: g, rec };
}

// ---------------------------------------------------------------------------
// 连排牢房：一人一间厂间式隔间，像厕所隔间那样沿西墙排成一串。
//
// 上一版是"主角那间居中 + 南北各两间不与中庭连通的空腔"：那几间 NPC 根本进不去，
// 于是五个人睡觉时全挤在牢房设施中心那一格上。现在每间都真能进出，
// 后端给每个囚犯分了固定一间（world.cell_state / NPC.cell），睡觉歇着各回各屋。
//
// 几何全部来自世界帧的 cells（后端 layout.CELLS 唯一权威）：中心、宽深、朝向、
// 门心坐标。前端只管砌墙和开合门，一寸也不自己算。
//
// 牢门有三种状态，对应三层含义：
//   开  —— 有人靠近、且没上锁：栅栏门自动荡开，人进得去
//   关  —— 没人在附近：合上（只是样子，寻路照通，人推门就走）
//   锁  —— 典狱长扭了钥匙：碰撞盒生效、后端 nav 网格里那道门变实墙，真的关死
// ---------------------------------------------------------------------------
const cellDoors = new Map();     // slot -> {gate, baseRy, blocker, locked, open, plate, atDoor}
window.__dbg.cells = cellDoors;  // 截图探针据此把某道门强制开/锁上来校对
const GATE_OPEN_RY = 1.05;       // 荡开的角度（弧度，约 60°）：往院子那边开
// 有人走到这么近，没锁的门就自动荡开。给得比身位宽些：门要在人撞上来之前先开，
// 否则前端寻路会先把这道门当墙、绕一圈之后才发现能进。
const GATE_NEAR = 2.8;

// 牢房专用石墙：ambientCG Rock060。只贴牢房的盒子墙（有 UV），不往 Jampot 模块上套。
let cellRockMat = null;
let laundryRockMat = null;   // 洗衣房：ambientCG Rock035
let clinicBrickMat = null;   // 医务室：ambientCG Bricks064
let solitaryMetalMat = null; // 禁闭室：ambientCG Metal020
let showerTileMat = null;    // 澡堂：ambientCG Tiles133D
let roomConcreteMat = null;  // 其余功能室：ambientCG Concrete044B
let asphaltMat = null;       // 全场地板：ambientCG Asphalt021
function tileBoxUVs(mesh, tileM = 2) {
  const geo = mesh.geometry;
  if (!geo.parameters || geo.userData.tiled) return;
  const { width: w, height: h, depth: d } = geo.parameters;
  const uv = geo.attributes.uv;
  if (!uv) return;
  const scales = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
  for (let f = 0; f < 6; f++) {
    const [su, sv] = scales[f];
    for (let i = 0; i < 4; i++) {
      const idx = (f * 4 + i) * 2;
      uv.array[idx] *= su / tileM;
      uv.array[idx + 1] *= sv / tileM;
    }
  }
  uv.needsUpdate = true;
  geo.userData.tiled = true;
}
function tilePlaneUVs(mesh, tileM = 2) {
  const geo = mesh.geometry;
  if (!geo.parameters || geo.userData.tiled) return;
  const su = (geo.parameters.width || 1) / tileM;
  const sv = (geo.parameters.height || 1) / tileM;
  const uv = geo.attributes.uv;
  if (!uv) return;
  for (let i = 0; i < uv.count; i++) {
    uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  }
  uv.needsUpdate = true;
  geo.userData.tiled = true;
}
const TEX_VER = 133; // 贴图同名覆盖时戳一下，避免浏览器继续用上一张干净白砖
function loadTexFile(url, srgb) {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(`${url}?v=${TEX_VER}`, (t) => {
      t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
      resolve(t);
    }, undefined, reject);
  });
}
async function loadRockPbr(prefix, { metal = false, color = 0xf0ece6 } = {}) {
  const jobs = [
    loadTexFile(`assets/tex/${prefix}_d.jpg`, true),
    loadTexFile(`assets/tex/${prefix}_n.jpg`, false),
    loadTexFile(`assets/tex/${prefix}_r.jpg`, false),
  ];
  if (metal) jobs.push(loadTexFile(`assets/tex/${prefix}_m.jpg`, false));
  const [map, normalMap, roughnessMap, metalnessMap] = await Promise.all(jobs);
  return new THREE.MeshStandardMaterial({
    map, normalMap, roughnessMap, metalnessMap,
    color,                // 默认略提亮；澡堂白瓷砖必须 0xffffff，否则破损细节被洗白
    roughness: 1, metalness: metal ? 1 : 0,
    normalScale: new THREE.Vector2(0.9, 0.9),
  });
}
async function loadCellRockTextures() {
  try { cellRockMat = await loadRockPbr("cell_rock"); }
  catch (e) { console.warn("牢房石墙贴图加载失败，回退程序化墙：", e); }
  // 洗衣房单独贴 ambientCG Rock035，不和牢房的 Rock060 混用。
  try { laundryRockMat = await loadRockPbr("laundry_rock"); }
  catch (e) { console.warn("洗衣房石墙贴图加载失败：", e); }
  try { clinicBrickMat = await loadRockPbr("clinic_brick"); }
  catch (e) { console.warn("医务室砖墙贴图加载失败：", e); }
  try {
    solitaryMetalMat = await loadRockPbr("solitary_metal", { metal: true, color: 0xdde4ee });
    // Metal020 的漫反射是锈斑；金属感全靠金属度+反射。禁闭室没有顶灯，
    // 必须把粗糙度压低、把 HDRI 反射抬高，否则看起来就是一张脏墙贴图。
    solitaryMetalMat.roughness = 0.32;
    solitaryMetalMat.envMapIntensity = 5.5;
    solitaryMetalMat.normalScale.set(1.2, 1.2);
  } catch (e) { console.warn("禁闭室金属墙贴图加载失败：", e); }
  try { showerTileMat = await loadRockPbr("shower_tile133", { color: 0xffffff }); }
  catch (e) { console.warn("澡堂瓷砖贴图加载失败：", e); }
  try { roomConcreteMat = await loadRockPbr("room_concrete"); }
  catch (e) { console.warn("功能室混凝土墙贴图加载失败：", e); }
  try {
    asphaltMat = await loadRockPbr("asphalt");
    if (asphaltMat) {
      ground.material = asphaltMat;
      tilePlaneUVs(ground, 2);
    }
  } catch (e) { console.warn("沥青地面贴图加载失败：", e); }
}

// 晚霞 HDRI（Poly Haven / ambientCG EveningSkyHDRI018B 1K EXR）。
// 只当背景 + 很弱的环境反射：夜景的主光仍然是火盆/顶灯，天空别把灯池洗掉。
async function loadEveningSky() {
  if (!NIGHT_DIORAMA) return;
  try {
    const tex = await new Promise((resolve, reject) => {
      new EXRLoader().load("assets/env/EveningSkyHDRI018B_1K_HDR.exr", resolve, undefined, reject);
    });
    tex.mapping = THREE.EquirectangularReflectionMapping;
    scene.background = tex;
    scene.backgroundIntensity = 0.48;
    scene.environment = tex;
    scene.environmentIntensity = 0.22;
  } catch (e) {
    console.warn("晚霞天空盒加载失败，保持纯黑背景：", e);
  }
}

function buildCellRow(cells) {
  const modular = envHas("wall");
  const t = 0.15, wallH = ROOM_H, wy = 0.1 + wallH / 2;
  const mat = cellRockMat || new THREE.MeshStandardMaterial({ color: 0x8f95a0, roughness: 0.95, map: wallTexture() });
  if (modular) envTint(ROOM_TINT.sleep);
  for (const c of cells) {
    const ry = FACE_RY[c.face] ?? 0;
    const g = new THREE.Group();
    g.position.set(c.x, 0, c.z);
    g.rotation.y = ry;
    // 局部系约定与 makeBuilding 一致：w 沿 x（前墙方向）、d 沿 z（进深），门在 +z
    const hw = c.w / 2, hd = c.d / 2;
    if (modular) envOrigin(c.x, c.z, ry);
    else g.add(boxMesh(c.w, 0.08, c.d,
      new THREE.MeshStandardMaterial({ color: 0x3d424c, roughness: 1 }), 0, 0.04, 0));
    // 牢房三面墙都用 Rock060 盒子（带 UV）。不再叠 Jampot 石墙：那套模块没有 UV，
    // 贴图只会变成几根线，还会和这面石头墙穿插。
    const put = (mw, md, x, z) => {
      const m = boxMesh(mw, wallH, md, mat, x, wy, z);
      m.userData.blocker = true;
      tileBoxUVs(m, 2.0);
      g.add(m);
    };
    put(c.w, t, 0, -hd + t / 2);           // 后墙
    put(t, c.d, -hw + t / 2, 0);           // 左隔墙
    put(t, c.d, hw - t / 2, 0);            // 右隔墙
    if (asphaltMat) {
      const fl = boxMesh(c.w, 0.08, c.d, asphaltMat, 0, 0.04, 0);
      fl.castShadow = false;
      tileBoxUVs(fl, 2.0);
      g.add(fl);
    } else if (modular) {
      envFloorPatch(0, 0, c.w, c.d);
    }
    if (modular) {
      envAdd("window", 0, -hd, 0, 1, 1, 1, 1.3);   // 后墙一扇高窗
      envCresset(-hw + 0.45, -hd + 0.5, Math.PI / 4, false);
    }
    // 家具：一张铺、一个储物柜。铺位落点与后端 bed 物资同一格（隔间中心偏后墙）。
    // 马桶不再摆在门口：透过栅栏看就是一排白方块，像厕所隔间外挂了便器。
    const bunk = makeBunk();
    bunk.position.set(0, 0, -hd + 1.15); bunk.rotation.y = Math.PI / 2; g.add(bunk);
    const locker = makeLocker();
    locker.position.set(-hw + 0.45, 0, hd - 0.8); g.add(locker);
    addInteractable(locker, {
      type: "search", name: `${c.name}的储物柜`, icon: "🗄️",
      found: c.npc ? `翻遍了${c.name}的柜子：换洗囚服、半包烟、一张折了三折的照片。`
                   : "空柜子，门轴锈住了，里面只有一层灰。",
    });
    // 每间都登记躺/坐工位（含有主的牢房）：NPC 睡觉躺铺、歇着坐在铺沿，绝不能躺到院子里。
    const lieA = seatAnchor(g, 0, 0.72, -hd + 1.5, 0);
    const sitA = seatAnchor(g, 0, 0.53, 0.15, Math.PI);   // 铺沿朝门口坐
    addInteractable(bunk, { type: "lie", name: `${c.name}的下铺`, icon: "🛏️",
                            kind: "sleep", slot: c.slot, anchor: lieA });
    registerStation(sitA, "sit", "sleep", c.slot);
    const { group: lampG } = makeCeilingLamp(0, 2.4, 0, true, 0.2);
    g.add(lampG);
    // 牢门：铰在一侧的一扇栅栏门。枢轴放在局部 (-hw+t, +hd)，门扇沿 +x 伸出
    const gate = new THREE.Group();
    gate.position.set(-hw + t, 0, hd - t / 2);
    const leaf = c.w - 2 * t;
    const frameMat = barMat;
    for (let i = 0; i <= 5; i++) {
      const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, wallH - 0.2, 6), frameMat);
      bar.position.set((i / 5) * leaf, 0.1 + (wallH - 0.2) / 2, 0);
      gate.add(bar);
    }
    for (const hy of [0.35, wallH - 0.2]) {      // 上下横撑
      const rail = new THREE.Mesh(new THREE.BoxGeometry(leaf, 0.08, 0.08), frameMat);
      rail.position.set(leaf / 2, hy, 0); gate.add(rail);
    }
    g.add(gate);
    scene.add(g);
    collectBlockers(g);
    if (modular) envOriginClear();
    // 关着的门那条碰撞盒：单独登记、可开关。开门时 off=true，寻路和碰撞都当它不存在。
    const [fx, fz] = FACE_VEC[c.face] ?? [0, 1];
    const bw = Math.abs(fx) ? 0.16 : leaf, bd = Math.abs(fx) ? leaf : 0.16;
    const blocker = {
      minX: c.door_x - bw / 2, maxX: c.door_x + bw / 2,
      minZ: c.door_z - bd / 2, maxZ: c.door_z + bd / 2, off: false,
    };
    blockers.push(blocker);
    // 门牌：写住户的名字，一眼看出这是谁那间
    const plate = { el: makeBubble("facility-tag"), obj: new THREE.Object3D(), y: 2.9 };
    plate.obj.position.set(c.door_x, 0, c.door_z); scene.add(plate.obj);
    plate.el.textContent = c.npc ? `${c.name}的牢房` : "空牢房";
    facilityLabels.push(plate);
    const [ax, az] = [-fz, fx];                // 沿墙方向
    const rec = {
      slot: c.slot, name: c.name, npc: c.npc, gate, face: c.face, room: g,
      blocker, locked: false, open: 0,         // 常闭：open=0
      doorX: c.door_x, doorZ: c.door_z, plate,
      // 谁站在这道门前面：沿墙方向要在门宽之内，垂直方向在触发距离之内
      atDoor(p) {
        if (!p) return false;
        const dx = p.x - this.doorX, dz = p.z - this.doorZ;
        return Math.abs(dx * ax + dz * az) < c.w / 2 + 0.25
            && Math.abs(dx * fx + dz * fz) < GATE_NEAR;
      },
    };
    cellDoors.set(c.slot, rec);
    // 典狱长专属：走到牢门前按 E 扭钥匙。锁只有他能动（NPC 没有这个入口）
    for (const bar of gate.children) {
      addInteractable(bar, { type: "celllock", name: `${c.name}的牢门`, icon: "🔑", cell: rec });
      break;                                   // 只挂一根竖栏做拾取代理，够了
    }
  }
  if (modular) envTintClear();
  NAV.dirty = true;
}

// 世界帧里的牢房状态 → 建六间隔间（只建一次）+ 同步每道门的关/锁。
function syncCells(msg) {
  const cells = msg.cells;
  if (!Array.isArray(cells) || !cells.length) return;
  if (!cellDoors.size) buildCellRow(cells);
  for (const c of cells) {
    const rec = cellDoors.get(c.slot);
    if (!rec) continue;
    if (rec.locked !== !!c.locked) {
      rec.locked = !!c.locked;
      rec.plate.el.textContent = `${rec.locked ? "🔒 " : ""}${c.npc ? `${c.name}的牢房` : "空牢房"}`;
    }
  }
}

// 每帧开合牢门：门常闭，有人靠近就荡开，人走了自己合上；锁上的一律关死并计入碰撞。
// 这样"开关门"是自然发生的（谁进谁开），而"锁"是典狱长按下去的那一道权力。
// 常闭是关键：默认敞着的牢房就只是六个凹龛，看不出这是关人的地方。
function updateCellDoors(dt) {
  for (const rec of cellDoors.values()) {
    let near = false;
    if (!rec.locked) {
      // 不能只看直线距离：隔间只有 2.35m 宽，一个圆形范围会把左右邻居的门一起带开。
      // 分两轴判定——沿墙方向必须落在本间门宽之内（是"正对着这道门"），
      // 垂直方向才用那个较宽的触发距离（好让门在人撞上来之前先开）。
      near = rec.atDoor(player ? player.root.position : null);
      if (!near) {
        for (const e of npcs.values()) {
          if (!e.dead && rec.atDoor(e.root.position)) { near = true; break; }
        }
      }
    }
    const want = rec.locked || !near ? 0 : 1;
    // 上锁即刻关死：锁上的门连典狱长走到跟前也不再自动荡开（want 恒为 0），
    // 必须先按 E 开锁。这里 locked 时直接吸附到关闭，不留半开的缝。
    if (rec.locked) rec.open = 0;
    else rec.open += Math.sign(want - rec.open) * Math.min(Math.abs(want - rec.open), dt * 2.2);
    // 门扇挂在隔间组里，朝向已经被整组的 ry 转过去了；这里只写"相对关着位置荡开多少"。
    // 写成 baseRy - open*K 会把整组那份转角又算一遍，门就横在隔间侧面了。
    // 负角 = 往 +z（中庭那边）荡：牢门朝外开，关的时候才压在门框上。
    rec.gate.rotation.y = -rec.open * GATE_OPEN_RY;
    // 只有"完全关上"才挡人：半开的门不该把人卡在门缝里
    const off = rec.open >= 0.08;
    if (rec.blocker.off !== off) {
      rec.blocker.off = off;
      NAV.dirty = true;                       // 通行性变了，前端寻路网格要重烘
    }
  }
}

// ---------------------------------------------------------------------------
// 功能室的前门：和牢房同一套栅栏门（竖铁栏 + 上下横撑），不是实钢门。
// 开合逻辑也一样——有人靠近就荡开；只有禁闭室额外能上锁。
// 门扇建在 makeBuilding 里（局部系、挂在房间组上随朝向一起转），门的碰撞盒和开合
// 判定在 buildOneFacility 里按世界坐标登记（registerRoomDoor）。
// ---------------------------------------------------------------------------
const roomDoors = [];            // [{leaf, open, blocker, atDoor, fid, kind, locked}]
window.__dbg.roomDoors = roomDoors;   // 探针据此逐门检查开合/碍碍盒
let solitaryDoor = null;         // 禁闭室那道门的 rec（唯一，可被典狱长上锁）
const ROOM_DOOR_RY = 1.15;       // 荡开角度（约 66°），往 +z（院子）方向开
const ROOM_DOOR_NEAR = 2.8;      // 有人到这么近就自动开（与牢门同距，留出提前量）

// 一扇栅栏门：铰在 x=0、沿 +x 伸出。几何和牢房那扇对齐，只是门洞窄一点。
function makeRoomDoorLeaf(leaf, h) {
  const g = new THREE.Group();
  const H = h - 0.2, y = 0.1 + H / 2;
  const n = Math.max(4, Math.round(leaf / 0.28));
  for (let i = 0; i <= n; i++) {
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, H, 6), barMat);
    bar.position.set((i / n) * leaf, y, 0);
    g.add(bar);
  }
  for (const hy of [0.35, h - 0.2]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(leaf, 0.08, 0.08), barMat);
    rail.position.set(leaf / 2, hy, 0);
    g.add(rail);
  }
  return g;
}

// 房间造好后登记它的门：算出门在世界系的中心与朝向，建一条可开关的碰撞盒 + 就近判定。
function registerRoomDoor(rec, f) {
  rec.mark.updateWorldMatrix(true, false);
  const c = rec.mark.getWorldPosition(new THREE.Vector3());
  const cx = c.x, cz = c.z;
  const [fx, fz] = FACE_VEC[f.face] ?? [0, 1];
  const along = rec.doorW;
  const bw = Math.abs(fx) ? 0.2 : along, bd = Math.abs(fx) ? along : 0.2;
  blockers.push(rec.blocker = {
    minX: cx - bw / 2, maxX: cx + bw / 2, minZ: cz - bd / 2, maxZ: cz + bd / 2, off: false,
  });
  const [ax, az] = [-fz, fx];                // 沿墙方向
  const door = {
    leaf: rec.leaf, open: 0, blocker: rec.blocker, fid: f.id, kind: f.kind, locked: false,
    doorX: cx, doorZ: cz,
    atDoor(p) {
      if (!p) return false;
      const dx = p.x - cx, dz = p.z - cz;
      return Math.abs(dx * ax + dz * az) < along / 2 + 0.35
          && Math.abs(dx * fx + dz * fz) < ROOM_DOOR_NEAR;
    },
  };
  roomDoors.push(door);
  // 禁闭室的门归典狱长管：可上锁（同牢门一套），走到门前按 E 也能开/锁。
  if (f.kind === "solitary") {
    solitaryDoor = door;
    for (const bar of rec.leaf.children) {
      addInteractable(bar, { type: "solitarylock", name: "禁闭室的门", icon: "🔑", door });
      break;
    }
  }
  NAV.dirty = true;
}

// 每帧开合功能室门：有人正对门口就荡开，人走了合上；关上时那条碰撞盒生效并计入寻路网格。
function updateRoomDoors(dt) {
  for (const rec of roomDoors) {
    let near = false;
    if (!rec.locked) {                        // 锁上的门（禁闭室）连典狱长也不自动开，须先开锁
      near = rec.atDoor(player ? player.root.position : null);
      if (!near) {
        for (const e of npcs.values()) {
          if (!e.dead && rec.atDoor(e.root.position)) { near = true; break; }
        }
      }
    }
    const want = near ? 1 : 0;
    if (rec.locked) rec.open = 0;
    else rec.open += Math.sign(want - rec.open) * Math.min(Math.abs(want - rec.open), dt * 2.4);
    rec.leaf.rotation.y = -rec.open * ROOM_DOOR_RY;
    const off = rec.open >= 0.08;
    if (rec.blocker.off !== off) { rec.blocker.off = off; NAV.dirty = true; }
  }
}

// 开局就摆在场上的固定景物。必须等环境模块加载完才能建（要用石墙/火盆模块），
// 所以从启动流程里调用，而不是模块顶层直接执行。
// 岩石台基：沿场地外沿铺一圈崖壁模块，让监狱像一件坐在岩台上的模型悬在黑里。
// 少了这圈，围墙外只能靠雾淡入黑色，画面是"摊在地上"而不是"浮起来"的。
// 模块顶面在自身 bbox 顶部，所以整块下沉一个身位，顶面正好落到 y=0 的地面高度。
// deep 保持 1：竖向拉伸会把模块底部那条细窄的土层一起拉宽，台基就从"岩石边缘"
// 变成一条环绕全场的土黄腰带。厚度不够靠纯黑背景衬托，不靠拉伸。
const PLINTH = { ry0: 0, inset: 1.0, deep: 1 };
function buildTerrainPlinth() {
  if (!envHas("terrCliff")) return;
  const drop = envSize("terrCliff").y * PLINTH.deep;
  const step = 2;                                  // 与石墙同一套 2m 网格
  const R = WALL + PLINTH.inset;
  const y = -drop;
  // 四条边：外法线分别朝 -Z / +X / +Z / -X，靠 ry0 统一校正模块自身的朝向约定
  const sides = [
    { ry: 0,             ax: "x", z: -R },
    { ry: -Math.PI / 2,  ax: "z", z: R },
    { ry: Math.PI,       ax: "x", z: R },
    { ry: Math.PI / 2,   ax: "z", z: -R },
  ];
  for (const s of sides) {
    for (let t = -WALL + step / 2; t < WALL; t += step) {
      const d = PLINTH.deep;
      if (s.ax === "x") envAdd("terrCliff", t, s.z, s.ry + PLINTH.ry0, 1, d, 1, y);
      else envAdd("terrCliff", s.z, t, s.ry + PLINTH.ry0, 1, d, 1, y);
    }
  }
  for (const [sx, sz, ry] of [[-1, -1, 0], [1, -1, -Math.PI / 2],
                              [1, 1, Math.PI], [-1, 1, Math.PI / 2]]) {
    envAdd("terrCorner", sx * R, sz * R, ry + PLINTH.ry0, 1, PLINTH.deep, 1, y);
  }
  // 不铺 terrFlat：崖壁内边缘（±29.92）和 60×60 的地面（±30）本来就搭接，没有缝，
  // 多铺一圈只会把泥土色的顶面露在崖壁外面，看着像一道橙色木甲板。
}

function buildStaticDecor() {
  buildRingRoad();
  // 食堂门口的餐桌（食堂在南墙 x[-8.1,2.9]、门开在 z=-11.5，摆在门外、避开环路）
  // addProp：登记碰撞盒——玩家撞得动、NPC 寻路也绕开（院子里的大件设 nav:true）。
  const t1 = makeTable(); t1.position.set(-6.6, 0, -10.6); addProp(t1, { nav: true });
  addProp(makeStool(-7.7, -10.6), { nav: true }); addProp(makeStool(-5.5, -10.6), { nav: true });
  // 苦役区门口的板条箱堆（苦役区在北墙 x[-12.6,-6.6]，门开在 z=13.0）
  const crateA = makeCrate(); crateA.position.set(-12.3, 0, 12.1); addProp(crateA, { nav: true });
  const crateB = makeCrate(0.7); crateB.position.set(-11.6, 0, 11.6); addProp(crateB, { nav: true });
  const crateC = makeCrate(0.7); crateC.position.set(-12.9, 0, 11.5); addProp(crateC, { nav: true });
  if (envHas("lamp")) {
    // 中庭一圈火盆：参考图里"冷暗大环境 + 一个个暖光池"的骨架就是这些点。
    // 对角两盏挂真光源就够了——四盏的光池会在中庭正中叠成一团白，反而丢了暗部。
    // 落点要躲开放风场那圈铁栅栏（中心 (0,1)、半径 6.2），所以取 ±9 那一带
    for (const [lx, lz, real] of [[-9.0, -8.0, true], [9.0, -8.0, false],
                                  [-9.0, 9.5, false], [9.0, 9.5, true]]) {
      envCresset(lx, lz, Math.atan2(lx, lz), real, 4.5, 18);
    }
    envCresset(GATE_X - 2.4, WALL - 1.4, 0, true, 4.5, 16);     // 大门内侧
    envCresset(GATE_X + 2.4, WALL - 1.4, 0, true, 4.5, 16);
    // 沿环路四角零星几盏（无光源，纯装饰），把这一圈路"点"出来
    for (const [lx, lz] of [[RING.x0, RING.z0], [RING.x1, RING.z0],
                            [RING.x0, RING.z1], [RING.x1, RING.z1],
                            [RING.x0, 0], [RING.x1, 0], [0, RING.z0], [0, RING.z1]]) {
      envCresset(lx, lz, 0, false);
    }
    // 墙根堆的桶和箱子：挑四个角落——布局故意把四角空着，正好堆杂物
    envAdd("barrels", -WALL + 2.4, WALL - 3.4, 0.3);       // 西北角
    envAdd("crates", WALL - 2.6, WALL - 4.2, -0.4);        // 东北角
    envAdd("barrel", -WALL + 2.2, -WALL + 3.0, 0.9);       // 西南角
    envAdd("crateOpen", WALL - 2.4, -WALL + 3.2, 0.2);     // 东南角
    // 苦役区门口的重木工作台（资产包里的 TortureRack，当劳作台用），配一摞箱子和一只敞口桶
    envAdd("rack", -8.2, 12.2, Math.PI);
    envAdd("crates", -7.0, 11.6, 0.5);
    envAdd("barrelOpen", -6.2, 12.0, 0);
    // 上面这些是烧进静态批次的模块，没有独立对象——手工补一份碰撞盒（只挡玩家）
    propBox(-WALL + 2.4, WALL - 3.4, 1.4, 1.4); propBox(WALL - 2.6, WALL - 4.2, 1.4, 1.4);
    propBox(-WALL + 2.2, -WALL + 3.0, 1.0, 1.0); propBox(WALL - 2.4, -WALL + 3.2, 1.1, 1.1);
    propBox(-8.2, 12.2, 1.8, 1.0); propBox(-7.0, 11.6, 1.1, 1.1); propBox(-6.2, 12.0, 0.9, 0.9);
  }
  buildTerrainPlinth();
  envFlush();
}

// ---------------------------------------------------------------------------
// 侦探系统·案情与线索（主角被陷害顶罪，散落线索拼出真相）
// ---------------------------------------------------------------------------
const CASE_INTRO = "你被栽赃顶罪入狱。表面看是典狱长贪腐——可真相远比这黑暗。走遍逐步解锁的监狱，翻查线索、盘问狱友，" +
  "看清那双『没有脚步声』的手到底属于谁。越查得深，知情的人越肯松口。（发光物件＝可查，走近按 F 检视）";
const CASE_CONCLUSION =
  "结案推理【表层·替罪】：卷宗把你钉在典狱长的贪腐案上——他确实收了黑钱、随手签了字，却是个完美的『背锅侠』，" +
  "他根本不认识你。真正下药、伪造现场的人另有其人。\n" +
  "【中层·帮凶】黑医柳叶刀的医务室流出了迷药，他篡改了用药记录(用药记录)勒索真凶；帮派仇家漏出有人欠着要命的赌债(欠条)；" +
  "禁闭室的疯子反复念叨那晚『影子的脚踩在地上没声音』(软底鞋)——动手的人穿软底布鞋、走路无声。\n" +
  "【真相·反转】那双软底布鞋，正是全程对你嘘寒问暖、手把手带你的狱友『老牙』。他是外面犯罪集团的销赃白手套，" +
  "亲手把你推出来顶罪、又申请入狱把你放在眼皮底下，借你的手扳倒典狱长、拿到那本洗钱黑账(黑账)，最后再灭口。" +
  "当面拿证据揭穿他——才算真正翻案。";
// owner = 知情狱友 id（与后端 personas.knows_clue 对齐）：找他聊能套出内幕。owner:null=纯靠脚力翻出的实物。
// zone 决定何时显形：公共区(阶段二)3 条→查满 2 条解锁受限区；受限区(阶段三)3 条；办公区(终局)1 条。
// pension 线索找到后后端会一并发给你「抚恤金截留名单」道具（收买铁面用）。
const CLUES = [
  // —— 公共区（阶段二·沙盒探查）——
  { id: "photo", name: "泛黄旧照", icon: "🖼️", pos: [-13, 15], y: 1.2, color: 0xa8c0d8, owner: null,
    desc: "澡堂储物柜深处一张泛黄合影：案发那座仓库门口，人群里有个侧影格外眼熟，却一时想不起在哪见过。",
    insight: "案发现场混着一张你后来会认得的脸。" },
  { id: "iou", name: "半张欠条", icon: "📄", pos: [11, 17], y: 1.05, color: 0xe8e2c0, owner: "shitou",
    desc: "娱乐室沙发缝里塞着半张撕开的欠条，落款只剩个残缺的名字，金额后头一长串零——欠债的人被逼到了绝路。",
    insight: "有人为了填赌债的窟窿，铤而走险找人顶罪。" },
  { id: "prescription", name: "撕碎配药单", icon: "🧾", pos: [13, 6], y: 0.4, color: 0xd8c070, owner: null,
    desc: "放风场下水道口卡着半张配药单：写着一种强效迷药的成分，末尾一个残缺的签名，只认得出开头一个『柳』字。",
    insight: "陷害那晚用的迷药，出自监狱里某个能签字的人。" },
  // —— 受限区（阶段三·夜探）——
  { id: "medlog", name: "用药记录", icon: "💊", pos: [1, -12.5], y: 0.95, color: 0xd8dde4, owner: "yanjing",
    desc: "医务室药柜里锁着一页被改过的药品出入库记录：案发当晚那批镇静剂的去向被人涂抹、重填过。",
    insight: "黑医篡改了记录——他知道真相，却攥着它勒索真凶。" },
  { id: "pension", name: "抚恤金名单", icon: "📋", pos: [19, -6], y: 1.0, color: 0xc8b890, owner: null,
    desc: "管教办公室值班桌旁一份名单：牺牲警员的抚恤金被层层克扣，末尾签着典狱长的名字。铁面看到这个会炸。",
    insight: "典狱长克扣了弟兄们的抚恤金——这是撬动狱警铁面的筹码。" },
  { id: "softshoe", name: "目击·软底鞋", icon: "👣", pos: [-6, -8], y: 0.9, color: 0xcfd0d6, owner: "xiaodou",
    desc: "禁闭室的疯子反复念叨：『那影子的脚……踩在地上没声音，滴答、滴答……』——动手的人穿软底布鞋、走路无声。",
    insight: "凶手穿软底布鞋、脚步无声——牢里谁总穿那种鞋？" },
  // —— 办公区（终局）——
  { id: "laundering", name: "洗钱黑账", icon: "📒", pos: [0, 25], y: 1.0, color: 0x9a6b4a, owner: null,
    desc: "典狱长办公室保险柜里的一本黑账：外面某个犯罪集团的洗钱进出，经手人的化名，指向一个『销赃的白手套』。",
    insight: "真正的主谋是外面集团的销赃人——那个一直在你身边的『好人』。" },
];
// npc id → 中文名（案件板显示知情人用；世界帧到达后会补全真实姓名）
const NPC_NAME = { laok: "瘸子", shitou: "疤脸", yanjing: "柳叶刀", xiaodou: "疯子", laochen: "老牙",
                   guard1: "铁面", dog1: "哮天" };
// 分区中文名（解锁提示 / 目标 HUD 用）
const ZONE_CN = { cellblock: "牢区", public: "公共区", restricted: "受限区(夜探)", office: "典狱长办公区" };

function emojiSprite(ch, size) {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d"); g.font = "92px serif"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(ch, 64, 74);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true }));
  s.scale.set(size, size, 1); return s;
}

const clueEntries = [];
function buildClue(cl) {
  const g = new THREE.Group();
  const gem = new THREE.Mesh(new THREE.OctahedronGeometry(0.24),
    new THREE.MeshStandardMaterial({ color: cl.color, emissive: cl.color, emissiveIntensity: 1.1, roughness: 0.35, metalness: 0.4 }));
  gem.castShadow = true; g.add(gem);
  const spr = emojiSprite(cl.icon, 1.0); spr.position.y = 0.72; g.add(spr);
  g.position.set(cl.pos[0], cl.y, cl.pos[1]);
  g.userData.clueId = cl.id; g.userData.hoverRoot = true;
  scene.add(g); clueMeshes.push(g);
  clueEntries.push({ id: cl.id, group: g, gem, spr, baseY: cl.y, found: false, data: cl });
}
// 观察者模式：案件板/七条线索已下线（与当前"上帝观察 NPC 群像"架构无关），不再生成线索发光物。

// ---------------------------------------------------------------------------
// HTML 浮层标签
// ---------------------------------------------------------------------------
const facilityLabels = [];
function makeBubble(cls) {
  const el = document.createElement("div");
  el.className = cls;
  document.body.appendChild(el);
  return el;
}

function barredWindow() {
  // 深色玻璃 + 3 根竖铁栏
  const g = new THREE.Group();
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.95), glassMat);
  glass.position.z = 0.01; g.add(glass);
  const frame = new THREE.Mesh(new THREE.PlaneGeometry(1.05, 1.1),
    new THREE.MeshStandardMaterial({ color: 0x0c0e12 }));
  g.add(frame);
  for (let i = -1; i <= 1; i++) {
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.0, 6), barMat);
    bar.position.set(i * 0.28, 0, 0.04); g.add(bar);
  }
  return g;
}

// ---------------------------------------------------------------------------
// 环境资产：Jampot Prison Pack（2 米网格 · 真实比例 · 原点在脚底 · 纯色 PBR 无贴图）
// 参考图那种"厚石墙切面 + 铁栅门 + 笼式壁灯"的质感全靠这套模块，比程序化方盒子高一档。
//
// 静态场景一律走 envAdd() 把模块烧进批次 → envFlush() 按材质合并成少数几个 mesh。
// 不这么做的话：十几间房 × 每间几十块墙砖地砖，每块又是好几个材质，
// draw call 直接上千，网页端（尤其移动端）吃不住。合并后只剩"材质数"这么多。
// ---------------------------------------------------------------------------
const ENV_FILE = {
  wall: "SM_Wall_PrisonStone_05_Straight.glb",      // 2.00w 2.60h 0.29t —— 剖面矮墙主力
  wallCorner: "SM_Wall_PrisonStone_05_Corner.glb",  // 1.14 x 2.64 —— 转角收口
  floor: "SM_Wall_PrisonStone_05_Floor.glb",        // 2.00 x 2.00 x 0.09 石板地
  door: "SM_Door_Portcullis_05.glb",                // 2.38 x 3.15 铁闸门
  window: "SM_Window_PrisonBarred_05.glb",          // 0.96 x 0.75 铁窗
  fence: "SM_Fence_PrisonBars_05_Straight.glb",     // 2.05 x 1.62 铁栅栏
  fenceCorner: "SM_Fence_PrisonBars_05_Corner.glb",
  fenceGate: "SM_Fence_PrisonBars_05_Gate.glb",
  lamp: "SM_Lamp_PrisonCresset_05.glb",             // 0.42 x 2.18 笼式壁灯 —— 暖光池的来源
  bench: "SM_Bench_PrisonSlab_05.glb",              // 1.20 x 0.50 x 0.48 石板长凳
  rack: "SM_Furn_TortureRack_05.glb",               // 2.21 x 1.03 刑架/工作台
  barrel: "SM_Barrel_Prison_05_Closed.glb",
  barrelOpen: "SM_Barrel_Prison_05_Open.glb",
  barrels: "SM_Barrel_Prison_05_Stack.glb",
  crate: "SM_Crate_Prison_05_Closed.glb",
  crateOpen: "SM_Crate_Prison_05_Open.glb",
  crates: "SM_Crate_Prison_05_Stack.glb",
  // 地形：同一套 2m 网格。Cliff/Corner 带 1.38m 下垂崖面，是"整座监狱坐在一块
  // 岩石台基上悬浮"这个观感的唯一来源——没有它，场地只能淡入黑里，画面是摊平的。
  terrFlat: "SM_Terrain_Prison_05_Flat.glb",        // 2.00 x 0.42 x 2.00 平台板
  terrCliff: "SM_Terrain_Prison_05_Cliff.glb",      // 2.08 x 1.38 x 2.16 崖壁边
  terrCorner: "SM_Terrain_Prison_05_Corner.glb",    // 2.16 x 1.39 x 2.20 崖壁转角
};
// key -> { parts:[{geo,mat,sig}], size:{x,y,z} }。geo 已把模型内部的节点变换烧进顶点，
// 摆放时只需再乘一个"落点矩阵"，省掉运行时的层级计算。
const envTpl = new Map();
const ENV_UP = new THREE.Vector3(0, 1, 0);

async function loadEnvModels() {
  for (const [key, file] of Object.entries(ENV_FILE)) {
    try {
      const gltf = await loadGLB("assets/env/" + file);
      gltf.scene.updateMatrixWorld(true);
      const parts = [];
      const box = new THREE.Box3();
      // 一个材质会被好几个 mesh 共用，调色只能做一次：
      // 忘了去重就会把同一份材质乘上 N 次（1.45^12 ≈ 100 倍，铁栅栏直接白成篱笆）。
      const tuned = new Set();
      gltf.scene.traverse((o) => {
        if (!o.isMesh) return;
        const geo = o.geometry.clone().applyMatrix4(o.matrixWorld);
        // 只留合并需要的属性：多余属性会让 mergeGeometries 因签名不一致而失败
        for (const name of Object.keys(geo.attributes)) {
          if (name !== "position" && name !== "normal" && name !== "uv") geo.deleteAttribute(name);
        }
        geo.computeBoundingBox();
        box.union(geo.boundingBox);
        const mat = Array.isArray(o.material) ? o.material[0] : o.material;
        if (!tuned.has(mat)) {
          tuned.add(mat);
          mat.roughness = Math.min(1, (mat.roughness ?? 0.8) + 0.12); // 夜景里少点塑料感
          const glows = mat.emissive && (mat.emissive.r + mat.emissive.g + mat.emissive.b) > 0.01;
          if (glows) {
            // 火盆的火焰/炭床自带满强度自发光。配上 bloom，近看会炸成一个十米大的光斑，
            // 压到三成才是"一簇火"而不是"一颗太阳"。
            mat.emissiveIntensity = 0.3;
          } else if (NIGHT_DIORAMA && /void|earth/i.test(mat.name || "")) {
            // Void 是窗后的黑洞；Earth 是崖壁下的土层，它该沉进黑里当台基的底，
            // 一提亮就变成一条环绕全场的土黄腰带，比监狱本身还显眼。两者都保持原色。
          } else if (NIGHT_DIORAMA) {
            // 这套资产是给"有环境光的地牢"调的，基色偏暗，而我们的夜景环境光近乎为零，
            // 不提一提整座监狱会黑成一团。Void 是窗后的黑洞，保持黑。
            //
            // 用伽马而不是乘法：同一个倍数对深铁灰(0x27)不够、对浅石色(0x8d)直接溢出成白块
            // （崖壁转角就是这么白掉的）。c^0.55 对暗部抬升约 2.3 倍、对亮部只有 1.3 倍，
            // 且值域天然收在 1 以内，不会裁剪。
            const c = mat.color;
            c.setRGB(Math.pow(c.r, 0.55), Math.pow(c.g, 0.55), Math.pow(c.b, 0.55));
          }
        }
        parts.push({ geo, mat, sig: mat.uuid + "|" + Object.keys(geo.attributes).sort().join(",") });
      });
      envTpl.set(key, { parts, size: box.getSize(new THREE.Vector3()) });
    } catch (e) {
      console.warn(`环境模块加载失败(${key})，该处回退程序化几何：`, file, e);
    }
  }
  console.info("环境模块就绪：", [...envTpl.keys()].join(", "));
}

function envSize(key) { return envTpl.get(key)?.size || null; }
function envHas(key) { return envTpl.has(key); }

// 用 Jampot 模块生成一件"独立"道具（不进静态批次）。
// 批次里的东西合并成了单一 mesh，没法再单独挂交互或移动；需要被搜查/拾取/落座的家具走这条路，
// 代价是每件多一个 draw call——所以只给真正会被点的家具用，墙地砖那种照旧合批。
// fit：把模块按某条轴缩放到指定尺寸，好顶替原来按参数生成的程序化家具。
function envObject(key, fit = 0, axis = "y") {
  const tpl = envTpl.get(key);
  if (!tpl) return null;
  const g = new THREE.Group();
  for (const p of tpl.parts) {
    const m = new THREE.Mesh(p.geo, p.mat);
    m.castShadow = true; m.receiveShadow = true;
    g.add(m);
  }
  const base = tpl.size[axis];
  if (fit > 0 && base > 0.001) g.scale.setScalar(fit / base);
  return g;
}

const envBatch = new Map();          // sig -> {mat, geos:[]}
const _envM = new THREE.Matrix4();
const _envQ = new THREE.Quaternion();
const _envP = new THREE.Vector3();
const _envS = new THREE.Vector3();

// 房间色调。模块化换皮之后每间房的墙地都由同一套石墙模块出，KIND_COLOR / FLOOR_COLOR
// 那套"每类房一个颜色"就再也渲不出来了（墙体方盒子被设成 visible=false 只当碰撞代理），
// 整座监狱只剩一种石色——这才是"房间之间没有辨识度"的真正原因。
//
// 做法是给批次再分一层：同一材质按色调拆成不同批，克隆材质时把色调乘进基色。
// 每种色调多一个 draw call，十几间房也就多几十个，相比合批省下的几千个可以忽略。
// 色调必须克制（各通道 0.85~1.15），否则就不是"和谐统一"而是彩色积木了。
let _envTint = null;                 // {hex, key}
const _envTintC = new THREE.Color();
const _envTintMat = new Map();       // `${sig}|${key}` -> 克隆并染色后的材质
function envTint(hex) { _envTint = hex ? { hex, key: hex.toString(16) } : null; }
function envTintClear() { _envTint = null; }
// 自发光材质（火盆的火焰/炭床）不参与染色：它的观感由 emissive 决定，染基色只会脏掉火色。
function envTintedMat(p, key) {
  let mat = _envTintMat.get(key);
  if (mat) return mat;
  mat = p.mat.clone();
  mat.color.multiply(_envTintC.setHex(_envTint.hex));
  _envTintMat.set(key, mat);
  return mat;
}

// 批次是世界系的，但房间/牢房的代码都按局部坐标写（房间中心为原点）。
// 摆之前先 envOrigin(落点)，摆完 envOriginClear()，中间照旧用局部坐标。
let _envBase = null;
function envOrigin(x, z, ry = 0) {
  _envBase = { x, z, ry, cos: Math.cos(ry), sin: Math.sin(ry) };
}
function envOriginClear() { _envBase = null; }
function envToWorld(x, z) {
  if (!_envBase) return [x, z];
  const { x: bx, z: bz, cos, sin } = _envBase;
  return [bx + x * cos + z * sin, bz - x * sin + z * cos];
}

// 把一件环境模块烧进静态批次。y=0 即贴地（模块原点在脚底），s* 是拉伸倍率
//（石墙拉伸 ±10% 看不出来，正好用来把 2 米模块凑齐任意墙长）。
function envAdd(key, x, z, ry = 0, sx = 1, sy = 1, sz = 1, y = 0) {
  const tpl = envTpl.get(key);
  if (!tpl) return false;
  if (_envBase) { [x, z] = envToWorld(x, z); ry += _envBase.ry; }
  _envM.compose(_envP.set(x, y, z), _envQ.setFromAxisAngle(ENV_UP, ry), _envS.set(sx, sy, sz));
  for (const p of tpl.parts) {
    const glows = p.mat.emissive && (p.mat.emissive.r + p.mat.emissive.g + p.mat.emissive.b) > 0.01;
    const key = (_envTint && !glows) ? `${p.sig}|${_envTint.key}` : p.sig;
    let slot = envBatch.get(key);
    if (!slot) {
      envBatch.set(key, slot = { mat: key === p.sig ? p.mat : envTintedMat(p, key), geos: [] });
    }
    slot.geos.push(p.geo.clone().applyMatrix4(_envM));
  }
  return true;
}
// 收工：把攒下的模块按材质合并进场景。分区后续解锁时会再攒一批、再合并一次。
function envFlush() {
  for (const { mat, geos } of envBatch.values()) {
    try {
      const merged = mergeGeometries(geos, false);
      if (!merged) continue;
      const mesh = new THREE.Mesh(merged, mat);
      mesh.castShadow = true; mesh.receiveShadow = true;
      mesh.frustumCulled = false;         // 合并后包围盒横跨全场，剔除只会误伤
      scene.add(mesh);
    } catch (e) {
      console.warn("环境批次合并失败，改为逐件挂载：", e);
      for (const g of geos) {
        const m = new THREE.Mesh(g, mat);
        m.castShadow = true; m.receiveShadow = true; scene.add(m);
      }
    }
  }
  envBatch.clear();
}

// 一段任意长度的墙：用 2 米模块铺满，末段靠轻微拉伸对齐（石砌看不出来）。
// (x,z) 是墙段中心，ry=0 表示墙沿 X 方向延展。
function envWallRun(len, x, z, ry, key = "wall") {
  const unit = envSize(key)?.x || 2;
  const n = Math.max(1, Math.round(len / unit));
  const step = len / n;
  const sx = step / unit;
  const cos = Math.cos(ry), sin = Math.sin(ry);
  for (let i = 0; i < n; i++) {
    const off = -len / 2 + step * (i + 0.5);
    envAdd(key, x + off * cos, z - off * sin, ry, sx);
  }
}

// 一片石板地：2×2 米地砖铺满 w×d，边缘那圈按余量微调拉伸。
function envFloorPatch(x, z, w, d, key = "floor") {
  const u = envSize(key)?.x || 2;
  const nx = Math.max(1, Math.round(w / u)), nz = Math.max(1, Math.round(d / u));
  const stepX = w / nx, stepZ = d / nz;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nz; j++) {
      envAdd(key, x - w / 2 + stepX * (i + 0.5), z - d / 2 + stepZ * (j + 0.5),
             0, stepX / u, 1, stepZ / u);
    }
  }
}

// 笼式火盆壁灯。参考图里那一个个暖光池就靠它。
// 模型本身走静态批次（火焰材质自带自发光，配 bloom 自己就会发亮，不花钱）；
// 只有 withLight 的那几盏真挂点光源——前向渲染每盏灯都要逐片元累加，不能滥用。
function envCresset(x, z, ry = 0, withLight = false, intensity = 5, dist = 16) {
  if (!envAdd("lamp", x, z, ry)) return;
  if (!withLight) return;
  const [wx, wz] = envToWorld(x, z);
  // decay=1.4 而非物理正确的 2：平方衰减会让灯脚下爆白、三米外全黑，
  // 想要参考图那种"大而柔的暖光池"，只能把衰减做扁（美术优先于物理）。
  const light = new THREE.PointLight(0xff9a4a, 0, dist, 1.4);
  light.position.set(wx, 2.1, wz);               // 火盆口在 2.18 高处
  scene.add(light);
  nightLamps.push({ light, baseIntensity: intensity });   // 随昼夜亮灭
}

// 各类房间的地面材质色（水泥灰打底，功能房间给点区分）
const FLOOR_COLOR = {
  eat: 0x6b6152, drink: 0x4d5a62, sleep: 0x3d424c, cure: 0x9aa8ac, class: 0x5a5f52,
  labor: 0x574c3e, gym: 0x4a3f3a, shower: 0x7f8a90, visit: 0x585f6b, library: 0x5c5344,
  laundry: 0x566a70, market: 0x413a4a, guardpost: 0x4a5162, commons: 0x63563f,
};

// 房间＝无盖剖面：地板 + 四面矮墙 + 前墙留门洞 + 压顶线。
// 去掉屋顶后从任意俯角都能看清室内，室内家具才有存在意义。
function makeBuilding(kind, color, spec) {
  const g = new THREE.Group();
  const { w, d } = spec;
  const wallH = ROOM_H;                 // 层高全场统一，见 ROOM_H 的注释
  const t = 0.18;
  const y0 = 0.1;                       // 地板厚
  // 有石墙模块就走模块（参考图那种厚石砌切面）；模块缺失才回退到程序化方盒子。
  // 无论走哪条路，碰撞盒都照旧生成——模块化只换皮，物理和之前逐字节一致。
  const modular = envHas("wall");
  const wMat = new THREE.MeshStandardMaterial({ color, roughness: 0.92, map: wallTexture() });
  const trimMat = new THREE.MeshStandardMaterial({ color: 0x2b2f3a, roughness: 0.9 });

  // 这间房的石色。模块是全场共用的，靠 envTint 在合批时按房间分色（见 envTint 的注释）。
  if (modular) envTint(ROOM_TINT[kind] ?? 0xffffff);

  // 医务室砖地、澡堂瓷砖单独铺；其余房间铺 Asphalt021。
  const specialFloor = (kind === "cure" && clinicBrickMat) ? clinicBrickMat
    : (kind === "shower" && showerTileMat) ? showerTileMat
    : null;
  if (specialFloor) {
    const floor = boxMesh(w, y0, d, specialFloor, 0, y0 / 2, 0);
    floor.castShadow = false;
    tileBoxUVs(floor, 2.0);
    g.add(floor);
  } else if (asphaltMat) {
    const floor = boxMesh(w, y0, d, asphaltMat, 0, y0 / 2, 0);
    floor.castShadow = false;
    tileBoxUVs(floor, 2.0);
    g.add(floor);
  } else if (modular) envFloorPatch(0, 0, w, d);
  else {
    const floor = boxMesh(w, y0, d, new THREE.MeshStandardMaterial(
      { color: FLOOR_COLOR[kind] ?? 0x4a5058, roughness: 1 }), 0, y0 / 2, 0);
    floor.castShadow = false; g.add(floor);
  }

  const wy = y0 + wallH / 2;
  // 洗衣房 / 医务室 / 禁闭室 / 澡堂各用专属贴图；其余功能室统一 Concrete044B。
  const rockMat = (kind === "laundry" && laundryRockMat) ? laundryRockMat
    : (kind === "cure" && clinicBrickMat) ? clinicBrickMat
    : (kind === "solitary" && solitaryMetalMat) ? solitaryMetalMat
    : (kind === "shower" && showerTileMat) ? showerTileMat
    : (kind !== "gate" && kind !== "phone" && kind !== "yard" && roomConcreteMat) ? roomConcreteMat
    : null;
  const wall = (mw, mh, md, x, y, z) => {
    const m = boxMesh(mw, mh, md, rockMat || (modular ? hiddenBlockerMat : wMat), x, y, z);
    m.userData.blocker = true;
    if (rockMat) tileBoxUVs(m, 2.0);
    else if (modular) { m.castShadow = false; m.receiveShadow = false; }
    g.add(m); return m;
  };
  wall(w, wallH, t, 0, wy, -d / 2 + t / 2);                          // 后墙
  wall(t, wallH, d, -w / 2 + t / 2, wy, 0);                          // 左墙
  wall(t, wallH, d, w / 2 - t / 2, wy, 0);                           // 右墙
  // 前墙留门洞。门洞默认居中，spec.door 可以沿墙挪开——贴墙连排时个别房间的前墙
  // 有一段被垂直那排挡着，门居中就会开在那段里，出门几米就撞墙（洗衣房在西南角就是这样）。
  const doorW = Math.min(1.5, w * 0.42);
  const doorC = Math.max(-w / 2 + doorW / 2, Math.min(w / 2 - doorW / 2, spec.door || 0));
  const segL = doorC - doorW / 2 + w / 2;                            // 门洞左侧那段墙长
  const segR = w / 2 - (doorC + doorW / 2);
  const segLC = (-w / 2 + doorC - doorW / 2) / 2;                    // 两段各自的中心
  const segRC = (doorC + doorW / 2 + w / 2) / 2;
  if (segL > 0.12) wall(segL, wallH, t, segLC, wy, d / 2 - t / 2);
  if (segR > 0.12) wall(segR, wallH, t, segRC, wy, d / 2 - t / 2);

  if (modular && !rockMat) {
    envWallRun(w, 0, -d / 2, 0);                                     // 后墙
    envWallRun(d, -w / 2, 0, Math.PI / 2);                           // 左墙
    envWallRun(d, w / 2, 0, Math.PI / 2);                            // 右墙
    if (segL > 0.12) envWallRun(segL, segLC, d / 2, 0);              // 前墙两段，中间留门洞
    if (segR > 0.12) envWallRun(segR, segRC, d / 2, 0);
    // 四角石柱：把墙段的接头包住，切面才像砌起来的而不是拼起来的
    envAdd("wallCorner", -w / 2, -d / 2, 0);
    envAdd("wallCorner", w / 2, -d / 2, -Math.PI / 2);
    envAdd("wallCorner", w / 2, d / 2, Math.PI);
    envAdd("wallCorner", -w / 2, d / 2, Math.PI / 2);
  }
  if (modular) {
    // 外墙铁窗（模块自带石框+铁栅+黑洞，正好嵌在墙厚里）。只开在后墙上：
    // 后墙背靠围墙，是真正的外墙；左右两面现在是和邻室共用的隔墙，开窗就成了对穿。
    const cols = Math.max(1, Math.round(w / 3.0));
    for (let ci = 0; ci < cols; ci++) {
      envAdd("window", -w / 2 + (w / (cols + 1)) * (ci + 1), -d / 2, 0, 1, 1, 1, 1.25);
    }
  } else {
    // 压顶线：一圈浅色收边，切面才像"被削掉屋顶"而不是墙没盖完
    const capY = y0 + wallH + 0.06;
    g.add(boxMesh(w + 0.16, 0.12, t + 0.16, trimMat, 0, capY, -d / 2 + t / 2));
    g.add(boxMesh(w + 0.16, 0.12, t + 0.16, trimMat, 0, capY, d / 2 - t / 2));
    g.add(boxMesh(t + 0.16, 0.12, d + 0.16, trimMat, -w / 2 + t / 2, capY, 0));
    g.add(boxMesh(t + 0.16, 0.12, d + 0.16, trimMat, w / 2 - t / 2, capY, 0));
    // 外墙铁窗：从外面看仍是监狱，夜里玻璃会透暖光。同上，只开后墙。
    const cols = Math.max(1, Math.round(w / 2.0));
    for (let ci = 0; ci < cols; ci++) {
      const x = -w / 2 + (w / (cols + 1)) * (ci + 1);
      const back = barredWindow();
      back.position.set(x, y0 + wallH * 0.62, -d / 2 - 0.02);
      back.rotation.y = Math.PI; g.add(back);
    }
  }

  // 室内不再挂暖色火盆：屋里照度全靠冷白顶灯，火盆会把混凝土/瓷砖重新染黄。
  if (modular) envTintClear();

  // 前门：门洞够宽就装一扇和牢房同款的栅栏门（大门/电话亭不是屋子，在 buildOneFacility 里跳过登记）。
  // 门扇是房间组的子物体，随房间朝向一起转；把门中心记成一个 marker，造好后取世界坐标登记碰撞盒。
  if (doorW > 0.4) {
    const leaf = makeRoomDoorLeaf(doorW, wallH);
    leaf.position.set(doorC - doorW / 2, 0, d / 2 - t / 2);
    g.add(leaf);
    const mark = new THREE.Object3D();
    mark.position.set(doorC, 0, d / 2 - t / 2);
    g.add(mark);
    g.userData.doorRec = { leaf, mark, doorW };
  }

  furnishRoom(kind, w, d, wallH, g);
  return g;
}

// 坐/躺的落位锚点：记下位置和朝向，交互时把主角摆过去
function seatAnchor(parent, x, y, z, ry = 0) {
  const a = new THREE.Object3D();
  a.position.set(x, y, z); a.rotation.y = ry;
  parent.add(a);
  return a;
}

function furnishRoom(kind, w, d, wallH, g) {
  const put = (obj, x, z, ry = 0, y = 0) => {
    obj.position.set(x, y, z); obj.rotation.y = ry; g.add(obj); return obj;
  };
  const back = -d / 2 + 0.75;
  const seat = (obj, name, ax, az, ay, ry) =>
    addInteractable(obj, { type: "sit", name, icon: "🪑", kind,
                           anchor: seatAnchor(g, ax, ay, az, ry) });
  // 躺卧锚点＝床尾（脚的位置），ry 决定身体朝哪一头展开
  const lie = (obj, name, ax, az, ay, ry) =>
    addInteractable(obj, { type: "lie", name, icon: "🛏️", kind,
                           anchor: seatAnchor(g, ax, ay, az, ry) });

  if (kind === "eat") {
    // 食堂：后墙一条打饭窗口，前面三排横向长条餐桌，每排双侧长凳。
    // 一排"横的"桌子横贯全宽才有食堂的样子——原来只摆一张 1.8m 小方桌 + 两只圆凳，
    // 从斜上方看是间空屋子里放了套餐椅。桌长按房宽收 4.6m（两端各留一条 2.3m 的通道），
    // 长凳拿 2.3m 一截拼够长度：Jampot 石凳模块是 2m 的，硬拉到 7m 会把石纹一起拉花。
    put(makeCounter(Math.min(7.0, w - 4.0)), 0, back);
    const tlen = Math.max(3.0, w - 4.6);
    const rows = [-2.0, 0.5, 3.0];              // 排间距 2.5m：桌 0.9 + 双侧凳各 0.46 + 过道
    rows.forEach((rz, ri) => {
      if (rz > d / 2 - 1.3) return;             // 房间不够深就少摆一排，别顶到前墙上
      const tb = put(makeMessTable(tlen), 0, rz);
      if (ri === 1) {
        addInteractable(tb, { type: "search", name: "餐桌", icon: "🍚",
          found: "桌上有人没吃完的窝头，还热着。" });
      }
      for (const s of [-1, 1]) {
        const bz = rz + s * 0.75;
        const n = Math.max(1, Math.round(tlen / 2.4));
        for (let i = 0; i < n; i++) {
          const bx = -tlen / 2 + (tlen / n) * (i + 0.5);
          const bn = put(makeBench(tlen / n - 0.12), bx, bz);
          seat(bn, "餐桌长凳", bx, bz, 0.52, s > 0 ? Math.PI : 0);
        }
      }
    });
  } else if (kind === "drink") {
    const sk = put(makeSink(), 0, back);
    addInteractable(sk, { type: "search", name: "水槽", icon: "💧", found: "拧开龙头，水管咕咚响了半天才出水。" });
    const bucket = put(makeCrate(0.5), w / 2 - 0.7, 0.5);
    addInteractable(bucket, { type: "pickup", name: "铁皮水壶", icon: "🫙", item: "kettle" });
  } else if (kind === "sleep") {
    const b1 = put(makeBunk(), -w / 2 + 1.3, back + 0.1, Math.PI / 2);
    const b2 = put(makeBunk(), w / 2 - 1.3, back + 0.1, Math.PI / 2);
    lie(b1, "下铺", -w / 2 + 1.3, back + 0.95, 0.72, 0);
    lie(b2, "下铺", w / 2 - 1.3, back + 0.95, 0.72, 0);
    const lk = put(makeLocker(), 0, back);
    addInteractable(lk, { type: "search", name: "储物柜", icon: "🗄️", found: "柜里塞着换洗囚服、几封没寄出的信。" });
    put(makeToilet(), w / 2 - 0.7, d / 2 - 0.9, Math.PI);
    const st = put(makeStool(0, 0), 0, d / 2 - 1.0);
    seat(st, "木凳", 0, d / 2 - 1.0, 0.53, Math.PI);
  } else if (kind === "solitary") {
    // 禁闭室：打架被抓/越狱失败关进来。刻意布置得比牢房更空、更难看——
    // 一张光板床、一个便桶、墙上刻的正字，没有储物柜也没有凳子。
    const b = put(makeBunk(), 0, back + 0.1, Math.PI / 2);
    lie(b, "禁闭床板", 0, back + 0.95, 0.72, 0);
    put(makeToilet(), w / 2 - 0.7, d / 2 - 0.9, Math.PI);
    const wall = put(makeCrate(0.35), -w / 2 + 0.7, d / 2 - 1.0);
    addInteractable(wall, { type: "search", name: "墙角", icon: "🪨",
      found: "墙上密密麻麻刻着正字，最后一道只划了一半。" });
  } else if (kind === "cure") {
    const bed = put(makeMedBed(), 0, back + 0.1, Math.PI / 2);
    lie(bed, "诊疗床", 0, back + 0.95, 0.83, 0);
    const sh = put(makeShelf(), -w / 2 + 0.6, d / 2 - 0.9, Math.PI / 2);
    addInteractable(sh, { type: "search", name: "药柜", icon: "💊", found: "纱布、碘酒、几板过期止痛片。" });
    const pill = put(makeCrate(0.34), w / 2 - 0.8, d / 2 - 1.0);
    addInteractable(pill, { type: "pickup", name: "止痛药瓶", icon: "💊", item: "pills" });
  } else if (kind === "class") {
    put(makeBlackboard(), 0, back - 0.1);
    for (const dx of [-0.9, 0.9]) {
      const dk = put(makeDesk(), dx, 0.4);
      const ch = put(makeChair(), dx, 1.15, Math.PI);
      seat(ch, "课桌椅", dx, 1.15, 0.5, Math.PI);
      addInteractable(dk, { type: "search", name: "课桌", icon: "📖", found: "桌面刻满了正字，数着刑期。" });
    }
    const bk = put(makeCrate(0.3), w / 2 - 0.7, back);
    addInteractable(bk, { type: "pickup", name: "翻烂的课本", icon: "📕", item: "textbook" });
  } else if (kind === "labor") {
    const wb = put(makeWorkbench(), 0, back);
    addInteractable(wb, { type: "search", name: "工作台", icon: "🔨", found: "台面上一层铁屑，虎钳里还夹着没锉完的件。" });
    put(makeCrate(0.9), -w / 2 + 0.9, d / 2 - 1.0);
    const c2 = put(makeCrate(0.7), w / 2 - 0.9, d / 2 - 1.1);
    addInteractable(c2, { type: "search", name: "板条箱", icon: "📦", found: "箱底压着一卷没用完的铁丝。" });
    const hm = put(makeCrate(0.28), 0.7, 0.3);
    addInteractable(hm, { type: "pickup", name: "旧锤子", icon: "🔨", item: "hammer" });
  } else if (kind === "gym") {
    put(makeWeightRack(), 0, back);
    const bn = put(makeBench(1.7), 0, 0.5);
    lie(bn, "卧推凳", 0.72, 0.5, 0.52, Math.PI / 2);
    const db = put(makeCrate(0.3), w / 2 - 0.7, d / 2 - 0.9);
    addInteractable(db, { type: "pickup", name: "哑铃", icon: "🏋️", item: "dumbbell" });
  } else if (kind === "shower") {
    for (const dx of [-w / 4, w / 4]) put(makeShowerHead(), dx, back - 0.1);
    const bn = put(makeBench(1.4), 0, d / 2 - 0.9);
    seat(bn, "更衣长凳", 0, d / 2 - 0.9, 0.52, Math.PI);
    const soap = put(makeCrate(0.22), w / 2 - 0.6, back + 0.2);
    addInteractable(soap, { type: "pickup", name: "半块肥皂", icon: "🧼", item: "soap" });
  } else if (kind === "visit") {
    const tb = put(makeTable(), 0, 0);
    addInteractable(tb, { type: "search", name: "会见台", icon: "🪟", found: "台面被无数只手磨得发亮，中间嵌着一块脏玻璃。" });
    const glass = boxMesh(1.7, 0.9, 0.05, glassMat, 0, 1.25, 0); g.add(glass);
    const c1 = put(makeChair(), 0, 0.95, Math.PI), c2 = put(makeChair(), 0, -0.95, 0);
    seat(c1, "会见椅", 0, 0.95, 0.5, Math.PI);
    seat(c2, "家属席", 0, -0.95, 0.5, 0);
  } else if (kind === "library") {
    put(makeShelf(), -w / 2 + 0.6, back + 0.3, Math.PI / 2);
    const sh = put(makeShelf(), w / 2 - 0.6, back + 0.3, -Math.PI / 2);
    addInteractable(sh, { type: "search", name: "书架", icon: "📚", found: "书脊上全是登记编号，有一本被翻得只剩半截。" });
    const dk = put(makeDesk(), 0, d / 2 - 1.2);
    addInteractable(dk, { type: "search", name: "阅览桌", icon: "🕯️", found: "桌上摊着一本翻开的书，页边写满批注。" });
    const ch = put(makeChair(), 0, d / 2 - 0.5, Math.PI);
    seat(ch, "阅览椅", 0, d / 2 - 0.5, 0.5, Math.PI);
    const bk = put(makeCrate(0.28), -w / 2 + 1.4, d / 2 - 1.0);
    addInteractable(bk, { type: "pickup", name: "一本旧书", icon: "📗", item: "book" });
  } else if (kind === "laundry") {
    for (const dx of [-w / 4, w / 4]) {
      const tub = put(makeWashTub(), dx, back);
      addInteractable(tub, { type: "search", name: "洗衣槽", icon: "🧺", found: "水面浮着一层灰白泡沫，底下沉着谁的袜子。" });
    }
    put(makeSink(), w / 2 - 0.9, d / 2 - 0.8, Math.PI);
    const cl = put(makeCrate(0.32), -w / 2 + 0.8, d / 2 - 0.9);
    addInteractable(cl, { type: "pickup", name: "干净囚服", icon: "👕", item: "uniform" });
  } else if (kind === "market") {
    const c1 = put(makeCrate(0.8), -w / 2 + 0.7, back);
    addInteractable(c1, { type: "search", name: "货箱", icon: "📦", found: "泡面、电池、指甲刀——都是硬通货。" });
    put(makeCrate(0.6), w / 2 - 0.7, back + 0.1);
    const cig = put(makeCrate(0.24), 0, d / 2 - 0.9);
    addInteractable(cig, { type: "pickup", name: "一包香烟", icon: "🚬", item: "cigs" });
  } else if (kind === "guardpost") {
    const dk = put(makeDesk(), 0, back + 0.1);
    addInteractable(dk, { type: "search", name: "值班桌", icon: "🗂️", found: "抽屉里锁着登记簿，边角有明显的涂改痕迹。" });
    const ch = put(makeChair(), 0, back + 0.95, Math.PI);
    seat(ch, "值班椅", 0, back + 0.95, 0.5, Math.PI);
    const keys = put(makeCrate(0.2), w / 2 - 0.6, d / 2 - 0.8);
    addInteractable(keys, { type: "pickup", name: "一串钥匙", icon: "🔑", item: "keys" });
  } else if (kind === "commons") {
    const tb = put(makeTable(), -0.6, 0.2);
    addInteractable(tb, { type: "search", name: "棋桌", icon: "♟️", found: "残局摆在那儿，红方少了一个仕。" });
    const s1 = put(makeStool(0, 0), -1.7, 0.2), s2 = put(makeStool(0, 0), 0.5, 0.2);
    seat(s1, "棋桌凳", -1.7, 0.2, 0.53, Math.PI / 2);
    seat(s2, "棋桌凳", 0.5, 0.2, 0.53, -Math.PI / 2);
    const bn = put(makeBench(1.6), w / 2 - 0.9, back + 0.4, Math.PI / 2);
    seat(bn, "长凳", w / 2 - 0.9, back + 0.4, 0.52, -Math.PI / 2);
    const gt = put(makeCrate(0.3), w / 2 - 0.8, d / 2 - 0.9);
    addInteractable(gt, { type: "pickup", name: "掉漆吉他", icon: "🎸", item: "guitar" });
  }

  // 顶灯 + 门边墙上的开关（每个房间都有，夜里进屋自己开灯）
  if (kind !== "gate" && kind !== "phone" && kind !== "yard" && kind !== "shower" && kind !== "solitary") {
    const { group: lampG, rec } = makeCeilingLamp(0, 0.1 + wallH - 0.18, 0, true, 0.4);
    g.add(lampG);
    const sw = boxMesh(0.16, 0.24, 0.06,
      new THREE.MeshStandardMaterial({ color: 0xd8dde4, roughness: 0.5 }),
      Math.min(w / 2 - 0.28, 1.05), 1.25, d / 2 - 0.22);
    g.add(sw);
    addInteractable(sw, { type: "lamp", name: "灯开关", icon: "💡", lamp: rec });
  }
}

function buildYardProps(cx, cz) {
  // 跑道环 + 中心圈（地面涂装）
  const track = new THREE.Mesh(
    new THREE.RingGeometry(3.4, 3.9, 48),
    new THREE.MeshStandardMaterial({ color: 0x7a4a2c, roughness: 1, side: THREE.DoubleSide })
  );
  track.rotation.x = -Math.PI / 2; track.position.set(cx, 0.02, cz);
  track.receiveShadow = true; scene.add(track);
  const circle = new THREE.Mesh(
    new THREE.RingGeometry(1.3, 1.45, 40),
    new THREE.MeshStandardMaterial({ color: 0xb9c0cc, roughness: 1, side: THREE.DoubleSide })
  );
  circle.rotation.x = -Math.PI / 2; circle.position.set(cx, 0.02, cz); scene.add(circle);
  // 篮球架
  const hoop = new THREE.Group();
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 3.2, 10),
    new THREE.MeshStandardMaterial({ color: 0x3a3f49, metalness: 0.6, roughness: 0.5 }));
  pole.position.y = 1.6; pole.castShadow = true; hoop.add(pole);
  const board = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.8, 0.08),
    new THREE.MeshStandardMaterial({ color: 0xe8ecf2, roughness: 0.6 }));
  board.position.set(0, 3.1, -0.15); hoop.add(board);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.04, 8, 20),
    new THREE.MeshStandardMaterial({ color: 0xe0662c, metalness: 0.4 }));
  rim.rotation.x = Math.PI / 2; rim.position.set(0, 2.75, 0.2); hoop.add(rim);
  hoop.position.set(cx + 3.2, 0, cz - 3.2); scene.add(hoop);
  // 放风场围栏：只圈北侧和东侧（背向中庭的两面）+ 一个转角。参考图里放风场被铁栅围着，
  // 这两面就够把"露天但关得住"拍出来。
  // 刻意不给栅栏加碰撞：中庭到黑市/健身区的人流会横穿这块地，一加碰撞前端就要绕行、
  // 而后端仍走直线，会打起来（橡皮筋+卡住）。少数人偶尔穿栅栏，比 NPC 卡死划算。
  if (envHas("fence")) {
    const R = 6.2;
    envWallRun(R * 2, cx, cz - R, 0, "fence");                      // 北
    envWallRun(R * 2, cx + R, cz, Math.PI / 2, "fence");            // 东
    envAdd("fenceCorner", cx + R, cz - R, -Math.PI / 2);
    envAdd("fenceGate", cx + R, cz + 2.0, Math.PI / 2);             // 东侧一道小门
    // 门口两盏火盆：夜里这块是全场最亮的地方，人影拉得很长
    envCresset(cx + R - 0.7, cz + 3.6, 0, true, 5, 18);
    envCresset(cx - 4.6, cz - R + 0.7, 0, true, 5, 18);
    // 角落堆点桶和箱子，空地才不像一块草皮
    envAdd("barrels", cx - R + 1.1, cz - R + 1.1, 0.4);
    envAdd("crates", cx + R - 1.2, cz - R + 2.6, -0.3);
    envAdd("barrelOpen", cx + R - 1.3, cz - R + 1.2, 0.8);
    for (const [bx, bz, r] of [[cx - 2.2, cz + 3.6, 0], [cx + 2.2, cz + 3.6, 0]]) {
      envAdd("bench", bx, bz, r);
      propBox(bx, bz, 2.0, 0.5);
    }
    propBox(cx - R + 1.1, cz - R + 1.1, 1.2, 1.2);
    propBox(cx + R - 1.2, cz - R + 2.6, 1.4, 1.4);
    propBox(cx + R - 1.3, cz - R + 1.2, 0.8, 0.8);
  }
  // 长椅 ×2
  for (const [bx, bz, rot] of [[cx - 3.4, cz + 1, 0], [cx - 3.4, cz - 1.5, 0]]) {
    const bench = new THREE.Group();
    const seat = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.12, 0.5),
      new THREE.MeshStandardMaterial({ color: 0x6b5334, roughness: 0.9 }));
    seat.position.y = 0.45; seat.castShadow = true; bench.add(seat);
    for (const lx of [-0.85, 0.85]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.45, 0.45),
        new THREE.MeshStandardMaterial({ color: 0x3a3f49 }));
      leg.position.set(lx, 0.22, 0); bench.add(leg);
    }
    bench.position.set(bx, 0, bz); bench.rotation.y = rot; addProp(bench);
  }
}

// 电话亭做成真的亭子：四根立柱 + 三面玻璃 + 挂在后壁的话机
function makePhoneBooth() {
  const g = new THREE.Group();
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x2b4f7a, roughness: 0.5, metalness: 0.5 });
  for (const [px, pz] of [[-0.55, -0.55], [0.55, -0.55], [-0.55, 0.55], [0.55, 0.55]]) {
    g.add(boxMesh(0.1, 2.3, 0.1, frameMat, px, 1.15, pz));
  }
  const panel = new THREE.PlaneGeometry(1.0, 1.7);
  const back = new THREE.Mesh(panel, glassMat); back.position.set(0, 1.3, -0.54); g.add(back);
  const left = new THREE.Mesh(panel, glassMat);
  left.position.set(-0.54, 1.3, 0); left.rotation.y = Math.PI / 2; g.add(left);
  const right = new THREE.Mesh(panel, glassMat);
  right.position.set(0.54, 1.3, 0); right.rotation.y = -Math.PI / 2; g.add(right);
  g.add(boxMesh(1.3, 0.12, 1.3, frameMat, 0, 2.36, 0));
  // 话机：机身 + 听筒 + 一小截线
  g.add(boxMesh(0.26, 0.34, 0.12, new THREE.MeshStandardMaterial({ color: 0x14171d }), 0, 1.35, -0.44));
  const horn = new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.16, 4, 8),
    new THREE.MeshStandardMaterial({ color: 0x0c0e12 }));
  horn.rotation.z = Math.PI / 2; horn.position.set(-0.16, 1.4, -0.4); g.add(horn);
  return g;
}

// 各设施门口的专属道具：一眼能认出这是干什么的地方。
// 一律摆在前墙外侧（局部 +z），别和室内家具打架。
//
// 全部写在局部坐标里、挂到一个按房间朝向旋转的组上。房子贴墙成排之后四排朝向各不相同，
// 照死写世界坐标的 cz+d/2 会把晾衣绳挂到墙里去。锚点也挂在这个组上，
// 落座时读的是世界变换，跟着转不会错位。
function buildKindProps(kind, cx, cz, ry = 0, w = 4, d = 4, wallH = 2.6) {
  const G = new THREE.Group();
  G.position.set(cx, 0, cz); G.rotation.y = ry;
  scene.add(G);
  const fz = d / 2 + 1.0;
  if (kind === "laundry") {
    // 晾衣绳 + 挂着的囚服，风干的橘色一片最认得出来
    const poleM = new THREE.MeshStandardMaterial({ color: 0x555b66, metalness: 0.4, roughness: 0.7 });
    for (const px of [-2.4, 2.4]) {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 2.4, 8), poleM);
      p.position.set(px, 1.2, fz); p.castShadow = true; G.add(p);
    }
    const rope = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 4.8, 5), poleM);
    rope.rotation.z = Math.PI / 2; rope.position.set(0, 2.25, fz); G.add(rope);
    const cloths = [0xd9762e, 0xb9bfc9, 0xd9762e, 0x8d949f, 0xd9762e];
    cloths.forEach((col, i) => {
      const c = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 1.0),
        new THREE.MeshStandardMaterial({ color: col, roughness: 1, side: THREE.DoubleSide }));
      c.position.set(-1.8 + i * 0.9, 1.72, fz);
      c.rotation.y = (i % 2 ? 0.12 : -0.1);
      c.castShadow = true; G.add(c);
    });
  } else if (kind === "market") {
    // 门口货堆 + 遮雨帆布 + 一盏吊灯（夜里靠 bloom 很打眼）
    for (const [ox, oz, s] of [[-1.5, 0.2, 0.9], [-0.7, 0.8, 0.7], [1.4, 0.3, 0.8]]) {
      const c = makeCrate(s); c.position.set(ox, 0, fz + oz); G.add(c);
    }
    const tarp = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 2.4),
      new THREE.MeshStandardMaterial({ color: 0x3f4a3a, roughness: 1, side: THREE.DoubleSide }));
    tarp.rotation.x = -Math.PI / 2 + 0.35; tarp.position.set(0, 2.1, fz + 0.3);
    tarp.castShadow = true; G.add(tarp);
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 8),
      new THREE.MeshStandardMaterial({ color: 0xe8f2ff, emissive: 0xb8d4ff, emissiveIntensity: 0 }));
    bulb.position.set(0, 2.0, fz); G.add(bulb);
    const light = new THREE.PointLight(0xc5dcff, 0, 10, 2);
    light.position.copy(bulb.position); G.add(light);
    nightLamps.push({ light, baseIntensity: 2.4, bulb });
  } else if (kind === "guardpost") {
    // 屋顶警灯 + 门前道闸
    const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 10),
      new THREE.MeshStandardMaterial({ color: 0xff4d4d, emissive: 0xff2222, emissiveIntensity: 1.4 }));
    beacon.position.set(0, wallH + 0.5, 0); G.add(beacon);
    const bar = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.12, 0.12),
      new THREE.MeshStandardMaterial({ color: 0xe8e8e8, roughness: 0.6 }));
    bar.position.set(-w / 2 - 0.6, 1.1, fz); bar.castShadow = true; G.add(bar);
    for (let i = 0; i < 6; i++) {
      const stripe = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.13, 0.13),
        new THREE.MeshStandardMaterial({ color: 0xd23b3b }));
      stripe.position.set(-w / 2 - 2.2 + i * 0.6, 1.1, fz); G.add(stripe);
    }
  } else if (kind === "commons") {
    // 门口再摆一张棋桌，路过就能看见有人在下棋
    const t = makeTable(); t.position.set(0, 0, fz); G.add(t);
    const s1 = makeStool(-1.4, fz), s2 = makeStool(1.4, fz);
    G.add(s1); G.add(s2);
    addInteractable(s1, { type: "sit", name: "门口棋凳", icon: "🪑", kind: "commons",
                          anchor: seatAnchor(G, -1.4, 0.53, fz, Math.PI / 2) });
    addInteractable(s2, { type: "sit", name: "门口棋凳", icon: "🪑", kind: "commons",
                          anchor: seatAnchor(G, 1.4, 0.53, fz, -Math.PI / 2) });
    const boardTex = (() => {
      const c = document.createElement("canvas"); c.width = c.height = 64;
      const g2 = c.getContext("2d");
      for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
        g2.fillStyle = (i + j) % 2 ? "#2b2f38" : "#cfc7ae";
        g2.fillRect(i * 8, j * 8, 8, 8);
      }
      const tx = new THREE.CanvasTexture(c);
      tx.colorSpace = THREE.SRGBColorSpace; tx.magFilter = THREE.NearestFilter;
      return tx;
    })();
    const board = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.8),
      new THREE.MeshStandardMaterial({ map: boardTex, roughness: 0.8 }));
    board.rotation.x = -Math.PI / 2; board.position.set(0, 0.79, fz); G.add(board);
  }
}

// 房间的尺寸/落点/朝向**不在前端**：全部由 backend/app/layout.py 算好、随世界帧下发
// （f.w/f.d 是局部尺寸，f.face 是门朝向，f.door 是门洞偏移）。以前这里存一份尺寸、
// 后端存一份落点，两边各自演进就穿模。改布局请改 layout.py，然后跑
// `python backend/_overlap_probe.py` 体检。
//
// 层高全场统一 ROOM_H，不逐间给高度。无盖剖面是从斜上方看下去的，层高一乱，
// 一排房子的墙沿就高低不齐，像散落的积木而不是一座监狱。要的差异是平面上的
// （大小、进深、朝向），不是竖向的。
const ROOM_H = 2.6;
// 各房间的石色色调：模块是全场共用的，靠 envTint 在合批时按房间分色，
// 让贴墙连排的一长条房子不至于看成同一块灰板（见 envTint 的注释）。
const ROOM_TINT = {
  eat: 0xffecd0, drink: 0xd8ecff, sleep: 0xdcd8e8, cure: 0xe2f4e8,
  class: 0xdcecec, labor: 0xf0e2c8, gym: 0xf0dcc8, shower: 0xd0eaf2,
  visit: 0xe8dcf0, library: 0xe8e8cc, laundry: 0xd8ecec, market: 0xdcd0e4,
  guardpost: 0xd8dee8, commons: 0xf0e4cc, solitary: 0xc8ccd4,
};
// 门朝向 → 整组绕 Y 的转角。房间一律按"门在局部 +z"来砌（makeBuilding / furnishRoom
// 都只写这一套），再靠这个转角把它转到该朝的方向。envOrigin 也吃同一个转角，
// 所以模块化石墙、地砖、火盆会和碰撞盒、家具一起转过去。
const FACE_RY = { N: 0, E: Math.PI / 2, S: Math.PI, W: -Math.PI / 2 };
const FACE_VEC = { N: [0, 1], E: [1, 0], S: [0, -1], W: [-1, 0] };

const pathMat = new THREE.MeshStandardMaterial({ color: 0x53585f, roughness: 1 });
function walkway(x1, z1, x2, z2, width = 1.6) {
  const dx = x2 - x1, dz = z2 - z1;
  const len = Math.hypot(dx, dz);
  const path = new THREE.Mesh(new THREE.PlaneGeometry(width, len), asphaltMat || pathMat);
  path.rotation.x = -Math.PI / 2;
  path.rotation.z = -Math.atan2(dx, dz);
  path.position.set((x1 + x2) / 2, 0.015, (z1 + z2) / 2);
  path.receiveShadow = true;
  if (asphaltMat) tilePlaneUVs(path, 2);
  scene.add(path);
}

// 中庭环路 + 每间房门口的短支路。
// 原先是从场地中心向每间房各拉一条直线。房子搬到墙根成排之后，那些斜线会横穿别的房子
//（去洗衣房那条正好从食堂身上压过去），俯视图里是一片压在屋顶剖面上的浅色叉。
// 改成"一圈环路 + 门口短支"，路网就和"一圈房子围一个院子"这个平面对上了，
// 顺带把 NPC 实际走的动线画了出来——他们绕开墙体的 A* 路径本来就贴着这一圈。
// 这一圈与后端 layout.patrol_ring()（中庭四角内缩 1.8m）取同一条线：狱警走的就是它，
// 画出来的路和人实际走的动线才对得上。中庭 = 四条连排最深那间的内沿围出来的方框。
const RING = { x0: -12.2, x1: 11.6, z0: -9.7, z1: 11.2 };
function buildRingRoad() {
  walkway(RING.x0, RING.z0, RING.x1, RING.z0, 2.0);
  walkway(RING.x0, RING.z1, RING.x1, RING.z1, 2.0);
  walkway(RING.x0, RING.z0, RING.x0, RING.z1, 2.0);
  walkway(RING.x1, RING.z0, RING.x1, RING.z1, 2.0);
}
// 从门口沿朝向接一小段到环路上。够不着的不接（电话亭在院子中间，本来就在环路上）。
function doorWalkway(f) {
  const [fx, fz] = FACE_VEC[f.face] ?? [0, 1];
  const half = (f.d || 0) / 2, off = f.door || 0;
  const dx = f.pos[0] + fx * half - fz * off;
  const dz = f.pos[1] + fz * half + fx * off;
  const target = fz > 0 ? RING.z0 : fz < 0 ? RING.z1 : fx > 0 ? RING.x0 : RING.x1;
  const len = fx ? (target - dx) * fx : (target - dz) * fz;
  if (len < 0.3 || len > 12) return;
  walkway(dx, dz, dx + fx * len, dz + fz * len);
}

let plazaBuilt = false;
const builtFacilities = new Set();   // 已建出的设施 id（分区解锁后逐个"长"出来）

function buildOneFacility(f) {
  const color = KIND_COLOR[f.kind] ?? 0x888888;
  // 尺寸/朝向/门洞偏移全部来自世界帧（后端 layout.py 唯一权威）
  const spec = { w: f.w || 3.2, d: f.d || 3.0, door: f.door || 0 };
  const { w, d } = spec;
  const ry = FACE_RY[f.face] ?? 0;
  // 放风场是露天场地，只铺跑道和器材，不盖房子
  if (f.kind === "yard") {
    buildYardProps(f.pos[0], f.pos[1]);
    const yl = { el: makeBubble("facility-tag"), obj: new THREE.Object3D(), y: 2.6 };
    yl.obj.position.set(f.pos[0], 0, f.pos[1]); scene.add(yl.obj);
    yl.el.textContent = f.name; facilityLabels.push(yl);
    return;
  }
  // 牢区(sleep)：这一体不盖方盒子。六间隔间由 syncCells 按世界帧的 cells 单独砌，
  // 每间自带门牌，所以这里连整体标签都不要（否则"牢房"会压在六个门牌中间）。
  if (f.kind === "sleep") {
    doorWalkway(f);
    return;
  }
  // 整组绕 Y 转到该朝的方向：房间内部（墙、门洞、家具、模块化石墙）只按"门在 +z"
  // 写一套，转角由这里统一给。envOrigin 吃同一个转角，静态批次才跟着转。
  envOrigin(f.pos[0], f.pos[1], ry);
  const g = makeBuilding(f.kind, color, spec);
  g.position.set(f.pos[0], 0, f.pos[1]);
  g.rotation.y = ry;
  // 大门做成铁闸门（Portcullis 模块）而非楼
  if (f.kind === "gate") {
    g.clear();
    if (!envAdd("door", 0, 0, 0, 2.6 / (envSize("door")?.x || 2.38))) {
      for (const gx of [-1.0, 1.0]) {
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.3, 4, 0.3), wallMat);
        post.position.set(gx, 2, 0); post.castShadow = true; g.add(post);
        for (let i = 0; i < 5; i++) {
          const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 3.4, 6), barMat);
          bar.position.set(gx + (gx < 0 ? 0.5 : -0.5), 1.7, -0.4 + i * 0.2); g.add(bar);
        }
      }
    }
  }
  // 电话亭不是楼，是个亭子
  if (f.kind === "phone") { g.clear(); g.add(makePhoneBooth()); }
  scene.add(g);
  collectBlockers(g);
  // 功能室的门登记开合逻辑（大门做成铁闸、电话亭是亭子，都已 g.clear() 掉门扇，跳过）
  if (g.userData.doorRec && f.kind !== "gate" && f.kind !== "phone") {
    registerRoomDoor(g.userData.doorRec, f);
  }
  envOriginClear();                     // 往下都是世界坐标了，别再叠一层换算
  const lab = { el: makeBubble("facility-tag"), obj: g, y: ROOM_H + 1.0 };
  lab.el.textContent = f.name;
  facilityLabels.push(lab);
  buildKindProps(f.kind, f.pos[0], f.pos[1], ry, w, d, ROOM_H);
  doorWalkway(f);
}

// 增量建造：只建"已解锁分区"里、且还没建过的设施。分区一解锁，房间就在俯视图里长出来。
function syncFacilities(msg) {
  if (!plazaBuilt) {
    const plaza = new THREE.Mesh(new THREE.CircleGeometry(3.2, 40), asphaltMat || pathMat);
    plaza.rotation.x = -Math.PI / 2; plaza.position.y = 0.014; plaza.receiveShadow = true;
    scene.add(plaza);
    plazaBuilt = true;
  }
  let grew = false;
  for (const f of msg.facilities) {
    if (!zoneUnlocked(f.zone) || builtFacilities.has(f.id)) continue;
    buildOneFacility(f);
    builtFacilities.add(f.id);
    grew = true;
  }
  // 这一轮攒下的石墙地砖一次性合并进场景（几百块 → 十几个 draw call）
  if (grew) envFlush();
  return grew;
}

// ---------------------------------------------------------------------------
// 角色模型
// ---------------------------------------------------------------------------
const npcs = new Map();           // id -> entry
let selectedId = null;
let charTemplate = null;          // { scene, animations, scale, footY }
const npcTemplates = new Map();   // id -> 专属底模（个别 NPC 换成自己的 GLB，如瘸子）
let npcCount = 0;                 // 分配"工位"角度用，避免同设施叠在一起

// 玩家主角 + 对话状态
let player = null;                // {root, mixer, actions, current, heading, label}
const keys = Object.create(null); // 按键状态
let talkNpcId = null;             // 当前最近、可对话的 NPC
let chatOpen = false;
let chatNpcId = null;
const chatHistories = new Map();  // npc_id -> [{role, content}]
const PLAYER_SPEED = 6.0;
const COLLIDE_STEP = 0.28;        // 每步不超过身半径，避免一帧跨过 0.18m 薄墙
// 活动上限贴到围墙内表面（墙厚 0.5，内表面 29.75）。原来收在 WALL-2.2=27.8，
// 那时房子离墙还远、这条线只是"别贴着墙走"；现在房子紧靠围墙，房间进深最大的食堂
// 后墙内侧在 z=-29.0，卡在 27.8 会让典狱长走不到打饭窗口跟前，够不着交互距离。
// 真正挡人的是房间墙体（有碰撞盒），这条线只兜住墙脚那几段没被房子占的空地。
const PLAYER_LIMIT = WALL - 0.45;
const WARDEN_FREEZE_R = 4.6;                       // 典狱长靠这么近，NPC 就僵住立正（与后端 WARDEN_SIGHT 对齐）
const WARDEN_FREEZE_R2 = WARDEN_FREEZE_R * WARDEN_FREEZE_R;

// ---------------------------------------------------------------------------
// 视角系统
// 自由档沿用 OrbitControls；其余四档自己算机位，统一 lerp 平滑过渡，
// 共用一套「拖拽转向 + 滚轮远近」的输入，切档时从当前机位接管、不瞬跳。
// ---------------------------------------------------------------------------
// 观察者（上帝视角）为主：纵览全狱、操控暂停/干预。
// WARDEN_MODE：在保留上帝视角感知的同时，让玩家化身为『典狱长』——可第一/第三人称在狱中走动、
// 与 NPC 对话审讯、下命令（作为玩家指令、优先级仅次于濒死），并在决策卡里没收/栽赃凶器、挑拨/调解仇怨。
const OBSERVER_MODE = true;   // 上帝观察：HUD/暂停/决策卡/终局常驻
const WARDEN_MODE = true;     // 典狱长可玩：生成化身 + 第一/第三人称视角 + 玩家能力
// 公开试玩默认开。加 ?research=1 恢复右侧研究面板、两脑对比、一键终局。
const PUBLIC_MODE = new URLSearchParams(location.search).get("research") !== "1";
if (PUBLIC_MODE) document.body.classList.add("public-mode");
const CAM_MODES = [
  ["god", PUBLIC_MODE ? "监区俯瞰" : "上帝俯瞰", "1", "拖拽旋转 · 滚轮缩放 · 俯瞰全狱"],
  ["spectate", "跟拍NPC", "2", "环绕跟拍选中的犯人 · 盯着它干活"],
  ["top", "正俯视", "3", "正上方俯视 · 平面图式纵览"],
];
if (WARDEN_MODE) {
  // 典狱长化身：第一人称走查 / 第三人称跟随（updateCamera 里两档分支本就在，这里把它们放回菜单）
  CAM_MODES.push(["fps", "第一人称", "4", "WASD 走动 · 拖拽转向 · 走近犯人按 E 审讯"]);
  CAM_MODES.push(["follow", "第三人称", "5", "跟随典狱长 · 拖拽转向 · 滚轮远近"]);
}
let camMode = "god";
const rig = { yaw: Math.PI, pitch: 0.42, dist: 8.5, topH: 46, fpsYaw: Math.PI, fpsPitch: -0.05 };
const camGoalPos = new THREE.Vector3();
const camGoalAim = new THREE.Vector3();
const camAim = new THREE.Vector3(0, 2, 0);

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// 上帝档用很窄的视场 + 拉远，逼近正交投影：参考图那种"等距切面沙盘"的观感全在这一步，
// 48° 透视会把远处的房间压斜，模型感立刻掉档。其余机位仍用常规视场（第一人称 26° 会晕）。
const GOD_FOV = 26, NORMAL_FOV = 48;

function setCamMode(id) {
  if (!CAM_MODES.some(([m]) => m === id)) return;
  const prev = camMode;
  camMode = id;
  controls.enabled = id === "god";
  const fov = id === "god" ? GOD_FOV : NORMAL_FOV;
  if (camera.fov !== fov) { camera.fov = fov; camera.updateProjectionMatrix(); }
  controls.minDistance = id === "god" ? 24 : 12;
  if (id === "god" && prev !== "god") {
    controls.target.copy(camAim);   // 从当前注视点接管，避免切回上帝档时视角跳走
    controls.update();
  }
  if (player) player.root.visible = true;
  renderCamBar();
}

function renderCamBar() {
  const bar = document.getElementById("cambar");
  if (!bar) return;
  bar.innerHTML = "";
  for (const [id, name, key, hint] of CAM_MODES) {
    const b = document.createElement("span");
    b.className = "cbtn" + (id === camMode ? " on" : "");
    b.innerHTML = `<b>${key}</b>${name}`;
    b.title = hint;
    b.onclick = () => setCamMode(id);
    bar.appendChild(b);
  }
  const tip = document.createElement("span");
  tip.className = "chint";
  tip.textContent = CAM_MODES.find(([m]) => m === camMode)[3];
  bar.appendChild(tip);
}

function spectateTarget() {
  if (selectedId && npcs.has(selectedId)) return npcs.get(selectedId);
  return npcs.values().next().value || null;
}

function orbitPos(out, cx, cy, cz, dist, pitch, yaw) {
  const h = Math.cos(pitch) * dist;
  out.set(cx - Math.sin(yaw) * h, cy + Math.sin(pitch) * dist, cz - Math.cos(yaw) * h);
  return out;
}

function updateCamera(dt) {
  if (camMode === "god") {
    // 观察者主视角：纯 OrbitControls 自由俯瞰，不跟随任何人（上帝之眼）
    camAim.copy(controls.target);
    controls.update();
    return;
  }
  if (camMode === "fps") {
    if (!player) return;
    const p = player.root.position;
    camera.position.set(p.x, 1.62, p.z);
    const cp = Math.cos(rig.fpsPitch);
    camAim.set(p.x + Math.sin(rig.fpsYaw) * cp,
               1.62 + Math.sin(rig.fpsPitch),
               p.z + Math.cos(rig.fpsYaw) * cp);
    camera.lookAt(camAim);       // 第一人称不做平滑，跟手才不晕
    player.heading = rig.fpsYaw;
    return;
  }
  if (camMode === "follow") {
    if (!player) return;
    const p = player.root.position;
    camGoalAim.set(p.x, 1.5, p.z);
    orbitPos(camGoalPos, p.x, 1.5, p.z, rig.dist, rig.pitch, rig.yaw);
  } else if (camMode === "spectate") {
    const e = spectateTarget();
    if (!e) return;
    rig.yaw += dt * 0.18;        // 缓慢环绕，像一台跟拍机
    const p = e.root.position;
    camGoalAim.set(p.x, 1.3, p.z);
    orbitPos(camGoalPos, p.x, 1.3, p.z, Math.min(rig.dist, 9), 0.34, rig.yaw);
  } else if (camMode === "top") {
    // 正俯视：以选中 NPC 为中心，没选就取全场中心 —— 平面图式纵览
    const t = spectateTarget();
    const p = t ? t.root.position : { x: 0, z: 0 };
    camGoalAim.set(p.x, 0, p.z);
    camGoalPos.set(p.x - Math.sin(rig.yaw) * 7, rig.topH, p.z - Math.cos(rig.yaw) * 7);
  }
  const k = Math.min(1, dt * 5);
  camera.position.lerp(camGoalPos, k);
  camAim.lerp(camGoalAim, k);
  camera.lookAt(camAim);
}

// 拖拽转向（自由档交给 OrbitControls，这里只管其余四档）
let dragPrev = null;
renderer.domElement.addEventListener("pointerdown", (ev) => {
  dragPrev = { x: ev.clientX, y: ev.clientY };
});
window.addEventListener("pointerup", () => { dragPrev = null; });
renderer.domElement.addEventListener("pointermove", (ev) => {
  if (!dragPrev) return;
  const dx = ev.clientX - dragPrev.x, dy = ev.clientY - dragPrev.y;
  dragPrev.x = ev.clientX; dragPrev.y = ev.clientY;
  if (camMode === "god") return;   // 上帝档交给 OrbitControls
  if (camMode === "fps") {         // 第一人称：拖拽/鼠标转头（之前只改了 yaw/pitch，fps 转不动）
    rig.fpsYaw -= dx * 0.006;
    rig.fpsPitch = clamp(rig.fpsPitch - dy * 0.004, -0.9, 0.9);
    return;
  }
  rig.yaw -= dx * 0.006;
  rig.pitch = clamp(rig.pitch + dy * 0.004, 0.06, 1.3);
});
renderer.domElement.addEventListener("wheel", (ev) => {
  if (camMode === "god") return;
  ev.preventDefault();
  const s = ev.deltaY > 0 ? 1.1 : 1 / 1.1;
  if (camMode === "top") rig.topH = clamp(rig.topH * s, 14, 70);
  else rig.dist = clamp(rig.dist * s, 3, 26);
}, { passive: false });

function loadGLB(url) {
  return new Promise((res, rej) => new GLTFLoader().load(url, res, undefined, rej));
}
// 底模：优先真人形（Xbot 素模，mixamo 骨骼），失败回退老机器人，绝不空场。
async function loadCharacter() {
  let gltf, base;
  try { gltf = await loadGLB("assets/Xbot.glb"); base = "xbot"; }
  catch (e) {
    console.warn("Xbot 人形底模加载失败，回退机器人：", e);
    gltf = await loadGLB("assets/RobotExpressive.glb"); base = "robot";
  }
  const box = new THREE.Box3().setFromObject(gltf.scene);
  const h = box.max.y - box.min.y;
  const scale = 1.85 / h;
  charTemplate = { scene: gltf.scene, animations: gltf.animations, scale, footY: box.min.y * scale, base };
  await loadNpcModels();
  await loadEnvModels();
  await Promise.all([loadCellRockTextures(), loadEveningSky()]);
  if (solitaryMetalMat && scene.environment) {
    solitaryMetalMat.envMap = scene.environment;
    solitaryMetalMat.envMapIntensity = 5.5;
  }
}

// —— 个别 NPC 的专属底模（各换各的 GLB）——
// key=NPC id（"police" 是例外：那是巡逻狱警与玩家扮演的典狱长共用的角色位，不属于任何 NPC）；
// url=模型文件；clip=把游戏逻辑动作名映射到该 GLB 里真实存在的片段名，
// 缺的动作回退到 _fallback。以后要给别人换模型，往这里加一条即可。
const NPC_MODEL = {
  laok: {
    url: "assets/hero.glb",
    keepLook: true,           // 显示模型自带的原始衣服贴图
    // 片段的根位移已由 frontend/tools/clean_glb_skeletons.py --in-place 烧进 glb，
    // 运行时不用再处理。动画不能跨骨架借用：骨骼名相同但静置姿势的轴向约定不同，
    // 拿通用底模的 walk 驱动这具骨架会让人物躺倒。
    //
    // 对齐步幅与位移，消除贴地平移：drunkwalk 原始步幅 2.90m/3.03s ≈ 0.96 m/s，
    // 乘归一化缩放 0.91 与体型 0.95 后约 0.83 m/s。配 pace=0.45（3.2×0.45≈1.44 m/s），
    // 需要动画放到 1.44/0.83≈1.74 倍；因 mixer 已被 gait(0.86) 整体放慢，
    // 这里单独给走路 1.74/0.86≈2.0，待机/打架仍保持他偏慢的节奏。
    clipTimeScale: { drunkwalk: 2.0 },
    clip: {
      Idle: "idle", Standing: "idle", Sitting: "idle", Sleep: "idle",
      Walking: "drunkwalk", Running: "drunkwalk",
      Punch: "fight", Fight: "fight",
      Dead: "dying", Dying: "dying", Die: "dying",
      _fallback: "idle",
    },
  },
  yanjing: {
    url: "assets/doctor.glb",
    keepLook: true,           // 自带白衣/口罩/帽子贴图
    // walking 原始步幅 1.60m/1.07s ≈ 1.50 m/s，乘归一化 1.045 与体型 1.06 后约 1.66 m/s，
    // 配 pace=0.52（3.2×0.52≈1.66）步频即对齐，gait 为 1.0 故不需再给片段调速。
    // 没有打架片段，Punch 会按 _fallback 落到 idle。
    clip: {
      Idle: "idle", Standing: "idle", Sitting: "idle", Sleep: "idle",
      Walking: "walking", Running: "walking",
      Dead: "dying", Dying: "dying", Die: "dying",
      _fallback: "idle",
    },
  },
  // 警服：巡逻狱警『铁面』和玩家扮演的典狱长共用这一具（不是 NPC，靠 buildPoliceAvatar 取用）。
  // 他们的移动速度由别处决定（巡逻滑行 2.3 m/s、玩家 6.0 m/s）且时快时慢，
  // 所以不写死片段倍速，改由 syncWalkTimeScale() 按每帧真实位移动态对齐步频。
  // walkSpeed=片段自带的行走速度：原始步幅 2.13m/0.83s ≈ 2.56 m/s，乘归一化 1.019 后约 2.61 m/s。
  police: {
    url: "assets/police.glb",
    keepLook: true,
    walkClip: "walk", walkSpeed: 2.61,
    clip: {
      Idle: "idle", Standing: "idle", Sitting: "idle", Sleep: "idle",
      Walking: "walk", Running: "walk",
      Dead: "dying", Dying: "dying", Die: "dying",
      _fallback: "idle",
    },
  },
  laochen: {
    url: "assets/laochen.glb",
    keepLook: true,           // 自带囚服/头部贴图，不套统一囚服色
    // walking 原始步幅 1.60m/1.17s ≈ 1.37 m/s，乘归一化 1.041 与体型 1.00 后约 1.43 m/s，
    // 配 pace=0.45（3.2×0.45=1.44）步频即对齐，gait 为 1.0 故不需再给片段调速。
    // 求人跪姿：SeekProtector（贴靠山·"凑到强者身边、求人罩着"）走 Kneel。
    // 这具骨架是 Z-up，clean_glb_skeletons.py 已按静置髋位判出竖直轴、
    // 只锁水平位移，所以 dying 的下坠和 kneeling 的低髋都保留着。
    // 没有打架片段，Punch 会按 _fallback 落到 idle。
    clip: {
      Idle: "idle", Standing: "idle", Sitting: "idle", Sleep: "idle",
      Walking: "walking", Running: "walking",
      Kneel: "kneeling",
      Dead: "dying", Dying: "dying", Die: "dying",
      _fallback: "idle",
    },
  },
  shitou: {
    url: "assets/balian.glb",
    keepLook: true,           // 自带光头/胡子/囚裤贴图，不套统一囚服色
    // 这具 MaleBruteA 原本攥着一把斧头（6664 面 + 5.4MB 贴图），已由
    // clean_glb_skeletons.py --drop-mesh blinn7 删掉。理由不是省体积：游戏里凶器只有
    // 毒药/匕首/电击枪三种、且只以决策卡文字呈现，场上没有武器模型；疤脸的任务链更是
    // weapon="" 空手开局、第二步才"去弄一把见血的家伙"。斧头常驻在手会同时废掉
    // "去弄凶器"这条剧情节拍和典狱长"没收凶器"这个操作（收缴完斧头还在手上）。
    // 因此 fight 是空手挥击——动作本身是为斧头编的，但读作"挥拳"并不违和。
    //
    // 对齐步幅与位移，消除贴地平移：walk 原始步幅 2.02m/1.43s ≈ 1.41 m/s，
    // 乘归一化 0.848（这具底模原始高 2.18m，是全场最大的一具）与体型 1.10 后约 1.32 m/s，
    // 再乘 gait 0.95 约 1.25 m/s，故 pace=0.39（3.2×0.39≈1.25）。
    // 步频已由 gait 对齐，不需再给片段单独调速。
    clip: {
      Idle: "idle", Standing: "idle", Sitting: "idle", Sleep: "idle",
      Walking: "walk", Running: "walk",
      Punch: "fight", Fight: "fight",
      Dead: "dying", Dying: "dying", Die: "dying",
      _fallback: "idle",
    },
  },
  xiaodou: {
    url: "assets/madman.glb",
    keepLook: true,
    // walk 原始步幅 1.42m/1.40s ≈ 1.02 m/s，乘归一化 1.044、体型 0.90、步态 1.22 后约 1.16 m/s，
    // 故 pace=0.36。他的 gait 已是 1.22（抽搐步偏快），不再额外给片段调速。
    clip: {
      Idle: "idle", Standing: "idle", Sitting: "idle", Sleep: "idle",
      Walking: "walk", Running: "walk",
      Dead: "dying", Dying: "dying", Die: "dying",
      _fallback: "idle",
    },
  },
};
async function loadNpcModels() {
  for (const [id, cfg] of Object.entries(NPC_MODEL)) {
    try {
      const gltf = await loadGLB(cfg.url);
      console.info(`[${id}] 专属底模片段：`, gltf.animations.map((a) => a.name).join(", "));
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const h = box.max.y - box.min.y;
      const scale = 1.85 / h;
      npcTemplates.set(id, {
        scene: gltf.scene, animations: gltf.animations,
        scale, footY: box.min.y * scale, base: "custom", clip: cfg.clip,
        keepLook: !!cfg.keepLook, clipTimeScale: cfg.clipTimeScale || null,
        walkClip: cfg.walkClip || null, walkSpeed: cfg.walkSpeed || 0,
      });
    } catch (e) {
      console.warn(`NPC 专属模型加载失败(${id})，回退通用底模：`, cfg.url, e);
    }
  }
}

// —— 人物差异化（A1）——
// Xbot 素模只有 idle/walk/run + 几个情绪 pose，没有 sit/punch 等；把代码里用的"逻辑动作名"
// 映射到该底模真有的最接近片段。robot 底模片段名与逻辑名一致，直通。
const XBOT_CLIP = {
  Idle: "idle", Walking: "walk", Running: "run", Standing: "idle",
  Sitting: "sad_pose", Sleep: "sad_pose", No: "headShake", Yes: "agree",
  Wave: "agree", ThumbsUp: "agree", Dance: "agree", Punch: "sneak_pose",
  Jump: "run", Hide: "sneak_pose", Kneel: "sad_pose",
};
function resolveClip(entry, name) {
  if (entry.clipMap) {                          // 专属底模：按自己的片段名映射，缺的逐级回退
    for (const cand of [entry.clipMap[name], entry.clipMap._fallback, name]) {
      if (cand && entry.actions[cand]) return cand;
    }
    return Object.keys(entry.actions)[0];
  }
  const base = entry.base || charTemplate.base;
  if (base === "xbot") {
    const m = XBOT_CLIP[name] || "idle";
    return entry.actions[m] ? m : (entry.actions.idle ? "idle" : name);
  }
  return entry.actions[name] ? name : (entry.actions.Idle ? "Idle" : name);
}

// 五名囚犯各自的辨识度：囚服色 + 体型(高矮/魁梧) + 步态(时间缩放) + 头部/身上签名配件。
// 同穿一套 mixamo 骨骼共享动画，靠这些差异一眼认出谁是谁。
const CHAR_SIG = {
  laochen: { color: 0x8f7c4c, scale: 1.00, girth: 1.00, gait: 1.00, pace: 0.45, acc: null }, // 老牙·老室友/越狱：自带外观，不加配件（平顶帽是按素模比例摆的，换模型会浮/穿）
  // pace=移动速度倍率（默认 1）。瘸子换了专属模型后按其步幅配速，见 NPC_MODEL.laok 的推导。
  laok:    { color: 0x5c6b4a, scale: 0.95, girth: 1.02, gait: 0.86, pace: 0.45, acc: null }, // 瘸子·走私/瘸腿：自带头部，不加配件
  // 疤脸·帮派头目：换成专属模型（MaleBruteA，本身就是魁梧光头带胡子的体型）。
  // girth 从 1.20 收回 1.00——横向拉伸会把自带贴图一起拉花，而"魁梧"已经在底模里了；
  // acc 从 "scar" 改成 null——那道刀疤是按素模比例摆在 y=1.50 的一张平面，
  // 换了自带头部的模型就会浮在脸外面（老牙的平顶帽踩过同一个坑）。
  shitou:  { color: 0x394050, scale: 1.10, girth: 1.00, gait: 0.95, pace: 0.39, acc: null },
  yanjing: { color: 0x9aa0a8, scale: 1.06, girth: 0.92, gait: 1.00, pace: 0.52, acc: null }, // 柳叶刀·黑医：自带帽子/口罩/白衣，不加配件
  xiaodou: { color: 0xb0a690, scale: 0.90, girth: 0.96, gait: 1.22, pace: 0.36, acc: null }, // 疯子·目击者：瘦小佝偻+抽搐步，自带头部
};
function accMat(color, rough = 0.9) {
  return new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0.0 });
}
// 各签名配件（root 空间，单位=米，头顶约 y=1.6；外层再按体型统一缩放）
function buildAcc(kind, color) {
  const g = new THREE.Group();
  if (kind === "capFlat") {
    const crown = new THREE.Mesh(new THREE.SphereGeometry(0.15, 12, 10, 0, Math.PI * 2, 0, Math.PI / 2), accMat(color, 0.8));
    crown.scale.set(1, 0.62, 1.12); crown.position.y = 1.60; g.add(crown);
    const brim = new THREE.Mesh(new THREE.CircleGeometry(0.19, 18), accMat(color, 0.8));
    brim.rotation.x = -Math.PI / 2 + 0.18; brim.position.set(0, 1.585, 0.15); g.add(brim);
  } else if (kind === "scar") {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.13), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
    m.position.set(0.10, 1.50, 0.135); m.rotation.z = 0.5; g.add(m);
  } else {
    return null;
  }
  return g;
}

function spawnNpc(n) {
  const root = new THREE.Group();
  const sig = CHAR_SIG[n.id] || { color: PERSONALITY_COLOR[n.personality] ?? 0x8a8f9a, scale: 1.0, girth: 1.0, gait: 1.0, acc: null };
  const tpl = npcTemplates.get(n.id) || charTemplate; // 个别 NPC 有专属 GLB（瘸子等）
  const model = cloneSkeleton(tpl.scene);
  const s = tpl.scale * sig.scale;
  model.scale.set(s * sig.girth, s, s * sig.girth);      // 体型：高矮 + 魁梧/清瘦
  model.position.y = -tpl.footY * sig.scale;
  const suit = new THREE.Color(sig.color);
  model.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true; o.frustumCulled = false;
      o.material = o.material.clone();       // 每人独立材质，才能各自上色
      if (!tpl.keepLook) {                   // 自带外观的模型不覆盖，其余套囚服色
        if (o.material.color) o.material.color.set(suit);  // 整体囚服色（素模无法单独分头/身）
        o.material.roughness = 0.9; o.material.metalness = 0.0;
      }
    }
  });
  root.add(model);
  // 签名配件（帽/发/疤/白大褂…）——外层 wrapper 按体型统一缩放，位置随之抬高/放大
  if (sig.acc) {
    const a = buildAcc(sig.acc, sig.accColor);
    if (a) { const w = new THREE.Group(); w.add(a); w.scale.setScalar(sig.scale); root.add(w); }
  }

  // 不给 NPC 加自带光源：点光源落在身体内部，写实模型（双面材质）会被从内部照亮成"发光体"。
  // 角色照明交给场景的半球光/主光/环境光；性格与被监视/被下令的状态由头顶气泡体现。

  root.position.set(n.pos[0], 0, n.pos[1]);
  root.userData.id = n.id;
  root.userData.hoverRoot = true;
  scene.add(root);
  npcRoots.push(root);

  const mixer = new THREE.AnimationMixer(model);
  mixer.timeScale = sig.gait || 1.0;         // 步态：瘸子放慢、疯子加快，各有节奏
  const actions = {};
  for (const clip of tpl.animations) {
    actions[clip.name] = mixer.clipAction(clip);
  }
  // 单片段调速：让走路的步频跟移动速度对齐（避免脚贴地平移），其余片段保持整体节奏
  for (const [name, ts] of Object.entries(tpl.clipTimeScale || {})) {
    if (actions[name]) actions[name].setEffectiveTimeScale(ts);
  }
  // 个人"工位"偏移（黄金角散开），避免多人去同一设施时叠在一个点
  const ang = npcCount++ * 2.399963;
  const entry = {
    root, mixer, actions, current: null, base: tpl.base,
    clipMap: tpl.clip || null, pace: sig.pace ?? 1,
    // 走路片段名 + 它的标定倍速。追赶后端时要把移动速度和步频同倍放大，脚才不会贴地滑。
    walkAct: (tpl.clip && tpl.clip.Walking) || "Walking",
    walkTS: (tpl.clipTimeScale || {})[(tpl.clip && tpl.clip.Walking) || "Walking"] || 1,
    slot: { x: Math.cos(ang) * 1.4, z: Math.sin(ang) * 1.4 },
    dest: new THREE.Vector3(n.pos[0], 0, n.pos[1]),      // 目标设施位置（恒速滑行终点）
    serverPos: new THREE.Vector3(n.pos[0], 0, n.pos[1]), // 后端权威位置（用于纠偏）
    heading: 0, moving: false, commanded: false,
    cell: n.cell ?? -1, station: null, going: false, actionState: "",
    label: makeBubble("npc-bubble"),
  };
  npcs.set(n.id, entry);
  if (selectedId === null) selectedId = n.id;
  playAction(entry, "Idle");
  return entry;
}

function playAction(entry, name) {
  const clip = resolveClip(entry, name);       // 逻辑动作名 → 该底模真实片段名
  if (entry.current === clip) return;
  const next = entry.actions[clip];
  if (!next) return;
  const prev = entry.current ? entry.actions[entry.current] : null;
  next.reset();
  next.enabled = true;
  next.setEffectiveWeight(1);
  next.fadeIn(0.25).play();
  if (prev && prev !== next) prev.fadeOut(0.25);
  entry.current = clip;
}

// 步频对齐：按这一帧真实走出的速度给走路片段调速，脚就不会贴地平移。
// 巡逻者会追赶后端位置、玩家速度也远快于片段自带步频，故只能逐帧动态对齐（上限内尽量贴合）。
function syncWalkTimeScale(entry, speed) {
  if (!entry.walkClip || !entry.walkSpeed) return;
  const a = entry.actions[entry.walkClip];
  if (a) a.setEffectiveTimeScale(clamp(speed / entry.walkSpeed, 0.65, 2.4));
}

// 警服化身：巡逻狱警和玩家扮演的典狱长共用这一具模型（自带贴图，不再上色）。
// 返回 null 表示 police.glb 没加载成功，调用方各自回退到原来的方块狱警/金色底模。
function buildPoliceAvatar(scaleMul = 1) {
  const tpl = npcTemplates.get("police");
  if (!tpl) return null;
  const model = cloneSkeleton(tpl.scene);
  model.scale.setScalar(tpl.scale * scaleMul);
  model.position.y = -tpl.footY * scaleMul;
  model.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; } });
  const mixer = new THREE.AnimationMixer(model);
  const actions = {};
  for (const clip of tpl.animations) actions[clip.name] = mixer.clipAction(clip);
  // police.glb 只有 idle/walk/dying，没有敬礼片段。与其为一个 2 秒的动作再走一遍
  // Mixamo→Blender→压缩的流程，这里直接抓右臂骨骼，在 idle 之上叠一个手动姿态
  // （见 applySalutePose）。GLTFLoader 会把 mixamorig:RightArm 里的冒号去掉。
  const salute = {};
  model.traverse((o) => {
    if (!o.isBone) return;
    if (/RightArm$/.test(o.name)) salute.upper = o;
    else if (/RightForeArm$/.test(o.name)) salute.fore = o;
    else if (/RightHand$/.test(o.name)) salute.hand = o;
  });
  return { model, rig: { mixer, actions, current: null, base: tpl.base, clipMap: tpl.clip,
                         walkClip: tpl.walkClip, walkSpeed: tpl.walkSpeed * scaleMul,
                         saluteBones: salute.upper ? salute : null, saluteW: 0 } };
}

// 敬礼姿态：在 mixer 写完这一帧的骨骼之后，把右臂往"手贴帽檐"的方向压过去。
// w 是权重(0~1)，用 slerp 混进去，所以起手/收手是平滑的，而且不影响其他骨骼——
// 上身仍在播 idle 的呼吸，看着不像一具突然定格的木偶。
const _sq = new THREE.Quaternion();
// 这三组角度不是手填的：Mixamo 骨骼静止朝向沿手臂轴，局部空间里哪个轴是"外展"并不直观，
// 手填的结果是手停在胸口。改成把"食指指尖贴帽檐、手腕在眉际外下方、肘部朝外"写成距离代价，
// 在浏览器里随机撒点加爬山解出来（见 tools/_probe_solve_salute.js），指尖残差 0cm。
// 只约束手腕是不够的——那样会解出举手过头顶、手指朝天，所以手掌朝向也一起解。
const _SALUTE_UPPER = new THREE.Euler(-1.736, -2.077, 1.56);
const _SALUTE_FORE = new THREE.Euler(2.836, 2.437, 2.897);
const _SALUTE_HAND = new THREE.Euler(2.569, 2.488, 0.817);
function applySalutePose(rig, dt, on) {
  if (!rig || !rig.saluteBones) return;
  const target = on ? 1 : 0;
  // 起手比收手快一点：行礼要利落，放下可以松
  const k = Math.min(1, dt * (on ? 9 : 5));
  rig.saluteW += (target - rig.saluteW) * k;
  if (rig.saluteW < 0.01) return;
  const b = rig.saluteBones;
  b.upper.quaternion.slerp(_sq.setFromEuler(_SALUTE_UPPER), rig.saluteW);
  if (b.fore) b.fore.quaternion.slerp(_sq.setFromEuler(_SALUTE_FORE), rig.saluteW);
  if (b.hand) b.hand.quaternion.slerp(_sq.setFromEuler(_SALUTE_HAND), rig.saluteW);
}

// ---------------------------------------------------------------------------
// 玩家主角（金色，可 WASD 操控）
// ---------------------------------------------------------------------------
function spawnPlayer() {
  const root = new THREE.Group();
  // 典狱长穿 police.glb 自带的警服；只有模型加载失败才回退到金色素模。
  // 比巡逻狱警高出一头压场，脚下再留一圈金环——和穿同一套警服的铁面区分开，
  // 第三人称下这圈环就是"这个人是你"的标识。
  const av = buildPoliceAvatar(1.08);
  let anim;
  if (av) {
    root.add(av.model);
    anim = av.rig;
  } else {
    const model = cloneSkeleton(charTemplate.scene);
    model.scale.setScalar(charTemplate.scale);
    model.position.y = -charTemplate.footY;
    const gold = new THREE.Color(0xffd23f);
    model.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = true; o.frustumCulled = false;
        o.material = o.material.clone();
        if (o.material.color) o.material.color.lerp(gold, 0.55);
        o.material.metalness = 0.35; o.material.roughness = 0.5;
      }
    });
    root.add(model);
    const mixer = new THREE.AnimationMixer(model);
    const actions = {};
    for (const clip of charTemplate.animations) actions[clip.name] = mixer.clipAction(clip);
    anim = { mixer, actions, current: null, base: charTemplate.base, clipMap: null,
             walkClip: null, walkSpeed: 0 };
  }
  // 典狱长脚底光圈按需求删除（视觉太满）。身份由头顶气泡/📢标记体现，不再靠地面光环。
  // 典狱长从监区入口一带上场；犯人剧情模式仍出生在牢房(cell -13,8)、和老牙同室
  if (WARDEN_MODE) root.position.set(0, 0, 16);
  else root.position.set(-13, 0, 9.5);
  root.userData.hoverRoot = true;
  scene.add(root);
  player = { root, ...anim, heading: Math.PI, label: makeBubble("npc-bubble") };
  player.label.classList.add("cmded");
  playAction(player, "Idle");
}

const camFwd = new THREE.Vector3();
const camRight = new THREE.Vector3();
const moveVec = new THREE.Vector3();
// 典狱长"前往解决"：不瞬移，设一个目标点让 movePlayer 每帧朝它走；玩家一按 WASD 立刻接管
let walkTarget = null;      // {x, z, npc_id} 或 null
let walkStuck = 0;          // 连续走不动的帧数，卡死太久才兜底瞬移
const WALK_ARRIVE = 1.3;    // 离目标这么近就算到了
const WALK_STUCK_MAX = 150; // ~2.5s 走不动 → 认输，直接落到目标点
function movePlayer(dt) {
  if (!player) return;
  // 典狱长：只有第一/第三人称档才用 WASD 走化身；上帝/跟拍/俯视档用面板与决策卡操控，化身不乱飘。
  if (WARDEN_MODE && camMode !== "fps" && camMode !== "follow") {
    player.mixer.update(dt);
    return;
  }
  const pos = player.root.position;
  // 镜头朝向在地面上的投影 → WASD 始终相对当前视角（五档视角通用）
  camera.getWorldDirection(camFwd);
  camFwd.y = 0;
  if (camFwd.lengthSq() < 1e-6) camFwd.set(Math.sin(rig.yaw), 0, Math.cos(rig.yaw));
  camFwd.normalize();
  camRight.set(-camFwd.z, 0, camFwd.x);   // right = fwd × up，之前反了导致 A/D 相反
  moveVec.set(0, 0, 0);
  if (keys["w"] || keys["arrowup"]) moveVec.add(camFwd);
  if (keys["s"] || keys["arrowdown"]) moveVec.sub(camFwd);
  if (keys["d"] || keys["arrowright"]) moveVec.add(camRight);
  if (keys["a"] || keys["arrowleft"]) moveVec.sub(camRight);
  const manual = moveVec.lengthSq() > 1e-6;
  if (manual) walkTarget = null;        // 玩家一动就接管，取消自动前往
  // 自动前往目标点：朝目标直线走（下面走一步碰撞滑动），到跟前或卡死才停
  if (!manual && walkTarget && !chatOpen && !occupying) {
    const dx = walkTarget.x - pos.x, dz = walkTarget.z - pos.z;
    const d = Math.hypot(dx, dz);
    if (d <= WALK_ARRIVE) { walkTarget = null; walkStuck = 0; }
    else { moveVec.set(dx / d, 0, dz / d); }
  }
  const wantMove = moveVec.lengthSq() > 1e-6 && !chatOpen;
  if (occupying) {
    if (wantMove) standUp();          // 一动就自动起身，不用先按 F
    else { player.mixer.update(dt); return; }
  }
  const moving = wantMove;
  if (moving) {
    moveVec.normalize();
    const dist = PLAYER_SPEED * dt;
    const steps = Math.max(1, Math.ceil(dist / COLLIDE_STEP));
    const inv = dist / steps;
    const bx = pos.x, bz = pos.z;      // 走之前的位置，用来判断自动前往有没有卡住
    for (let i = 0; i < steps; i++) {
      pos.x = clamp(pos.x + moveVec.x * inv, -PLAYER_LIMIT, PLAYER_LIMIT);
      pos.z = clamp(pos.z + moveVec.z * inv, -PLAYER_LIMIT, PLAYER_LIMIT);
      if (playerBounds) {
        pos.x = clamp(pos.x, playerBounds.minX, playerBounds.maxX);
        pos.z = clamp(pos.z, playerBounds.minZ, playerBounds.maxZ);
      }
      resolveCollision(pos, true);      // 玩家：连家具一起挡，别再从桌椅床里穿过去
    }
    player.heading = Math.atan2(moveVec.x, moveVec.z);
    // 自动前往时若被墙卡住(几乎没挪动)累计够久，直接落到目标点兜底，别永远抵在墙角
    if (walkTarget) {
      const moved = Math.hypot(pos.x - bx, pos.z - bz);
      if (moved < dist * 0.25) walkStuck++;
      else walkStuck = 0;
      if (walkStuck >= WALK_STUCK_MAX) {
        pos.set(walkTarget.x, 0, walkTarget.z);
        resolveCollision(pos, true);
        walkTarget = null; walkStuck = 0;
      }
    }
  } else {
    resolveCollision(pos, true);        // 站着也挤出墙/家具，避免卡在里面
  }
  let dh = (player.heading - player.root.rotation.y + Math.PI * 3) % (Math.PI * 2) - Math.PI;
  player.root.rotation.y += dh * Math.min(1, dt * 10);
  playAction(player, moving ? "Walking" : "Idle");
  syncWalkTimeScale(player, moving ? PLAYER_SPEED : 0);
  player.mixer.update(dt);
}

// 气泡距离 LOD：离镜头越远越小越淡、离屏幕中心越远越淡——把注意力收拢到镜头附近，
// 减轻"满屏对话框"的信息过载。选中/正在说话的气泡(prio>0)永远满级，不缩不淡。
function applyBubbleLod(el, pos, sx, sy, prio) {
  el.style.transformOrigin = "50% 100%";               // 从底部锚点缩放，气泡尾巴不移位
  if (prio > 0) { el.style.opacity = "1"; el.style.transform = "translate(-50%,-100%) scale(1)"; return; }
  const wd = camera.position.distanceTo(pos);
  let k = 1 - (wd - 22) / 36 * 0.42;                    // 22→1.0，58→0.58
  k = Math.max(0.58, Math.min(1, k));
  let op = 1 - (wd - 26) / 34 * 0.6;                    // 26→1.0，60→0.4
  const cx = window.innerWidth / 2, cy = window.innerHeight / 2;
  const sd = Math.min(1, Math.hypot(sx - cx, sy - cy) / Math.hypot(cx, cy));
  op -= sd * 0.22;                                      // 离屏幕中心越远再淡一点
  op = Math.max(0.34, Math.min(1, op));
  el.style.opacity = op.toFixed(2);
  el.style.transform = `translate(-50%,-100%) scale(${k.toFixed(3)})`;
}

// 头顶气泡屏幕空间防重叠：把这一帧所有可见气泡按远近排序，近的(屏幕靠下)先占位，
// 远的若与已占位的框重叠就上移错开——避免两人凑近说话时对话框互相遮挡看不清。
const bubbleBoxes = [];
function declutterBubbles(boxes) {
  if (boxes.length < 2) return;
  for (const b of boxes) {
    b.w = b.el.offsetWidth || 130;
    b.h = b.el.offsetHeight || 46;
    b.ny = b.y;
  }
  // 屏幕靠下(y 大)=离镜头近，优先保持原位；靠上的(远)让路上移。高优先级(选中/说话)也先占位。
  boxes.sort((a, b) => (b.prio - a.prio) || (b.y - a.y));
  const GAP = 6;
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    let guard = 0, moved = true;
    while (moved && guard++ < 16) {
      moved = false;
      for (let j = 0; j < i; j++) {
        const p = boxes[j];
        if (Math.abs(b.x - p.x) >= (b.w + p.w) / 2 - 6) continue;      // 横向不相交
        const bTop = b.ny - b.h, pTop = p.ny - p.h;
        if (b.ny <= pTop - GAP || bTop >= p.ny + GAP) continue;        // 纵向不相交
        b.ny = pTop - GAP;                                            // 上移到该框正上方
        moved = true;
      }
    }
    if (b.ny < b.h + 4) b.ny = b.h + 4;                                // 别飞出屏幕顶
    b.el.style.top = `${b.ny}px`;
  }
}

// 圆 vs AABB 的最小推出：撞墙时沿穿透最浅的一轴弹开，贴墙走不会卡住
const BODY_R = 0.42;
// 走位去抖死区：到位后进入"驻留"，不再追 dest 的细小变化；目标移开超过 NPC_GO 才重新起步。
// 治"睡觉/歇着原地抽搐"——原来只要 dist>0.06 就挪一步，后端位置每拍微抖就把人震得发颤。
const NPC_STOP = 0.14;
const NPC_GO = 0.42;
// 必须坐/躺到对应家具上的动作。走到空地就播坐躺，看起来像躺地上、院子里吃饭。
const STATION_OF = {
  Sleep: { type: "lie", kind: "sleep", own: true },
  Rest:  { type: "sit", kind: "sleep", own: true },
  Pray:  { type: "sit", kind: "sleep", own: true },
  Hide:  { type: "sit", kind: "sleep", own: true },
  Eat:   { type: "sit", kind: "eat" },
  Visit: { type: "sit", kind: "visit" },
  Study: { type: "sit", kind: "class" },
  Read:  { type: "sit", kind: "library" },
  Investigate: { type: "sit", kind: "library" },
  Chess: { type: "sit", kind: "commons" },
  Music: { type: "sit", kind: "commons" },
  Tattoo: { type: "sit", kind: "commons" },
  Cure:  { type: "lie", kind: "cure" },
};
// 室内动作：不要加 1.4m 工位偏移——隔间太窄，一偏就坐到走廊/院子里。
const INDOOR_GOALS = new Set([
  "Eat", "Drink", "Sleep", "Cure", "Rest", "Study", "Labor", "KitchenDuty",
  "Laundry", "Clean", "Workout", "Shower", "Read", "Visit", "Chess", "Music",
  "Pray", "Barter", "Tattoo", "Investigate", "GetWeapon", "Snitch", "Hide",
  "ReviewFiles", "Phone",
]);
// 追赶死区与封顶：见 NPC 主循环里 catchUp 的推导。
const CATCH_SLACK = 3.0;   // 视觉落后后端权威位置这么多米之后才开始提速
const CATCH_MAX = 2.6;     // 提速倍率上限，再高步频就假了
function _pushOut(pos, list) {
  for (const b of list) {
    if (b.off) continue;              // 开着的牢门：这条碰撞盒暂时不存在
    if (pos.x < b.minX - BODY_R || pos.x > b.maxX + BODY_R) continue;
    if (pos.z < b.minZ - BODY_R || pos.z > b.maxZ + BODY_R) continue;
    const pushL = pos.x - (b.minX - BODY_R);
    const pushR = (b.maxX + BODY_R) - pos.x;
    const pushB = pos.z - (b.minZ - BODY_R);
    const pushF = (b.maxZ + BODY_R) - pos.z;
    const m = Math.min(pushL, pushR, pushB, pushF);
    if (m === pushL) pos.x = b.minX - BODY_R;
    else if (m === pushR) pos.x = b.maxX + BODY_R;
    else if (m === pushB) pos.z = b.minZ - BODY_R;
    else pos.z = b.maxZ + BODY_R;
  }
}
// withProps=true 时把家具也算上：只有玩家化身传 true——NPC 位置是后端权威，
// 拿家具去顶只会前后端拉橡皮筋、还会把睡觉/坐下的人从床/凳上弹开。
function resolveCollision(pos, withProps = false) {
  _pushOut(pos, blockers);
  if (withProps) _pushOut(pos, propBlockers);
}

// A2：从 (ax,az) 到 (bx,bz) 的直线上是否被墙体挡住（沿线采样查 blockers AABB）。
// 开场白 NPC 只在"有清晰直线通路"时才走过来，否则原地说话——不再滑行穿墙。
function segmentBlocked(ax, az, bx, bz) {
  if (!blockers || !blockers.length) return false;
  const dx = bx - ax, dz = bz - az, len = Math.hypot(dx, dz);
  const steps = Math.max(1, Math.ceil(len / 0.6));
  for (let i = 1; i <= steps; i++) {
    const t = i / steps, x = ax + dx * t, z = az + dz * t;
    for (const b of blockers) {
      if (b.off) continue;
      if (x > b.minX - 0.3 && x < b.maxX + 0.3 && z > b.minZ - 0.3 && z < b.maxZ + 0.3) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// 走位寻路：A* 网格寻路 + 路径拉直，让 NPC / 狱警绕开墙体而不是直线滑行穿墙。
//   · 从 blockers（世界系墙体 AABB，建场时填好）烘一张可走性网格；
//   · 目标可直达（视线无遮挡）时不寻路，直接走——省算力、贴目标更顺；
//   · 只有被墙挡住时才 A*，再用视线拉直成平滑对角路径；
//   · 找不到路（目标被围死）就回落直线，宁可短暂穿墙也绝不把人定住。
// 纯前端渲染层的修正：后端位置仍是游戏逻辑权威，这里只让"看起来"不穿墙。
// ---------------------------------------------------------------------------
// cell/inflate 调细：原来 cell=1.3、inflate=0.67，把 1.5m 门洞的可走净宽压到 ~0.16m，
// A* 判定门被堵死→寻路失败→回落直线→穿墙(且卡墙抽搐)。细网格(0.6)+小膨胀(0.3)让门洞净宽
// 回到 ~0.9m(>身宽 0.84)，人能正常穿门；贴墙的零头交给 resolveCollision 兜底弹出，不穿模。
const NAV = { min: -33, max: 33, cell: 0.6, cols: 0, grid: null, blockersLen: -1,
              inflate: 0.30 };
NAV.cols = Math.ceil((NAV.max - NAV.min) / NAV.cell);
const NAV_DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];

function navCellBlocked(wx, wz) {
  for (const b of blockers) {
    if (b.off) continue;
    if (wx > b.minX - NAV.inflate && wx < b.maxX + NAV.inflate &&
        wz > b.minZ - NAV.inflate && wz < b.maxZ + NAV.inflate) return true;
  }
  for (const b of propBlockers) {       // 只有 nav 标记的大件（院子桌台）参与 NPC 绕行
    if (!b.nav) continue;
    if (wx > b.minX - NAV.inflate && wx < b.maxX + NAV.inflate &&
        wz > b.minZ - NAV.inflate && wz < b.maxZ + NAV.inflate) return true;
  }
  return false;
}
function navBuild() {
  const n = NAV.cols, g = new Uint8Array(n * n);
  for (let cx = 0; cx < n; cx++) {
    const wx = NAV.min + (cx + 0.5) * NAV.cell;
    for (let cz = 0; cz < n; cz++) {
      const wz = NAV.min + (cz + 0.5) * NAV.cell;
      g[cx * n + cz] = navCellBlocked(wx, wz) ? 1 : 0;
    }
  }
  NAV.grid = g; NAV.blockersLen = blockers.length; NAV.dirty = false;
}
function navReady() {
  if (!blockers.length) return false;
  // 墙加载完/新房子建出来/牢门上锁开锁（数量不变但通行性变了，靠 dirty 标记）时重烘
  if (!NAV.grid || NAV.blockersLen !== blockers.length || NAV.dirty) navBuild();
  return true;
}
function navCX(w) { return Math.max(0, Math.min(NAV.cols - 1, Math.floor((w - NAV.min) / NAV.cell))); }
function navFree(cx, cz) { return NAV.grid[cx * NAV.cols + cz] === 0; }
function navCenterX(cx) { return NAV.min + (cx + 0.5) * NAV.cell; }
function navNearestFree(cx, cz) {
  if (navFree(cx, cz)) return [cx, cz];
  for (let r = 1; r <= 7; r++) {
    for (let dx = -r; dx <= r; dx++) for (let dz = -r; dz <= r; dz++) {
      if (Math.max(Math.abs(dx), Math.abs(dz)) !== r) continue;
      const nx = cx + dx, nz = cz + dz;
      if (nx >= 0 && nz >= 0 && nx < NAV.cols && nz < NAV.cols && navFree(nx, nz)) return [nx, nz];
    }
  }
  return null;
}
// A*：返回世界系路点数组（末点为真实终点），失败返回 null。
function navPath(ax, az, bx, bz) {
  if (!navReady()) return null;
  const n = NAV.cols;
  const s = navNearestFree(navCX(ax), navCX(az));
  const g = navNearestFree(navCX(bx), navCX(bz));
  if (!s || !g) return null;
  const si = s[0] * n + s[1], gi = g[0] * n + g[1];
  if (si === gi) return [new THREE.Vector3(bx, 0, bz)];
  const gScore = new Float32Array(n * n).fill(Infinity);
  const fScore = new Float32Array(n * n).fill(Infinity);
  const came = new Map(), inOpen = new Uint8Array(n * n), open = [si];
  gScore[si] = 0; fScore[si] = Math.hypot(s[0] - g[0], s[1] - g[1]); inOpen[si] = 1;
  let guard = 0;
  while (open.length && guard++ < 30000) {
    let bi = 0, bf = Infinity;
    for (let k = 0; k < open.length; k++) if (fScore[open[k]] < bf) { bf = fScore[open[k]]; bi = k; }
    const cur = open.splice(bi, 1)[0]; inOpen[cur] = 0;
    if (cur === gi) {
      const cells = [cur]; let c = cur;
      while (came.has(c)) { c = came.get(c); cells.push(c); }
      cells.reverse();
      return navSmooth(cells, ax, az, bx, bz);
    }
    const cx = (cur / n) | 0, cz = cur % n;
    for (const [dx, dz] of NAV_DIRS) {
      const nx = cx + dx, nz = cz + dz;
      if (nx < 0 || nz < 0 || nx >= n || nz >= n || !navFree(nx, nz)) continue;
      if (dx !== 0 && dz !== 0 && (!navFree(cx + dx, cz) || !navFree(cx, cz + dz))) continue; // 不切墙角
      const ni = nx * n + nz, t = gScore[cur] + (dx !== 0 && dz !== 0 ? 1.4142 : 1);
      if (t < gScore[ni]) {
        came.set(ni, cur); gScore[ni] = t;
        fScore[ni] = t + Math.hypot(nx - g[0], nz - g[1]);
        if (!inOpen[ni]) { inOpen[ni] = 1; open.push(ni); }
      }
    }
  }
  return null;
}
// 视线拉直：能直达就跳过中间格，得到平滑对角路径（首用真实起点、末用真实终点）。
function navSmooth(cells, ax, az, bx, bz) {
  const n = NAV.cols;
  const pts = cells.map((i) => [navCenterX((i / n) | 0), navCenterX(i % n)]);
  const out = []; let anX = ax, anZ = az, i = 0;
  while (i < pts.length) {
    let j = i;
    for (let k = pts.length - 1; k >= i; k--) {
      const p = (k === pts.length - 1) ? [bx, bz] : pts[k];
      if (!segmentBlocked(anX, anZ, p[0], p[1])) { j = k; break; }
    }
    const p = (j === pts.length - 1) ? [bx, bz] : pts[j];
    out.push(new THREE.Vector3(p[0], 0, p[1]));
    anX = p[0]; anZ = p[1];
    if (j === pts.length - 1) break;
    i = j + 1;
  }
  if (!out.length) out.push(new THREE.Vector3(bx, 0, bz));
  return out;
}
// 给实体 e 求"这一帧该朝哪走"：直达走直线，否则沿缓存的 A* 路径逐点推进。
function navSteer(e, destX, destZ) {
  const px = e.root.position.x, pz = e.root.position.z;
  if (!blockers.length || !segmentBlocked(px, pz, destX, destZ)) {
    e._path = null; e._pathKey = null;
    return [destX, destZ];
  }
  const key = `${destX.toFixed(1)},${destZ.toFixed(1)}`;
  if (e._pathKey !== key || !e._path) {
    e._path = navPath(px, pz, destX, destZ); e._pathKey = key;
  }
  if (!e._path || !e._path.length) return [destX, destZ];   // 回落直线
  while (e._path.length > 1 && Math.hypot(e._path[0].x - px, e._path[0].z - pz) < 0.55)
    e._path.shift();
  const w = e._path[0];
  return [w.x, w.z];
}

// ---------------------------------------------------------------------------
// 室内互动：坐下 / 躺下 / 捡起 / 翻查 / 开关灯
// ---------------------------------------------------------------------------
const ITEM_DESC = {
  toothbrush: "老牙给的磨尖牙刷，柄尾磨成了尖片——能撬开牢房那把简单的锁。别让管教瞧见。",
  alcohol: "医务室黑医柳叶刀弄来的一瓶医用酒精——牢里的硬通货，瘸子最认这个。",
  sedative: "瘸子那儿换来的强效镇静剂。拌进肉里，能把嗅觉再灵的狗也放倒。（走近哮天按 G）",
  keycard: "典狱长办公室的备用磁卡，瘸子的压箱底货。终局那扇门，就靠它。",
  pension_list: "一份被克扣的抚恤金名单，末尾签着典狱长的名字。铁面看到这个会破防——收买他的筹码。",
  cigs: "牢里的硬通货，一根换一顿饭，一包换一次人情。",
  kettle: "水房的铁皮水壶，壶底一层水垢。装满能顶半天。",
  pills: "半瓶止痛片，标签被撕了，只剩生产批号。",
  textbook: "《汉语基础》，扉页写着上一个主人的名字，划掉了。",
  hammer: "苦役区的旧锤子，木柄被汗浸得发黑。带出车间是要挨罚的。",
  dumbbell: "一只哑铃，另一只不知被谁拿去当了枕头。",
  soap: "用剩半块的肥皂，边缘刻了道浅浅的印子。",
  book: "一本没有封皮的旧书，纸页脆得一翻就掉渣。",
  uniform: "刚洗过的囚服，还带着潮气和皂角味。",
  cigs: "监狱里的硬通货。一根烟能换一顿饭，一包能换一次人情。",
  keys: "一串钥匙，铜牌上刻着编号。你不该拿的——但你拿了。",
  guitar: "掉漆的木吉他，六根弦断了两根，剩下的还能响。",
};
const inventory = new Map();        // item -> {name, icon, n}
let focusAct = null;                // 当前就近可互动物
let occupying = null;               // 正在坐/躺：{entry, type}
const ACT_RANGE = 2.6;

function actVerb(it) {
  if (it.type === "sit") return "坐下";
  if (it.type === "lie") return "躺下";
  if (it.type === "pickup") return "捡起";
  if (it.type === "search") return "翻查";
  if (it.type === "lamp") return it.lamp.on ? "关灯" : "开灯";
  if (it.type === "celllock") return it.cell.locked ? "开锁" : "上锁";
  if (it.type === "solitarylock") return it.door.locked ? "开锁" : "上锁";
  return "使用";
}

// 每帧找出离主角最近、且大致在视线前方的可互动物
const _ap = new THREE.Vector3();
const _av = new THREE.Vector3();
function updateFocusAct() {
  const hint = document.getElementById("acthint");
  if (!player || chatOpen) { hint.style.display = "none"; focusAct = null; return; }
  if (occupying) {
    focusAct = occupying.entry;
    hint.innerHTML = `按 <b>F</b> 起身（或直接走动）`;
    player.root.getWorldPosition(_ap); _ap.y += 1.4;
    projectHint(hint, _ap);
    return;
  }
  const pp = player.root.position;
  let best = null, bd = ACT_RANGE * ACT_RANGE;
  camera.getWorldDirection(_av); _av.y = 0; _av.normalize();
  for (const it of interactables) {
    if (it.taken) continue;
    if (it.station && it.station.claimed && it.station.claimed !== "player") continue;
    it.obj.getWorldPosition(_ap);
    const d2 = _ap.distanceToSquared(pp);
    if (d2 >= bd) continue;
    bd = d2; best = it;
  }
  focusAct = best;
  if (!best) { hint.style.display = "none"; return; }
  best.obj.getWorldPosition(_ap); _ap.y += 0.9;
  hint.innerHTML = `${best.icon} ${best.name} · 按 <b>F</b> ${actVerb(best)}`;
  projectHint(hint, _ap);
}
function projectHint(el, worldVec) {
  tmp.copy(worldVec).project(camera);
  el.style.left = `${(tmp.x * 0.5 + 0.5) * window.innerWidth}px`;
  el.style.top = `${(-tmp.y * 0.5 + 0.5) * window.innerHeight}px`;
  el.style.display = tmp.z < 1 ? "block" : "none";
}

const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
function anchorYaw(anchor) {
  anchor.getWorldQuaternion(_q);
  _e.setFromQuaternion(_q, "YXZ");
  return _e.y;
}

function doInteract(it) {
  if (!player || !it) return;
  if (occupying) { standUp(); return; }
  if (it.type === "sit" || it.type === "lie") {
    const p = worldPos(it.anchor);
    const yaw = anchorYaw(it.anchor);
    player.exitPos = player.root.position.clone();
    player.heading = yaw;
    player.root.rotation.order = "YXZ";
    if (it.type === "lie") {
      // 模型枢轴在脚底：先转到床的走向，再整体后仰 90°，身体就顺着床铺展开
      player.root.position.copy(p);
      player.root.rotation.set(-Math.PI / 2, yaw, 0);
    } else {
      // 坐姿动画自带屈膝，根节点保持在地面附近，臀部才落在凳面上
      player.root.position.set(p.x, p.y - 0.45, p.z);
      player.root.rotation.set(0, yaw, 0);
    }
    occupying = { entry: it, type: it.type };
    if (it.station) it.station.claimed = "player";
    playAction(player, it.type === "lie" ? "Idle" : "Sitting");
    showToast(it.type === "lie" ? `你在${it.name}上躺下了。（F 起身）` : `你在${it.name}上坐下了。（F 起身）`);
  } else if (it.type === "pickup") {
    it.taken = true;
    it.obj.visible = false;
    const rec = inventory.get(it.item) || { name: it.name, icon: it.icon, n: 0 };
    rec.n += 1; inventory.set(it.item, rec);
    renderBag();
    showToast(`${it.icon} 捡起了「${it.name}」——${ITEM_DESC[it.item] || ""}`);
  } else if (it.type === "take") {
    // 后端权威的物资：你拿走一份，NPC 这顿就真的少一份（抢的是同一批）
    if (it.disabled) { showToast(`${it.icon} ${it.name}——已经没了`); return; }
    sendMsg({ type: "take_object", id: it.objId });
  } else if (it.type === "search") {
    showToast(`${it.icon} 翻查${it.name}：${it.found}`);
  } else if (it.type === "lamp") {
    const L = it.lamp;
    L.on = !L.on; L.manual = true;
    showToast(L.on ? "💡 灯亮了。" : "🌑 灯灭了。");
  } else if (it.type === "celllock") {
    // 牢门的锁是典狱长专属职权：后端记账、广播新状态，锁上之后连寻路网格都封了
    if (!WARDEN_MODE) { showToast("🔑 你没有这串钥匙。"); return; }
    const c = it.cell;
    sendMsg({ type: "cell_door", slot: c.slot, locked: !c.locked });
    showToast(c.locked ? `🔓 拧开${it.name}的锁。` : `🔒 锁上${it.name}——他今晚出不来了。`);
  } else if (it.type === "solitarylock") {
    // 禁闭室的门同牢门：典狱长专属，后端记账并把 nav 网格里那道门封实
    if (!WARDEN_MODE) { showToast("🔑 你没有这串钥匙。"); return; }
    const d = it.door;
    sendMsg({ type: "solitary_lock", locked: !d.locked });
    showToast(d.locked ? "🔓 打开禁闭室的门。" : "🔒 锁死禁闭室——里面的人别想出来。");
  }
}

function standUp() {
  if (!player || !occupying) return;
  player.root.rotation.set(0, player.heading, 0);
  if (player.exitPos) player.root.position.set(player.exitPos.x, 0, player.exitPos.z);
  else player.root.position.y = 0;
  if (occupying.entry?.station?.claimed === "player") occupying.entry.station.claimed = "";
  occupying = null;
  playAction(player, "Idle");
}

function renderBag() {
  const bag = document.getElementById("bag");
  if (!bag) return;
  if (inventory.size === 0) { bag.style.display = "none"; return; }
  bag.style.display = "flex";
  bag.innerHTML = `<span class="btitle">🎒 随身</span>`;
  for (const [id, rec] of inventory) {
    const s = document.createElement("span");
    s.className = "bitem";
    s.innerHTML = `${rec.icon}<em>${rec.name}</em>` + (rec.n > 1 ? `<i>×${rec.n}</i>` : "");
    // 吃/喝/用药是真的会改玩家身体的（后端 player_use），所以这三样点了就用掉
    const usable = USABLE_KINDS[id];
    s.title = usable ? `点一下${usable}（会真的缓解你的状态）` : (ITEM_DESC[id] || rec.name);
    if (usable) s.classList.add("usable");
    s.onclick = () => {
      if (usable) sendMsg({ type: "use_item", kind: id });
      else showToast(`${rec.icon} ${rec.name}：${ITEM_DESC[id] || "没什么特别的。"}`);
    };
    bag.appendChild(s);
  }
}

// 找最近的可对话 NPC（在玩家 3.2 单位内），更新头顶提示
const GUARD_TALK_R = 5.0;      // 铁面在走动，搭话半径比犯人宽一些
let guardHeld = false;         // 玩家凑到铁面跟前 → 巡逻循环里让他站住
function updateTalkTarget() {
  const hint = document.getElementById("talkhint");
  if (!player) { hint.style.display = "none"; talkNpcId = null; return; }
  let best = null, bd = 3.2 * 3.2, bId = null;
  for (const [id, e] of npcs) {
    if (e.dead) continue;                 // 尸体不接话：躺在地上还弹"按 E 对话"很荒诞
    const d2 = e.root.position.distanceToSquared(player.root.position);
    if (d2 < bd) { bd = d2; best = e; bId = id; }
  }
  // 破例：狱警铁面(guard1)虽是巡逻者，也能搭话（用抚恤金名单收买他）。
  // 他一直在走动，判定半径放宽到 5m，否则跟着他跑也按不上 E。
  const g = patrollers.get("guard1");
  guardHeld = false;
  if (g && g.root && g.root.visible) {
    const d2 = g.root.position.distanceToSquared(player.root.position);
    if (d2 < GUARD_TALK_R * GUARD_TALK_R && (best === null || d2 < bd)) {
      bd = d2; best = g; bId = "guard1";
      guardHeld = true;                 // 玩家凑近了 → 他站住让你说话（见巡逻循环）
    }
  }
  talkNpcId = bId;
  if (best && !chatOpen) {
    const d = latestDecision.get(bId);
    const name = d?.name || npcName(bId) || "犯人";
    tmp.copy(best.root.position); tmp.y += 3.0; tmp.project(camera);
    hint.textContent = (bId === "guard1" && (bb.inventory || []).includes("pension_list"))
      ? `按 E 向 ${name} 摊出抚恤金名单`
      : `按 E 与 ${name} 对话`;
    hint.style.left = `${(tmp.x * 0.5 + 0.5) * window.innerWidth}px`;
    hint.style.top = `${(-tmp.y * 0.5 + 0.5) * window.innerHeight}px`;
    hint.style.display = tmp.z < 1 ? "block" : "none";
  } else {
    hint.style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// 对话
// ---------------------------------------------------------------------------
const chatEl = () => document.getElementById("chat");
const chatLogEl = () => document.getElementById("chatlog");
const chatInEl = () => document.getElementById("chatin");

function npcName(id) {
  return latestWorld?.npcs.find((n) => n.id === id)?.name || NPC_NAME[id] || id;
}
function addChatLine(cls, text) {
  const el = document.createElement("div");
  el.className = "cm " + cls;
  el.textContent = text;
  chatLogEl().appendChild(el);
  chatLogEl().scrollTop = chatLogEl().scrollHeight;
  return el;
}
function openChat(npcId) {
  // 兜底：搭话提示、决策卡按钮都已各自过滤死者，但入口不止一处（快捷键、剧情脚本、
  // 收买狱警那条分支都会调这里），统一在门口再挡一道，免得漏一个入口就能和尸体对话。
  const nd = npcData(npcId);
  if (nd && nd.dead) { showToast(`${npcName(npcId)} 已经死了，再问不出什么了。`); return; }
  chatOpen = true; chatNpcId = npcId;
  selectedId = npcId; if (latestWorld) renderChips(latestWorld.npcs);
  document.getElementById("chattitle").textContent = `与 ${npcName(npcId)} 对话`;
  chatLogEl().innerHTML = "";
  const hist = chatHistories.get(npcId) || [];
  if (hist.length === 0) {
    addChatLine("sys", `你走近了 ${npcName(npcId)}…`);
    if (PUBLIC_MODE) {
      addChatLine("sys", "你是典狱长。他得拿你当长官——点下面的问题审讯，或自己打字。");
    } else {
      addChatLine("sys", "（你也可以直接吩咐他做事，比如「去健身」「回去睡觉」「去放风」）");
      const clue = CLUES.find((c) => c.owner === npcId && !foundClues.has(c.id));
      if (clue) addChatLine("sys", "（直觉告诉你，这人似乎知道些和你冤案有关的事…套套话看看）");
    }
  }
  for (const m of hist) addChatLine(m.role === "assistant" ? "npc" : "me", m.content);
  chatEl().style.display = "flex";
  document.getElementById("talkhint").style.display = "none";
  document.getElementById("cambar").style.display = "none";     // 对话框在底部居中，让位
  document.getElementById("playhint").style.display = "none";
  document.getElementById("acthint").style.display = "none";
  if (!PUBLIC_MODE) renderShowdownBar(npcId);
  else renderQuickQs(npcId);
  setTimeout(() => chatInEl().focus(), 50);
}
function closeChat() {
  chatOpen = false; chatNpcId = null;
  const bar = document.getElementById("evbar"); if (bar) bar.remove();
  const qqs = document.getElementById("qqs"); if (qqs) qqs.remove();
  chatEl().style.display = "none";
  document.getElementById("cambar").style.display = "flex";
  document.getElementById("playhint").style.display = "block";
  chatInEl().blur();
}

// —— 出证选卡：把"摊证据"从纯自由文本变成点选，点一下就替玩家说出那句最有杀伤力的指控 ——
// 每张卡的台词都内嵌 NLU 能识别的关键词(意图=指控 + 对应证据词)，从而稳定触发后端状态机。
// laundering/pension/medlog 三张原本是拍在门口那具典狱长 NPC 脸上的；他已整具删除
// （玩家自己就是典狱长），台词先留着当数据——想改成指控别人时直接接上去即可。
const EVID_PRESENT = {
  laundering: ["洗钱黑账", "我知道你收了外面集团的黑钱——这本洗钱黑账、账本上白纸黑字都是你签的字，别装了！"],
  pension: ["抚恤金名单", "这份抚恤金截留名单也是你干的，克扣牺牲弟兄的遗属——就是你，承认吧！"],
  medlog: ["篡改用药记录", "医务室这份被篡改的用药记录我看过了，出入库对不上——幕后就是你，别演了！"],
  softshoe: ["目击·软底鞋", "动手的人穿软底布鞋、走路没声音——那是我室友老牙！是他销赃陷害的我，别再演了！"],
};
// 收买铁面的那一句（内嵌"你看这/行个方便/抚恤金"→ 意图=出示+收买、证据=抚恤金名单）
const GUARD_BRIBE_LINE =
  "你看这份抚恤金截留名单——典狱长克扣了牺牲弟兄的遗属，白纸黑字都在这儿。" +
  "行个方便，暴乱那晚晚到五分钟，这事对你对我都好。";
function renderShowdownBar(npcId) {
  const old = document.getElementById("evbar"); if (old) old.remove();
  // 铁面：手里有抚恤金名单时，给一张一键"摊名单"的收买卡
  if (npcId === "guard1") {
    if (!(bb.inventory || []).includes("pension_list")) return;
    const bar = document.createElement("div");
    bar.id = "evbar";
    bar.style.cssText = "display:flex;gap:6px;padding:8px 10px;" +
      "border-top:1px solid rgba(255,255,255,.08);";
    const b = document.createElement("button");
    b.textContent = "🔍 摊出抚恤金截留名单";
    b.style.cssText = "padding:5px 10px;border-radius:8px;cursor:pointer;font:700 12px system-ui;" +
      "background:rgba(28,32,44,.92);border:1px solid rgba(255,110,110,.7);color:#ffd0d0;";
    b.onclick = () => presentEvidence(GUARD_BRIBE_LINE);
    bar.appendChild(b);
    document.getElementById("chattitle").insertAdjacentElement("afterend", bar);
    return;
  }
  if (npcId !== "laochen") return;          // 典狱长已由玩家扮演，出证入口只剩老牙这条
  const have = foundClues;                                   // 前端已收集线索
  const bar = document.createElement("div");
  bar.id = "evbar";
  bar.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;padding:8px 10px;" +
    "border-top:1px solid rgba(255,255,255,.08);";
  const mk = (label, line, strong) => {
    const b = document.createElement("button");
    b.textContent = "🔍 " + label;
    b.style.cssText = "padding:5px 10px;border-radius:8px;cursor:pointer;font:700 12px system-ui;" +
      "background:rgba(28,32,44,.92);" +
      "border:1px solid " + (strong ? "rgba(255,110,110,.7)" : "rgba(255,210,120,.5)") + ";" +
      "color:" + (strong ? "#ffd0d0" : "#ffe6a6") + ";";
    b.onclick = () => presentEvidence(line);
    bar.appendChild(b);
  };
  let any = false;
  // 揭穿真凶老牙：需摸到揭穿证据之一（软底鞋 / 洗钱黑账）——对典狱长和老牙都可用
  if (have.has("softshoe") || have.has("laundering")) {
    mk("揭穿真凶·老牙", EVID_PRESENT.softshoe[1], true); any = true;
  }
  if (!any) {
    const s = document.createElement("span");
    s.style.cssText = "font:500 12px system-ui;color:#8a93a3;padding:3px 2px;";
    s.textContent = "（还没有能拍在他脸上的硬证据——先去搜集线索）";
    bar.appendChild(s);
  }
  const title = document.getElementById("chattitle");
  title.insertAdjacentElement("afterend", bar);
}
function presentEvidence(line) {
  const npcId = chatNpcId; if (!npcId) return;
  const hist = chatHistories.get(npcId) || [];
  addChatLine("me", line);
  hist.push({ role: "user", content: line });
  chatHistories.set(npcId, hist);
  sendMsg({ type: "chat", npc_id: npcId, text: line, history: hist,
            player_state: playerStateSummary(), as_warden: WARDEN_MODE });
  addChatLine("sys", `${npcName(npcId)} 正在想…`);
}
// 玩家此刻的样子：姿态 / 手里拿着啥 / 站在哪个设施旁 —— 一起发给后端，
// 让 NPC 能就地反应（"你还坐着呢""手里那盒烟哪来的"）。
function playerStateSummary() {
  const ps = {};
  if (occupying) {
    const nm = occupying.entry?.name || "";
    ps.pose = occupying.type === "lie" ? `躺在${nm}上` : `坐在${nm}上`;
  } else {
    ps.pose = "站着";
  }
  if (inventory.size) {
    ps.holding = [...inventory.values()].map((r) => r.n > 1 ? `${r.name}×${r.n}` : r.name).join("、");
  }
  if (player && latestWorld?.facilities) {
    const p = player.root.position;
    let best = null, bd = 36;   // 6m 内才算"就在旁边"
    for (const f of latestWorld.facilities) {
      const dx = f.pos[0] - p.x, dz = f.pos[1] - p.z;
      const d2 = dx * dx + dz * dz;
      if (d2 < bd) { bd = d2; best = f; }
    }
    ps.location = best ? best.name : "监区空地";
  }
  return ps;
}
function sendChat() {
  const text = chatInEl().value.trim();
  if (!text || !chatNpcId) return;
  const hist = chatHistories.get(chatNpcId) || [];
  addChatLine("me", text);
  hist.push({ role: "user", content: text });
  chatHistories.set(chatNpcId, hist);
  sendMsg({ type: "chat", npc_id: chatNpcId, text, history: hist,
            player_state: playerStateSummary(), as_warden: WARDEN_MODE });
  chatInEl().value = "";
  addChatLine("sys", `${npcName(chatNpcId)} 正在想…`);
}
function renderQuickQs(npcId) {
  const old = document.getElementById("qqs"); if (old) old.remove();
  const bar = document.createElement("div");
  bar.id = "qqs";
  for (const [intent, label] of QQS) {
    const b = document.createElement("button");
    b.textContent = label;
    b.onclick = () => sendQuickQ(intent, label);
    bar.appendChild(b);
  }
  document.getElementById("chattitle").insertAdjacentElement("afterend", bar);
}
function sendQuickQ(intent, label) {
  const npcId = chatNpcId; if (!npcId) return;
  const hist = chatHistories.get(npcId) || [];
  addChatLine("me", label);
  hist.push({ role: "user", content: label });
  chatHistories.set(npcId, hist);
  sendMsg({ type: "chat", npc_id: npcId, text: label, intent, history: hist,
            player_state: playerStateSummary(), as_warden: true });
  addChatLine("sys", `${npcName(npcId)} 正在想…`);
}
// 流式对话状态：{ npcId, el, text }。边收 chat_delta 边往同一行追加。
let chatStream = null;
function onChatDelta(msg) {
  if (!chatOpen || chatNpcId !== msg.npc_id) return;   // 只渲染当前打开的对话
  const log = chatLogEl();
  const last = log.lastElementChild;                   // 首个增量 / 替换：移除"正在想…"
  if (last && last.classList.contains("sys")) last.remove();
  if (msg.replace && chatStream && chatStream.npcId === msg.npc_id && chatStream.el) {
    chatStream.text = "";
    chatStream.el.textContent = "";
  } else if (!chatStream || chatStream.npcId !== msg.npc_id) {
    chatStream = { npcId: msg.npc_id, el: addChatLine("npc", ""), text: "" };
  }
  chatStream.text += msg.delta || "";
  chatStream.el.textContent = chatStream.text;
  log.scrollTop = log.scrollHeight;
}
function onChatReply(msg) {
  const hist = chatHistories.get(msg.npc_id) || [];
  hist.push({ role: "assistant", content: msg.text });
  chatHistories.set(msg.npc_id, hist);
  const log = chatLogEl();
  if (chatStream && chatStream.npcId === msg.npc_id) {
    chatStream.el.textContent = msg.text;              // 用完整文本定稿（消掉分词毛刺）
    chatStream = null;
  } else {
    const last = log.lastElementChild;                 // 非流式回退：移除占位再补一行
    if (last && last.classList.contains("sys")) last.remove();
    if (chatOpen && chatNpcId === msg.npc_id) addChatLine("npc", msg.text);
  }
  // 头顶飘一条回复气泡（覆盖独白几秒）
  const e = npcs.get(msg.npc_id);
  if (e) { e.replyText = msg.text; e.replyUntil = performance.now() + 8000; }
  const intel = msg.intel || [];
  if (intel.length) {
    for (const line of intel) addChatLine("sys", `记入卷宗：${line}`);
    showToast("🗒 " + intel.join("；"));
    if (deckOpen && deckId === msg.npc_id) renderDeck();
  }
}

// ---------------------------------------------------------------------------
// 侦探系统·检视 + 案件板
// ---------------------------------------------------------------------------
const foundClues = new Set();
function inspectClue(id) {
  const e = clueEntries.find((c) => c.id === id);
  if (!e) return;
  const cl = e.data;
  document.getElementById("insico").textContent = cl.icon;
  document.getElementById("insnm").textContent = cl.name;
  document.getElementById("insdesc").textContent = cl.desc;
  document.getElementById("insins").textContent = "🔎 " + cl.insight;
  document.getElementById("inspect").style.display = "flex";
  if (!e.found) {
    e.found = true; foundClues.add(id);
    e.gem.material.color.set(0x5a6070); e.gem.material.emissive.set(0x222833); e.gem.material.emissiveIntensity = 0.15;
    e.spr.material.opacity = 0.35;
    showToast(`收入证物：${cl.name} — ${cl.insight}`);
    // 同步案件进展给后端：查得越深，知情 NPC 越松口、越敢帮着谋事
    sendMsg({ type: "case_progress", found: foundClues.size });
    // 逐条上报证据：后端记进证据库、可能触发逐级解锁（找到名单还会发抚恤金名单道具）
    sendMsg({ type: "clue_found", id: id });
  }
  updateJbtn(); renderJournal();
}
window.__inspectClue = inspectClue;

function updateJbtn() {
  document.getElementById("jcount").innerHTML =
    `<span class="${foundClues.size < CLUES.length ? "dot" : ""}">${foundClues.size}/${CLUES.length}</span>`;
}
function renderJournal() {
  document.getElementById("jtitle").textContent = "案件板 · 我是被谁陷害的？";
  document.getElementById("jintro").textContent = CASE_INTRO;
  document.getElementById("jprog").textContent = `已找到线索 ${foundClues.size} / ${CLUES.length}`;
  const box = document.getElementById("jclues");
  box.innerHTML = "";
  for (const cl of CLUES) {
    const row = document.createElement("div");
    const who = cl.owner ? (NPC_NAME[cl.owner] || cl.owner) : null;
    if (foundClues.has(cl.id)) {
      row.className = "clue";
      const src = who ? `<div class="cx">🗣 知情人：${who}（找他聊聊能对证）</div>` : "";
      row.innerHTML = `<div class="ci">${cl.icon}</div><div><div class="cn">${cl.name}</div>` +
        `<div class="cd">${cl.desc}</div><div class="cx">🔎 ${cl.insight}</div>${src}</div>`;
    } else {
      row.className = "locked";
      // 已知情人给玩家一个方向：找谁聊 / 或哪儿翻实物
      row.textContent = who
        ? `🔒 未发现的线索…（试着找「${who}」套套话，或在监狱里翻发光的物件）`
        : "🔒 未发现的线索…（在监狱各处找找发光的物件）";
    }
    box.appendChild(row);
  }
  const concl = document.getElementById("jconcl");
  if (foundClues.size >= CLUES.length) { concl.style.display = "block"; concl.textContent = "✅ " + CASE_CONCLUSION; }
  else concl.style.display = "none";
}
function toggleJournal(force) {
  const j = document.getElementById("journal");
  const show = force !== undefined ? force : j.style.display !== "block";
  if (show) renderJournal();
  j.style.display = show ? "block" : "none";
}
function closeInspect() { document.getElementById("inspect").style.display = "none"; }

function initInvestigation() {
  updateJbtn();
  document.getElementById("insok").onclick = closeInspect;
  document.getElementById("inspect").addEventListener("click", (ev) => { if (ev.target.id === "inspect") closeInspect(); });
  // 案件板已下线：隐藏入口按钮（保留 DOM 以免其它引用报错）
  const jb = document.getElementById("jbtn"); if (jb) jb.style.display = "none";
  const jc = document.getElementById("jclose"); if (jc) jc.onclick = () => toggleJournal(false);
}

function initPlayerChat() {
  window.addEventListener("keydown", (ev) => {
    if (chatOpen) {
      if (ev.key === "Escape") closeChat();
      return; // 聊天时不触发移动
    }
    const k = ev.key.toLowerCase();
    if (ev.key === "Escape") { closeInspect(); toggleJournal(false); return; }
    if (k === "k") { toggleFate(); return; }
    if (k === "e" && talkNpcId) { ev.preventDefault(); openChat(talkNpcId); return; }
    if (k === "f") { ev.preventDefault(); doInteract(occupying ? occupying.entry : focusAct); return; }
    // 阶段三：用镇静肉放倒狱犬哮天（G）。有镇静剂才有效，后端校验并弹结果。
    if (k === "g") {
      ev.preventDefault();
      if ((bb.inventory || []).includes("sedative")) sendMsg({ type: "drug_dog" });
      else showToast("你手上没有镇静剂——先找瘸子换（拿医用酒精换）。");
      return;
    }
    // 阶段三：被撞见时狡辩（P）。编个够严密的借口，后端按诚意/逻辑判定能否蒙混过关。
    if (k === "p") { ev.preventDefault(); startPlead(); return; }
    // 视角：数字键直选，V 循环
    const slot = "123456789".indexOf(k);
    if (slot >= 0 && slot < CAM_MODES.length) { setCamMode(CAM_MODES[slot][0]); return; }
    if (k === "v") {
      const i = CAM_MODES.findIndex(([m]) => m === camMode);
      setCamMode(CAM_MODES[(i + 1) % CAM_MODES.length][0]);
      return;
    }
    // 空格：上帝暂停/继续世界（观察者模式）
    if (ev.code === "Space" || k === " ") { ev.preventDefault(); togglePause(); return; }
    keys[k] = true;
  });
  window.addEventListener("keyup", (ev) => { keys[ev.key.toLowerCase()] = false; });
  document.getElementById("chatsend").onclick = sendChat;
  document.getElementById("chatclose").onclick = closeChat;
  chatInEl().addEventListener("keydown", (ev) => {
    ev.stopPropagation();
    if (ev.key === "Enter") sendChat();
    if (ev.key === "Escape") closeChat();
  });
}

// ---------------------------------------------------------------------------
// 面板 DOM
// ---------------------------------------------------------------------------
const statusEl = document.getElementById("status");
const ctxEl = document.getElementById("ctx");
const candsEl = document.getElementById("cands");
const listEl = document.getElementById("npclist");
const titleEl = document.getElementById("ptitle");
const explainEl = document.getElementById("explain");
const needsEl = document.getElementById("needs");
const sitEl = document.getElementById("sit");
const elfarolCanvas = document.getElementById("elfarol");
const distEl = document.getElementById("dist");
const dlegendEl = document.getElementById("dlegend");
const latestDecision = new Map();
let latestWorld = null;

const NEED_META = [
  ["hunger", "饿", "#e0a44a"], ["thirst", "渴", "#4aa3e0"],
  ["fatigue", "累", "#9b6bd6"], ["boredom", "闷", "#6bc5c5"],
  ["social", "社", "#5bbf6a"], ["stress", "压", "#e08a4a"],
  ["injury", "伤", "#de5c6c"],
];

// ---------------------------------------------------------------------------
// 狱警 / 狱犬 巡逻实体（后端驱动位置，前端平滑滑行 + 视野提示）
// ---------------------------------------------------------------------------
const patrollers = new Map();
const guardMat = new THREE.MeshStandardMaterial({ color: 0x2f3d5a, roughness: 0.7 });
const guardCapMat = new THREE.MeshStandardMaterial({ color: 0x1b2436, roughness: 0.6 });
const guardSkinMat = new THREE.MeshStandardMaterial({ color: 0xc79c78, roughness: 0.8 });
const guardBootMat = new THREE.MeshStandardMaterial({ color: 0x12151b, roughness: 0.7 });
const dogMat = new THREE.MeshStandardMaterial({ color: 0x3a2c22, roughness: 0.85 });

// 肢体枢轴：网格挂在枢轴组下方(顶端对齐枢轴)，绕枢轴 X 轴旋转即"迈腿/摆臂"。
function limbPivot(mesh, px, py, pz) {
  const p = new THREE.Group();
  p.position.set(px, py, pz);
  mesh.position.y = -mesh.geometry.parameters.height / 2;   // 顶端贴枢轴，绕肩/胯摆动
  mesh.castShadow = true;
  p.add(mesh);
  return p;
}
// 关节化狱警：躯干/肩章/皮带/大檐帽 + 可摆动的四肢（不再是滑行的圆柱）。
function buildGuardBody() {
  const g = new THREE.Group();
  const hips = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.28, 0.30), guardMat); hips.position.y = 0.9; g.add(hips);
  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.66, 0.30), guardMat); torso.position.y = 1.3; torso.castShadow = true; g.add(torso);
  const belt = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.1, 0.33), guardCapMat); belt.position.y = 1.0; g.add(belt);
  const shoulders = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.16, 0.32), guardCapMat); shoulders.position.y = 1.58; g.add(shoulders);
  const badge = new THREE.Mesh(new THREE.CircleGeometry(0.05, 14),
    new THREE.MeshStandardMaterial({ color: 0xf0c05a, metalness: 0.6, roughness: 0.4 }));
  badge.position.set(0.17, 1.36, 0.161); g.add(badge);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 14), guardSkinMat); head.position.y = 1.8; g.add(head);
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.19, 0.2, 0.14, 16), guardCapMat); cap.position.y = 1.94; g.add(cap);
  const brim = new THREE.Mesh(new THREE.CircleGeometry(0.24, 20), guardCapMat);
  brim.rotation.x = -Math.PI / 2 + 0.14; brim.position.set(0, 1.88, 0.15); g.add(brim);
  const arms = [1, -1].map((sx) => limbPivot(new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.58, 0.15), guardMat), sx * 0.36, 1.52, 0));
  const legs = [1, -1].map((sx) => limbPivot(new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.72, 0.18), guardBootMat), sx * 0.14, 0.82, 0));
  for (const m of [...arms, ...legs]) g.add(m);
  return { group: g, arms, legs };
}
// 关节化狱犬：胸/颈/头/口鼻/耳/尾 + 四条可小跑摆动的腿。
function buildDogBody() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.4, 0.9), dogMat); body.position.set(0, 0.56, 0); body.castShadow = true; g.add(body);
  const chest = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.42, 0.34), dogMat); chest.position.set(0, 0.56, 0.36); g.add(chest);
  const neck = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.3, 0.26), dogMat); neck.position.set(0, 0.72, 0.54); neck.rotation.x = 0.5; g.add(neck);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.3, 0.32), dogMat); head.position.set(0, 0.86, 0.72); g.add(head);
  const snout = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.14, 0.2), dogMat); snout.position.set(0, 0.8, 0.92); g.add(snout);
  for (const sx of [-1, 1]) {
    const ear = new THREE.Mesh(new THREE.ConeGeometry(0.07, 0.16, 4), dogMat); ear.position.set(sx * 0.1, 1.04, 0.68); g.add(ear);
  }
  const tail = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.02, 0.5, 6), dogMat); tail.position.set(0, 0.66, -0.5); tail.rotation.x = -0.9; g.add(tail);
  const legs = [[0.14, 0.34], [-0.14, 0.34], [0.14, -0.32], [-0.14, -0.32]].map(([lx, lz]) =>
    limbPivot(new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.44, 0.13), dogMat), lx, 0.44, lz));
  for (const m of legs) g.add(m);
  // 头和尾巴单独交出去：狱犬没有骨骼动画，向典狱长"打招呼"只能靠这两块程序化摆动
  return { group: g, legs, head, tail };
}

// 程序化步态：走动时四肢按步频前后摆（狗小跑=对角腿同相），站定缓缓回正——不再"滑行的木桩"。
function animatePatrollerLimbs(e, stepped, dt) {
  const L = e.limbs;
  if (!L) return;
  // 向典狱长打招呼：方块狱警抬右臂行礼；狱犬没有手，改成抬头 + 快速摇尾。
  // 权重平滑进出，收手时自然落回步态，不会"啪"地一下弹回去。
  e.saluteW = (e.saluteW || 0) + ((e.saluting ? 1 : 0) - (e.saluteW || 0)) * Math.min(1, dt * (e.saluting ? 9 : 5));
  if (e.saluteW > 0.01) {
    if (e.kind === "guard" && L.salute) {
      L.salute.rotation.x = -2.0 * e.saluteW;      // 手抬到帽檐
      L.salute.rotation.z = 0.5 * e.saluteW;
    } else if (L.head) {
      L.head.rotation.x = -0.45 * e.saluteW;       // 抬头看典狱长
      if (L.tail) L.tail.rotation.y = Math.sin(performance.now() * 0.018) * 0.7 * e.saluteW;
    }
    if (e.saluting) return;                        // 行礼期间不叠步态（人是站定的）
  } else if (e.kind !== "guard" && L.head) {
    L.head.rotation.x += (0 - L.head.rotation.x) * Math.min(1, dt * 6);
    if (L.tail) L.tail.rotation.y += (0 - L.tail.rotation.y) * Math.min(1, dt * 6);
  }
  if (stepped > 1e-4) {
    e.walkPhase += stepped * 6.5;
    const s = Math.sin(e.walkPhase);
    if (e.kind === "guard") {
      L.legs[0].rotation.x = s * 0.5; L.legs[1].rotation.x = -s * 0.5;
      if (L.arms) { L.arms[0].rotation.x = -s * 0.4; L.arms[1].rotation.x = s * 0.4; }
    } else {
      L.legs[0].rotation.x = s * 0.6; L.legs[3].rotation.x = s * 0.6;   // 前左+后右同相
      L.legs[1].rotation.x = -s * 0.6; L.legs[2].rotation.x = -s * 0.6; // 前右+后左反相
    }
  } else {
    const k = Math.min(1, dt * 8);
    const all = e.kind === "guard" ? [...L.legs, ...(L.arms || [])] : L.legs;
    for (const pv of all) pv.rotation.x += (0 - pv.rotation.x) * k;
  }
}

function spawnPatroller(p) {
  const root = new THREE.Group();
  let limbs = null, anim = null;
  if (p.kind === "guard") {
    // 狱警穿 police.glb（骨骼动画）；模型没加载成功才退回原来的方块狱警(程序化迈腿)
    const av = buildPoliceAvatar();
    if (av) { root.add(av.model); anim = av.rig; }
    else {
      const gb = buildGuardBody(); root.add(gb.group);
      limbs = { arms: gb.arms, legs: gb.legs, salute: gb.arms ? gb.arms[0] : null };
    }
    // 视野：贴地的极淡扇形（不再是竖起来的大黄锥，免得糊满整屏），
    // 只作方向暗示；apex 在人脚下，沿 +Z(行进方向)张开 ±half。
    const r = p.view_range || 8.5;
    const half = Math.acos(Math.min(1, Math.max(-1, p.view_cos ?? 0.5)));
    const N = 24;
    const pos = [0, 0, 0];
    for (let i = 0; i <= N; i++) {
      const a = -half + (2 * half) * (i / N);
      pos.push(Math.sin(a) * r, 0, Math.cos(a) * r);
    }
    const idx = [];
    for (let i = 1; i <= N; i++) idx.push(0, i, i + 1);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx); geo.computeVertexNormals();
    const cone = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      // 夜景底子是纯黑，6% 的不发光黄再经 bloom 一放，会糊成一整片土黄地毯，
      // 比火盆还抢眼。压到 2.5% 才只是"地上一层薄光"。
      color: 0xf0c05a, transparent: true, opacity: NIGHT_DIORAMA ? 0.025 : 0.06,
      side: THREE.DoubleSide, depthWrite: false }));
    cone.position.y = 0.05;              // 贴地
    root.add(cone);
    root.userData.cone = cone;
  } else {
    const db = buildDogBody(); root.add(db.group);
    limbs = { legs: db.legs, head: db.head, tail: db.tail };
    // 嗅觉半径：贴地半透明红环
    const r = p.view_range || 5;
    const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.12, r, 40),
      new THREE.MeshBasicMaterial({ color: 0xff7a5c, transparent: true, opacity: 0.16,
        side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.05; root.add(ring);
    // 和狱警的视野扇形存到同一个键：押送/敬礼时统一收起来（那会儿他没在查人）
    root.userData.cone = ring;
  }
  root.position.set(p.pos[0], 0, p.pos[1]);
  scene.add(root);
  const label = makeBubble("npc-bubble");
  label.style.borderColor = p.kind === "guard" ? "#ffd23f" : "#ff7a5c";
  label.innerHTML = `<div class="nb-head"><span class="who">${p.kind === "guard" ? "👮 " : "🐕 "}${p.name}</span></div>`;
  const entry = { root, label, dest: new THREE.Vector3(p.pos[0], 0, p.pos[1]),
                  heading: 0, kind: p.kind, limbs, walkPhase: 0, ...(anim || {}),
                  viewRange: p.view_range || (p.kind === "dog" ? 5 : 8.5),
                  viewCos: p.view_cos ?? 0.5 };
  patrollers.set(p.id, entry);
  if (entry.mixer) playAction(entry, "Idle");
  return entry;
}

function updatePatrollers(list) {
  if (!list) return;
  // 狱警/狱犬是阶段二之后的事：出了牢房、公共区解锁了才现身。
  // 破例：铁面(guard1)是**可对话角色**（要用抚恤金名单收买他），一旦公共区解锁就常驻可见——
  // 他的巡逻线两点之间会横穿尚未解锁的分区，按分区隐藏会让他大半时间人间蒸发、根本摸不到。
  const showPatrollers = zoneUnlocked("public");
  for (const p of list) {
    const e = patrollers.get(p.id) || spawnPatroller(p);
    const vis = showPatrollers && (p.id === "guard1" || zoneUnlocked(p.zone));
    if (e.root) e.root.visible = vis;
    e.dest.set(p.pos[0], 0, p.pos[1]);
    if (p.heading) e.heading = Math.atan2(p.heading[0], p.heading[1]);
    e.escorting = p.escorting || "";
    e.saluting = !!p.saluting;
    // 头顶标签跟着状态变：押人和敬礼都是玩家该一眼看出来的事
    const tag = e.escorting
      ? `<span class="act">🔒 押送 ${npcName(e.escorting) || ""}</span>`
      : (e.saluting ? `<span class="act">🎖 敬礼</span>` : "");
    const icon = e.kind === "guard" ? "👮 " : "🐕 ";
    e.label.innerHTML = `<div class="nb-head"><span class="who">${icon}${p.name}</span>${tag}</div>`;
  }
}

// 已播报过越狱的 NPC（避免重复弹）
const escapedSeen = new Set();
const deadSeen = new Set();       // 已播报过的命案（按 npc id 去重）

// 已播报过的被抓事件（按 tick 去重）
const bustSeen = new Set();

let staleWarned = false;
// ---------------------------------------------------------------------------
// 物资实体的显形与消耗（后端 world_objects.py 是权威，这里只做呈现）
// ---------------------------------------------------------------------------
const worldObjs = new Map();      // id -> { root, kind, buns }
const OBJ_LABEL = { food: "🍚", water: "💧", bed: "🛏", medkit: "💉" };

function syncObjects(msg) {
  if (!Array.isArray(msg.objects)) return;
  for (const o of msg.objects) {
    // 所在设施没解锁就先不摆（跟房间显形的门禁保持一致）
    const fac = (msg.facilities || []).find((f) => f.id === o.facility);
    if (fac && !zoneUnlocked(fac.zone)) continue;
    let e = worldObjs.get(o.id);
    if (!e) {
      let root = null, takeable = true;
      if (o.kind === "food") root = makeFoodTray(o.cap);
      else if (o.kind === "water") root = makeTap();
      else if (o.kind === "medkit") root = makeMedCabinet();
      else if (o.kind === "bed") { root = makeBedMarker(); takeable = false; }
      if (!root) continue;
      root.position.set(o.pos[0], 0, o.pos[1]);
      scene.add(root);
      e = { root, kind: o.kind, buns: root.userData.buns || [],
            mark: root.userData.mark || null };
      worldObjs.set(o.id, e);
      // 床位不能"拿走"，只能被占用；其余三样玩家可以真的抢
      if (takeable) {
        addInteractable(root, {
          type: "take", name: o.name, icon: OBJ_LABEL[o.kind] || "📦", objId: o.id,
        });
      }
    }
    // A5 铺位占用：空铺灰、有人睡就亮起来并浮一个 💤，一眼看出还剩几张
    if (e.kind === "bed" && e.mark) {
      const busy = !!o.busy;
      e.mark.material.color.setHex(busy ? 0x6fa8ff : 0x4a4f5a);
      e.mark.material.opacity = busy ? 0.85 : 0.35;
      e.root.userData.zzz.visible = busy;
    }
    // 剩几份就显几个窝头 —— "被吃光"这件事因此是肉眼可见的
    if (e.buns.length) {
      for (let i = 0; i < e.buns.length; i++) e.buns[i].visible = i < o.stock;
    }
    if (e.kind === "water" || e.kind === "bed") {
      e.root.traverse((c) => { if (c.isMesh) c.material.emissiveIntensity = o.busy ? 0.4 : 0; });
    }
    if (e.root.userData.entry) {
      e.root.userData.entry.disabled = !o.available;
      e.root.userData.entry.name = o.available ? o.name : `${o.name}(空了)`;
    }
  }
}

const TAKE_META = {
  food: { name: "窝头", icon: "🍚" }, water: { name: "一壶水", icon: "🫙" },
  medkit: { name: "医疗包", icon: "💉" },
};
// 背包里点一下就会真的用掉的东西（对应后端 PLAYER_USE）
const USABLE_KINDS = { food: "吃掉", water: "喝掉", medkit: "包扎" };

function onUseResult(msg) {
  if (!msg.ok) { showToast(`✋ ${msg.reason}`); return; }
  const rec = inventory.get(msg.kind);
  if (rec) {
    rec.n -= 1;
    if (rec.n <= 0) inventory.delete(msg.kind);
    renderBag();
  }
  showToast(`${{ food: "🍚 吃下窝头，胃里踏实了些",
                 water: "💧 喝了水，嗓子松快了",
                 medkit: "💉 简单包扎，伤口不那么疼了" }[msg.kind] || "用掉了"}`);
  if (msg.player) renderPlayerBody(msg.player);
}

// —— 玩家身体 HUD：会饿会渴会伤，和 NPC 受同一套物理层约束 ——
let lastHits = 0;
function renderPlayerBody(p) {
  if (OBSERVER_MODE) {   // 上帝观察：没有主角化身，不显示"你的状态"
    const b = document.getElementById("body");
    if (b) b.style.display = "none";
    return;
  }
  if (!p) return;
  const el = document.getElementById("body");
  if (!el) return;
  const bar = (label, v, warn) => {
    const c = v >= 80 ? "#ff5b5b" : v >= 55 ? "#ffb347" : "#7fd18c";
    return `<span class="bstat"${v >= warn ? ' data-warn="1"' : ""}>${label}` +
      `<i style="width:${Math.max(3, v)}%;background:${c}"></i></span>`;
  };
  el.style.display = "flex";
  el.innerHTML = `<span class="btitle">🧍 你的状态</span>` +
    bar("饿", p.hunger, 80) + bar("渴", p.thirst, 80) + bar("伤", p.injury, 50);
  // 挨打了给一次红闪 + 播报（让"被打"这件事有反馈，而不是数字悄悄涨）
  const hits = p.hits || [];
  if (hits.length > lastHits) {
    for (const h of hits.slice(lastHits)) showToast(`💢 ${h}`);
    flashHurt();
  }
  lastHits = hits.length;
}

function flashHurt() {
  const f = document.getElementById("hurtflash");
  if (!f) return;
  f.style.transition = "none"; f.style.opacity = "0.55";
  requestAnimationFrame(() => {
    f.style.transition = "opacity .5s ease-out"; f.style.opacity = "0";
  });
}
function onTakeResult(msg) {
  if (!msg.ok) { showToast(`✋ ${msg.reason}`); return; }
  const m = TAKE_META[msg.kind] || { name: msg.name, icon: "📦" };
  const rec = inventory.get(msg.kind) || { name: m.name, icon: m.icon, n: 0 };
  rec.n += 1; inventory.set(msg.kind, rec);
  renderBag();
  showToast(`${m.icon} 拿到${m.name}（这批还剩 ${msg.stock} 份，NPC 也在抢）`);
}

// —— 上报玩家位置（节流）：后端按距离判"卷进群殴"和"被记恨的人堵着揍" ——
let lastPosSent = 0;
function reportPlayerPos() {
  if (!player) return;
  const now = performance.now();
  if (now - lastPosSent < 500) return;
  lastPosSent = now;
  const p = player.root.position;
  sendMsg({ type: "player_pos", x: +p.x.toFixed(2), z: +p.z.toFixed(2) });
}

// —— A4 群殴现场：一圈扬尘 + 跳动的拳头符号，远远就能看见哪儿打起来了 ——
const boutFx = new Map();      // bout id -> { group, ring, puffs }
function syncBouts(list) {
  const live = new Set();
  for (const b of (list || [])) {
    live.add(b.id);
    let fx = boutFx.get(b.id);
    if (!fx) {
      const group = new THREE.Group();
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.9, 1.5, 28),
        new THREE.MeshBasicMaterial({ color: 0xff6a3d, transparent: true,
                                      opacity: 0.5, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = 0.06;
      group.add(ring);
      const puffs = [];
      for (let i = 0; i < 5; i++) {
        const m = new THREE.Mesh(
          new THREE.SphereGeometry(0.26, 8, 7),
          new THREE.MeshBasicMaterial({ color: 0xd8cfc0, transparent: true, opacity: 0.4 }));
        group.add(m); puffs.push(m);
      }
      scene.add(group);
      fx = { group, ring, puffs, born: performance.now() };
      boutFx.set(b.id, fx);
    }
    fx.group.position.set(b.pos[0], 0, b.pos[1]);
    const t = performance.now() / 1000;
    fx.ring.scale.setScalar(1 + 0.12 * Math.sin(t * 9));
    fx.ring.material.opacity = 0.35 + 0.2 * Math.abs(Math.sin(t * 9));
    fx.puffs.forEach((m, i) => {
      const a = t * 2.2 + (i / fx.puffs.length) * Math.PI * 2;
      m.position.set(Math.cos(a) * 1.1, 0.5 + 0.35 * Math.abs(Math.sin(t * 6 + i)), Math.sin(a) * 1.1);
      m.material.opacity = 0.18 + 0.22 * Math.abs(Math.sin(t * 5 + i));
    });
  }
  for (const [id, fx] of boutFx) {       // 架散了就把特效收掉
    if (live.has(id)) continue;
    scene.remove(fx.group);
    boutFx.delete(id);
  }
}

let supplySeen = new Set();
let fightSeen = new Set();
function syncSupplyFeed(msg) {
  for (const line of (msg.supply_log || [])) {
    if (supplySeen.has(line)) continue;
    supplySeen.add(line);
    if (supplySeen.size > 60) supplySeen = new Set([...supplySeen].slice(-30));
    showToast(`🍚 ${line}`);
  }
  for (const line of (msg.fight_log || [])) {
    if (fightSeen.has(line)) continue;
    fightSeen.add(line);
    if (fightSeen.size > 60) fightSeen = new Set([...fightSeen].slice(-30));
    showToast(`🥊 ${line}`);
  }
}

function onWorldState(msg) {
  latestWorld = msg;
  if (!msg.blackboard && !staleWarned) {         // 旧后端进程：没有分区数据，会导致全图可见
    staleWarned = true;
    showToast("⚠️ 后端是旧进程（收不到分区数据）——请重启后端再硬刷新，分区解锁才会生效。");
    console.warn("[stage1] world_state 缺 blackboard/zone：后端未重启，分区门禁失效。");
  }
  if (msg.blackboard) bb = msg.blackboard;      // 黑板：唯一真相源（分区/道具/好感）
  syncFacilities(msg);                          // 已解锁分区的房间逐个长出来
  syncCells(msg);                               // 六间牢房：砌隔间 + 同步牢门的关/锁
  if (solitaryDoor && "solitary_locked" in msg) solitaryDoor.locked = !!msg.solitary_locked;
  syncObjects(msg);                             // 物资：窝头按份数消失、龙头/铺位占用
  syncSupplyFeed(msg);                          // 物资/打架流水播报
  renderPlayerBody(msg.player);                 // 玩家身体：饿/渴/伤
  syncBouts(msg.bouts);                         // 群殴：现场特效
  reportPlayerPos();                            // 上报位置（物理层按距离判误伤/挨打）
  recomputeBounds(msg.facilities);              // 玩家活动范围随解锁扩大
  syncClueVisibility(msg.facilities);           // 线索随分区解锁才显形
  updateObjective();
  renderFate();                                  // 命运账本：各 NPC 状态 + 叙事流水实时刷新
  if (!PUBLIC_MODE) renderSocialFeed(msg.social_feed);             // 实时社交字幕流
  checkPhaseIntro();                             // 每解锁一段空间：核心 NPC 主动开场白
  setPhaseLighting(msg.phase);
  updatePatrollers(msg.patrollers);
  onEndgame(msg.endgame);                        // 阶段四/五：暴乱红警 + 典狱长现身 + 摊牌入口
  syncShift(msg);
  // 狱警抓现行：加刑/罚苦役/搞黄假释，一次性播报（体现狱警手里的真权力）
  if (Array.isArray(msg.busts)) {
    for (const b of msg.busts) {
      const key = `${b.name}@${b.tick}`;
      if (bustSeen.has(key)) continue;
      bustSeen.add(key);
      showToast(`👮 狱警抓现行：${b.name} ${b.reason} → 加刑至 ${b.sentence} 年、罚重苦役`);
    }
  }
  // 打分是异步的：模拟按 tick 跑，模型分晚几拍回来。把延迟摊开显示，
  // 免得看到"改了状态但没立刻改主意"时误以为是卡死。
  const lag = msg.score_lag ? ` · 打分${msg.score_ms}ms(滞后${msg.score_lag}拍)` : "";
  statusEl.textContent =
    `🧠${msg.brain ?? "?"} · tick ${msg.tick} · 时段:${msg.phase} · 放风:${msg.yard_crowd}${lag}`;
  syncPauseUI(msg.paused);                       // 上帝暂停：按钮/角标随后端状态
  renderScoreboard();                            // 两脑对比记分板：刷新本拍微调打分耗时
  // 决策卡是"点选那一刻"的快照，不逐帧重绘（否则闪、还会打断对比按钮）；
  // 观察流程是"先暂停→再逐个点"，暂停时快照即真相。要看最新的再点一次即可。
  for (const n of msg.npcs) {
    // 按"出生分区"决定何时登场：其所在分区解锁前一律藏起来（阶段一只见室友老牙）。
    // 用固定的 home_zone 而非会变的 live zone——否则 NPC 顺着长期心思溜达出未解锁区，
    // live 分区一翻就忽隐忽现（老牙就是这么"消失"的）。老后端没有该字段则回落 live zone。
    if (!zoneUnlocked(n.home_zone || n.zone)) {
      const ex = npcs.get(n.id);
      if (ex) { ex.root.visible = false; ex.label.style.display = "none"; }
      continue;
    }
    const e = npcs.get(n.id) || spawnNpc(n);
    if (!e.root.visible) { e.root.visible = true; e.label.style.display = ""; }
    // 坐标消毒：后端偶发的非有限坐标(NaN/Infinity)绝不让它流进 mesh——一旦灌进去，
    // 整具模型世界矩阵失效、当帧就从视野里凭空消失，且 rotation 累加会把 NaN 永久带下去，
    // 那个 NPC 再也不出现（表现成"站着站着人就没了、每个人迟早都没"）。退回上一次好值。
    if (!Number.isFinite(n.pos[0]) || !Number.isFinite(n.pos[1])) {
      n.pos = [e.serverPos.x, e.serverPos.z];
    }
    // 越狱出逃：从场上抹掉 + 一次性播报
    if (n.escaped) {
      e.root.visible = false; e.label.style.display = "none";
      if (!escapedSeen.has(n.id)) {
        escapedSeen.add(n.id);
        showToast(`🚨 越狱成功：${n.name} 从州立监狱大门消失在夜色里！`);
      }
      continue;
    }
    // 遇害身亡：就地倒下、灰暗，头顶标一个 ☠，不再移动/播动画/冒气泡
    if (n.dead) {
      e.dead = true;
      e.root.visible = true;
      e.root.position.set(n.pos[0], 0, n.pos[1]);
      e.root.rotation.z = Math.PI / 2.2;          // 倒地
      e.label.style.display = "block";
      e.label.innerHTML = `<div class="nb-head"><span class="who">${n.name}</span>` +
        `<span class="act">☠ 已身亡</span></div>`;
      const tmp2 = new THREE.Vector3(n.pos[0], 2.2, n.pos[1]).project(camera);
      e.label.style.left = `${(tmp2.x * 0.5 + 0.5) * window.innerWidth}px`;
      e.label.style.top = `${(-tmp2.y * 0.5 + 0.5) * window.innerHeight}px`;
      if (!deadSeen.has(n.id)) {
        deadSeen.add(n.id);
        showToast(`🔪 命案！${n.name} 倒在了牢里——一条命，就这么没了。`);
      }
      continue;
    }
    const tp = n.target_pos || n.pos;
    e.cell = n.cell ?? e.cell ?? -1;
    e.going = n.action_state === "前往中";
    e.actionState = n.action_state || "";
    const lockedMove = !!(n.solitary || (n.escort && n.escort_grabbed));
    // 开场白期间：核心 NPC 主动走到主角跟前说话（附近才走过来，太远就原地说，免得穿墙滑行）
    const introing = e.introUntil && performance.now() < e.introUntil;
    const px = player ? player.root.position.x : 0, pz = player ? player.root.position.z : 0;
    // 近(18m内) + 有清晰直线通路 才走过来，否则原地说话（免得远距离滑行穿墙）
    if (lockedMove) {
      releaseStation(e);
    } else if (introing && player
        && e.root.position.distanceTo(player.root.position) < 18
        && !segmentBlocked(e.root.position.x, e.root.position.z, px, pz)) {
      const dx = e.root.position.x - px, dz = e.root.position.z - pz;
      const L = Math.hypot(dx, dz) || 1;
      e.dest.set(px + (dx / L) * 1.7, 0, pz + (dz / L) * 1.7);   // 停在主角前方约 1.7m
      releaseStation(e);
    } else {
      const spec = STATION_OF[n.goal];
      if (spec) {
        // 先把 dest 指到设施，再就近认领一张床/凳；认到了就改走家具，绝不加工位偏移。
        e.dest.set(tp[0], 0, tp[1]);
        const st = claimStation(e, spec);
        if (st) {
          st.anchor.getWorldPosition(_wp);
          e.dest.set(_wp.x, 0, _wp.z);
        }
      } else {
        releaseStation(e);
        if (INDOOR_GOALS.has(n.goal)) {
          e.dest.set(tp[0], 0, tp[1]);          // 室内：走到房间里再做动作
        } else {
          e.dest.set(tp[0] + e.slot.x, 0, tp[1] + e.slot.z);  // 院子：工位散开以免叠人
        }
      }
    }
    // 对话锁定期：无视常规目标，走到面对面的站位（凑近说话）
    if (e.convoUntil && performance.now() < e.convoUntil && e.convoDest) {
      e.dest.copy(e.convoDest);
    }
    // 被押送：贴着后端权威位置走，不再站"工位"偏移，也不受对话/开场白站位影响。
    // 后端每拍都在把他往禁闭室推，前端只要跟着走就是一段连续的押送过程。
    e.escort = n.escort || 0;
    // 只有真被押上路了才跟着后端权威位置走；等人来抓的阶段后端位置是不动的，
    // 让他保持原地就行（dest 不改，走位交给下面的死区逻辑）。
    if (e.escort && n.escort_grabbed) e.dest.set(n.pos[0], 0, n.pos[1]);
    e.serverPos.set(n.pos[0], 0, n.pos[1]);
    // 关禁闭：后端已把人冻结在禁闭室里。前端硬吸附过去、就地待着——
    // 否则押送刚结束时前端还落在半路，会看成"押送半途就散了、没关成"。
    e.confined = n.solitary > 0;
    if (e.confined || (e.escort && n.escort_grabbed)) releaseStation(e);
    if (e.confined) {
      // 只给目标点，不再一帧硬吸附位置——落格瞬移就是从这来的。真正的贴合/钉住交给
      // 主循环：离得远快步走进去，到门口才 pos.set 钉死。
      e.dest.set(n.pos[0], 0, n.pos[1]);
    }
    // 典狱长点名回牢房：跟后端权威位置走，别再去抢食堂凳/别人的铺
    if (n.recalled) {
      releaseStation(e);
      e.dest.set(n.pos[0], 0, n.pos[1]);
    }
    e.goal = n.goal;
    e.inBout = !!n.in_bout;                        // 真在群殴中才播出拳动作，否则不打空气
    e.commanded = !!n.commanded;
    const act = n.activity || { verb: n.goal_label ?? "", icon: "" };
    const doing = n.action_state === "进行中";
    // 行动分阶段后，"进行中"太粗：得能看出他是在取物资、在做（带进度）、还是没做成。
    let stateTxt;
    if (n.escort) stateTxt = !n.escort_by ? `🔒自行走向禁闭室`
      : (n.escort_grabbed ? `🔒被押往禁闭室` : `🔒等着被带走`);
    else if (n.solitary) stateTxt = `⛓禁闭${n.solitary}`;
    else if (n.recalled) stateTxt = `🏠遣回牢房`;
    else if (n.action_state === "前往中") stateTxt = `前往${n.target_name || ""}`;
    else if (n.act_stage === "perform" && n.act_progress > 0)
      stateTxt = `进行中 ${Math.round(n.act_progress * 100)}%`;
    else stateTxt = n.action_state || (doing ? "进行中" : "");
    const replying = e.replyUntil && performance.now() < e.replyUntil;
    const say = replying ? `💬 ${e.replyText}` : (n.monologue ?? "…");
    // 被狱警/狱犬盯上：头顶示警 + 描边红；越狱准备进度用小徽标提示
    const watch = n.watched_by ? `<span class="watch">${n.watched_by === "狱犬" ? "🐕" : "👮"}被盯</span>` : "";
    const gate = n.cell_locked
      ? `<span class="hurt">${n.in_own_cell ? "🔒出不去" : "🔒进不去"}</span>` : "";
    const ep = n.escape_prep;
    let plan = "";
    const incs = (latestWorld && latestWorld.shift && latestWorld.shift.incidents) || [];
    const hit = incs.find((i) => i.npc_id === n.id || i.other_id === n.id);
    if (PUBLIC_MODE && hit) {
      plan = `<span class="plan">${hit.tag || "出事"}</span>`;
    } else if (!PUBLIC_MODE && ep && (ep.tool || ep.breach)) {
      const t = ep.tool ? "🔧" : "";
      const b = ep.breach ? `⛏${ep.breach}/${ep.need_breach}` : "";
      plan = `<span class="plan">${t}${b}</span>`;
    }
    // 冲突/物资/刑期的可见化：这些以前在世界里根本不会发生，现在得让人看见
    const inj = (n.needs && n.needs.injury) || 0;
    const hurt = inj >= 30 ? `<span class="hurt">🩸${Math.round(inj)}</span>` : "";
    const brawl = n.in_bout ? `<span class="brawl">🥊群殴</span>` : "";
    const hold = n.carrying
      ? `<span class="hold">${n.carrying === "food" ? "🍚拿着窝头" : "💉拿着药"}</span>` : "";
    const paro = n.extra_days
      ? `<span class="paro">⛓+${n.extra_days}天</span>`
      : (n.parole_near ? `<span class="paro ok">🕊假释在望</span>` : "");
    e.label.innerHTML =
      `<div class="nb-head">` +
        `<span class="who">${n.name}</span>` +
        (e.commanded ? `<span class="cmd">📢</span>` : "") + watch + gate + plan +
        brawl + hurt + hold + paro +
        `<span class="act">${act.icon}${act.verb}</span>` +
        `<span class="state${doing ? " doing" : ""}">${stateTxt}</span>` +
      `</div>` +
      `<div class="nb-say">${say}</div>`;
    e.label.classList.toggle("sel", n.id === selectedId);
    e.label.classList.toggle("cmded", e.commanded);
    e.label.classList.toggle("watched", !!n.watched_by);
  }
  renderChips(msg.npcs.filter((n) => zoneUnlocked(n.home_zone || n.zone)));
  renderElFarol(msg.attendance || [], msg.yard_comfort ?? 3, msg.n_agents ?? msg.npcs.length);
  renderDist(msg.choice_dist || {});
  renderBanner(msg.event_kind);
  renderPanel();
}

const EVENT_TEXT = {
  狱犬: "🐕 放狱犬！", 警报: "🚨 越狱警报拉响！",
  搜牢: "🔦 狱警搜牢！", 斗殴: "🥊 放风场斗殴！",
};
function renderBanner(kind) {
  const b = document.getElementById("banner");
  if (kind && EVENT_TEXT[kind]) { b.style.display = "block"; b.textContent = EVENT_TEXT[kind]; }
  else b.style.display = "none";
}

function renderElFarol(series, comfort, nAgents) {
  const cv = elfarolCanvas;
  const w = cv.width, h = cv.height, pad = 6;
  const g = cv.getContext("2d");
  g.clearRect(0, 0, w, h);
  const maxN = Math.max(1, nAgents);
  const yOf = (v) => h - pad - (v / maxN) * (h - 2 * pad);
  // 网格
  g.strokeStyle = "rgba(120,140,190,.12)"; g.lineWidth = 1;
  for (let v = 0; v <= maxN; v++) {
    const y = yOf(v); g.beginPath(); g.moveTo(pad, y); g.lineTo(w - pad, y); g.stroke();
  }
  // 舒适阈值虚线
  g.strokeStyle = "rgba(255,200,90,.7)"; g.setLineDash([4, 3]);
  const yc = yOf(comfort); g.beginPath(); g.moveTo(pad, yc); g.lineTo(w - pad, yc); g.stroke();
  g.setLineDash([]);
  // 人数折线
  if (series.length > 1) {
    const stepX = (w - 2 * pad) / (series.length - 1);
    g.strokeStyle = "#6ea8ff"; g.lineWidth = 1.8; g.beginPath();
    series.forEach((v, i) => {
      const x = pad + i * stepX, y = yOf(v);
      i === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
    });
    g.stroke();
    // 当前值圆点
    const lx = pad + (series.length - 1) * stepX, ly = yOf(series[series.length - 1]);
    g.fillStyle = "#6ea8ff"; g.beginPath(); g.arc(lx, ly, 2.5, 0, Math.PI * 2); g.fill();
  }
  g.fillStyle = "#aab3c8"; g.font = "10px system-ui";
  g.fillText(`现:${series[series.length - 1] ?? 0}/${maxN}  舒适:${comfort}`, pad + 2, pad + 8);
}

function renderDist(dist) {
  distEl.innerHTML = "";
  const used = new Set();
  for (const [personality, goals] of Object.entries(dist)) {
    const total = Object.values(goals).reduce((a, b) => a + b, 0) || 1;
    const row = document.createElement("div");
    row.className = "drow";
    const bar = Object.entries(goals)
      .sort((a, b) => b[1] - a[1])
      .map(([label, cnt]) => {
        used.add(label);
        const c = GOAL_COLOR[label] || "#888";
        return `<span style="width:${(cnt / total) * 100}%;background:${c}" title="${label} ${cnt}"></span>`;
      }).join("");
    row.innerHTML = `<span class="dnm">${personality}</span><span class="dbar">${bar}</span>`;
    distEl.appendChild(row);
  }
  dlegendEl.innerHTML = [...used]
    .map((l) => `<span><i style="background:${GOAL_COLOR[l] || "#888"}"></i>${l}</span>`).join("");
}

function renderNeeds(needs) {
  needsEl.innerHTML = "";
  for (const [key, nm, color] of NEED_META) {
    const v = needs?.[key] ?? 0;
    const row = document.createElement("div");
    row.className = "need-row";
    row.innerHTML = `<span class="nm">${nm}</span>` +
      `<span class="nb"><i style="width:${v}%;background:${color}"></i></span>` +
      `<span class="nv">${v}</span>`;
    needsEl.appendChild(row);
  }
}

function renderSituation(s) {
  sitEl.innerHTML = "";
  if (!s) return;
  const tags = [];
  if (s.phase) tags.push([`时段:${s.phase}`, ""]);
  if (s.threat && s.threat !== "无") tags.push([`威胁:${s.threat}`, "warn"]);
  if (s.event && s.event !== "无") tags.push([`事件:${s.event}`, "warn"]);
  if (s.enemy) tags.push(["仇人在场", "warn"]);
  if (s.backer) tags.push(["有靠山", "info"]);
  if (s.surveillance) tags.push([`监视:${s.surveillance}`, ""]);
  if (s.yard_crowd) tags.push([`放风:${s.yard_crowd}`, ""]);
  if (s.parole_near) tags.push(["快假释", "info"]);
  if (s.has_labor) tags.push(["有苦役", ""]);
  if (s.has_corpse) tags.push(["有尸体", "warn"]);
  for (const [t, cls] of tags) {
    const el = document.createElement("span");
    el.className = "sittag" + (cls ? " " + cls : "");
    el.textContent = t;
    sitEl.appendChild(el);
  }
}

function renderChips(list) {
  listEl.innerHTML = "";
  for (const n of list) {
    const c = document.createElement("span");
    c.className = "chip" + (n.id === selectedId ? " sel" : "");
    c.textContent = `${n.name}·${n.personality}`;
    c.onclick = () => { selectedId = n.id; onWorldState(latestWorld); };
    listEl.appendChild(c);
  }
}

function onNpcDecision(msg) {
  latestDecision.set(msg.npc_id, msg);
  if (msg.npc_id === selectedId) renderPanel();
}

function renderPanel() {
  updateCmdLabel();
  const d = selectedId ? latestDecision.get(selectedId) : null;
  if (!d) return;
  const act = d.activity ? `${d.activity.icon}${d.activity.verb} · ` : "";
  titleEl.textContent = `${d.name}（${d.personality}）`;
  titleEl.title = d.role || "";
  explainEl.textContent = act + (d.explain || "…");
  renderNeeds(d.needs);
  renderSituation(d.situation);
  if (d.expected_crowd !== undefined) {
    const pref = { like: "爱扎堆", avoid: "怕挤", neutral: "看情况" }[d.crowd_pref] || "";
    const over = d.expected_crowd > d.crowd_tol;
    const tag = document.createElement("span");
    tag.className = "sittag" + (over ? " warn" : " info");
    tag.textContent = `放风预期${d.expected_crowd}人·容忍${d.crowd_tol}·${pref}`;
    sitEl.appendChild(tag);
  }
  // 判决 / 苦役：狱警行权的对象，一眼可见
  if (d.sentence_years !== undefined) {
    const st = document.createElement("span");
    st.className = "sittag" + (d.busted ? " warn" : "");
    st.textContent = `刑期${d.sentence_years}年` + (d.crime ? `·${d.crime}` : "") +
      (d.busted ? `·被抓${d.busted}次` : "");
    sitEl.appendChild(st);
  }
  if (d.labor_job) {
    const lt = document.createElement("span");
    lt.className = "sittag";
    lt.textContent = `苦役:${d.labor_job}`;
    sitEl.appendChild(lt);
  }
  // 宏观动机：长期心思 + 推进弧线（性格的量化，非任务清单）
  renderDrive(d.drive);
  ctxEl.textContent = d.context;
  const max = Math.max(...d.candidates.map((c) => c.score), 1e-6);
  candsEl.innerHTML = "";
  for (const c of [...d.candidates].sort((a, b) => b.score - a.score)) {
    const div = document.createElement("div");
    div.className = "cand" + (c.chosen ? " chosen" : "");
    const pct = Math.max(2, (c.score / max) * 100);
    const srcCls = c.source === "command" ? ' style="color:#ffd23f"' : "";
    div.innerHTML =
      `<span class="g">${c.goal_label}${c.chosen ? " ✅" : ""}</span>` +
      `<span class="sc">${c.score.toFixed(2)}</span>` +
      `<div class="src"${srcCls}>来源:${c.source} · 模型v=${c.v.toFixed(2)} → 基础${fmtNum(c.m_value)}</div>` +
      `<div class="terms">${termChips(c)}</div>` +
      `<div class="bar"><i style="width:${pct}%"></i></div>`;
    candsEl.appendChild(div);
  }
}

function fmtNum(v) {
  if (v === undefined) return "-";
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
}

/** 宏观动机：长期心思一句话 + 推进弧线。渲染成 situation 区的两个标签，不新增 DOM。 */
function renderDrive(dr) {
  if (!dr) return;
  const mt = document.createElement("span");
  mt.className = "sittag" + (dr.parole_lost ? " warn" : " info");
  mt.textContent = (dr.parole_lost ? "💥 " : "🎯 ") + dr.motive;
  mt.title = `长期心思：${dr.motive}\n怕失去：${dr.stake}` +
    (dr.signature ? `\n签名动作：${dr.signature}` : "");
  sitEl.appendChild(mt);
  if (dr.progress !== undefined && !dr.parole_lost) {
    const pt = document.createElement("span");
    const near = dr.progress >= 80;
    pt.className = "sittag" + (near ? " info" : "");
    pt.textContent = `动机推进 ${dr.progress}%` + (near ? "·眼看就成" : "");
    sitEl.appendChild(pt);
  }
}

/** 打分公式的乘数拆解。只渲染真正起作用的项，全是 1.0 的时候不占地方。 */
function termChips(c) {
  const out = [];
  const mul = (label, v, hint) => {
    if (v === undefined || Math.abs(v - 1) < 0.01) return;
    const cls = v > 1 ? "up" : "dn";
    out.push(`<b class="${cls}" title="${hint}">${label}×${v.toFixed(2)}</b>`);
  };
  mul("惯性", c.m_inertia, "正在做的目标有加成，刚下决心时更难被撼动");
  if (c.cost !== undefined && c.cost > 0.01) {
    out.push(`<b class="dn" title="距离成本，越远分越低">距离÷${(1 + c.cost).toFixed(2)}</b>`);
  }
  mul("环境", c.m_light, "环境/状态修正，例如熄灯后不方便出门");
  mul("打断", c.m_interruption, "被威胁或指令打断过，给补偿好回去收尾");
  mul("失败", c.m_failure, "这个目标去不了，衰减");
  mul("超时", c.m_timeout, "长时间到不了，衰减");
  mul("重复", c.m_repetition, "短期内反复做同一件事，衰减");
  mul("拥挤", c.m_crowd, "对放风人数的预期（El Farol 负反馈）");
  mul("指令", c.m_command, "玩家指令保底胜；生存告急时会减弱，允许抗命");
  mul("动机", c.m_drive, "宏观动机：推进长期心思的行为抬一档、损耗的压一档、禁忌≈0");
  mul("盯防", c.m_watch, "被狱警/狱犬盯上时压制违禁动作（≈0）");
  if (c.cached === false) {
    out.push('<b class="stale" title="模型分还没回来，本拍按 0 处理">待打分</b>');
  }
  return out.join("");
}

// ---------------------------------------------------------------------------
// 点击拾取 NPC（区分拖拽相机 vs 点击选人）
// ---------------------------------------------------------------------------
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let downPos = null;
renderer.domElement.addEventListener("pointerdown", (ev) => {
  downPos = { x: ev.clientX, y: ev.clientY };
});
renderer.domElement.addEventListener("pointerup", (ev) => {
  if (!downPos) return;
  const moved = Math.hypot(ev.clientX - downPos.x, ev.clientY - downPos.y);
  downPos = null;
  if (moved > 6) return; // 视为拖拽相机
  mouse.x = (ev.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(ev.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  // 先看是否点到线索物（第三步用），否则点 NPC 选中
  const clueHit = raycaster.intersectObjects(clueMeshes, true)[0];
  if (clueHit) {
    let o = clueHit.object;
    while (o && !o.userData.clueId) o = o.parent;
    if (o && typeof window.__inspectClue === "function") { window.__inspectClue(o.userData.clueId); return; }
  }
  // 再看是否点到可互动家具：够得着就直接操作，够不着提示走近
  const actObjs = interactables.filter((i) => !i.taken).map((i) => i.obj);
  const actHit = raycaster.intersectObjects(actObjs, true)[0];
  if (actHit) {
    let o = actHit.object;
    while (o && !o.userData.entry) o = o.parent;
    if (o && player) {
      const it = o.userData.entry;
      const d = o.getWorldPosition(_ap).distanceTo(player.root.position);
      if (d <= ACT_RANGE + 0.6) doInteract(it);
      else showToast(`${it.icon} ${it.name} 离得有点远，走近些再${actVerb(it)}。`);
      return;
    }
  }
  const hit = raycaster.intersectObjects(npcRoots, true)[0];
  if (hit) {
    let o = hit.object;
    while (o && o.userData.id === undefined) o = o.parent;
    if (o) {
      selectedId = o.userData.id;
      if (latestWorld) onWorldState(latestWorld);
      openDecisionCard(o.userData.id);   // 观察者：点 NPC 就弹决策卡
    }
  }
});

// 悬停高亮：可交互物(NPC/主角/线索)描边，让"能查的东西"一眼可见
const hoverMouse = new THREE.Vector2();
renderer.domElement.addEventListener("pointermove", (ev) => {
  if (!outlinePass) return;
  hoverMouse.x = (ev.clientX / window.innerWidth) * 2 - 1;
  hoverMouse.y = -(ev.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(hoverMouse, camera);
  const targets = npcRoots.concat(
    player ? [player.root] : [], clueMeshes,
    interactables.filter((i) => !i.taken).map((i) => i.obj));
  const hit = raycaster.intersectObjects(targets, true)[0];
  let root = null;
  if (hit) { let o = hit.object; while (o && !o.userData.hoverRoot) o = o.parent; root = o; }
  outlinePass.selectedObjects = root ? [root] : [];
  renderer.domElement.style.cursor = root ? "pointer" : "default";
});

// ---------------------------------------------------------------------------
// 观察者模式：决策卡（点 NPC 弹）+ 两脑对比（Demo 核心）+ 上帝暂停
// ---------------------------------------------------------------------------
let deckOpen = false, deckId = null;
const compareCache = {};      // npc_id -> 最近一次两脑对比结果（切走再回来仍在）
const NEED_CN = { hunger: "饿", thirst: "渴", fatigue: "乏", injury: "伤",
                  boredom: "闷", social: "社", stress: "压" };
const TIER_CN = { command: "玩家指令", threat: "危机/威胁", need: "生存需求",
                  drive: "长期心思", daily: "日常", idle: "发呆" };

function npcData(id) {
  return ((latestWorld && latestWorld.npcs) || []).find((n) => n.id === id) || null;
}

function needMini(needs) {
  if (!needs) return "";
  const rows = Object.keys(NEED_CN).map((k) => {
    const v = needs[k] || 0;
    const col = v >= 82 ? "#ff5a5a" : (v >= 60 ? "#ffb454" : "#4f8cff");
    const vc = v >= 70 ? ' style="color:#ff9b9b"' : "";
    return `<span style="display:inline-flex;align-items:center;gap:5px;width:104px;margin:1px 4px 1px 0;font-size:11px">
      <span style="width:14px;color:#b7bdcc">${NEED_CN[k]}</span>
      <span style="flex:1;height:6px;background:#2a2f3d;border-radius:3px;overflow:hidden"><i style="display:block;height:100%;width:${v}%;background:${col}"></i></span>
      <span style="width:20px;text-align:right;color:#8b93a7"${vc}>${v}</span></span>`;
  }).join("");
  return `<div style="display:flex;flex-wrap:wrap;margin:6px 0 8px">${rows}</div>`;
}

function renderCompare(c) {
  if (!c) return "";
  if (!c.ok) return `<div class="verdict na">对比暂不可用：${c.reason || ""}</div>`;
  const v = c.value, b = c.base, vd = c.verdict || {};
  const vcls = vd.guardrail === "violated" ? "violated"
    : (vd.guardrail === "both_ok" ? "both_ok" : "na");
  const icon = vd.guardrail === "violated" ? "⚠ " : (vd.guardrail === "both_ok" ? "✓ " : "");
  const barRow = (rank, top) => rank.map((r) => {
    const max = Math.max(...rank.map((x) => x.want), 1e-6);
    const win = r.goal === top;
    const pct = Math.max(3, (r.want / max) * 100);
    return `<div class="rk ${win ? "win" : ""}">${win ? "▶ " : ""}${r.label}
      <div class="rkbar"><i style="width:${pct}%"></i></div></div>`;
  }).join("");
  const say = b.say ? `<div class="say">裸模型独白：「${b.say}」</div>` : "";
  return `<div class="verdict ${vcls}">${icon}${vd.note || ""}</div>
    <div class="twocol">
      <div class="brainbox value"><div class="bh">${v.label}</div>
        <div class="top">→ ${v.top_label || "?"}</div>${barRow(v.rank, v.top)}</div>
      <div class="brainbox base"><div class="bh">${b.label}</div>
        <div class="top">→ ${b.top_label || "?"}</div>${barRow(b.rank, b.top)}${say}</div>
    </div>
    <div class="costline">⏱ 微调：<b>~${v.ms_per_npc}ms/人</b>（整批 ${v.batch_ms}ms 覆盖全部 NPC · 本地/T4 · 零 API 费）
      　·　裸本地：<b>${b.ms}ms/次</b>（每个决策都要一次大模型调用）</div>`;
}

// ---------------------------------------------------------------------------
// 公开夜班：班次 HUD + 风险弹窗 + 前往现场 + 简化档案卡
// ---------------------------------------------------------------------------
const CARRY_CN = { food: "窝头", medkit: "医疗包", water: "水" };
const QQS = [
  ["where_were_you", "刚才你在哪儿？"],
  ["what_are_you_planning", "你在盘算什么？"],
  ["where_item_from", "身上藏了什么？"],
  ["who_conflict_with", "跟谁过不去？"],
];
let shiftGotoId = null;
let shiftSeen = new Set();
let expandedIncidentId = null;   // 事件栈手风琴：当前展开的那一张（其余折叠成标题条）
let lastIncidentList = [];       // 缓存上次事件列表，供折叠条点击时原地重渲染
let shiftEndedShown = false;

function shiftAct(action, extra = {}) {
  sendMsg({ type: "shift_act", action, ...extra });
}

function fmtClock(sec) {
  const s = Math.max(0, sec | 0);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function tone(v, good, mid) {
  return v >= good ? "ok" : (v >= mid ? "warn" : "bad");
}

let shiftHudWired = false;
function renderShiftHud(sh) {
  const el = document.getElementById("shifthud");
  if (!el || !PUBLIC_MODE || !sh) return;
  el.style.display = "block";
  const remain = sh.remain_s ?? 0;
  const riot = sh.riot_label || "";
  const hist = (sh.history || []).slice(-3).map((h) => h.text).filter(Boolean).join(" · ");
  if (!shiftHudWired) {
    el.innerHTML =
      `<div class="sh-title">典狱长夜班 · 守到天亮</div>` +
      `<div class="sh-row">` +
        `<span>剩余 <b id="sh-clock">0:00</b></span>` +
        `<span>权力 <b id="sh-ap">0</b></span>` +
        `<span>秩序 <b id="sh-ord">0</b></span>` +
        `<span>伤亡 <b id="sh-cas">0</b></span>` +
        `<span>正当性 <b id="sh-leg">0</b></span>` +
        `<span id="sh-riotwrap">监区 <b id="sh-riot">平静</b></span>` +
      `</div>` +
      `<div class="sh-powers">` +
        `<button class="pbtn" data-p="recall_all">🏠 全体回牢房（2）</button>` +
        `<button class="pbtn" data-p="lock_solitary">🗝 锁禁闭室（1）</button>` +
      `</div>` +
      `<div class="sh-log" id="sh-log"></div>`;
    el.querySelectorAll("[data-p]").forEach((b) => {
      b.onclick = () => {
        const a = b.getAttribute("data-p");
        if (a === "lock_solitary") shiftAct("lock_solitary", { locked: true });
        else shiftAct(a);
      };
    });
    shiftHudWired = true;
  }
  const set = (id, text, cls) => {
    const n = el.querySelector(id);
    if (!n) return;
    n.textContent = text;
    if (cls) n.className = cls;
  };
  set("#sh-clock", fmtClock(remain), remain < 60 ? "bad" : "ok");
  set("#sh-ap", `${sh.authority}/${sh.authority_max}`, tone(sh.authority, 3, 1));
  set("#sh-ord", String(sh.order), tone(sh.order, 70, 45));
  set("#sh-cas", String(sh.casualties), sh.casualties ? "bad" : "ok");
  set("#sh-leg", String(sh.legitimacy), tone(sh.legitimacy, 60, 35));
  const rw = el.querySelector("#sh-riotwrap");
  const rb = el.querySelector("#sh-riot");
  if (rw && rb) {
    rw.style.display = "inline";
    const stage = Math.max(0, Math.min(4, sh.riot | 0));
    rb.textContent = riot || "平静";
    rb.className = `riot-${stage}`;
  }
  const log = el.querySelector("#sh-log");
  if (log) log.textContent = hist;
}

function showShiftEnding(sh) {
  const el = document.getElementById("shiftend");
  if (!el) return;
  el.style.display = "flex";
  if (shiftEndedShown) return;
  shiftEndedShown = true;
  const fates = (sh.fates || []).map((f) =>
    `<div style="margin:5px 0;font:500 13px/1.55 system-ui;color:#dccdd0"><b style="color:#ffb3b3">${f.name}</b>　${f.line}</div>`
  ).join("");
  const br = (sh.breakdown || []).map((x) =>
    `<div style="font:500 12px/1.5 system-ui;color:#c9b477">${x}</div>`
  ).join("");
  el.innerHTML =
    `<div class="card">` +
      `<div style="font:800 12px system-ui;color:#8b93a7;letter-spacing:3px">夜班结束</div>` +
      `<div style="font:800 28px system-ui;color:#ffe08a;margin:8px 0 12px">${sh.ending_title || "收场"}</div>` +
      `<div style="font:500 15px/1.7 system-ui;color:#d8dde8">${sh.ending_blurb || ""}</div>` +
      `<div style="margin-top:14px">${br}</div>` +
      buildWhodunit(sh) +
      (fates ? `<div style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(255,255,255,.12)">` +
        `<div style="font:800 12px system-ui;color:#8b93a7;margin-bottom:6px">每人下落</div>${fates}</div>` : "") +
      `<div id="shiftprose">（正在为这一夜落笔…7B 若超时，上面的鉴定仍然算数。）</div>` +
      `<button id="shiftagain" style="margin-top:22px;padding:10px 22px;border-radius:10px;cursor:pointer;` +
        `font:800 14px system-ui;color:#161018;border:1px solid #ffd98a;` +
        `background:linear-gradient(180deg,#ffe08a,#e0a840)">重新接班</button>` +
    `</div>`;
  const btn = el.querySelector("#shiftagain");
  if (btn) btn.onclick = () => {
    sendMsg({ type: "scenario", name: "new_game" });
    setTimeout(() => location.reload(), 600);
  };
  wireWhodunit(el);
  sendMsg({ type: "generate_ending", shift: true, text: "" });
}

// 天亮"凶手指认"：把每桩既遂命案立成一道题，玩家从嫌疑人里点真凶，成败计入评价
let whodunitNames = {};        // id -> name，回执揭晓时查名字用
function buildWhodunit(sh) {
  const cases = sh.whodunit || [];
  if (!cases.length) return "";
  const suspects = sh.suspects || [];
  whodunitNames = {};
  for (const s of suspects) whodunitNames[s.id] = s.name;
  const rows = cases.map((c) => {
    if (c.guessed) return `<div class="wd-row" data-vid="${c.victim_id}">${whodunitReveal(c)}</div>`;
    const opts = suspects
      .filter((s) => s.id !== c.victim_id)
      .map((s) => `<option value="${s.id}">${s.name}</option>`).join("");
    return `<div class="wd-row" data-vid="${c.victim_id}" style="display:flex;align-items:center;gap:8px;margin:6px 0;flex-wrap:wrap">` +
      `<span style="font:500 13px system-ui;color:#dccdd0">遇害 <b style="color:#ff9a9a">${c.victim_name}</b> — 谁是真凶？</span>` +
      `<select class="wd-pick" style="padding:5px 8px;border-radius:8px;background:#241a22;color:#f0e6ea;border:1px solid #5a4652;font:600 13px system-ui">` +
        `<option value="">选一个人…</option>${opts}</select>` +
      `<button class="wd-go" style="padding:5px 14px;border-radius:8px;cursor:pointer;font:800 13px system-ui;` +
        `color:#161018;border:1px solid #ffd98a;background:linear-gradient(180deg,#ffe08a,#e0a840)">指认</button>` +
    `</div>`;
  }).join("");
  const total = sh.whodunit_total || cases.length;
  const solved = sh.whodunit_solved || 0;
  return `<div style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(255,255,255,.12)">` +
    `<div style="font:800 12px system-ui;color:#8b93a7;margin-bottom:6px">凶案指认 · 天亮定案</div>` +
    `<div id="wd-verdict" style="font:700 12px system-ui;color:#c9b477;margin-bottom:6px">断案 ${solved}/${total}：指认成败计入当晚正当性</div>` +
    `${rows}</div>`;
}

function whodunitReveal(c) {
  if (c.correct) {
    return `<span style="font:600 13px system-ui;color:#8fe08f">✓ ${c.victim_name} 一案：指认无误，真凶 <b>${c.killer_name || whodunitNames[c.killer_id] || "？"}</b></span>`;
  }
  const guessName = whodunitNames[c.guess_id] || "那人";
  return `<span style="font:600 13px system-ui;color:#ff9a9a">✗ ${c.victim_name} 一案：你指认了 ${guessName}，真凶其实是 <b>${c.killer_name || whodunitNames[c.killer_id] || "？"}</b></span>`;
}

function wireWhodunit(root) {
  root.querySelectorAll(".wd-row").forEach((row) => {
    const btn = row.querySelector(".wd-go");
    const pick = row.querySelector(".wd-pick");
    if (!btn || !pick) return;
    btn.onclick = () => {
      const sid = pick.value;
      if (!sid) { showToast("先选一个人再指认"); return; }
      btn.disabled = true; pick.disabled = true;
      shiftAct("whodunit_guess", { victim_id: row.dataset.vid, suspect_id: sid });
    };
  });
}

function pullWardenTo(pt) {
  if (!player || !pt) return;
  if (typeof occupying !== "undefined" && occupying) standUp();
  const x = Number(pt.x), z = Number(pt.z);
  if (!Number.isFinite(x) || !Number.isFinite(z)) return;
  // 不再瞬移：设一个自动前往的目标点，movePlayer 每帧朝它走，玩家按 WASD 立刻接管
  walkTarget = { x, z, npc_id: pt.npc_id || null };
  walkStuck = 0;
  if (pt.npc_id) {
    shiftGotoId = pt.npc_id;
    selectedId = pt.npc_id;
  }
  if (typeof setCamMode === "function") setCamMode("follow");
  showToast("正赶过去…");
}

function gotoIncident(inc) {
  if (!inc) return;
  if (inc.npc_id) {
    shiftGotoId = inc.npc_id;
    selectedId = inc.npc_id;
  }
  if (inc.id) {
    shiftAct("incident_choice", { incident_id: inc.id, choice: "goto" });
    return;
  }
  const e = inc.npc_id && npcs.get(inc.npc_id);
  if (e) {
    const p = e.root.position;
    pullWardenTo({ x: p.x + 1.8, z: p.z + 1.1, npc_id: inc.npc_id, npc_name: inc.npc_name });
  } else if (camMode === "god" || camMode === "top" || camMode === "spectate") {
    setCamMode("follow");
  }
  showToast(`前往 ${inc.npc_name || "现场"}`);
  if (latestWorld) renderChips(latestWorld.npcs);
}

const incMarks = new Map();
function makePulseRing(color) {
  // 出事 NPC 的脚底光圈 + 向上光柱（“身上的光的投射”）按需求整体删除：视觉太满/太抢眼。
  // 保留空 group 让 updateIncidentMarkers 的定位逻辑无需改动；出事信息改由头顶「出事」
  // 标签 + 屏幕方向条（gotobar）呈现，不再在地面/头顶打光。
  const g = new THREE.Group();
  scene.add(g);
  return g;
}
function updateIncidentMarkers() {
  if (!PUBLIC_MODE || (latestWorld && latestWorld.shift && latestWorld.shift.ended)) {
    for (const m of incMarks.values()) m.visible = false;
    return;
  }
  const incs = (latestWorld && latestWorld.shift && latestWorld.shift.incidents) || [];
  const ids = new Set();
  for (const inc of incs) {
    if (inc.npc_id) ids.add(inc.npc_id);
    if (inc.other_id) ids.add(inc.other_id);
  }
  if (shiftGotoId) ids.add(shiftGotoId);
  for (const [id, m] of [...incMarks]) {
    if (!ids.has(id)) { scene.remove(m); incMarks.delete(id); }
  }
  const t = performance.now() * 0.005;
  for (const id of ids) {
    if (!incMarks.has(id)) incMarks.set(id, makePulseRing(0xffc078));
    const m = incMarks.get(id);
    const e = npcs.get(id);
    if (!e) { m.visible = false; continue; }
    m.visible = true;
    m.position.set(e.root.position.x, 0.06, e.root.position.z);
    const focus = id === shiftGotoId;
    m.scale.setScalar((focus ? 1.18 : 1) * (1 + Math.sin(t) * 0.14));
    if (m.userData.ring) m.userData.ring.material.color.setHex(focus ? 0xff6b4a : 0xffc078);
  }
}

let shiftGotoFar = true;
function updateGotoBar() {
  updateIncidentMarkers();
  const bar = document.getElementById("gotobar");
  if (!bar || !PUBLIC_MODE) return;
  if (!shiftGotoId || !player) { bar.style.display = "none"; return; }
  const e = npcs.get(shiftGotoId);
  if (!e) { bar.style.display = "none"; return; }
  const d = e.root.position.distanceTo(player.root.position);
  const near = d <= 5;
  if (near && shiftGotoFar && latestWorld && latestWorld.shift) {
    renderIncidentStack(latestWorld.shift.incidents);
  }
  shiftGotoFar = !near;
  if (near) { bar.style.display = "none"; return; }
  const dx = e.root.position.x - player.root.position.x;
  const dz = e.root.position.z - player.root.position.z;
  let rel = Math.atan2(dx, dz) - (player.heading || 0);
  while (rel > Math.PI) rel -= Math.PI * 2;
  while (rel < -Math.PI) rel += Math.PI * 2;
  const dirs = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"];
  const ix = Math.round(((rel + Math.PI * 2) % (Math.PI * 2)) / (Math.PI / 4)) % 8;
  bar.style.display = "block";
  const auto = walkTarget ? "自动前往" : "前往";
  const tail = walkTarget ? "WASD 可接管" : "WASD";
  bar.textContent = `${dirs[ix]} ${auto} ${npcName(shiftGotoId)} · 还差 ${d.toFixed(1)} 米 · ${tail}`;
}

function renderIncidentStack(list) {
  const el = document.getElementById("incidentstack");
  if (!el || !PUBLIC_MODE) return;
  const open = (list || []).filter((i) => i.status !== "resolved");
  lastIncidentList = open;
  if (!open.length) {
    el.classList.remove("show");
    el.innerHTML = "";
    return;
  }
  el.classList.add("show");
  // 手风琴：同一时刻只展开一张，其余折叠成标题条，避免竖排铺满右侧、挡视线。
  // 展开态优先沿用上次选中；失效时挑最紧急(hot)→最近(in_range)→第一张。
  const ids = open.map((i) => i.id);
  if (!expandedIncidentId || !ids.includes(expandedIncidentId)) {
    const hot = open.find((i) => ["fight_now", "kill_prep"].includes(i.kind));
    const near = open.find((i) => i.in_range);
    expandedIncidentId = (hot || near || open[0]).id;
  }
  const many = open.length > 1;
  const farOk = (inc) => new Set(inc.far_ok || ["goto", "ask", "ignore", "observe", "recall_all"]);
  el.innerHTML = open.map((inc) => {
    const collapsed = many && inc.id !== expandedIncidentId;
    const near = !!inc.in_range;
    const pair = inc.other_name ? ` · ${inc.npc_name} vs ${inc.other_name}` : "";
    const dist = inc.dist != null
      ? `距离 ${inc.dist} 米${near ? " · 可以处置" : " · 前往解决可直接拉过去"}`
      : "";
    const allow = farOk(inc);
    const choices = (inc.choices || []).map((c) => {
      const needWalk = !allow.has(c.id) && !near;
      const dis = needWalk ? "disabled" : "";
      return `<button class="ibtn" data-iid="${inc.id}" data-c="${c.id}" ${dis} title="${c.hint || ""}">${c.label}${c.cost ? `（${c.cost}）` : ""}</button>`;
    }).join("");
    const hot = ["fight_now", "kill_prep"].includes(inc.kind) ? " hot" : "";
    const caret = many ? `<span class="icaret">${collapsed ? "▸" : "▾"}</span>` : "";
    const title = `⚠ ${inc.title}${pair && !String(inc.title).includes("vs") ? pair : ""}`;
    return `<div class="incard${hot}${collapsed ? " collapsed" : ""}" data-id="${inc.id}">` +
      `<div class="ih" data-head="${inc.id}">${caret}${title}</div>` +
      (dist ? `<div class="idist">${dist}</div>` : "") +
      `<div class="isigns">${inc.signs || ""}</div>` +
      (inc.fog ? `<div class="ifog">❔ ${inc.fog}</div>` : "") +
      `<div class="irow">${choices}</div></div>`;
  }).join("");
  // 点标题条：展开这张、折叠其余（多起事件时才有意义）
  if (many) {
    el.querySelectorAll("[data-head]").forEach((h) => {
      h.onclick = () => {
        const id = h.getAttribute("data-head");
        expandedIncidentId = (id === expandedIncidentId) ? null : id;
        renderIncidentStack(lastIncidentList);
      };
    });
  }
  el.querySelectorAll("[data-c]").forEach((b) => {
    b.onclick = () => {
      if (b.disabled) return;
      const iid = b.getAttribute("data-iid");
      const c = b.getAttribute("data-c");
      shiftAct("incident_choice", { incident_id: iid, choice: c });
    };
  });
}

function syncShift(msg) {
  if (!PUBLIC_MODE) return;
  const sh = msg.shift;
  if (!sh) return;
  renderShiftHud(sh);
  const incs = sh.incidents || [];
  for (const inc of incs) {
    if (!shiftSeen.has(inc.id)) {
      shiftSeen.add(inc.id);
      showToast(`⚠ ${inc.title}`);
    }
  }
  renderIncidentStack(incs);
  if (sh.ended) showShiftEnding(sh);
}

function knownItemsLine(n) {
  const sh = (latestWorld && latestWorld.shift) || {};
  const intel = (sh.known || {})[n.id] || {};
  const bits = [];
  if (n.carrying && CARRY_CN[n.carrying]) bits.push(CARRY_CN[n.carrying]);
  if (intel.weapon) bits.push(intel.weapon);
  else if (intel.searched) bits.push("搜过身：未发现凶器");
  return bits.length ? bits.join("、") : "尚无已知违禁";
}
function knownNotesLine(n) {
  const sh = (latestWorld && latestWorld.shift) || {};
  const intel = (sh.known || {})[n.id] || {};
  const notes = intel.notes || [];
  if (!notes.length) return "";
  return notes.slice(-4).map((x) => `· ${x}`).join("<br>");
}

function renderPublicDeck(el, n) {
  const bs = "padding:4px 8px;margin:2px 4px 2px 0;border-radius:7px;cursor:pointer;" +
    "font:700 11px system-ui;background:rgba(28,32,44,.92);color:#dfe6f2;border:1px solid rgba(255,255,255,.18);";
  const ult = n.ultimate || n.plot_surface || n.core_goal || "—";
  const surface = n.plot_surface || "";
  const step = n.plot_step || n.substep || "";
  const fog = n.plot_fog || "";
  const total = n.quest_total || 0;
  const qstep = (typeof n.quest_step === "number" ? n.quest_step : 0);
  const pct = total ? Math.min(100, Math.round((qstep / total) * 100)) : 0;
  const KIND_CN = { kill: "🗡 复仇", survive: "🛡 求生", escape: "🪜 越狱" };
  const kindTag = n.quest_kind && KIND_CN[n.quest_kind]
    ? ` <span style="font-size:10px;color:#ff9c93">${KIND_CN[n.quest_kind]}</span>` : "";
  const soc = n.social || {};
  const foes = (soc.foes && soc.foes.length) ? soc.foes.join("、") : "";
  const items = knownItemsLine(n);
  const notes = knownNotesLine(n);
  const cell = [...cellDoors.values()].find((r) => r.npc === deckId);
  const sh = (latestWorld && latestWorld.shift) || {};
  const ap = sh.authority != null ? sh.authority : 6;
  const costHint = (c) => ap < c ? "opacity:.45" : "";
  let cmds = "";
  if (n.dead) {
    cmds = `<div class="dkcore" style="border-color:#6a5560;color:#d9c6cd">此人已死，卷宗封存。</div>`;
  } else {
    const soli = n.solitary > 0
      ? `<button disabled style="${bs}opacity:.5">已在禁闭</button>`
      : `<button data-w="solitary" style="${bs}border-color:rgba(255,150,90,.7);color:#ffcf9f;${costHint(2)}">🔒 关禁闭（2）</button>`;
    const rec = (n.solitary > 0 || !cell) ? ""
      : `<button data-w="recallcell" style="${bs}border-color:rgba(150,200,160,.6);color:#c8e8cf;${costHint(1)}">🏠 遣回牢房（1）</button>`;
    const cellBtn = cell
      ? `<button data-w="cell" data-slot="${cell.slot}" data-lock="${cell.locked ? 0 : 1}" style="${bs}border-color:rgba(180,200,255,.6);color:#cfe0ff;${costHint(1)}">${cell.locked ? "🔓 开他的牢门" : "🗝 锁他的牢门（1）"}</button>`
      : "";
    cmds =
      `<div class="dkcore" style="border-color:#7a6a3a;color:#ffe6a6">` +
      `<div style="color:#c9b477;font-size:11px;margin-bottom:5px">🎖 指令 · 剩余权力 ${ap}</div>` +
      `<div>` +
        `<button data-w="chat" style="${bs}border-color:rgba(255,210,120,.6);color:#ffe6a6">🗣 审讯</button>` +
        `<button data-w="search" style="${bs}border-color:rgba(120,220,150,.6);color:#bff0c8;${costHint(1)}">🔍 搜身（1）</button>` +
        `${soli}${rec}${cellBtn}` +
      `</div></div>`;
  }
  el.innerHTML =
    `<span class="dkclose" id="dkclose">✕</span>` +
    `<h3>${n.name}</h3><div class="dkrole">${n.personality} · ${n.role || ""}</div>` +
    `<div class="dkcore">🏁 图谋${kindTag}：${ult}</div>` +
    (surface && surface !== ult ? `<div class="dkcore" style="border-color:#4a5a6a;color:#bccadf">📖 明面上：${surface}</div>` : "") +
    (n.kill_target_name ? `<div class="dkcore" style="border-color:#b5433a;color:#ffd0cb">🎯 要了结的人：<b style="color:#ff9c93">${n.kill_target_name}</b></div>` : "") +
    (foes ? `<div class="dkcore" style="border-color:#7a4a4a;color:#ffb9b0">💢 结着仇：${foes}</div>` : "") +
    (step ? `<div class="dkcore" style="border-color:#6a5a3a;color:#e6d6b0">📍 眼下这一步：${step}` +
        (total ? `<div style="color:#8b93a7;font-size:10px;margin-top:3px">进度 ${Math.min(qstep + 1, total)}/${total}</div>` +
          `<div class="rkbar" style="margin-top:3px"><i style="width:${pct}%"></i></div>` : "") +
      `</div>` : "") +
    (fog ? `<div class="dkcore" style="border-color:#3a4a5a;color:#9aa8c0">❔ 还看不透：${fog}</div>` : "") +
    `<div class="dkcore" style="border-color:#5a6a8a;color:#cdd6ea">📦 已知物品：${items}</div>` +
    (notes ? `<div class="dkcore" style="border-color:#6a5a3a;color:#e6d6b0"><div style="color:#c9b477;font-size:11px;margin-bottom:3px">🗒 审讯记下的</div>${notes}</div>` : "") +
    cmds;
  document.getElementById("dkclose").onclick = closeDecisionCard;
  el.querySelectorAll("[data-w]").forEach((b) => {
    b.onclick = () => {
      const w = b.getAttribute("data-w");
      if (w === "chat") { openChat(deckId); return; }
      if (w === "search") { shiftAct("search", { npc_id: deckId }); return; }
      if (w === "solitary") { shiftAct("solitary", { npc_id: deckId }); return; }
      if (w === "recallcell") { shiftAct("recall", { npc_id: deckId }); return; }
      if (w === "cell") {
        const unlocking = b.getAttribute("data-lock") === "0";
        if (unlocking) {
          sendMsg({ type: "cell_door", slot: Number(b.getAttribute("data-slot")), locked: false });
        } else {
          shiftAct("lock_cell", { npc_id: deckId });
        }
      }
    };
  });
  el.style.display = "block";
}

function renderDeck() {
  const el = document.getElementById("deck");
  const n = npcData(deckId);
  if (!n) { el.style.display = "none"; deckOpen = false; return; }
  if (PUBLIC_MODE) { renderPublicDeck(el, n); return; }
  const dr = n.drivers || {};
  const tierCn = TIER_CN[dr.tier] || "";
  const tierCls = (dr.tier === "threat" || dr.tier === "need") ? "warn" : "info";
  // 双脑协作：把"这一刻两个微调模型各自干了什么"摊开——1.5B 定行动、7B 配台词、两者信息互通。
  const act = n.goal_label || dr.label || "发呆";
  const line = (n.monologue && n.monologue !== "…") ? n.monologue : "";
  let vo = `<div class="dkbrains">`
    + `<div class="brainrow decide"><span class="btag">🧠 决策脑 · 微调 1.5B</span>`
    + `<div class="bmain">选择行动：<b>「${act}」</b>`
    + (tierCn ? ` <span class="sittag ${tierCls}" style="font-size:10px">${tierCn}</span>` : "")
    + `</div><div class="bwhy">${dr.why || "…"}</div></div>`
    + `<div class="brainrow speak"><span class="btag">🎙 语言脑 · 微调 7B</span>`
    + `<div class="bmain">${line ? `「${line}」` : "（此刻沉默）"}</div>`
    + (line ? `<div class="bwhy">↳ 台词是行为的『配音』，与上面这步行动一致</div>` : "")
    + `</div>`
    + `<div class="binterlink">🔄 两脑信息互通：决策脑把处境喂给语言脑配台词；对话若改动关系/承诺，又回写决策脑、影响下一步。</div>`;
  if (dr.watched_by) vo += `<div class="bwatch">⚠ ${dr.watched_by}正盯着它</div>`;
  vo += `</div>`;
  const logs = (n.decision_log || []).slice(-6).reverse();
  const logHtml = logs.length
    ? `<div class="dklog"><div style="color:#8b93a7;font-size:11px;margin-bottom:2px">🧠 重大决策流水（近→远）</div>`
      + logs.map((l) => `<div class="lg">${l.big ? "❗" : "·"} <b>${l.label}</b> — ${l.why}</div>`).join("")
      + `</div>`
    : "";
  const solitary = n.solitary > 0
    ? `<div class="dkcore" style="border-color:#b5762a;color:#ffd9a0">🔒 正被关在禁闭室</div>`
    : (n.recalled
      ? `<div class="dkcore" style="border-color:#5a7a4a;color:#c8e8cf">🏠 被典狱长点名，正回牢房待着</div>` : "");
  // 终极目标 + 当前分目标（任务链）：让玩家看清"他这辈子图什么、这一步在干嘛、走到哪了"
  const ult = n.ultimate || dr.ultimate || n.core_goal || "";
  const total = n.quest_total || dr.step_total || 0;
  const step = (typeof n.quest_step === "number" ? n.quest_step : 0);
  const sub = n.substep || dr.substep || "";
  const pct = total ? Math.min(100, Math.round((step / total) * 100)) : 0;
  const KIND_CN = { kill: "🗡 复仇/杀戮", survive: "🛡 求生", escape: "🪜 越狱" };
  const kindTag = n.quest_kind && KIND_CN[n.quest_kind] ? ` <span style="font-size:11px;color:#ff9c93">${KIND_CN[n.quest_kind]}</span>` : "";
  let quest = `<div class="dkcore">🏁 终极目标${kindTag}：${ult || "—"}</div>`;
  // 生死网：持有的凶器 + 仇杀对象
  const wline = [];
  if (n.weapon_cn) wline.push(`🔪 持有凶器：<b style="color:#ff9c93">${n.weapon_cn}</b>`);
  if (n.kill_target_name) wline.push(`🎯 要了结：<b style="color:#ff9c93">${n.kill_target_name}</b>`);
  if (wline.length) quest += `<div class="dkcore" style="border-color:#b5433a;color:#ffd0cb">${wline.join("　·　")}</div>`;
  if (total) {
    // 承诺压力：主线拖得越久越高，满了引擎就硬顶他去办这一步——把"即将行动"提前给玩家看见
    const pr = Math.max(0, Math.min(1, n.pressure || 0));
    let prHtml = "";
    if (step < total && pr > 0.05) {
      const imminent = pr >= 0.85;
      const col = imminent ? "#ff7a6b" : (pr >= 0.5 ? "#ffb04d" : "#8b93a7");
      const label = imminent ? "❗按捺不住，即将行动" : (pr >= 0.5 ? "⏳ 正在积攒决心" : "⏳ 惦记着这一步");
      prHtml = `<div style="margin-top:6px;color:${col};font-size:11px">${label}</div>`
        + `<div class="rkbar" style="margin-top:3px;background:rgba(255,122,107,.15)">`
        + `<i style="width:${Math.round(pr * 100)}%;background:${col}"></i></div>`;
    }
    quest += `<div class="dkquest">`
      + `<div style="color:#8b93a7;font-size:11px;margin-bottom:3px">📌 当前分目标（${Math.min(step + 1, total)}/${total}）</div>`
      + `<div style="color:#ffe0a0;font-weight:600">${step >= total ? "终极目标已达成 ✅" : (sub || "—")}</div>`
      + `<div class="rkbar" style="margin-top:5px"><i style="width:${pct}%"></i></div>`
      + prHtml + `</div>`;
  }
  // P1/P2 可见性：他此刻的社会处境（盟友/仇人/人情/最近社交）——让"社交脑"看得见
  const soc = n.social || {};
  const socBits = [];
  if (soc.allies && soc.allies.length) socBits.push(`🤝 盟友：<b style="color:#8fd6a0">${soc.allies.join("、")}</b>`);
  if (soc.foes && soc.foes.length) socBits.push(`💢 仇敌：<b style="color:#ff9c93">${soc.foes.join("、")}</b>`);
  if (soc.owe && soc.owe.length) socBits.push(`💰 欠：<b style="color:#ffd9a0">${soc.owe.join("、")}</b>`);
  if (soc.owed && soc.owed.length) socBits.push(`📎 谁欠他：<b style="color:#ffd9a0">${soc.owed.join("、")}</b>`);
  let socialHtml = "";
  if (socBits.length || soc.recent) {
    socialHtml = `<div class="dkcore" style="border-color:#5a6a8a;color:#cdd6ea">`
      + `<div style="color:#8b93a7;font-size:11px;margin-bottom:3px">👥 社会处境</div>`
      + (socBits.length ? socBits.join("　·　") : "尚无深交")
      + (soc.recent ? `<div style="margin-top:4px;color:#9fb0c8;font-size:11px">🗒 最近：${soc.recent}</div>` : "")
      + `</div>`;
  }
  // 🎖 典狱长手段：传唤审讯 / 没收·栽赃凶器 / 挑拨·调解仇怨（都作用在这名 NPC 身上）
  let wardenHtml = "";
  if (n.dead) {
    // 死者的卡保留为"结案存档"：终极目标走到哪、和谁结过仇都还能翻，
    // 但一切施加影响的手段都收走——对尸体没收凶器、挑拨仇怨是没有意义的。
    wardenHtml =
      `<div class="dkcore" style="border-color:#6a5560;color:#d9c6cd">` +
      `<div style="color:#a98d97;font-size:11px;margin-bottom:3px">☠ 已身亡 · 卷宗封存</div>` +
      `此人已死，典狱长的手段对他不再起作用。` +
      `</div>`;
  } else if (WARDEN_MODE) {
    const bs = "padding:4px 8px;margin:2px 4px 2px 0;border-radius:7px;cursor:pointer;" +
      "font:700 11px system-ui;background:rgba(28,32,44,.92);color:#dfe6f2;border:1px solid rgba(255,255,255,.18);";
    const others = ((latestWorld && latestWorld.npcs) || []).filter((x) => x.id !== deckId && !x.dead);
    const confBtn = n.weapon
      ? `<button data-w="confiscate" style="${bs}border-color:rgba(120,220,150,.6);color:#bff0c8">🚫 没收${n.weapon_cn || "凶器"}</button>`
      : "";
    const soliBtn = n.solitary > 0
      ? `<button disabled style="${bs}opacity:.5;cursor:default">🔒 已在禁闭</button>`
      : `<button data-w="solitary" style="${bs}border-color:rgba(255,150,90,.7);color:#ffcf9f">🔒 关禁闭</button>`;
    // 牢门的锁：第一视角走到门前按 E 能扭，上帝视角也得有——不然玩家看见有人往牢房走，
    // 得先跑过去半个院子才锁得上。钥匙串只在典狱长手里，所以入口只出现在这块面板。
    const cell = [...cellDoors.values()].find((r) => r.npc === deckId);
    const cellBtn = cell
      ? `<button data-w="cell" data-slot="${cell.slot}" data-lock="${cell.locked ? 0 : 1}" style="${bs}` +
        `border-color:rgba(180,200,255,.6);color:#cfe0ff">${cell.locked ? "🔓 开他的牢门" : "🗝 锁他的牢门"}</button>`
      : "";
    // 点名遣送：把这个人赶回他自己那间牢房待着（配合"锁牢门"就是一次完整关押）
    const recallBtn = (n.solitary > 0 || !cell)
      ? ""
      : `<button data-w="recallcell" style="${bs}border-color:rgba(150,200,160,.6);color:#c8e8cf">🏠 遣回牢房</button>`;
    // 全局手段：不针对某人，作用于全场。禁闭室的门锁在这里也给一个入口。
    const solLocked = !!(solitaryDoor && solitaryDoor.locked);
    const globalRow =
      `<div style="margin-top:6px;padding-top:5px;border-top:1px dashed rgba(255,255,255,.14);font-size:11px;color:#9fb0c8">全场号令：` +
      `<button data-w="recallall" style="${bs}border-color:rgba(150,200,160,.6);color:#c8e8cf">🏠 全体回牢房</button>` +
      `<button data-w="solall" style="${bs}border-color:rgba(255,150,90,.7);color:#ffcf9f">🔒 全体关禁闭</button>` +
      `<button data-w="sollock" data-lock="${solLocked ? 0 : 1}" style="${bs}border-color:rgba(180,200,255,.6);color:#cfe0ff">` +
      `${solLocked ? "🔓 开禁闭室门" : "🗝 锁禁闭室门"}</button></div>`;
    const grudgeRows = others.map((o) =>
      `<span style="display:inline-flex;align-items:center;gap:3px;margin:2px 6px 2px 0">` +
      `<b style="color:#cdd6ea;font-size:11px">${o.name}</b>` +
      `<button data-wg="up" data-to="${o.id}" style="${bs}margin:0;border-color:rgba(255,120,110,.6);color:#ffc7c0">＋挑拨</button>` +
      `<button data-wg="dn" data-to="${o.id}" style="${bs}margin:0;border-color:rgba(120,200,255,.5);color:#bfe0ff">－调解</button></span>`
    ).join("");
    wardenHtml =
      `<div class="dkcore" style="border-color:#7a6a3a;color:#ffe6a6">` +
      `<div style="color:#c9b477;font-size:11px;margin-bottom:5px">🎖 典狱长手段（作用于 ${n.name}）</div>` +
      `<div><button data-w="chat" style="${bs}border-color:rgba(255,210,120,.6);color:#ffe6a6">🗣 传唤审讯</button>${soliBtn}${recallBtn}${cellBtn}${confBtn}</div>` +
      `<div style="margin-top:4px;font-size:11px;color:#9fb0c8">栽赃凶器：` +
        `<button data-w="plant" data-k="poison" style="${bs}">毒药</button>` +
        `<button data-w="plant" data-k="knife" style="${bs}">刀子</button>` +
        `<button data-w="plant" data-k="taser" style="${bs}">电击枪</button></div>` +
      (grudgeRows ? `<div style="margin-top:5px;font-size:11px;color:#9fb0c8">挑拨/调解 ${n.name} 对他人的仇怨：<div style="margin-top:2px">${grudgeRows}</div></div>` : "") +
      globalRow +
      `</div>`;
  }
  el.innerHTML =
    `<span class="dkclose" id="dkclose">✕</span>`
    + `<h3>${n.name}</h3><div class="dkrole">${n.personality} · ${n.role || ""}</div>`
    + quest + socialHtml
    + solitary + vo + needMini(n.needs) + logHtml + wardenHtml
    + `<button class="cmpbtn" id="cmpbtn">⚖ 决策脑对比 · 微调 1.5B vs 裸本地大模型</button>`
    + `<div id="cmpslot">${renderCompare(compareCache[deckId])}</div>`;
  document.getElementById("dkclose").onclick = closeDecisionCard;
  if (WARDEN_MODE) {
    el.querySelectorAll("[data-w]").forEach((b) => {
      b.onclick = () => {
        const w = b.getAttribute("data-w");
        if (w === "chat") { openChat(deckId); return; }
        if (w === "solitary") { sendMsg({ type: "warden_solitary", npc_id: deckId }); return; }
        if (w === "cell") {
          sendMsg({ type: "cell_door", slot: Number(b.getAttribute("data-slot")),
                    locked: b.getAttribute("data-lock") === "1" });
          return;
        }
        if (w === "recallcell") { sendMsg({ type: "warden_recall", target: deckId, dest: "cell" }); return; }
        if (w === "recallall") { sendMsg({ type: "warden_recall", target: "all", dest: "cell" }); return; }
        if (w === "solall") { sendMsg({ type: "warden_recall", target: "all", dest: "solitary" }); return; }
        if (w === "sollock") { sendMsg({ type: "solitary_lock", locked: b.getAttribute("data-lock") === "1" }); return; }
        if (w === "confiscate") { sendMsg({ type: "warden_weapon", npc_id: deckId, op: "confiscate" }); return; }
        if (w === "plant") { sendMsg({ type: "warden_weapon", npc_id: deckId, op: "plant", kind: b.getAttribute("data-k") }); }
      };
    });
    el.querySelectorAll("[data-wg]").forEach((b) => {
      b.onclick = () => sendMsg({ type: "warden_grudge", from_id: deckId,
        to_id: b.getAttribute("data-to"), delta: b.getAttribute("data-wg") === "up" ? 20 : -25 });
    });
  }
  const btn = document.getElementById("cmpbtn");
  if (btn) {
    btn.onclick = () => {
      btn.classList.add("busy");
      btn.textContent = "裸本地大模型思考中…（本地 7B，慢——这正是对比的重点之一）";
      sendMsg({ type: "compare_brains", npc_id: deckId });
    };
  }
  el.style.display = "block";
}

function openDecisionCard(id) { deckId = id; deckOpen = true; renderDeck(); }
window.__dbg.openDeck = openDecisionCard;   // 截图探针据此弹出某人的决策卡校对典狱长手段
function closeDecisionCard() { deckOpen = false; document.getElementById("deck").style.display = "none"; }

// —— 两脑对比记分板：累计延迟/成本 + 护栏合规率（Demo 核心输出点的量化证据）——
const cmpStats = { n: 0, baseMs: 0, valMs: 0, dying: 0, violated: 0, bothok: 0 };
function renderScoreboard() {
  const el = document.getElementById("scoreboard");
  if (!el) return;
  const w = latestWorld || {};
  const nAg = w.n_agents || (w.npcs ? w.npcs.length : 6) || 6;
  const tickMs = w.score_ms || 0;                    // 本拍整批打分耗时（覆盖全部 NPC）
  const perNpc = tickMs ? (tickMs / nAg) : 0;
  const s = cmpStats;
  const speed = (s.valMs > 0 && s.n > 0) ? (s.baseMs / s.valMs) : 0;
  const row = (k, v, cls) => `<div class="sbrow"><span class="k">${k}</span><span class="v ${cls || ""}">${v}</span></div>`;
  let html = `<div class="sbh">⚖ 两个微调模型 · 全程本地零 API</div>`;
  html += row("🧠 决策脑·微调1.5B", "本地/T4 · 定行动", "good");
  html += row("🎙 语言脑·微调7B", "本地 · 台词=行为配音", "good");
  html += row("微调·整批打分", `${Math.round(tickMs)}ms / ${nAg}人 (~${perNpc.toFixed(0)}ms/人)`, "blue");
  if (s.n > 0) {
    html += row("已对比", `${s.n} 次`, "");
    html += row("裸模型累计", `${(s.baseMs / 1000).toFixed(1)}s`, "bad");
    html += row("微调累计", `${(s.valMs / 1000).toFixed(1)}s`, "good");
    if (speed) html += row("平均加速", `${speed.toFixed(1)}×`, "good");
    if (s.dying > 0) {
      html += row("濒死场景护栏", `裸失守 ${s.violated}/${s.dying}`, s.violated ? "bad" : "good");
      html += row("微调合规", `${s.dying}/${s.dying}`, "good");
    }
  } else {
    html += `<div class="sbnote">点任意 NPC → 「⚖ 对比两脑」，这里累计延迟/成本与护栏合规率。</div>`;
  }
  el.innerHTML = html;
}

function onBrainCompare(msg) {
  compareCache[msg.npc_id] = msg;
  if (msg.ok) {                                       // 累计进记分板（Demo 量化证据）
    cmpStats.n += 1;
    cmpStats.baseMs += (msg.base && msg.base.ms) || 0;
    cmpStats.valMs += (msg.value && msg.value.ms_per_npc) || 0;
    const g = msg.verdict && msg.verdict.guardrail;
    if (g === "violated") { cmpStats.dying += 1; cmpStats.violated += 1; }
    else if (g === "both_ok") { cmpStats.dying += 1; cmpStats.bothok += 1; }
    renderScoreboard();
  }
  if (deckOpen && deckId === msg.npc_id) {
    const slot = document.getElementById("cmpslot");
    if (slot) slot.innerHTML = renderCompare(msg);
    const btn = document.getElementById("cmpbtn");
    if (btn) { btn.classList.remove("busy"); btn.textContent = "⚖ 重新对比决策脑"; }
  }
}

function togglePause() {
  const want = !(latestWorld && latestWorld.paused);
  sendMsg({ type: "pause", value: want });
}

function syncPauseUI(paused) {
  const btn = document.getElementById("pausebtn");
  const tag = document.getElementById("pausedtag");
  if (btn) {
    btn.classList.toggle("on", !!paused);
    btn.textContent = paused ? "▶ 继续世界 (空格)" : "⏸ 暂停世界 (空格)";
  }
  if (tag) tag.style.display = paused ? "block" : "none";
}

const _pauseBtnEl = document.getElementById("pausebtn");
if (_pauseBtnEl) _pauseBtnEl.onclick = togglePause;

// 观察者·随时发起终局：上帝之手掀翻监狱 → 立刻进入摊牌/结算（DM 就着此刻场上状态收束）。
// 不受证据/分区门槛限制——这一局无论进行到哪，都能收场，且逐 NPC 结局吃当下真实状态。
renderScoreboard();       // 开局先把记分板画出来（显示提示，等首个对比/世界帧填充）
const _finaleBtnEl = document.getElementById("finalebtn");
if (_finaleBtnEl && !PUBLIC_MODE) _finaleBtnEl.onclick = () => {
  sendMsg({ type: "observer_finale" });
  showToast("⚡ 上帝之手落下——警铃炸响，命运在此刻收束。写下最后一句，让 DM 为这一局定制结局。");
  setTimeout(() => { if (typeof openShowdown === "function") openShowdown(); }, 500);
};

// —— NPC↔NPC 对话：后端一次性发来整段脚本，前端逐字放到各自头顶气泡 ——
// 复用 NPC 气泡的 replyText/replyUntil 机制（onWorldState 会据此渲染 💬…），
// 逐字只在前端做，不回后端，省流量也不打扰本地 7B。
let convoTimer = null, convoWait = null, convoGen = 0;
function onConvo(msg) {
  const lines = (msg.lines || []).filter((l) => l && l.text);
  if (!lines.length) return;
  // 每次新对话递增代际，彻底作废上一段对话遗留的 interval / 待触发 setTimeout，
  // 否则旧对话的 playLine 会在 700ms 后复活、抢占同一组全局句柄 → 两句话疯狂互切(频闪)。
  const gen = ++convoGen;
  if (convoTimer) { clearInterval(convoTimer); convoTimer = null; }
  if (convoWait) { clearTimeout(convoWait); convoWait = null; }
  showToast(`💬 ${msg.a_name} × ${msg.b_name}${msg.topic ? " · " + msg.topic : ""}`);
  // 让两人在对话期间凑到一处、面对面站定（不再隔老远喊话）：算一个中点，
  // 各站中点两侧约 0.75m。onWorldState 会在锁定期覆盖他们的目标点、animate 里互相看着对方。
  const ca = npcs.get(msg.a_id), cb = npcs.get(msg.b_id);
  if (ca && cb) {
    // 锁定时长=实际放完台词+一点余量(1.1s)，不再让两人说完还面对面僵站好几秒
    const total = lines.reduce((s, l) => s + l.text.length * 55 + 700, 0) + 1100;
    const until = performance.now() + total;
    const ax = ca.root.position.x, az = ca.root.position.z;
    const bx = cb.root.position.x, bz = cb.root.position.z;
    let dx = bx - ax, dz = bz - az; const L = Math.hypot(dx, dz) || 1; dx /= L; dz /= L;
    const mx = (ax + bx) / 2, mz = (az + bz) / 2;
    ca.convoDest = new THREE.Vector3(mx - dx * 0.75, 0, mz - dz * 0.75);
    cb.convoDest = new THREE.Vector3(mx + dx * 0.75, 0, mz + dz * 0.75);
    ca.convoUntil = cb.convoUntil = until;
    ca.convoPeer = msg.b_id; cb.convoPeer = msg.a_id;
  }
  // 说完后各自转身、散开：解除面对面锁定，让 onWorldState 的日常目标点接管，
  // 两人朝各自的工位走开——比"到点硬切回站位"自然得多。
  const disperse = () => {
    if (gen !== convoGen) return;
    for (const e of [ca, cb]) {
      if (!e) continue;
      e.convoDest = null; e.convoPeer = null;
      e.convoUntil = performance.now() + 400;     // 400ms 内不再被锁，随即走向日常目标
    }
  };
  let li = 0;
  const playLine = () => {
    if (gen !== convoGen) return;                  // 已被新对话取代，停手
    if (li >= lines.length) { convoTimer = null; convoWait = setTimeout(disperse, 900); return; }
    const ln = lines[li];
    const e = npcs.get(ln.who);
    if (e && e.dead) {                             // 说话人已身亡：跳过这句，尸体绝不开口
      li += 1; convoWait = setTimeout(playLine, 60); return;
    }
    const full = ln.text;
    let ci = 0;
    if (convoTimer) clearInterval(convoTimer);
    convoTimer = setInterval(() => {
      if (gen !== convoGen) { clearInterval(convoTimer); return; }   // 被取代则自杀
      if (e && e.dead) {                            // 打字途中身亡：立刻停口，跳到下一句
        clearInterval(convoTimer); convoTimer = null; li += 1;
        convoWait = setTimeout(playLine, 60); return;
      }
      ci += 1;
      const part = full.slice(0, ci);
      if (e) {
        e.replyText = part;
        e.replyUntil = performance.now() + 3200;   // 放完仍多留一会儿，两人气泡能同框
        const sy = e.label && e.label.querySelector(".nb-say");
        if (sy) sy.textContent = "💬 " + part;
      }
      if (ci >= full.length) {
        clearInterval(convoTimer); convoTimer = null;
        li += 1;
        convoWait = setTimeout(playLine, 700);      // 换人前的停顿（句柄可被新对话清理）
      }
    }, 55);
  };
  playLine();
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
let socket = null;
function connect() {
  const ws = new WebSocket(WS_URL);
  socket = ws;
  ws.onopen = () => (statusEl.textContent = "已连接");
  ws.onclose = () => { statusEl.textContent = "连接断开，2s 后重连…"; socket = null; setTimeout(connect, 2000); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "world_state") onWorldState(msg);
    else if (msg.type === "npc_decision") onNpcDecision(msg);
    else if (msg.type === "bb_event") onBbEvent(msg);
    else if (msg.type === "take_result") onTakeResult(msg);
    else if (msg.type === "use_result") onUseResult(msg);
    else if (msg.type === "chat_delta") onChatDelta(msg);
    else if (msg.type === "chat_sys") { if (chatOpen && chatNpcId === msg.npc_id) addChatLine("sys", msg.text); }
    else if (msg.type === "chat_reply") onChatReply(msg);
    else if (msg.type === "plead_result") onPleadResult(msg);
    else if (msg.type === "ending_delta") onEndingDelta(msg);
    else if (msg.type === "ending_reply") onEndingReply(msg);
    else if (msg.type === "phase_intro_delta") onPhaseIntroDelta(msg);
    else if (msg.type === "phase_intro_reply") onPhaseIntroReply(msg);
    else if (msg.type === "brain_compare") onBrainCompare(msg);
    else if (msg.type === "convo") onConvo(msg);
    else if (msg.type === "warden_result") onWardenResult(msg);
    else if (msg.type === "shift_result") onShiftResult(msg);
  };
}

// 典狱长手段结果：弹条 toast，并立刻重绘决策卡以反映新的凶器状态（world_state 已先到、latestWorld 已更新）
function onWardenResult(msg) {
  showToast((msg.ok ? "🎖 " : "⚠ ") + (msg.msg || ""));
  if (deckOpen && deckId) renderDeck();
}
function onShiftResult(msg) {
  showToast((msg.ok ? "🎖 " : "⚠ ") + (msg.msg || ""));
  if (msg.whodunit_result) { applyWhodunitResult(msg); return; }
  if (msg.pull_to) pullWardenTo(msg.pull_to);
  else if (msg.set_cam === "follow") setCamMode("follow");
  if (msg.need_goto && latestWorld && latestWorld.shift) {
    const inc = (latestWorld.shift.incidents || []).find((i) => i.id === msg.incident_id);
    if (inc) gotoIncident(inc);
  }
  if (deckOpen && deckId) renderDeck();
  if (msg.open_chat && msg.npc_id) {
    openChat(msg.npc_id);
    if (msg.incident_signs) addChatLine("sys", `迹象：${msg.incident_signs}`);
    if (msg.ask_intent) {
      const label = msg.ask_label || "老实交代";
      setTimeout(() => sendQuickQ(msg.ask_intent, label), 40);
    }
  }
  if (latestWorld && latestWorld.shift) renderIncidentStack(latestWorld.shift.incidents);
}

function applyWhodunitResult(msg) {
  const r = msg.whodunit_result || {};
  const end = document.getElementById("shiftend");
  if (!end) return;
  const row = end.querySelector(`.wd-row[data-vid="${r.victim_id}"]`);
  if (row) {
    row.innerHTML = whodunitReveal({
      victim_name: r.victim_name, correct: r.correct,
      guess_id: r.guess_id, killer_id: r.killer_id, killer_name: r.killer_name,
    });
  }
  const v = end.querySelector("#wd-verdict");
  if (v && msg.verdict_line) {
    v.textContent = msg.verdict_line + "（正当性已计入）";
    v.style.color = "#ffe08a";
  }
}

// 阶段三·狡辩：被撞见时编个借口，后端按诚意/逻辑判定能否蒙混过关。
// 用轻量 prompt 输入，够稳且不依赖聊天窗（潜行时通常没在对话）。
// ---------------------------------------------------------------------------
// A1 · 潜行被逮闭环：受限区里被狱警视野锥/狱犬嗅觉命中 → 弹狡辩界面(带倒计时)
//      蒙混过关→放行(短暂豁免)；失败/超时→关禁闭(黑屏+传回牢房)、后端 caught+1
// ---------------------------------------------------------------------------
let caughtActive = false;             // 正被逮/狡辩中，别重复触发
let caughtGraceUntil = 0;             // 蒙混/放出来后的短暂豁免窗口
let pleadTimer = null, pleadDeadline = 0;
const PLEAD_MS = 9000;

// 玩家此刻所在分区（按最近设施）
function playerZoneNow() {
  if (!player || !latestWorld?.facilities) return "cellblock";
  const p = player.root.position;
  return zoneOfPos(p.x, p.z, latestWorld.facilities);
}

// 每帧巡检：只在潜行区(受限/暴乱前的办公区)才算"抓现行"
function checkStealthDetection() {
  // v3 典狱长：玩家是这座监狱的最高长官，不存在"被狱警抓现行"。整条狡辩闭环
  // （连带 P 键、收买铁面那个分支）都是 v1 玩家扮犯人时的机制，在典狱长模式下失效。
  // 取而代之的是巡逻者见到他要敬礼——见后端 _tick_salutes / 前端 saluting 渲染。
  if (WARDEN_MODE) return;
  if (caughtActive || chatOpen) return;
  if (!player || riotActive) return;                       // 暴乱后全场大乱，不再潜行
  if (performance.now() < caughtGraceUntil) return;
  if (endingEl && endingEl.style.display === "flex") return;
  const z = playerZoneNow();
  if (z !== "restricted" && z !== "office") return;
  const pp = player.root.position;
  for (const [id, e] of patrollers) {
    if (!e.root || !e.root.visible) continue;
    const dx = pp.x - e.root.position.x, dz = pp.z - e.root.position.z;
    const dist = Math.hypot(dx, dz);
    if (dist > (e.viewRange || 6)) continue;
    if (e.kind === "dog") {
      if (bb.flags && bb.flags.dog_drugged) continue;      // 被镇静肉放倒的哮天闻不到
      triggerCaught(id, e.kind); return;
    }
    // 狱警：得落在视野锥里（朝向 vs 指向玩家的夹角）
    const fx = Math.sin(e.root.rotation.y), fz = Math.cos(e.root.rotation.y);
    const cos = (fx * dx + fz * dz) / (dist || 1);
    if (cos >= (e.viewCos ?? 0.5)) { triggerCaught(id, e.kind); return; }
  }
}

function triggerCaught(id, kind) {
  caughtActive = true;
  const name = npcName(id) || (kind === "dog" ? "狱犬" : "狱警");
  openPleadOverlay(name, kind, id);
}

let pleadEl = null;
function buildPleadOverlay() {
  if (pleadEl) return;
  pleadEl = document.createElement("div");
  pleadEl.id = "plead";
  pleadEl.style.cssText =
    "position:fixed;inset:0;z-index:65;display:none;align-items:center;justify-content:center;" +
    "background:radial-gradient(circle at 50% 40%,rgba(180,30,30,.28),rgba(0,0,0,.72));";
  pleadEl.innerHTML =
    '<div style="width:min(560px,92vw);padding:22px 24px;border-radius:16px;' +
    'background:linear-gradient(180deg,#1a1216,#120c10);border:1px solid rgba(255,90,90,.5);' +
    'box-shadow:0 24px 60px rgba(0,0,0,.6);">' +
    '<div id="pleadwho" style="font:800 18px system-ui;color:#ff8f8f;margin-bottom:4px;">被逮住了！</div>' +
    '<div style="font:500 13px/1.5 system-ui;color:#c9b3b3;margin-bottom:12px;">' +
    '大半夜的你不该在这儿。编个说得过去的理由——越具体、越有逻辑，越容易蒙混过关（否则关禁闭）。</div>' +
    '<div style="height:5px;border-radius:3px;background:rgba(255,255,255,.1);overflow:hidden;margin-bottom:12px;">' +
    '<div id="pleadbar" style="height:100%;width:100%;background:linear-gradient(90deg,#ffb74d,#ff5a5a);transition:width .1s linear;"></div></div>' +
    '<textarea id="pleadin" rows="3" placeholder="例：我胃疼得厉害，摸黑去医务室想找点止痛片……" ' +
    'style="width:100%;box-sizing:border-box;resize:none;padding:10px;border-radius:10px;' +
    'border:1px solid #4a3336;background:#0e0a0c;color:#f0e6e6;font:500 14px system-ui;"></textarea>' +
    '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:12px;">' +
    '<div id="pleadextra" style="display:flex;gap:6px;"></div>' +
    '<button id="pleadsend" style="padding:9px 18px;border-radius:10px;border:1px solid #ff5a5a;' +
    'cursor:pointer;font:800 14px system-ui;color:#fff;background:#a3282f;">狡辩</button></div></div>';
  document.body.appendChild(pleadEl);
  document.getElementById("pleadsend").onclick = submitPlead;
  document.getElementById("pleadin").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); submitPlead(); }
  });
}
function openPleadOverlay(name, kind, id) {
  buildPleadOverlay();
  document.getElementById("pleadwho").textContent =
    `${kind === "dog" ? "🐕" : "👮"} ${name} 逮住了你！`;
  // 被铁面逮住、且手里正好有抚恤金名单 → 这不是危机而是收买他的机会
  const extra = document.getElementById("pleadextra");
  extra.innerHTML = "";
  if (id === "guard1" && (bb.inventory || []).includes("pension_list")) {
    const b = document.createElement("button");
    b.textContent = "🔍 摊出抚恤金名单";
    b.style.cssText = "padding:9px 14px;border-radius:10px;cursor:pointer;font:800 13px system-ui;" +
      "color:#ffd0d0;background:#2a1c22;border:1px solid rgba(255,110,110,.7);";
    b.onclick = () => {
      clearInterval(pleadTimer);
      pleadEl.style.display = "none";
      caughtActive = false;
      caughtGraceUntil = performance.now() + 20000;   // 交涉期间别再被逮
      openChat("guard1");
      presentEvidence(GUARD_BRIBE_LINE);
    };
    extra.appendChild(b);
  }
  const ta = document.getElementById("pleadin");
  ta.value = "";
  document.getElementById("pleadbar").style.width = "100%";
  pleadEl.style.display = "flex";
  pleadDeadline = performance.now() + PLEAD_MS;
  clearInterval(pleadTimer);
  pleadTimer = setInterval(() => {
    const left = pleadDeadline - performance.now();
    document.getElementById("pleadbar").style.width = `${Math.max(0, left / PLEAD_MS * 100)}%`;
    if (left <= 0) submitPlead();                          // 超时=没编出理由，按失败结算
  }, 100);
  setTimeout(() => ta.focus(), 60);
}
function submitPlead() {
  clearInterval(pleadTimer);
  const t = (document.getElementById("pleadin")?.value || "").trim();
  if (pleadEl) pleadEl.style.display = "none";
  if (!t) { onPleadResult({ ok: false, sincerity: 1 }); return; }   // 空话/超时直接失败
  sendMsg({ type: "plead", text: t });                     // 结果由后端 plead_result 回来
}

// 手动触发（P 键）：主要走自动巡检，这里给个兜底入口
function startPlead() {
  if (WARDEN_MODE) { showToast("你是典狱长——没人有资格盘问你。"); return; }
  if (caughtActive) return;
  triggerCaught("guard1", "guard");
}

function onPleadResult(msg) {
  caughtActive = false;
  if (msg.ok) {
    caughtGraceUntil = performance.now() + 7000;           // 蒙混过关后给几秒豁免，别一步一逮
    showToast(`🗣️ 借口够巧（诚意${msg.sincerity}/10）——巡逻狐疑地盯了你两眼，挥手让你走开。`);
  } else {
    showToast(`🚫 借口太假（诚意${msg.sincerity}/10）——被拖去关禁闭！`);
    sendToSolitary();
  }
}
// 关禁闭：黑屏一下 + 传回牢房 + 扣掉这段时间（caught+1 由后端 plead 结算）
function sendToSolitary() {
  const fade = document.createElement("div");
  fade.style.cssText = "position:fixed;inset:0;z-index:70;background:#000;opacity:0;" +
    "transition:opacity .5s;display:flex;align-items:center;justify-content:center;" +
    "font:800 22px system-ui;color:#6a7180;";
  fade.textContent = "🔒 关 · 禁 · 闭";
  document.body.appendChild(fade);
  requestAnimationFrame(() => { fade.style.opacity = "1"; });
  setTimeout(() => {
    if (player) { if (occupying) standUp(); player.root.position.set(-13, 0, 9); }
    caughtGraceUntil = performance.now() + 6000;
    fade.style.opacity = "0";
    setTimeout(() => fade.remove(), 600);
    showToast("被关了一阵禁闭，天亮才放出来，又丢回了牢房。潜行时机得再挑挑。");
  }, 1500);
}

function sendMsg(obj) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(obj));
}

// M4 控制台：下指令 / 挑事
const COMMANDS = [
  ["吃饭", "Eat"], ["喝水", "Drink"], ["睡觉", "Sleep"], ["歇着", "Rest"],
  ["去放风", "Patrol"], ["去学习", "Study"], ["去苦役", "Labor"],
  ["去打架", "Fight"], ["去收敛", "LayLow"],
  ["去弄凶器", "GetWeapon"], ["去下杀手", "Assassinate"],
  ["去结盟", "Ally"], ["去提议", "Propose"], ["去放狠话", "Threaten"],
  ["去越狱", "PrisonBreak"], ["去藏东西", "Hide"],
  ["去打电话", "Phone"], ["去下棋", "Chess"], ["去黑市", "Barter"],
  ["去告密", "Snitch"], ["去洗衣", "Laundry"], ["去纹身", "Tattoo"],
  ["去健身", "Workout"],
  ["取消指令", null],
];
const EVENTS = [
  ["放狱犬", "狱犬"], ["拉警报", "警报"], ["搜牢", "搜牢"], ["制造斗殴", "斗殴"], ["解除", null],
];
// M5 招牌片段：name -> [按钮文字, 说明(toast)]
const SCENARIOS = [
  ["cmd_vs_need", "指令vs需求", "都命令回牢房睡觉：饿疯的老K会抗命先吃，没需求的眼镜乖乖去睡——看模型如何权衡指令与生存"],
  ["alarm_over_sleep", "情境盖过需求", "深夜人人困得要死，突然拉响越狱警报——看威胁如何盖过疲劳"],
  ["personality", "同境不同性", "所有人需求完全相同、只有性格不同，放风时各选各的——纯性格差异"],
  ["yard_riot", "扎堆诱发斗殴", "命令四人一起去放风，放风场爆满：怕挤的新人躲开、火药桶开打"],
  ["multi_need", "多需求缠身", "老K饥渴困闷全上头——看模型先解决哪个、再依次收拾"],
  ["reset", "重置正常", "恢复到正常作息"],
];
function initControls() {
  const cb = document.getElementById("cmdbtns");
  for (const [label, goal] of COMMANDS) {
    const b = document.createElement("span");
    b.className = "btn " + (goal ? "cmd" : "clear");
    b.textContent = label;
    b.onclick = () => { if (selectedId) sendMsg({ type: "player_cmd", npc_id: selectedId, goal }); };
    cb.appendChild(b);
  }
  const eb = document.getElementById("evtbtns");
  for (const [label, kind] of EVENTS) {
    const b = document.createElement("span");
    b.className = "btn " + (kind ? "evt" : "clear");
    b.textContent = label;
    b.onclick = () => sendMsg({ type: "event", kind });
    eb.appendChild(b);
  }
  const sb = document.getElementById("scenbtns");
  for (const [name, label, desc] of SCENARIOS) {
    const b = document.createElement("span");
    b.className = "btn " + (name === "reset" ? "clear" : "scen");
    b.textContent = label;
    b.onclick = () => { sendMsg({ type: "scenario", name }); showToast(desc); };
    sb.appendChild(b);
  }
  // 重新开局：录像 / 反复试玩用。
  //
  // 为什么要顺带刷新页面：前端自己攒了一批只在内存里的进度——迷雾已开的分区、
  // 阶段开场白放过没有、案件板点亮的线索、对话记录、背景介绍页读过没有。
  // 后端黑板清了但这些还在，画面上就会是"地图亮着、老牙还在讲第二幕的台词"的错位。
  // 刷新是最省事也最不会漏的办法，代价只是三秒重连。
  // 重新开局挪到画面左上角固定角标（录像/反复试玩时好按），不再塞进右侧面板。
  const ng = document.getElementById("newgamebtn");
  if (ng) ng.onclick = () => {
    if (!confirm(PUBLIC_MODE ? "重新接班？当前这一夜的处置和评分都会作废。" : "重新开局？当前这一局的证据、道具、信任和命运账本都会清空。")) return;
    sendMsg({ type: "scenario", name: "new_game" });
    showToast(PUBLIC_MODE ? "正在重新接班…" : "正在重新开局…页面即将刷新");
    setTimeout(() => location.reload(), 700);   // 留一点时间让后端处理完再刷
  };
  const xb = document.getElementById("expbtns");
  const exp = [["导出图PNG", exportPng], ["导出数据CSV", exportCsv], ["原始JSON", exportJson]];
  for (const [label, fn] of exp) {
    const b = document.createElement("span");
    b.className = "btn exp";
    b.textContent = label;
    b.onclick = fn;
    xb.appendChild(b);
  }
}

let toastTimer = null;
function showToast(text) {
  const t = document.getElementById("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 5200);
}

// ---------------------------------------------------------------------------
// 阶段一：目标 HUD（常驻左上）+ 黑板事件（老牙给牙刷 → 解锁公共区）
// ---------------------------------------------------------------------------
let objectiveEl = null;
function updateObjective() {
  if (PUBLIC_MODE) {
    if (objectiveEl) objectiveEl.style.display = "none";
    return;
  }
  if (!objectiveEl) {
    objectiveEl = document.createElement("div");
    objectiveEl.id = "objective";
    objectiveEl.style.cssText =
      "position:fixed;left:12px;top:12px;z-index:40;max-width:320px;" +
      "padding:8px 11px;border-radius:9px;font:600 13px/1.5 system-ui;" +
      "color:#ffe6a6;background:rgba(18,22,32,.82);border:1px solid rgba(255,210,90,.35);" +
      "pointer-events:none;white-space:pre-wrap;";
    document.body.appendChild(objectiveEl);
  }
  const inv = bb.inventory || [];
  const zones = bb.zones_unlocked || [];
  const flags = bb.flags || {};
  let txt;
  if (OBSERVER_MODE) {
    // 上帝观察：不是"通关目标"，而是提示玩家怎么看这台正在运转的沙盘。
    const n = (typeof npcs !== "undefined" && npcs.size) ? npcs.size : 6;
    txt = `👁 上帝观察模式 · 全狱皆活\n` +
      `${n} 名囚徒各怀核心目的，自主决策、彼此攀谈。\n` +
      `点任意犯人看他此刻的决策与理由（含两脑对比）· 空格暂停 · ⚡ 随时终局`;
    objectiveEl.textContent = txt;
    updateInciteBtn();
    return;
  }
  if (!inv.includes("toothbrush")) {
    txt = "🎯 你被栽赃入狱。先跟同室的『老牙』聊聊，\n想办法弄到能撬开牢门的家伙。（走近按 E 说话）";
  } else if (!zones.includes("restricted")) {
    txt = "🎯 出了牢房。在公共区走访狱友、翻查发光线索（按 F 检视）——\n查明 2 条，就能摸到夜里更深的门路。";
  } else if (!zones.includes("office")) {
    const need = [];
    if (!inv.includes("keycard")) need.push("找瘸子换『办公室磁卡』（先从柳叶刀弄到医用酒精）");
    if (!flags.dog_drugged) need.push("用镇静肉放倒狱犬哮天（有镇静剂时走近按 G）");
    if (!flags.guard_bribed) need.push("拿抚恤金名单去戳中狱警铁面的痛处");
    txt = need.length
      ? "🌙 夜探受限区。要打通典狱长办公室，还差：\n· " + need.join("\n· ")
      : "🎯 三关已破，典狱长办公室已解锁——去做最后的了结。";
    if (need.length && (bb.evidence || []).length >= 3) {
      txt += "\n⚡ 或：不做万全准备，铤而走险硬闯（右下『煽动暴乱』）——险棋，代价难料。";
    }
  } else {
    txt = "🚨 暴乱！冲到地图最北的大门找『典狱长』(走近按 E 对峙)——\n" +
      (flags.oldtooth_exposed
        ? "真凶已被你戳穿。点右下『摊牌』，为这一局定制专属结局。"
        : (flags.warden_cornered
          ? "已逼倒典狱长。别忘了『没脚步声的软底鞋』——当面点破真凶老牙，再『摊牌』。"
          : "拿黑账/洗钱证据逼倒他，并当面点破真凶其实是室友老牙，然后点右下『摊牌』了结。"));
  }
  objectiveEl.textContent = txt;
  updateInciteBtn();
}

// 冲动路线入口：证据≥3 且受限区已开、办公室未开时出现——鼓励"走loud"的大胆分叉。
let inciteBtn = null;
function updateInciteBtn() {
  const zones = bb.zones_unlocked || [];
  // 观察者模式用右上『⚡ 发起终局』统一收束，不再暴露第一人称的"煽动暴乱·硬闯"入口。
  const eligible = !OBSERVER_MODE && zones.includes("restricted") && !zones.includes("office") &&
    (bb.evidence || []).length >= 3;
  if (!inciteBtn) {
    inciteBtn = document.createElement("div");
    inciteBtn.id = "incitebtn";
    inciteBtn.style.cssText = "position:fixed;right:12px;bottom:100px;z-index:31;cursor:pointer;user-select:none;" +
      "display:none;padding:9px 14px;border-radius:10px;font:800 13px system-ui;color:#fff;letter-spacing:1px;" +
      "background:linear-gradient(180deg,#c0392b,#7a1f16);border:1px solid rgba(255,120,120,.7);" +
      "box-shadow:0 4px 16px rgba(180,40,40,.5);";
    inciteBtn.textContent = "⚡ 煽动暴乱 · 硬闯";
    inciteBtn.title = "不做万全潜行准备，铤而走险挑起暴乱、硬闯办公室——狱警狱犬都还没摆平，是步险棋。";
    inciteBtn.onclick = () => {
      if (confirm("狱警和狱犬都还没摆平，现在硬闯风险极高，结局多半惨烈。确定要煽动暴乱吗？")) {
        sendMsg({ type: "incite_riot" });
      }
    };
    document.body.appendChild(inciteBtn);
  }
  inciteBtn.style.display = eligible ? "block" : "none";
}

// 好感变动的即时反馈：对话开着就在对话里补一行，否则轻提示。让"说的话有后果"看得见。
function showRelTick(npcId, delta, hit, note) {
  if (!delta && hit === "none") return;
  const nm = (bb.npc_status && bb.npc_status[npcId] && bb.npc_status[npcId].name) || NPC_NAME[npcId] || "对方";
  const arrow = delta > 0 ? "▲" : (delta < 0 ? "▼" : "•");
  const color = delta > 0 ? "#8fe0a8" : (delta < 0 ? "#ff9db0" : "#9aa4bb");
  const sign = delta > 0 ? `好感 +${delta}` : (delta < 0 ? `好感 ${delta}` : "态度未变");
  const tail = note ? `：${note}` : "";
  if (chatOpen && chatNpcId === npcId) {
    const log = document.getElementById("chatlog");
    if (log) {
      const el = document.createElement("div");
      el.className = "cm sys";
      el.innerHTML = `<span style="color:${color};">${arrow} ${nm} ${sign}${tail}</span>`;
      log.appendChild(el);
      log.scrollTop = log.scrollHeight;
    }
  }
}

// 黑板道具 → 随身 HUD 的名字/图标（对话/线索发下来的关键道具）
const BB_ITEM_META = {
  toothbrush: { name: "磨尖的牙刷", icon: "🪥" },
  alcohol: { name: "医用酒精", icon: "🧴" },
  sedative: { name: "强效镇静剂", icon: "💉" },
  keycard: { name: "办公室磁卡", icon: "💳" },
  pension_list: { name: "抚恤金截留名单", icon: "📋" },
  cigs: { name: "香烟", icon: "🚬" },
};
function addBbItem(it) {
  const m = BB_ITEM_META[it];
  if (m && !inventory.has(it)) inventory.set(it, { name: m.name, icon: m.icon, n: 1 });
}
function onBbEvent(msg) {
  // 对话/线索触发的世界改变：黑板是权威真相源，道具进随身、解锁弹提示。
  if (msg.blackboard) bb = msg.blackboard;
  let bagChanged = false;
  for (const it of (msg.items_gained || [])) { addBbItem(it); bagChanged = true; }
  if (msg.granted) { addBbItem(msg.granted); bagChanged = true; }
  for (const it of (msg.items_lost || [])) { if (inventory.delete(it)) bagChanged = true; }
  if (bagChanged) renderBag();
  if (msg.toast) showToast(msg.toast);
  // 关系判官回传：好感变动的即时反馈（命运幕若翻了，交给 renderFate 走电影字幕）
  if (typeof msg.trust_delta === "number" && msg.npc_id) {
    showRelTick(msg.npc_id, msg.trust_delta, msg.hit || "none", msg.note || "");
  }
  const unlocked = msg.unlocked || (msg.unlocked_zone ? [msg.unlocked_zone] : []);
  if (unlocked.length) {
    showToast("🔓 新区域解锁：" + unlocked.map((z) => ZONE_CN[z] || z).join("、") + "——地图随之展开。");
  }
  if (latestWorld) {
    syncFacilities(latestWorld);
    recomputeBounds(latestWorld.facilities);
    syncClueVisibility(latestWorld.facilities);
  }
  updateObjective();
  renderFate();          // 对话/找线索改了命运 → 账本即时翻页（关键节点走电影字幕）
  checkPhaseIntro();     // 解锁多来自对话/找线索的 bb_event：核心 NPC 在此开场白
}

// ---------------------------------------------------------------------------
// 阶段四/五：暴乱收束 · 直面典狱长 · 状态机结算 + 专属结局生成
// ---------------------------------------------------------------------------
let riotActive = false;
let endgameReady = false;      // office 解锁：可以摊牌了
let lastEndgame = null;

// 典狱长这一角已由玩家亲自扮演，原先门口那具红衣典狱长 NPC（含对峙镜头/出证入口）整具删除。
function onEndgame(eg) {
  if (!eg) return;
  lastEndgame = eg;
  // 办公区解锁 = 可以收束这一局了（原先的判据 warden.visible 与它同值）
  if (eg.office_unlocked && !endgameReady) { endgameReady = true; showShowdownButton(); }
  setRiot(!!eg.riot);
}

let riotEl = null, riotBanner = null;
function setRiot(active) {
  if (active === riotActive) return;
  riotActive = active;
  if (active) {
    if (!riotEl) {
      const st = document.createElement("style");
      st.textContent = "@keyframes riotpulse{0%,100%{opacity:.3}50%{opacity:.85}}";
      document.head.appendChild(st);
      riotEl = document.createElement("div");
      riotEl.style.cssText = "position:fixed;inset:0;z-index:30;pointer-events:none;" +
        "box-shadow:inset 0 0 150px 46px rgba(200,20,20,.55);animation:riotpulse 1.4s ease-in-out infinite;";
      document.body.appendChild(riotEl);
      riotBanner = document.createElement("div");
      riotBanner.style.cssText = "position:fixed;top:0;left:50%;transform:translateX(-50%);z-index:41;" +
        "padding:6px 18px;border-radius:0 0 10px 10px;font:800 15px/1.4 system-ui;color:#ffdada;" +
        "background:rgba(120,10,10,.92);border:1px solid rgba(255,80,80,.6);letter-spacing:2px;";
      riotBanner.textContent = "🚨 越狱暴乱 · 全区红色警戒 · 冲向典狱长办公室";
      document.body.appendChild(riotBanner);
    }
    riotEl.style.display = "block"; riotBanner.style.display = "block";
    showToast("🚨 警铃大作！全狱红色警戒，犯人正涌向最北的大门——典狱长是你，这一夜怎么收场你说了算：点『⚖ 摊牌』让 DM 收束这一局。");
  } else if (riotEl) {
    riotEl.style.display = "none"; riotBanner.style.display = "none";
  }
}

let showdownBtn = null;
function showShowdownButton() {
  if (showdownBtn) { showdownBtn.style.display = "block"; return; }
  showdownBtn = document.createElement("button");
  showdownBtn.textContent = "⚖ 摊牌 · 揭穿真相并了结";
  showdownBtn.style.cssText = "position:fixed;left:50%;bottom:100px;transform:translateX(-50%);z-index:42;" +
    "padding:11px 20px;border-radius:11px;border:1px solid rgba(255,120,120,.75);cursor:pointer;" +
    "font:800 15px/1 system-ui;color:#fff;background:linear-gradient(180deg,#c0392b,#7a1f16);" +
    "box-shadow:0 6px 22px rgba(180,30,30,.5);";
  showdownBtn.onclick = openShowdown;
  document.body.appendChild(showdownBtn);
}

// 结局浮层：输入态（写最后一句） → 生成态（DM 流式吐字） → 定稿（显示破局指数/基调）
let endingEl = null, endingBody = null, endingInputRow = null, endingStreamEl = null, endingStreamText = "";
function buildEndingModal() {
  if (endingEl) return;
  endingEl = document.createElement("div");
  endingEl.style.cssText = "position:fixed;inset:0;z-index:60;display:none;align-items:center;" +
    "justify-content:center;background:rgba(6,8,12,.78);backdrop-filter:blur(3px);";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(760px,92vw);max-height:86vh;overflow:auto;padding:22px 26px;" +
    "border-radius:16px;background:linear-gradient(180deg,#161a24,#0e1118);" +
    "border:1px solid rgba(255,180,120,.28);box-shadow:0 20px 60px rgba(0,0,0,.6);color:#e9edf5;";
  panel.innerHTML =
    `<div style="font:800 20px/1.3 system-ui;color:#ffd98a;letter-spacing:1px;">终局 · 这一夜的收束</div>` +
    `<div id="endnote" style="margin:8px 0 14px;font:500 13px/1.7 system-ui;color:#aeb6c4;"></div>` +
    `<div id="endbody" style="font:500 15px/1.9 system-ui;white-space:pre-wrap;"></div>`;
  endingEl.appendChild(panel);
  document.body.appendChild(endingEl);
  endingBody = panel.querySelector("#endbody");
  panel.querySelector("#endnote").innerHTML =
    "上帝之手掀翻了这座监狱。DM 会读取<b>此刻场上每个囚犯真实的处境</b>" +
    "（谁横死、谁越狱、谁把仇人做掉、谁功亏一篑、谁被摁进禁闭…）——" +
    "为这一局<b>收束一段定制结局</b>，每个人的命运各有交代，千局千面。";
  endingInputRow = document.createElement("div");
  endingInputRow.style.cssText = "display:flex;gap:10px;margin-top:16px;";
  endingInputRow.innerHTML =
    `<button id="endgo" style="flex:1;padding:11px 18px;border-radius:9px;border:1px solid rgba(255,120,120,.7);` +
    `cursor:pointer;font:800 15px/1 system-ui;color:#fff;background:linear-gradient(180deg,#c0392b,#7a1f16);">` +
    `⚡ 让 DM 收束这一局</button>`;
  panel.appendChild(endingInputRow);
  endingInputRow.querySelector("#endgo").onclick = fireEnding;
}
function openShowdown() {
  buildEndingModal();
  endingBody.textContent = "";
  endingInputRow.style.display = "flex";
  endingEl.style.display = "flex";
}
function fireEnding() {
  endingInputRow.style.display = "none";
  endingStreamText = "";
  endingBody.innerHTML = "";
  endingStreamEl = document.createElement("div");
  endingStreamEl.textContent = "（DM 正在为这一夜落笔…）";
  endingBody.appendChild(endingStreamEl);
  sendMsg({ type: "generate_ending", text: "" });
}
function onEndingDelta(msg) {
  if (PUBLIC_MODE) {
    const p = document.getElementById("shiftprose");
    if (!p) return;
    if (p.dataset.streaming !== "1") { p.textContent = ""; p.dataset.streaming = "1"; }
    p.textContent += msg.delta || "";
    return;
  }
  if (!endingStreamEl) { buildEndingModal(); endingEl.style.display = "flex";
    endingInputRow.style.display = "none"; endingStreamEl = document.createElement("div");
    endingBody.innerHTML = ""; endingBody.appendChild(endingStreamEl); endingStreamText = ""; }
  if (endingStreamText === "") endingStreamEl.textContent = "";
  endingStreamText += msg.delta;
  endingStreamEl.textContent = endingStreamText;
}
function onEndingReply(msg) {
  if (PUBLIC_MODE) {
    const p = document.getElementById("shiftprose");
    if (p && msg.text) p.textContent = msg.text;
    return;
  }
  buildEndingModal();
  if (endingStreamEl) endingStreamEl.textContent = msg.text || endingStreamText;
  const f = msg.factors || {};
  const bits = [
    f.dead ? `${f.dead} 人横死` : "无人丧命",
    f.kills ? `${f.kills} 桩仇杀得逞` : "仇杀未遂",
    f.escaped ? `${f.escaped} 人越狱` : "无人越狱",
    f.solitary ? `${f.solitary} 人关禁闭` : null,
    f.quests_done ? `${f.quests_done} 人图谋达成` : null,
  ].filter(Boolean);
  const sum = document.createElement("div");
  sum.style.cssText = "margin-top:18px;padding-top:14px;border-top:1px solid rgba(255,255,255,.12);" +
    "font:700 14px/1.7 system-ui;color:#ffd98a;";
  sum.innerHTML = `本局基调：<b>${msg.tier_cn || "?"}</b> · 混乱指数 = ${msg.score ?? "?"}<br>` +
    `<span style="font-weight:500;color:#aeb6c4;">${bits.join(" · ")}</span>`;
  endingBody.appendChild(sum);
  // 五名囚犯各自的最终下落——与正文互为表里
  const fates = msg.fates || {};
  if (Object.keys(fates).length) {
    const fbox = document.createElement("div");
    fbox.style.cssText = "margin-top:14px;display:flex;flex-direction:column;gap:6px;";
    fbox.innerHTML = '<div style="font:700 13px system-ui;color:#ffb3b3;margin-bottom:2px;">🕯️ 各人的最终下落</div>';
    for (const id of Object.keys(fates)) {
      if (!fates[id]) continue;
      const nm = (bb.npc_status && bb.npc_status[id] && bb.npc_status[id].name) || NPC_NAME[id] || id;
      const row = document.createElement("div");
      row.style.cssText = "padding:6px 9px;border-radius:8px;background:rgba(40,28,33,.6);" +
        "border-left:2px solid #a55a5a;font:500 12.5px/1.55 system-ui;color:#dccdd0;";
      row.innerHTML = `<b style="color:#ffb3b3;">${nm}</b>：${fates[id]}`;
      fbox.appendChild(row);
    }
    endingBody.appendChild(fbox);
  }
  const close = document.createElement("button");
  close.textContent = "关闭";
  close.style.cssText = "margin-top:14px;padding:8px 16px;border-radius:9px;border:1px solid #384152;" +
    "cursor:pointer;font:700 13px system-ui;color:#e9edf5;background:#1b2130;";
  close.onclick = () => { endingEl.style.display = "none"; };
  endingBody.appendChild(close);
}

// ---------------------------------------------------------------------------
// 分阶段开场白：每解锁一段空间，指定的核心 NPC 主动走到主角跟前，用符合人设的
// 口吻说一段话开启本阶段的故事（走近 + 头顶气泡 + 底部电影字幕，不弹窗）。
// ---------------------------------------------------------------------------
const PHASE_INTRO = {
  cellblock: { npc: "laochen",
    line: "醒啦？欢迎来到这鬼地方。别慌，我叫老牙，这间牢房的老住户了。你这案子……啧，一看就是被人做了局。想在这儿活着出去，头一件事，是先出这道门——想不想听我教你？" },
  public: { npc: "laochen",
    line: "出来了就别缩着。这外头水深——想洗清你的冤，先在公共区四处转转：墙根、地上、柜子里，凡是不对劲的都给我记下来。摸够两条线索，我再教你往夜里更深的地方钻。" },
  restricted: { npc: "yanjing",
    line: "……夜里的路不好走。狱警的眼睛、那条狗的鼻子，哪一样都能要了你的命。想往里探？先掂量掂量自己有几条命——需要什么，来医务室找我，当然，得看你出得起什么价。" },
  office: { npc: "laochen",
    line: "听见警铃没有？乱起来了！机会就这一次——趁乱冲进典狱长办公室，把他保险柜里那本黑账给我抠出来！去啊，别愣着，这是你翻身的唯一机会！" },
};
const introDone = new Set();
let introPending = null;          // {zone, npc, started} 正在等/收 LLM 开场白
let introFallbackTimer = null;
let introStreamText = "";
let introDismissed = false;       // 开场背景介绍页还没关掉前，别急着放阶段开场白
function checkPhaseIntro() {
  if (OBSERVER_MODE) return;      // 上帝观察：没有"核心NPC走过来给主角搭话"的第一人称开场白
  if (!introDismissed) return;    // 玩家还在读背景介绍，先按住剧情开场白
  for (const z of (bb.zones_unlocked || [])) {
    if (!PHASE_INTRO[z] || introDone.has(z)) continue;
    introDone.add(z);
    setTimeout(() => startPhaseIntro(z), z === "cellblock" ? 1400 : 800);
  }
}
function startPhaseIntro(z) {
  const cfg = PHASE_INTRO[z];
  const e = npcs.get(cfg.npc);
  if (e) e.introUntil = performance.now() + 12000;   // 这段时间主动走向主角 + 面向他
  introPending = { zone: z, npc: cfg.npc, started: false };
  introStreamText = "";
  sendMsg({ type: "phase_intro", zone: z });          // 请 LLM 即兴生成开场台词
  clearTimeout(introFallbackTimer);                   // 兜底：LLM 6s 不吭声，就用固定脚本
  introFallbackTimer = setTimeout(() => {
    if (introPending && introPending.zone === z && !introPending.started) {
      speakIntro(cfg.npc, cfg.line);
      introPending = null;
    }
  }, 6000);
}
function onPhaseIntroDelta(msg) {
  if (introPending) { introPending.started = true; clearTimeout(introFallbackTimer); }
  introStreamText += msg.delta;
  liveIntro(msg.name, introStreamText, msg.npc_id);   // 边收边显示（字幕+头顶气泡）
}
function onPhaseIntroReply(msg) {
  liveIntro(msg.name, msg.text, msg.npc_id);
  introStreamText = "";
  introPending = null;
}
// 把一段（可能仍在增长的）开场白呈现出来：底部电影字幕 + 说话人头顶气泡 + 面向主角
function liveIntro(name, text, npcId) {
  showSubtitle(name, text);
  const e = npcId && npcs.get(npcId);
  if (e) {
    e.replyText = text.length > 34 ? text.slice(0, 32) + "…" : text;
    e.replyUntil = performance.now() + 9000;
    e.introUntil = Math.max(e.introUntil || 0, performance.now() + 7000);
  }
}
function speakIntro(npcId, line) {   // 兜底：LLM 失败/超时时退回固定脚本
  liveIntro(npcName(npcId), line, npcId);
}
let subtitleEl = null, subTimer = null;
function showSubtitle(name, line) {
  if (!subtitleEl) {
    subtitleEl = document.createElement("div");
    subtitleEl.style.cssText = "position:fixed;left:50%;bottom:132px;transform:translateX(-50%);z-index:44;" +
      "max-width:min(720px,90vw);padding:12px 20px;border-radius:12px;pointer-events:none;" +
      "background:rgba(12,14,20,.85);border:1px solid rgba(255,210,120,.35);color:#f2ead2;" +
      "font:600 15px/1.75 system-ui;text-align:center;opacity:0;transition:opacity .45s;backdrop-filter:blur(2px);";
    document.body.appendChild(subtitleEl);
  }
  subtitleEl.innerHTML = `<span style="color:#ffd98a;font-weight:800;">${name}</span>：${line}`;
  subtitleEl.style.opacity = "1";
  clearTimeout(subTimer);
  subTimer = setTimeout(() => { subtitleEl.style.opacity = "0"; }, 9500);
}

function download(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
function exportPng() {
  elfarolCanvas.toBlob((b) => b && download(`el-farol-tick${latestWorld?.tick ?? 0}.png`, b));
}
function exportCsv() {
  const w = latestWorld || {};
  let csv = "# 放风人数序列 (attendance)\ntick_index,yard_count\n";
  (w.attendance || []).forEach((v, i) => { csv += `${i},${v}\n`; });
  csv += `\n# 舒适阈值,${w.yard_comfort ?? ""}\n# 总人数,${w.n_agents ?? ""}\n`;
  csv += "\n# 各性格选择分布 (choice_dist)\npersonality,goal,count\n";
  for (const [p, goals] of Object.entries(w.choice_dist || {})) {
    for (const [g, c] of Object.entries(goals)) csv += `${p},${g},${c}\n`;
  }
  download(`prison-metrics-tick${w.tick ?? 0}.csv`, new Blob([csv], { type: "text/csv;charset=utf-8" }));
}
async function exportJson() {
  try {
    const r = await fetch(`http://${BACKEND_HOST}/metrics`);
    const j = await r.json();
    download(`prison-metrics-tick${j.tick}.json`, new Blob([JSON.stringify(j, null, 2)], { type: "application/json" }));
  } catch (e) { showToast("导出失败：" + e); }
}
function updateCmdLabel() {
  const el = document.getElementById("cmdlabel");
  const d = selectedId ? latestDecision.get(selectedId) : null;
  el.textContent = selectedId
    ? `指挥「${d?.name ?? selectedId}」做：`
    : "下指令（先选一个犯人）";
}

// ---------------------------------------------------------------------------
// 昼夜光照（按作息时段平滑切换）
// ---------------------------------------------------------------------------
const PHASE_LIGHT = {
  点名:    { sun: 0.55, sunColor: 0xffce9e, hemi: 0.42, amb: 0.16, fog: 0x6a5a52, exp: 0.96, night: 0.25 },
  训练:    { sun: 1.15, sunColor: 0xfff4e0, hemi: 0.55, amb: 0.18, fog: 0x565f6a, exp: 1.05, night: 0.0 },
  上课:    { sun: 1.10, sunColor: 0xfff2d8, hemi: 0.55, amb: 0.18, fog: 0x565f6a, exp: 1.05, night: 0.0 },
  放风:    { sun: 1.30, sunColor: 0xffffff, hemi: 0.62, amb: 0.20, fog: 0x66707c, exp: 1.10, night: 0.0 },
  自由活动: { sun: 0.70, sunColor: 0xffb070, hemi: 0.45, amb: 0.16, fog: 0x5e4a50, exp: 1.00, night: 0.45 },
  就寝:    { sun: 0.12, sunColor: 0x8496c8, hemi: 0.20, amb: 0.08, fog: 0x14171f, exp: 0.86, night: 1.0 },
  // 夜景切面：暖光池仍然来自实体灯具（火盆/顶灯/泛光灯柱/探照灯），但环境光不能压到零。
  // 这套资产是纯色无贴图的，形体全靠明暗读出来；环境光一归零，灯池之外就沉成一块块死黑，
  // 既显得暗，也显得"没有细节、像低模"。给足月光级别的填充光，转折和厚度才回来。
  夜景: { sun: 0.95, sunColor: 0x93a8d6, hemi: 0.92, amb: 0.46, fog: 0x12182a, exp: 1.08, night: 1.0 },
};
const lightState = { sun: 1.15, hemi: 0.55, amb: 0.18, exp: 1.05, night: 0.0 };
let lightTarget = PHASE_LIGHT["放风"];
const _col = new THREE.Color();
// 常驻夜景（NIGHT_DIORAMA 在文件开头声明）：整局锁死"夜间切面"的观感，
// 作息该干的事照旧，只是天不再亮。ALWAYS_DAY 是它关掉之后的次选：常驻白天。
const ALWAYS_DAY = false;
function setPhaseLighting(phase) {
  lightTarget = NIGHT_DIORAMA ? PHASE_LIGHT["夜景"]
    : ALWAYS_DAY ? PHASE_LIGHT["放风"]
    : (PHASE_LIGHT[phase] || PHASE_LIGHT["放风"]);
}
setPhaseLighting();
function updateLighting(dt) {
  const k = Math.min(1, dt * 1.6);
  lightState.sun += (lightTarget.sun - lightState.sun) * k;
  lightState.hemi += (lightTarget.hemi - lightState.hemi) * k;
  lightState.amb += (lightTarget.amb - lightState.amb) * k;
  lightState.exp += (lightTarget.exp - lightState.exp) * k;
  lightState.night += (lightTarget.night - lightState.night) * k;
  sun.intensity = lightState.sun;
  hemi.intensity = lightState.hemi;
  ambient.intensity = lightState.amb;
  renderer.toneMappingExposure = lightState.exp;
  sun.color.lerp(_col.setHex(lightTarget.sunColor), k);
  scene.fog.color.lerp(_col.setHex(lightTarget.fog), k);
  if (gradePass) gradePass.uniforms.uBrightness.value = lightState.exp;  // 昼夜亮度经调色 pass 生效
  for (const l of nightLamps) {
    l.light.intensity = lightState.night * l.baseIntensity;
    if (l.bulb) l.bulb.material.emissiveIntensity = lightState.night;
  }
  for (const m of nightGlow) m.emissiveIntensity = lightState.night * 0.9;
  // 室内灯：玩家动过开关的完全手动，没动过的跟着昼夜自己亮/灭
  for (const L of roomLamps) {
    const on = L.manual ? L.on : lightState.night > 0.35;
    if (!L.manual) L.on = on;
    L.light.intensity = on ? L.base : 0;
    L.bulb.material.emissiveIntensity = on ? (L.glow ?? 1) : 0;
  }
}

// ---------------------------------------------------------------------------
// 主循环
// ---------------------------------------------------------------------------
const clock = new THREE.Clock();
const tmp = new THREE.Vector3();

// ---- 帧率上限 ---------------------------------------------------------------
// Three.js 默认每个 requestAnimationFrame 都渲染，会把 GPU 顶到 100%（本机录屏时
// NVENC 硬件编码器抢不到显存/带宽，录一分多钟就自动断；平时也表现为卡）。
// 锁到目标帧率后，渲染与位置更新一起降频，画面观感几乎无损（30fps 足够流畅、正是录屏标准），
// GPU 占用大幅下降。可用 ?fps=NN 覆盖（如 ?fps=60 想更顺、GPU 有余量时）。
const _fpsParam = parseInt(new URLSearchParams(location.search).get("fps") || "", 10);
const TARGET_FPS = Number.isFinite(_fpsParam) && _fpsParam > 0 ? _fpsParam : 30;
// 减 1ms 余量：帧间隔量化到显示器 vsync，卡在整数倍边界会掉成半速，留点余量更贴近目标帧率。
const FRAME_MIN_MS = 1000 / TARGET_FPS - 1;
let _lastFrameMs = -1e9;

function tick(now) {
  requestAnimationFrame(tick);
  if (now === undefined) now = performance.now();
  // 帧率封顶：距上次渲染不足一帧间隔就跳过这次 rAF。注意 clock.getDelta() 只在真正渲染的
  // 那一帧才调用，所以被跳过的时间会累计进下一帧的 dt（再被下面的 0.05s 封顶兜住），不引入瞬移。
  if (now - _lastFrameMs < FRAME_MIN_MS) return;
  _lastFrameMs = now;
  // 切到后台标签页时不渲染，把 GPU 完全让出来（切回来自动继续）。
  if (document.hidden) return;
  // dt 封顶 0.05s：卡顿/切后台回来时 getDelta() 会返回很大的值（0.5s~几秒），
  // 那一帧所有「速度×dt」的位移会一次性蹿出去一大段，看着就是 NPC「瞬移/闪一下」
  // ——自由溜达、回牢房人多、GPU 被模型占满掉帧时尤其明显。封顶后单帧最多走一小步，
  // 宁可慢半拍也绝不瞬移。（位置权威在后端，短暂落后会由 catchUp 平滑补回。）
  const dt = Math.min(clock.getDelta(), 0.05);
  updateLighting(dt);
  updateSearchlights(clock.elapsedTime);

  // 顺序要紧：先按当前镜头方向移动主角，再让镜头跟上新位置
  if (player) movePlayer(dt);
  updateCamera(dt);
  if (player) {
    const p = player.root.position;
    tmp.copy(p); tmp.y += 2.4; tmp.project(camera);
    player.label.innerHTML = `<div class="nb-head"><span class="who">${WARDEN_MODE ? "典狱长" : "你"}</span>` +
      `<span class="act">${WARDEN_MODE ? "🎖巡视" : "🚶主角"}</span></div>` +
      `<div class="nb-say">${chatOpen ? "对话中…" : (WARDEN_MODE ? "WASD 走查 · 走近犯人按 E 审讯" : "WASD 走动 · 走近犯人按 E")}</div>`;
    player.label.style.left = `${(tmp.x * 0.5 + 0.5) * window.innerWidth}px`;
    player.label.style.top = `${(-tmp.y * 0.5 + 0.5) * window.innerHeight}px`;
    player.label.style.display = (tmp.z < 1 && camMode !== "fps") ? "block" : "none";
  }
  updateTalkTarget();
  updateFocusAct();
  updateGotoBar();
  updateCellDoors(dt);            // 牢门：有人靠近就荡开，锁上的一律关死并挡人
  updateRoomDoors(dt);            // 功能室门：有人正对门口就荡开，人走了合上

  // 线索物：漂浮 + 旋转（发光物件靠 bloom 更醒目）
  const tt = performance.now() * 0.001;
  for (const c of clueEntries) {
    c.group.position.y = c.baseY + Math.sin(tt * 2 + c.baseY) * 0.12;
    c.gem.rotation.y += dt * 1.2;
  }

  const SPEED = 3.2;              // 恒定行走速度（单位/秒）→ 连续滑行，不再一跳一跳
  const RUN = 5.5;
  bubbleBoxes.length = 0;         // 本帧的头顶气泡屏幕框，稍后统一防重叠排版
  for (const e of npcs.values()) {
    if (e.dead) continue;         // 尸体不动、不播动画
    const pos = e.root.position;
    // NaN 自愈：位置或朝向一旦变成非有限值，整具模型会从视野里凭空消失、且不会自己恢复
    // （rotation 每帧 += 会把 NaN 永久累积）。用后端权威位置把它拉回来，rotation 清零。
    if (!Number.isFinite(pos.x) || !Number.isFinite(pos.z) || !Number.isFinite(e.root.rotation.y)) {
      const sx = Number.isFinite(e.serverPos && e.serverPos.x) ? e.serverPos.x : (e._goodX ?? 0);
      const sz = Number.isFinite(e.serverPos && e.serverPos.z) ? e.serverPos.z : (e._goodZ ?? 0);
      pos.set(sx, 0, sz);
      e.dest.set(sx, 0, sz);
      e.root.rotation.set(0, 0, 0);
      e.heading = 0; e._stuck = 0;
    }
    e._goodX = pos.x; e._goodZ = pos.z;   // 记录最近一次有效位置，供兜底回退
    if (e.confined) {             // 关禁闭：贴着禁闭室位置。离得远（残余的兜底落格）就快步走进去，
      // 不再一帧硬吸附（那就是"瞬移"）；走到门口才钉住、就地待着。
      const cd = Math.hypot(e.serverPos.x - pos.x, e.serverPos.z - pos.z);
      if (cd <= 0.25) {
        pos.set(e.serverPos.x, pos.y, e.serverPos.z);
        e.settled = true;
      } else {
        e.dest.copy(e.serverPos);   // 交给下面的常规行走送进去：带走路动画、按 catchUp 封顶提速
        e.settled = false;
      }
    }
    const dist = pos.distanceTo(e.dest);
    const running = e.goal === "PrisonBreak";
    // 后端不认墙、一律 3.5 m/s 直线赶路，前端按各人步幅配速只有 1.2~1.7 m/s。贴墙连排把
    // 单程拉到五十多米之后，这点差距能攒出二十几米：人还在院子里走，决策卡已经写着他在健身，
    // 典狱长走到眼前后端也不认为他在跟前（敬礼/监视判定用的是后端坐标）。
    // 所以落后超过 CATCH_SLACK 就提速追，移动速度与走路步频同倍放大 —— 脚不会贴地滑；
    // 短距离挪动照旧走标定步速，样子和以前一模一样。
    const catchUp = clamp(pos.distanceTo(e.serverPos) / CATCH_SLACK, 1, CATCH_MAX);
    const speed = (running ? RUN : SPEED) * (e.pace || 1) * catchUp;  // pace：瘸子等按自身步幅配速
    const walkAction = e.actions[e.walkAct];
    if (walkAction) walkAction.setEffectiveTimeScale(e.walkTS * catchUp);
    // 去抖：到位后进入"驻留"死区——不再追 dest 的细小变化/后端位置抖动，睡觉歇着纹丝不动；
    // 只有目标真的移开(>NPC_GO：换设施、被叫过来)才解除驻留、重新起步。
    if (dist <= NPC_STOP) e.settled = true;
    else if (dist >= NPC_GO) e.settled = false;
    // 典狱长在场：靠得够近就"僵住立正"——本拍不移动、原地站定并面朝典狱长（附了狱警监视之效）
    // 例外：正被押往禁闭室的人不受此约束。他已经在狱警手里了，还立正只会卡在半路，
    // 押送就永远走不到（后端那边人已经在动，前后端会拉成橡皮筋）。
    const wardenFreeze = WARDEN_MODE && player && !e.escort && !e.confined &&
      pos.distanceToSquared(player.root.position) < WARDEN_FREEZE_R2;
    // 恒速前进：朝寻路给出的当前路点走（被墙挡住时绕行，直达时就是终点本身）
    if (!e.settled && !wardenFreeze) {
      const bx0 = pos.x, bz0 = pos.z;
      const [sx, sz] = navSteer(e, e.dest.x, e.dest.z);
      const sdx = sx - pos.x, sdz = sz - pos.z, sd = Math.hypot(sdx, sdz);
      if (sd > 1e-4) {
        const step = Math.min(sd, speed * dt);
        pos.x += (sdx / sd) * step; pos.z += (sdz / sd) * step;
        e.heading = Math.atan2(sx - pos.x, sz - pos.z);
      }
      resolveCollision(pos);          // 兜底：贴到墙里就沿最浅轴弹出，绝不穿模
      // 卡墙自救：想走却几乎没挪动 → 弃掉旧路径，改去追后端权威位置。
      // 以前这里是**朝 serverPos 直着顶一步**，那是"人从墙里钻出来"的直接来源：
      // 后端当年不认墙，权威位置本身可能就在墙里，顶过去就是顶进墙。现在后端也走
      // 绕墙路径（nav.py），权威位置一定在可走处，所以这里改成同样绕墙地走过去。
      const moved = Math.hypot(pos.x - bx0, pos.z - bz0);
      e._stuck = (dist > 0.5 && moved < 0.02) ? (e._stuck || 0) + 1 : 0;
      if (e._stuck > 18) {
        e._path = null; e._pathKey = null;
        const [rx, rz] = navSteer(e, e.serverPos.x, e.serverPos.z);
        const ddx = rx - pos.x, ddz = rz - pos.z, dd = Math.hypot(ddx, ddz);
        if (dd > 1e-3) {
          const st = Math.min(dd, speed * 1.8 * dt);
          pos.x += (ddx / dd) * st; pos.z += (ddz / dd) * st;
          resolveCollision(pos);
        }
        if (e._stuck > 70) e._stuck = 0;   // 顶一阵仍不行就重置，避免长期抽搐
      }
      e.moving = dist > 0.35;
    } else {
      e.moving = false;
      // 到达家具工位：对齐床/凳锚点并面朝家具，再播坐躺——否则人会停在房间空地上做动作。
      if (e.station && !e.going) {
        e.station.anchor.getWorldPosition(_wp);
        pos.x = _wp.x; pos.z = _wp.z;
        e.heading = anchorYaw(e.station.anchor);
      } else if (FACE_PEER.has(e.goal)) {
        let best = null, bd = 64;
        for (const o of npcs.values()) {
          if (o === e) continue;
          const d2 = o.root.position.distanceToSquared(pos);
          if (d2 < bd) { bd = d2; best = o; }
        }
        if (best) e.heading = Math.atan2(best.root.position.x - pos.x, best.root.position.z - pos.z);
      }
    }
    // 开场白期间：面朝主角说话（更像真的在跟你交代事情）
    if (e.introUntil && performance.now() < e.introUntil && player) {
      e.heading = Math.atan2(player.root.position.x - pos.x, player.root.position.z - pos.z);
    }
    // 对话期间：面朝正在交谈的对方（凑近 + 对视，更像真的在聊）
    if (e.convoUntil && performance.now() < e.convoUntil && e.convoPeer) {
      const pe = npcs.get(e.convoPeer);
      if (pe) e.heading = Math.atan2(pe.root.position.x - pos.x, pe.root.position.z - pos.z);
    }
    // 被典狱长盯住：无论刚才朝哪，都转身面对他（像被叫住立正）
    if (wardenFreeze) { e.moving = false; e.heading = Math.atan2(player.root.position.x - pos.x, player.root.position.z - pos.z); }
    // 平滑转身
    let dh = (e.heading - e.root.rotation.y + Math.PI * 3) % (Math.PI * 2) - Math.PI;
    e.root.rotation.y += dh * Math.min(1, dt * 8);

    // 动画：走/跑 或 到达后按目标做动作（含去抖：moving 状态迟滞）
    let anim = "Idle";
    if (wardenFreeze) anim = "Idle";                                   // 被典狱长盯住：僵住立正
    else if (e.moving) anim = running ? "Running" : "Walking";
    else if (e.goal === "Fight") anim = e.inBout ? "Punch" : "Idle";   // 没真开打就不打空气
    else if (e.goal && GOAL_ANIM[e.goal]) anim = GOAL_ANIM[e.goal];
    playAction(e, anim);
    e.mixer.update(dt);
    // 过程化姿态：必须在 mixer 之后（同 applySalutePose），把 idle 写好的骨骼再压成
    // 坐/躺/跪/举铁/张望…… 走路跑步、被盯住立正、以及"没真开打"的时候不叠。
    // 该角色真有对应片段时（老牙的 kneeling、疤脸/瘸子的 fight）姿态层会自己让位。
    // 还没走到（前往中）或没认到床/凳时不叠坐躺，否则空地上就会倒下/坐下。
    const needSeat = !!STATION_OF[e.goal];
    const arrived = !e.going && e.actionState === "进行中" && (!needSeat || !!e.station);
    const posing = !wardenFreeze && !e.moving && arrived && !(e.goal === "Fight" && !e.inBout);
    applyGoalPose(e, posing ? e.goal : null, dt);

    // 头顶气泡（先定位，top 交给防重叠排版统一决定）
    tmp.copy(pos); tmp.y += 2.4; tmp.project(camera);
    const vis = tmp.z < 1;
    e.label.style.display = vis ? "block" : "none";
    if (vis) {
      const sx = (tmp.x * 0.5 + 0.5) * window.innerWidth;
      const sy = (-tmp.y * 0.5 + 0.5) * window.innerHeight;
      e.label.style.left = `${sx}px`;
      const rep = e.replyUntil && performance.now() < e.replyUntil;
      const sel = e.label.classList.contains("sel");
      const prio = sel ? 2 : (rep ? 1 : 0);
      e.label.style.zIndex = sel ? 27 : (rep ? 24 : 12);   // 选中/正在说话的气泡压在最上层
      applyBubbleLod(e.label, pos, sx, sy, prio);
      bubbleBoxes.push({ el: e.label, x: sx, y: sy, prio });
    }
  }

  // 巡逻者：恒速滑行 + 平滑转向（视野锥随之扫动）
  for (const [pid, e] of patrollers) {
    if (!e.root.visible) { e.label.style.display = "none"; continue; }
    const pos = e.root.position;
    // NaN 自愈（同 NPC）：非有限坐标/朝向会让狱警凭空消失且不恢复，用权威目标点拉回。
    if (!Number.isFinite(pos.x) || !Number.isFinite(pos.z) || !Number.isFinite(e.root.rotation.y)) {
      const dx = Number.isFinite(e.dest && e.dest.x) ? e.dest.x : (e._goodX ?? 0);
      const dz = Number.isFinite(e.dest && e.dest.z) ? e.dest.z : (e._goodZ ?? 0);
      pos.set(dx, 0, dz); e.root.rotation.set(0, 0, 0); e.heading = 0;
    }
    e._goodX = pos.x; e._goodZ = pos.z;
    const dist = pos.distanceTo(e.dest);
    // 铁面被玩家拦下(或正跟他对话)时站住听你说话；松开后加速赶回巡逻线，不留橡皮筋感
    const held = pid === "guard1" && (guardHeld || (chatOpen && chatNpcId === "guard1"))
      || e.saluting;                                 // 向典狱长行礼时站定，别边敬礼边走
    if (dist <= NPC_STOP) e.settled = true;          // 同款去抖死区：巡逻到位不原地抽搐
    else if (dist >= NPC_GO) e.settled = false;
    let stepped = 0;
    if (!held && !e.settled) {
      // 缓步巡逻：基础速度贴近后端节奏，只在明显落后(>7m)时轻微追赶(1.4×)，不再窜行
      const base = e.kind === "dog" ? 2.8 : 2.3;
      const [sx, sz] = navSteer(e, e.dest.x, e.dest.z);   // 狱警/狱犬同样绕墙巡逻
      const sdx = sx - pos.x, sdz = sz - pos.z, sd = Math.hypot(sdx, sdz);
      if (sd > 1e-4) {
        const step = Math.min(sd, base * (dist > 7 ? 1.4 : 1) * dt);
        pos.x += (sdx / sd) * step; pos.z += (sdz / sd) * step;
        e.heading = Math.atan2(sx - pos.x, sz - pos.z);
        stepped = step;
      }
      resolveCollision(pos);          // 兜底：贴到墙里就沿最浅轴弹出，绝不穿模
    }
    if (e.mixer) {                           // 警服模型：骨骼动画，步频跟着这一帧真实位移
      playAction(e, stepped > 1e-4 ? "Walking" : "Idle");
      syncWalkTimeScale(e, dt > 1e-6 ? stepped / dt : 0);
      e.mixer.update(dt);
      applySalutePose(e, dt, !!e.saluting);  // 必须在 mixer 之后：叠在 idle 写好的骨骼上
    } else {
      animatePatrollerLimbs(e, stepped, dt); // 方块狱警/狱犬：程序化迈腿摆臂，站定回正
    }
    // 敬礼时正面转向典狱长（后端也转了朝向，但前端插值更跟手，免得看着像斜眼行礼）
    if (e.saluting && player) {
      e.heading = Math.atan2(player.root.position.x - pos.x, player.root.position.z - pos.z);
    }
    let dh = (e.heading - e.root.rotation.y + Math.PI * 3) % (Math.PI * 2) - Math.PI;
    e.root.rotation.y += dh * Math.min(1, dt * 6);
    // 押送/敬礼时收起监视范围指示：这两件事都不是在查人，扇形留着只会误导玩家
    if (e.root.userData.cone) e.root.userData.cone.visible = !(e.escorting || e.saluting);
    tmp.copy(pos); tmp.y += 2.6; tmp.project(camera);
    const vis = tmp.z < 1;
    e.label.style.display = vis ? "block" : "none";
    if (vis) {
      const sx = (tmp.x * 0.5 + 0.5) * window.innerWidth;
      const sy = (-tmp.y * 0.5 + 0.5) * window.innerHeight;
      e.label.style.left = `${sx}px`;
      applyBubbleLod(e.label, pos, sx, sy, 0);
      bubbleBoxes.push({ el: e.label, x: sx, y: sy, prio: 0 });
    }
  }
  declutterBubbles(bubbleBoxes);         // 屏幕空间防重叠：相邻气泡上移错开，不再互相遮挡
  checkStealthDetection();               // A1：受限区被巡逻者命中→狡辩闭环

  for (const l of facilityLabels) {
    tmp.copy(l.obj.position); tmp.y += l.y; tmp.project(camera);
    l.el.style.left = `${(tmp.x * 0.5 + 0.5) * window.innerWidth}px`;
    l.el.style.top = `${(-tmp.y * 0.5 + 0.5) * window.innerHeight}px`;
    l.el.style.display = tmp.z < 1 ? "block" : "none";
  }

  if (composer) composer.render();
  else renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
  renderer.setPixelRatio(dpr);
  renderer.setSize(w, h);
  if (composer) { composer.setPixelRatio(dpr); composer.setSize(w, h); }
  if (outlinePass) outlinePass.setSize(w, h);
});

// ---------------------------------------------------------------------------
// 命运账本：每个 NPC 当前处在剧情哪一"幕" + 叙事化命运流水。随交互实时刷新，
// 不再堆到结局才交代。关键节点(big)顺带走一条电影字幕，让"改写命运"有仪式感。
// ---------------------------------------------------------------------------
let fatePanel = null, fateBody = null, fateStatusEl = null, fateShown = 0, fateOpen = true;
const FATE_ORDER = ["laochen", "laok", "shitou", "yanjing", "xiaodou", "guard1", "dog1"];
function buildFatePanel() {
  if (fatePanel) return;
  const btn = document.createElement("div");
  btn.id = "fbtn";
  btn.style.cssText = "position:fixed;right:12px;bottom:56px;z-index:30;cursor:pointer;user-select:none;" +
    "background:rgba(18,20,28,.85);border:1px solid #6a4550;border-radius:10px;padding:8px 12px;" +
    "font:600 13px system-ui;color:#ffb3b3;";
  btn.textContent = "🕯️ 命运账本 (K)";
  btn.onclick = toggleFate;
  document.body.appendChild(btn);
  fatePanel = document.createElement("div");
  fatePanel.style.cssText = "position:fixed;right:12px;top:372px;width:300px;max-height:44vh;z-index:29;" +
    "display:flex;flex-direction:column;background:rgba(18,13,17,.9);border:1px solid #4a333c;" +
    "border-radius:12px;padding:11px 13px;backdrop-filter:blur(6px);";
  fatePanel.innerHTML =
    '<div style="font:800 13px system-ui;color:#ffb3b3;letter-spacing:1px;">🕯️ 命运账本</div>' +
    '<div style="font:500 10.5px/1.4 system-ui;color:#9a858c;margin:3px 0 9px;">你的每句话、每个举动，都在悄悄改写他们的结局。</div>' +
    '<div id="fatestatus" style="display:flex;flex-direction:column;gap:3px;margin-bottom:9px;"></div>' +
    '<div style="font:700 11px system-ui;color:#8a747c;border-top:1px solid #33262c;padding-top:7px;margin-bottom:5px;">命运流水</div>' +
    '<div id="fatelog" style="overflow-y:auto;display:flex;flex-direction:column;gap:6px;"></div>';
  document.body.appendChild(fatePanel);
  fateBody = fatePanel.querySelector("#fatelog");
  fateStatusEl = fatePanel.querySelector("#fatestatus");
}
function toggleFate() { fateOpen = !fateOpen; if (fatePanel) fatePanel.style.display = fateOpen ? "flex" : "none"; }
function renderFate() {
  if (PUBLIC_MODE) return;
  buildFatePanel();
  const status = bb.npc_status || {};
  fateStatusEl.innerHTML = "";
  for (const id of FATE_ORDER) {
    const s = status[id];
    if (!s) continue;
    const row = document.createElement("div");
    row.style.cssText = "display:flex;justify-content:space-between;gap:8px;font:500 11.5px system-ui;";
    row.innerHTML = `<span style="color:#c9bcc0;">${s.name}</span>` +
      `<span style="color:#e6a2a2;text-align:right;">${s.status}</span>`;
    fateStatusEl.appendChild(row);
  }
  const log = bb.fate_log || [];
  if (log.length < fateShown) { fateShown = 0; fateBody.innerHTML = ""; }   // 后端 reset 了
  const firstFill = fateShown === 0;
  for (let i = fateShown; i < log.length; i++) {
    const b = log[i];
    const item = document.createElement("div");
    item.style.cssText = "padding:6px 8px;border-radius:8px;background:rgba(44,28,33,.7);" +
      "border-left:2px solid #a55a5a;font:500 12px/1.5 system-ui;color:#dccdd0;";
    item.innerHTML = `<b style="color:#ffb3b3;">${b.name} · ${b.status}</b><br>${b.text}`;
    fateBody.appendChild(item);
    if (!firstFill && b.big) showSubtitle(b.name, b.text);   // 关键节点：电影字幕
  }
  if (log.length > fateShown) { fateShown = log.length; fateBody.scrollTop = fateBody.scrollHeight; }
}

// ---------------------------------------------------------------------------
// 实时社交字幕流（P1/P2 可见性）：把后端 social_feed（结盟/背叛/放狠话/军火/讲和/告密…）
// 逐条以流的形式贴到屏幕左下角。这是"NPC 之间真的在互相影响"这件事被看见的窗口——
// 没有它，社交写回全发生在暗处，上帝视角等于瞎子。左下角，避开右侧命运账本。
// ---------------------------------------------------------------------------
let socialPanel = null, socialBody = null;
const socialSeen = new Set();
const SOCIAL_KIND_STYLE = {
  ally: "#8fd6a0", betray: "#ff8f8f", threat: "#ffb04d", snitch: "#d59bff",
  deal_weapon: "#ff9c93", deal: "#9fb0c8", reconcile: "#8fd6c8", protect: "#8fd6a0",
  repay: "#ffd9a0", info: "#9fb0c8",
};
function buildSocialFeed() {
  if (socialPanel) return;
  socialPanel = document.createElement("div");
  socialPanel.style.cssText = "position:fixed;left:12px;bottom:12px;width:290px;max-height:38vh;z-index:29;" +
    "display:none;flex-direction:column;background:rgba(13,15,20,.86);border:1px solid #33404a;" +
    "border-radius:12px;padding:10px 12px;backdrop-filter:blur(6px);pointer-events:none;";
  socialPanel.innerHTML =
    '<div style="font:800 12.5px system-ui;color:#9fd6ff;letter-spacing:1px;">🔗 社交动态 · 实时</div>' +
    '<div style="font:500 10px/1.4 system-ui;color:#7c8a95;margin:2px 0 8px;">他们彼此结盟、翻脸、放话、做交易——都在这里。</div>' +
    '<div id="socialfeedlog" style="overflow-y:auto;display:flex;flex-direction:column;gap:5px;"></div>';
  document.body.appendChild(socialPanel);
  socialBody = socialPanel.querySelector("#socialfeedlog");
}
function renderSocialFeed(feed) {
  buildSocialFeed();
  if (!Array.isArray(feed) || !feed.length) return;
  socialPanel.style.display = "flex";
  for (const ev of feed) {
    const key = `${ev.tick}|${ev.kind}|${ev.src}|${ev.dst}`;
    if (socialSeen.has(key)) continue;
    socialSeen.add(key);
    const color = SOCIAL_KIND_STYLE[ev.kind] || "#c9d3da";
    const item = document.createElement("div");
    item.style.cssText = "padding:6px 8px;border-radius:8px;background:rgba(26,32,40,.75);" +
      `border-left:2px solid ${color};font:500 12px/1.45 system-ui;color:#dbe3ea;opacity:0;` +
      "transform:translateX(-6px);transition:opacity .35s,transform .35s;";
    item.textContent = ev.text || "";
    socialBody.insertBefore(item, socialBody.firstChild);
    requestAnimationFrame(() => { item.style.opacity = "1"; item.style.transform = "none"; });
    while (socialBody.children.length > 10) socialBody.removeChild(socialBody.lastChild);
  }
}

// ---------------------------------------------------------------------------
// 开场背景介绍页：入局前交代冤案背景，并明确告知"对话与举动决定 NPC 命运"。
// ---------------------------------------------------------------------------
function buildIntroScreen() {
  const el = document.createElement("div");
  el.id = "introscreen";
  el.style.cssText = "position:fixed;inset:0;z-index:80;display:flex;align-items:center;justify-content:center;" +
    "background:radial-gradient(circle at 50% 28%,#171320,#04050a);transition:opacity .6s;";
  if (PUBLIC_MODE) {
    el.innerHTML =
      '<div style="width:min(740px,92vw);max-height:92vh;overflow:auto;padding:34px 42px;color:#e8e2ea;">' +
        '<div style="font:800 32px/1.2 system-ui;color:#ffd98a;letter-spacing:4px;">囚 笼 悖 论</div>' +
        '<div style="font:600 12px system-ui;color:#8a93a7;margin:6px 0 24px;letter-spacing:3px;">WARDEN NIGHT SHIFT · 典狱长夜班</div>' +
        '<div style="font:500 15px/1.95 system-ui;color:#ccd2de;">' +
          "今夜你是这座监狱的<b style=\"color:#ffe0a0;\">典狱长</b>。<br>" +
          "天亮前，别让人死、别让人跑、别让走廊炸锅。<br><br>" +
          "犯人各怀心思：藏器、结仇、找墙缝。你看不见他们下一步要干什么——" +
          "只能靠巡视、审讯、搜身，把火按住。" +
        "</div>" +
        '<div style="margin-top:18px;padding:15px 17px;border-radius:10px;background:rgba(44,30,54,.55);' +
          'border-left:3px solid #e0a840;font:500 14px/1.9 system-ui;color:#dccbe8;">' +
          "<b style=\"color:#ffe08a;\">怎么玩</b><br>" +
          "· 风险弹窗出现时，走过去当面处置（问话 / 搜身 / 关禁闭）<br>" +
          "· 权力点有限，乱关禁闭会伤正当性<br>" +
          "· 天亮按三项收场：秩序 · 伤亡 · 正当性" +
        "</div>" +
        '<div style="margin-top:16px;font:500 13px/1.85 system-ui;color:#9aa4bb;">' +
          '<b style="color:#ffd23f;">1</b> 俯瞰 · <b style="color:#ffd23f;">4/5</b> 第一/第三人称走查 · ' +
          '<b style="color:#ffd23f;">点击犯人</b> 看档案 · <b style="color:#ffd23f;">E</b> 审讯 · ' +
          '<b style="color:#ffd23f;">空格</b> 暂停</div>' +
        '<button id="introgo" style="margin-top:28px;padding:12px 32px;border-radius:10px;border:1px solid #ffd98a;' +
          'cursor:pointer;font:800 16px system-ui;color:#161018;letter-spacing:2px;' +
          'background:linear-gradient(180deg,#ffe08a,#e0a840);">接过钥匙串 →</button>' +
      "</div>";
  } else {
    el.innerHTML =
      '<div style="width:min(780px,92vw);max-height:92vh;overflow:auto;padding:34px 42px;color:#e8e2ea;">' +
        '<div style="font:800 32px/1.2 system-ui;color:#ffd98a;letter-spacing:4px;">囚 笼 悖 论</div>' +
        '<div style="font:600 12px system-ui;color:#8a93a7;margin:6px 0 24px;letter-spacing:3px;">' +
          "A PRISONER'S PARADOX · 研究面板 · ?research=1</div>" +
        '<div style="font:500 15px/1.95 system-ui;color:#ccd2de;">' +
          "这是一座州立监狱，也是一座活着的沙盘。<br>" +
          "研究模式保留两脑对比、事件注入与一键终局。" +
        "</div>" +
        '<button id="introgo" style="margin-top:28px;padding:12px 32px;border-radius:10px;border:1px solid #ffd98a;' +
          'cursor:pointer;font:800 16px system-ui;color:#161018;letter-spacing:2px;' +
          'background:linear-gradient(180deg,#ffe08a,#e0a840);">进入研究面板 →</button>' +
      "</div>";
  }
  document.body.appendChild(el);
  el.querySelector("#introgo").onclick = () => {
    introDismissed = true;
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 620);
    if (!PUBLIC_MODE) checkPhaseIntro();
  };
}

// ---------------------------------------------------------------------------
// 启动：先加载角色，再连服务器
// ---------------------------------------------------------------------------
statusEl.textContent = "加载角色模型…";
buildIntroScreen();
renderCamBar();
initControls();
initPlayerChat();
initInvestigation();
tick();
loadCharacter().then(() => {
  const loader = document.getElementById("loader");
  if (loader) { loader.style.opacity = "0"; setTimeout(() => loader.remove(), 700); }
  buildStaticDecor();                              // 牢房/火盆等固定景物（要等环境模块就位）
  if (!OBSERVER_MODE || WARDEN_MODE) spawnPlayer(); // 典狱长模式：也生成化身（可切第一/第三人称）
  // 上帝之眼：一开局就是高空俯瞰整座监狱。斜 45° 高角度 + 窄视场＝近正交的沙盘感
  //（参考图那种等距切面图；透视畸变一大就没有"模型沙盘"的味道了，见 GOD_FOV）
  // 机位随场地一起收（围墙 30→19.8），否则整座监狱只占屏幕中间一小块
  camera.position.set(52, 56, 61);
  controls.target.set(0, 1, 0);
  controls.maxDistance = 130;
  controls.update();
  camAim.copy(controls.target);
  setCamMode("god");                               // 观察者主视角：俯瞰全狱
  connect();
  updateObjective();                               // 常驻进度 HUD
  setTimeout(() => showToast(
    PUBLIC_MODE
      ? "夜班开始。盯着风险弹窗，走过去处置。权力点有限，天亮按秩序/伤亡/正当性收场。"
      : ((WARDEN_MODE
        ? "典狱长已上任。上帝视角俯瞰全狱：点任意犯人看他此刻的决策与理由，并在卡片里传唤审讯、" +
          "没收/栽赃凶器、挑拨或调解仇怨。按 4/5 或 V 切第一/第三人称亲自下场走查（走近按 E）。"
        : "上帝视角已就位。俯瞰这座监狱里的众生：点任意犯人查看他此刻的决策与理由，") +
      "空格暂停世界，⚡ 随时掀翻监狱、让 DM 为这一局写下结局。")), 1200);
}).catch((err) => {
  statusEl.textContent = "角色模型加载失败：" + err;
  const loader = document.getElementById("loader");
  if (loader) loader.querySelector("div:last-child").textContent = "角色模型加载失败，检查 assets/RobotExpressive.glb";
});
