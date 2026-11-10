"""阶段0-c：hy3(混元) 标注成对偏好 + RA 护栏 → preferences.jsonl

流程（Motif 第①步）：
  1) 大模型读「情境 + 两个候选」，判"NPC 此刻更该做哪个"，给 choice/confidence/reason；
  2) **护栏覆盖**：跨优先级档位（危机/指令/需求/worktask/日常）时以硬规则为准，
     纠正大模型的系统性偏差（RA 报告里护栏纠正率 ~9.6%）；同档位听大模型细判。

输出每条：{situation_id, pair_id, context, chosen, rejected, confidence, label_by, reason}
label_by ∈ hy / rule(护栏覆盖) 。支持断点续标（重跑跳过已标 pair_id）。

用法：
    python annotate.py --in data/pairs.jsonl --out data/preferences.jsonl --workers 4
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from guardrails import guardrail
from hy_client import THROTTLE, HYClient

SYSTEM = (
    "你是监狱模拟游戏里的 NPC 行为仲裁官。给你一个 NPC 的当前情境和两个候选行动 A、B，"
    "判断此刻这个 NPC 更应该做哪一个。\n"
    "严格遵循优先级阶梯（高的压过低的）：\n"
    "  ① 濒死级生存危机（重伤/极度饥渴/被仇人或狱犬逼到眼前）——最高\n"
    "  ② 玩家指令（强制执行的命令）\n"
    "  ③ 应急威胁与突发事件（打架、警报、搜牢、被狱警狱犬盯防时收敛）\n"
    "  ④ 普通生存需求（饿了/困了/无聊/想社交，未到濒死）\n"
    "  ⑤ 待办工作（苦役、帮厨）\n"
    "  ⑥ 日常休闲（消遣、闲逛、歇着）——最低\n"
    "注意：狱警/狱犬在场或高监视度时，见不得光的违禁行为（赌钱/换货/挖墙/纹身）应被压制，"
    "老实收敛更该做。同一档位内，按情境数值和常识判断谁更急。\n"
    '只输出 JSON：{"choice":"A"或"B","confidence":0到1的小数,"reason":"一句中文理由"}'
)

_local = threading.local()
# 失败原因收集（诊断限流/超时/解析失败到底是哪一类）
_ERRORS: list[str] = []
_ERR_LOCK = threading.Lock()


def _note_error(msg: str) -> None:
    with _ERR_LOCK:
        _ERRORS.append(msg[:160])


def _client(env_path: str) -> HYClient:
    c = getattr(_local, "client", None)
    if c is None:
        c = HYClient(env_path)
        _local.client = c
    return c


def _cand_line(tag: str, c: dict) -> str:
    src_cn = {"need": "自身需求", "threat": "威胁应对", "command": "玩家指令",
              "worktask": "待办工作", "event": "突发事件", "daily": "日常行为"}.get(c["source"], c["source"])
    return f"{tag}. {c['goal']}（{c.get('desc', c['goal'])}；来源：{src_cn}）"


def annotate_one(pair: dict, env_path: str, check_override: bool = False) -> dict | None:
    a, b = pair["cand_a"], pair["cand_b"]
    ctx = pair["ctx"]

    # 先看护栏能不能直接裁（跨档位）。能裁且不需核对时，直接出标签、**不调 API**（省钱省时）。
    g = guardrail(ctx, a, b)
    if g is not None and not check_override:
        win_idx, reason = g
        chosen, rejected = (a, b) if win_idx == 0 else (b, a)
        return _pref(pair, chosen, rejected, 0.95, "rule", reason)

    user = (
        f"情境：{pair['context']}\n\n"
        f"{_cand_line('A', a)}\n{_cand_line('B', b)}\n\n"
        "这个 NPC 此刻更应该做 A 还是 B？"
    )
    try:
        js = _client(env_path).chat_json(SYSTEM, user, max_tokens=200)
        if js is None:
            _note_error("解析失败：返回不是可解析 JSON")
    except Exception as e:  # noqa: BLE001
        _note_error(f"{type(e).__name__}: {e}")
        js = None

    hy_choice = None
    hy_conf = 0.7
    hy_reason = ""
    if js and js.get("choice") in ("A", "B"):
        hy_choice = js["choice"]
        try:
            hy_conf = max(0.5, min(1.0, float(js.get("confidence", 0.7))))
        except Exception:  # noqa: BLE001
            hy_conf = 0.7
        hy_reason = str(js.get("reason", ""))[:80]

    if g is not None:
        # check_override 模式：护栏仍然说了算，但记录大模型是否一致（统计纠正率）
        win_idx, reason = g
        chosen, rejected = (a, b) if win_idx == 0 else (b, a)
        hy_agree = None
        if hy_choice is not None:
            hy_agree = ((0 if hy_choice == "A" else 1) == win_idx)
        return {**_pref(pair, chosen, rejected, 0.95, "rule", reason), "hy_agree": hy_agree}

    # 同档位：交给大模型
    if hy_choice is None:
        return None  # 标不出来，丢弃这条
    chosen, rejected = (a, b) if hy_choice == "A" else (b, a)
    return _pref(pair, chosen, rejected, hy_conf, "hy", hy_reason)


def _pref(pair, chosen, rejected, conf, label_by, reason):
    return {
        "situation_id": pair["situation_id"],
        "pair_id": pair["pair_id"],
        "context": pair["context"],
        "chosen": chosen,
        "rejected": rejected,
        "confidence": conf,
        "label_by": label_by,
        "reason": reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pairs.jsonl")
    ap.add_argument("--out", default="data/preferences.jsonl")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rps", type=float, default=1.0,
                    help="全局请求速率上限(次/秒)。网关对 opus 有突发限流，1.0 左右最稳")
    ap.add_argument("--limit", type=int, default=0, help="只标前 N 条（0=全部），先小跑验证")
    ap.add_argument("--check-override", action="store_true",
                    help="护栏能裁的也调大模型核对，统计纠正率（费 API，默认关）")
    args = ap.parse_args()

    pairs = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if args.limit:
        pairs = pairs[: args.limit]

    # 断点续标
    done: set[str] = set()
    outp = Path(args.out)
    if outp.exists():
        for line in outp.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["pair_id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [p for p in pairs if p["pair_id"] not in done]
    THROTTLE.set_rps(args.rps)
    need_api = sum(1 for p in todo if guardrail(p["ctx"], p["cand_a"], p["cand_b"]) is None)
    print(f"待标 {len(todo)} 条（已标 {len(done)}），其中需调大模型 {need_api} 条；"
          f"workers={args.workers} rps={args.rps}")

    lock = threading.Lock()
    stats = {"hy": 0, "rule": 0, "skip": 0, "rule_override": 0}
    with outp.open("a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(annotate_one, p, args.env, args.check_override): p for p in todo}
        n = 0
        for fut in as_completed(futs):
            rec = fut.result()
            n += 1
            if rec is None:
                stats["skip"] += 1
            else:
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                stats[rec["label_by"]] += 1
                if rec["label_by"] == "rule" and rec.get("hy_agree") is False:
                    stats["rule_override"] += 1
            if n % 50 == 0:
                print(f"  进度 {n}/{len(todo)}  {stats}")
    total_rule = stats["rule"]
    ov = stats["rule_override"]
    print("完成：", stats)
    if _ERRORS:
        from collections import Counter
        print(f"失败 {len(_ERRORS)} 条，原因 TOP5：")
        for msg, cnt in Counter(_ERRORS).most_common(5):
            print(f"  ×{cnt}  {msg}")
    if total_rule:
        print(f"护栏纠正率（大模型与护栏相左）：{ov}/{total_rule} = {ov / total_rule:.1%}")


if __name__ == "__main__":
    main()
