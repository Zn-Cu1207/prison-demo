"""测网关对 claude-opus-4-8 的可持续速率：串行发 N 次，记录每次耗时/状态码。

模型服务有突发限流（429 User burst limit exceeded），标注前先用这个探出稳定节奏，
再据此设 annotate.py 的 --rps / --workers，避免整轮标注被 429 打成筛子。

    python probe_rate.py --n 10 --gap 3
"""
from __future__ import annotations

import argparse
import time

import httpx

from hy_client import load_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--gap", type=float, default=3.0, help="每次请求间隔秒数")
    args = ap.parse_args()

    env = load_env(".env")
    key = env["HY_API_KEY"]
    url = env["HY_BASE_URL"].rstrip("/") + "/chat/completions"
    model = env["HY_MODEL"]
    verify = str(env.get("HY_VERIFY_SSL", "1")).lower() not in ("0", "false", "no")

    ok = 0
    codes: dict[int, int] = {}
    with httpx.Client(timeout=90, verify=verify) as c:
        for i in range(args.n):
            t0 = time.perf_counter()
            try:
                r = c.post(url, headers={"Authorization": f"Bearer {key}",
                                         "Content-Type": "application/json"},
                           json={"model": model,
                                 "messages": [{"role": "user", "content": "只回一个字:好"}],
                                 "max_tokens": 8})
                dt = time.perf_counter() - t0
                codes[r.status_code] = codes.get(r.status_code, 0) + 1
                if r.status_code == 200:
                    ok += 1
                    print(f"  #{i + 1} 200 {dt:.2f}s")
                else:
                    print(f"  #{i + 1} {r.status_code} {dt:.2f}s  {r.text[:110]}")
            except Exception as e:  # noqa: BLE001
                print(f"  #{i + 1} ERR {type(e).__name__} {str(e)[:80]}")
            if i < args.n - 1:
                time.sleep(args.gap)

    print(f"\n间隔 {args.gap}s：成功 {ok}/{args.n}  状态码分布 {codes}")
    print(f"→ 可持续速率约 {ok / (args.n * args.gap):.2f} 次/秒" if ok else "→ 该间隔仍被限流，加大 --gap")


if __name__ == "__main__":
    main()
