"""标注用大模型客户端（任意 OpenAI 兼容端点：a user-configured provider…）。

配置全走 .env（已 .gitignore，key 不入库）：
    HY_API_KEY      = sk-...
    HY_BASE_URL     = https://api.example.com/v1   # 末尾自动补 /chat/completions
    HY_MODEL        = claude-opus-4-8
    HY_VERIFY_SSL   = 1    # verify TLS certificates

只依赖 httpx。带指数退避重试 + JSON 解析容错（大模型偶尔在 JSON 外面裹一层文字）。
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from pathlib import Path

import httpx

_ENV_LOADED = False


class _Throttle:
    """全局请求节流器。

    模型服务对 claude-opus-4-8 有**突发限流**（429 User burst limit exceeded），
    靠多线程堆并发只会把失败率推高。这里用一个进程级最小间隔把请求摊平，
    所有线程共享同一个节奏（等待时持锁 = 天然排队）。
    """

    def __init__(self) -> None:
        self.min_interval = 0.0
        self._last = 0.0
        self._lock = threading.Lock()

    def set_rps(self, rps: float) -> None:
        self.min_interval = 0.0 if rps <= 0 else 1.0 / rps

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            target = self._last + self.min_interval
            if target > now:
                time.sleep(target - now)
                now = time.monotonic()
            self._last = now


THROTTLE = _Throttle()


def load_env(path: str | Path = ".env") -> dict[str, str]:
    """极简 .env 读取（不引第三方 dotenv）。也回落到进程环境变量。"""
    global _ENV_LOADED
    env: dict[str, str] = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    # 环境变量优先级更高（CI/临时覆盖）
    for k in ("HY_API_KEY", "HY_BASE_URL", "HY_MODEL", "HY_VERIFY_SSL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    _ENV_LOADED = True
    return env


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


class HYClient:
    def __init__(self, env_path: str | Path = ".env", timeout_s: float = 60.0,
                 model: str | None = None):
        env = load_env(env_path)
        self.key = env.get("HY_API_KEY", "")
        base = env.get("HY_BASE_URL", "https://api.example.com/v1").rstrip("/")
        self.url = base + "/chat/completions"
        self.model = model or env.get("HY_MODEL", "claude-opus-4-8")
        if not self.key:
            raise RuntimeError("缺少 HY_API_KEY，请在 value-model/.env 里配置")
        # Configure the provider endpoint and optional capabilities for your own environment.
        self.verify = str(env.get("HY_VERIFY_SSL", "1")).lower() not in ("0", "false", "no")
        self._client = httpx.Client(timeout=timeout_s, verify=self.verify)

    def close(self):
        self._client.close()

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 256,
        json_only: bool = False,
        retries: int = 6,
    ) -> str:
        """一次对话，返回 assistant 文本。json_only 时尝试请求 JSON 模式。

        temperature 默认**不发**：claude-opus-4-8 等新模型在网关侧已弃用该参数，
        发了会 400。需要时显式传值，遇到弃用报错会自动剥掉重试。
        """
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if json_only:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        last_err = None
        for attempt in range(retries):
            try:
                THROTTLE.wait()
                r = self._client.post(self.url, headers=headers, json=body)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                # 网关对某些参数不买账 → 剥掉再试（不消耗"真失败"配额）
                if r.status_code == 400:
                    low = r.text.lower()
                    if "temperature" in low and "temperature" in body:
                        body.pop("temperature", None)
                        continue
                    if "response_format" in low or "response_format" in body:
                        body.pop("response_format", None)
                        continue
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                # 429 突发限流：退避要够长，且带抖动避免所有线程同拍重试
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After", "")
                    delay = float(ra) if ra.replace(".", "", 1).isdigit() else min(60.0, 4.0 * (2 ** attempt))
                    time.sleep(delay + random.uniform(0.5, 2.0))
                    continue
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 8) + random.uniform(0, 1.0))
        raise RuntimeError(f"调用失败（{retries} 次）：{last_err}")

    def chat_json(self, system: str, user: str, **kw) -> dict | None:
        txt = self.chat(system, user, json_only=True, **kw)
        return _extract_json(txt)

    def ping(self) -> tuple[bool, str]:
        try:
            out = self.chat("你是测试助手。", "只回一个字：好", max_tokens=8, retries=1)
            return True, out.strip()
        except Exception as e:  # noqa: BLE001
            return False, str(e)[:200]


if __name__ == "__main__":
    c = HYClient()
    ok, msg = c.ping()
    print("base_url:", c.url)
    print("model   :", c.model)
    print("verify  :", c.verify)
    print("ping    :", "OK" if ok else "FAIL", "->", msg)
