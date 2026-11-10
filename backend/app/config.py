"""运行时配置。所有可调项集中在此，环境变量可覆盖。

铁律对应：后端配一个 MODEL_URL 可切（本地 3060 / T4 备用 / 本地 mock）。
"""
from __future__ import annotations

import os


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# 价值模型服务地址。默认指向本地 mock_value_server；接真模型时改这里或设环境变量。
#   本地真模型:  http://127.0.0.1:8009/score
#   T4(SSH隧道): http://127.0.0.1:8009/score
MODEL_URL: str = _env("MODEL_URL", "http://127.0.0.1:8009/score")

# 单次模型请求超时（秒）。影子/接管都对延迟不敏感，给足余量。
#
# 为什么给到 20s：打分是单飞的（_score_inflight 在途就跳过后续拍），所以超时短
# 并不能省下等待时间——一批打分要跑多久就是多久，超时只决定"这段时间白等还是
# 拿到结果"。显卡被别的程序挤满时一批要 10s 以上，5s 的超时会让每一批都失败、
# 分缓存永远填不上，NPC 于是集体呆立不动（历史上反复出现的那个现象）。
# 放宽到 20s 后同样的情况只是决策变慢、世界照旧在动。
MODEL_TIMEOUT_S: float = float(_env("MODEL_TIMEOUT_S", "20.0"))

# Simulation timestep in seconds.
TICK_INTERVAL_S: float = float(_env("TICK_INTERVAL_S", "0.5"))

# 后端监听地址/端口
HOST: str = _env("HOST", "127.0.0.1")
PORT: int = int(_env("PORT", "8000"))

# ---------------------------------------------------------------------------
# 决策"大脑"选择：
#   value    = 微调小价值模型(默认)
#   local    = 本地 Qwen(Ollama, OpenAI 兼容, 免 key)
#   deepseek = DeepSeek 云端 LLM(需 key)
# ---------------------------------------------------------------------------
BRAIN: str = _env("BRAIN", "value").lower()

# DeepSeek（OpenAI 兼容 Chat API）。需设置环境变量 DEEPSEEK_API_KEY 才启用。
DEEPSEEK_API_KEY: str = _env("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL: str = _env("DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL: str = _env("DEEPSEEK_MODEL", "deepseek-chat")

# 本地 LLM（Ollama 的 OpenAI 兼容端点；免 key）。
# 默认指向对话 SFT 微调模型 prison-convo:q4（Opus 蒸馏群像对白，q4_K_M 量化）；
# 想回退裸基座对比时，设环境变量 LOCAL_LLM_MODEL=qwen2.5:7b 即可。
LOCAL_LLM_URL: str = _env("LOCAL_LLM_URL", "http://127.0.0.1:11434/v1/chat/completions")
LOCAL_LLM_MODEL: str = _env("LOCAL_LLM_MODEL", "prison-convo:q4")

# LLM 请求超时（秒）。本地 7B 批量生成较慢，给足余量。
LLM_TIMEOUT_S: float = float(_env("LLM_TIMEOUT_S", "120.0"))

# ---------------------------------------------------------------------------
# 对话"大脑"后端（独立于决策大脑）：
#   local = 本地 Ollama（默认，免 key、免网络）
#   opus  = 用户配置的 OpenAI-compatible chat model（云端，试"上限效果"用）
# 只影响 NPC↔NPC 对话/开场白/结局的发声，不碰决策（决策始终价值模型）。
# 也不碰"两脑对比"里的裸本地大模型——那一路始终留在本地，卖点不被污染。
# ---------------------------------------------------------------------------
CHAT_BACKEND: str = _env("CHAT_BACKEND", "local").lower()


def _value_model_env() -> dict[str, str]:
    """复用 value-model/.env 里已配好的网关凭据（HY_*），免得再抄一份 key。
    后端进程环境变量优先级更高（可临时覆盖）。"""
    import os
    from pathlib import Path

    env: dict[str, str] = {}
    p = Path(__file__).resolve().parents[2] / "value-model" / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    for k in ("HY_API_KEY", "HY_BASE_URL", "HY_MODEL", "HY_VERIFY_SSL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


_vm = _value_model_env()
# Opus 网关（OpenAI 兼容：base_url 末尾补 /chat/completions；Bearer 鉴权）
_opus_base = _env("OPUS_BASE_URL", _vm.get("HY_BASE_URL", "https://api.example.com/v1")).rstrip("/")
OPUS_URL: str = _opus_base + "/chat/completions"
# 显式钉死 claude-opus-4-8：value-model/.env 的 HY_MODEL 被设成了网关别名 "hy3"，
# 那未必路由到 Opus；要试 Opus 就直接点名。需要别的模型时用 OPUS_MODEL 环境变量覆盖。
OPUS_MODEL: str = _env("OPUS_MODEL", "your-chat-model")
OPUS_API_KEY: str = _env("OPUS_API_KEY", _vm.get("HY_API_KEY", ""))
# Verify TLS certificates by default in the public release.
OPUS_VERIFY_SSL: bool = str(_env("OPUS_VERIFY_SSL", _vm.get("HY_VERIFY_SSL", "1"))).lower() not in ("0", "false", "no")
