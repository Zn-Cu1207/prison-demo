"""阶段4-b：价值模型推理服务（标准库 HTTP，常驻显存）。

契约（与 backend/app/model_client.py 完全对齐）：
    POST /score        {"context": str, "candidates":[{goal,desc,source,priority,bForcePlan},...]}
                       → {"scores":[v,...], "server_ms": float}   顺序与 candidates 对齐
    POST /score_multi  {"groups":[{context,candidates},...]}
                       → {"scores":[[v,...],...], "server_ms": float}   跨 NPC 合批
    GET  /health       → {"status":"ok", "model": <path>}

输入渲染只走 common.build_input（train-serve 命根子）。单标量 v 即 Bradley-Terry logit，
后端用 M_value=exp(v/2) 换算成打分。

用法：
    python value_server.py --model merged/value-1p5b --host 0.0.0.0 --port 8009
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import build_input

_LOCK = threading.Lock()
_TOK = None
_MODEL = None
_DEVICE = "cpu"
_MAXLEN = 256
_MODEL_PATH = ""


def load_model(path: str, maxlen: int):
    global _TOK, _MODEL, _DEVICE, _MAXLEN, _MODEL_PATH
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _MAXLEN = maxlen
    _MODEL_PATH = path
    _TOK = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if _TOK.pad_token is None:
        _TOK.pad_token = _TOK.eos_token
    _TOK.padding_side = "right"
    dtype = torch.float16 if _DEVICE == "cuda" else torch.float32
    _MODEL = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=1, torch_dtype=dtype, trust_remote_code=True,
    )
    _MODEL.config.pad_token_id = _TOK.pad_token_id
    _MODEL.to(_DEVICE).eval()
    print(f"模型已加载：{path}  device={_DEVICE}  dtype={dtype}")


_MAX_BATCH = 32  # 单次前向最多这么多条，防一口气来太多候选把显存打爆


@torch.inference_mode()
def _score_texts(texts: list[str]) -> list[float]:
    """给一批文本打分。超过 _MAX_BATCH 自动分块，避免大批量 OOM。"""
    if not texts:
        return []
    out: list[float] = []
    with _LOCK:
        for i in range(0, len(texts), _MAX_BATCH):
            chunk = texts[i:i + _MAX_BATCH]
            enc = _TOK(chunk, padding=True, truncation=True, max_length=_MAXLEN,
                       return_tensors="pt")
            enc = {k: v.to(_DEVICE) for k, v in enc.items()}
            logits = _MODEL(**enc).logits.squeeze(-1)
            # squeeze 后单条会退化成 0 维，统一成列表
            vals = logits.float().cpu().tolist()
            out.extend(vals if isinstance(vals, list) else [vals])
    return out


def score_group(context: str, candidates: list[dict]) -> list[float]:
    texts = [build_input(context, c) for c in candidates]
    return _score_texts(texts)


def score_multi(groups: list[dict]) -> list[list[float]]:
    # 拉平成一个大 batch 一次前向，再按组切回
    flat_texts, spans = [], []
    for g in groups:
        cands = g.get("candidates", [])
        start = len(flat_texts)
        flat_texts.extend(build_input(g.get("context", ""), c) for c in cands)
        spans.append((start, len(flat_texts)))
    flat = _score_texts(flat_texts)
    return [flat[s:e] for s, e in spans]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # 静默默认访问日志
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": _MODEL_PATH, "device": _DEVICE})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            t0 = time.perf_counter()
            if self.path == "/score":
                scores = score_group(payload.get("context", ""), payload.get("candidates", []))
                self._send(200, {"scores": scores, "server_ms": (time.perf_counter() - t0) * 1000})
            elif self.path == "/score_multi":
                rows = score_multi(payload.get("groups", []))
                self._send(200, {"scores": rows, "server_ms": (time.perf_counter() - t0) * 1000})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="merged/value-1p5b")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--maxlen", type=int, default=256)
    args = ap.parse_args()
    load_model(args.model, args.maxlen)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"value_server 监听 http://{args.host}:{args.port}  (/score /score_multi /health)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
