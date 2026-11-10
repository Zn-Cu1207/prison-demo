"""开发用静态服务器。

与 `python -m http.server` 的唯一区别：强制 no-store。
浏览器对 ES module 的磁盘缓存很激进，改完 main.js 常常刷不出新版，
调试时会误判成代码没生效，所以开发期一律禁缓存。
"""
from __future__ import annotations

import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "5173"))
ROOT = os.path.dirname(os.path.abspath(__file__))


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def send_response(self, code, message=None):  # 去掉每请求一行的噪声时间戳
        super().send_response(code, message)


def main() -> None:
    handler = partial(NoCacheHandler, directory=ROOT)
    with ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"serving {ROOT} at http://127.0.0.1:{PORT} (no-store)", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
