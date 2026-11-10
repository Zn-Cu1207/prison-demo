"""前端静态服务器（禁用缓存版）。

原来的 `python -m http.server` 不发缓存头，浏览器会把 index.html 缓存下来——
它没带 ?v= 版本号，于是普通刷新一直复用旧的 index.html，指向旧的 main.js，
改了前端也"看不到新版本"。这里给所有响应加 no-store，从此硬/软刷新都拿最新。
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = 5173


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    print(f"前端(禁缓存)运行在 http://localhost:{PORT}/")
    ThreadingHTTPServer(("0.0.0.0", PORT), NoCacheHandler).serve_forever()
