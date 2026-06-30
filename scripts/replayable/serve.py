#!/usr/bin/env python
"""Threaded static server for the demo.

`python -m http.server` is single-threaded and serializes requests, so streaming many
camera frames is slow. This serves the web/ bundle with a threading server (parallel
requests) + sane cache headers, which makes frame loading much faster.

    python serve.py [port]      # default 8000   (run from scripts/replayable/)
"""
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        path = self.path.split("?", 1)[0]
        # frames/assets are content-addressed by path -> let the browser cache them hard
        if any(path.endswith(e) for e in (".jpg", ".png", ".glb")):
            self.send_header("Cache-Control", "public, max-age=86400")
        # dynamic bundle metadata + code: always revalidate so a rebuilt bundle
        # (e.g. new scenes in scenes.json) shows up without a hard refresh
        elif any(path.endswith(e) for e in (".json", ".html", ".js")) or path in ("/", ""):
            self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def log_message(self, *args):
        pass  # quiet


def main():
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), partial(Handler, directory=WEB))
    print(f"serving {WEB}\n  http://localhost:{PORT}   (threaded)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
