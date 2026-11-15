"""OpenAI-compatible local gateway for Kilo Auto Free."""
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.getenv("KILO_GATEWAY_URL", "https://ai-gateway.kilo.ai/api/v1/chat/completions")
MODEL = "kilo-auto/free"


class UpstreamError(Exception):
    def __init__(self, status, body, headers):
        super().__init__(f"Kilo upstream returned HTTP {status}")
        self.status = status
        self.body = body
        self.headers = headers


def forward_completion(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ValueError("request must contain a messages array")
    forwarded = {key: value for key, value in payload.items()
                 if key not in {"model", "api_key", "timeout", "max_retries"}}
    forwarded["model"] = MODEL
    forwarded["stream"] = False
    request = urllib.request.Request(
        UPSTREAM, data=json.dumps(forwarded).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("KILO_TIMEOUT_SECONDS", "240"))) as response:
            body = response.read()
            if response.status < 200 or response.status >= 300:
                raise UpstreamError(response.status, body, response.headers)
    except urllib.error.HTTPError as exc:
        raise UpstreamError(exc.code, exc.read(), exc.headers) from exc
    return json.loads(body)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self.send_json(404, {"error": "Not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 2_000_000:
                return self.send_json(413, {"error": {"message": "Invalid body size"}})
            payload = json.loads(self.rfile.read(size))
            self.send_json(200, forward_completion(payload))
        except UpstreamError as exc:
            try:
                error = json.loads(exc.body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                error = {"error": {"message": exc.body.decode("utf-8", "replace")[:1000]}}
            print(f"Kilo upstream HTTP {exc.status}: {json.dumps(error, ensure_ascii=False)[:1000]}", flush=True)
            self.send_json(exc.status, error, retry_after=exc.headers.get("Retry-After"))
        except Exception as exc:
            print(f"Gateway error: {type(exc).__name__}: {exc}", flush=True)
            self.send_json(502, {"error": {"message": f"{type(exc).__name__}: {str(exc)[:500]}",
                                              "type": "gateway_error"}})

    def send_json(self, status, data, retry_after=None):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if retry_after:
            self.send_header("Retry-After", retry_after)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    bind = os.getenv("OPENKILO_BIND", "127.0.0.1")
    port = int(os.getenv("OPENKILO_PORT", "18765"))
    ThreadingHTTPServer((bind, port), Handler).serve_forever()
