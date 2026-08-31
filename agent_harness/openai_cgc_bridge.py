#!/usr/bin/env python3
"""OpenAI -> CGC bridge: /v1/chat/completions -> llama-simple.exe"""
import json, os, subprocess, sys, time, uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["PATH"] = "C:/msys64/mingw64/bin;" + os.environ.get("PATH", "")

LLAMA = os.environ.get("LLAMA_SIMPLE",
    "D:/alex/flashkv0516/cgcengine_full/src/llama.cpp/build/bin/llama-simple.exe")
MODEL = os.environ.get("MODEL",
    "D:/alex/flashkv0516/models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf")
PORT = int(os.environ.get("BRIDGE_PORT", "1234"))

def build_prompt(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(f"{role.capitalize()}: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/v1/models", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/v1/models":
                self.wfile.write(json.dumps({"object":"list","data":[{"id":os.path.basename(MODEL),"object":"model","owned_by":"cgc"}]}).encode())
            else:
                self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self.handle_chat()
        else:
            self.send_response(404); self.end_headers()

    def handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        max_tokens = min(body.get("max_tokens", 256), 512)
        temperature = body.get("temperature", 0)
        prompt = build_prompt(messages)

        cmd = [LLAMA, "-m", MODEL, "-p", prompt, "-n", str(max_tokens),
               "--temp", str(temperature), "-c", "2048", "-ngl", "4", "--log-disable"]
        try:
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=120)
            wall = time.time() - t0
            output = (proc.stdout or "").strip()
            # Remove prompt echo
            if output.startswith(prompt[:20]):
                output = output[len(prompt):].strip()
            # Remove thinking tags
            if "<think>" in output:
                parts = output.split("</think>", 1)
                output = parts[-1].strip() if len(parts) > 1 else output
        except Exception as e:
            output = f"<error: {e}>"
            wall = 0

        resp = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion", "created": int(time.time()),
                "model": os.path.basename(MODEL),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(output.split()), "total_tokens": len(prompt.split()) + len(output.split())},
                "cgc_summary": {"wall_s": round(wall, 2)}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp, ensure_ascii=False).encode())

    def log_message(self, fmt, *args):
        print(f"[bridge] {args[0]}", flush=True)

if __name__ == "__main__":
    print(f"Bridge on port {PORT}, model={os.path.basename(MODEL)}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
