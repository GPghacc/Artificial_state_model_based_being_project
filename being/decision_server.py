"""The decision organ: a small model that judges instead of writing.

LiquidAI d1-3B (open weights, LFM 1.0 license) reads a situation once and returns a probability for each
allowed answer to each question, in a single forward pass and without generating any text. The being uses
it for its quick judgments (how something lands, whether an act goes against a request, whether 현 is
doubting something it said). The being asks; this only answers.

Runs in its own Python environment (.venv-decision, made by decision_setup.bat) so the being itself stays
dependency-free. Serves on http://127.0.0.1:8770:
    GET  /health  -> {"ok": true, "model": ..., "device": ..., "vram_gb": ...}
    POST /decide  {"state": ..., "questions": {name: {"type": "noul"|"choice"|"score", "instructions": ...,
                   "criteria": ...}}}  -> {"answers": {...}, "usage": {...}, "ms": ...}
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "LiquidAI/d1-3B"


def load(model_id: str, device: str, compile_: bool):
    import torch
    from transformers import AutoModel

    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            device = "cuda" if free >= 7.0e9 else "cpu"
            if device == "cpu":
                print(f"[판단 기관] GPU 여유가 {free / 1e9:.1f}GB뿐이라 CPU로 돈다 (느림)")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    t0 = time.time()
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype).to(device)
    if compile_ and device == "cuda":
        try:
            model.compile(mode="reduce-overhead")
        except Exception as e:                      # compiling isn't always available on Windows
            print(f"[판단 기관] compile 생략: {e}")
    print(f"[판단 기관] {model_id} 를 {device}에 올림 ({time.time() - t0:.0f}초)")
    return model, device


def flash_works() -> bool:
    """Whether this PyTorch build has the flash attention kernels (Windows builds usually don't)."""
    import torch
    try:
        q = torch.zeros(1, 1, 16, 64, device="cuda", dtype=torch.bfloat16)
        torch.ops.aten._scaled_dot_product_flash_attention(q, q, q, 0.0, False, False)
        return True
    except Exception:
        return False


def avoid_missing_flash(device: str) -> str:
    """d1 answers several questions about one state as a tree (the state once, each question a branch), and on a
    recent GPU in bf16 its tree attention calls the flash kernels directly. Without them every multi-question
    request fails, so its gate (`_fused`) is pointed at the explicit path the model already has for other GPUs.
    Returns how the tree attention will run: flash / explicit / split (couldn't find the gate: questions go
    one at a time) / cpu."""
    if device != "cuda":
        return "cpu"
    if flash_works():
        return "flash"
    gates = [m for name, m in list(sys.modules.items())
             if name.startswith("transformers_modules") and callable(getattr(m, "_fused", None))]
    for m in gates:
        m._fused = lambda q: False
    return "explicit" if gates else "split"


def answer(model, state, questions: dict, mode: str) -> dict:
    """One judgment. With mode "split", each question is asked on its own (a plain pass, no tree)."""
    if mode != "split" or len(questions) == 1:
        return model.system_one(state, questions)
    res = {"answers": {}, "usage": {}}
    for name, q in questions.items():
        one = model.system_one(state, {name: q})
        res["answers"].update(one.get("answers", {}))
        for k, v in (one.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                res["usage"][k] = res["usage"].get(k, 0) + v
    return res


def make_handler(model, device: str, model_id: str, tree: str = ""):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                vram = None
                if device == "cuda":
                    try:
                        import torch
                        vram = round(torch.cuda.memory_allocated() / 1e9, 2)
                    except Exception:
                        pass
                return self._json({"ok": True, "model": model_id, "device": device, "vram_gb": vram, "tree": tree})
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self.path.startswith("/decide"):
                return self._json({"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                state, questions = req.get("state"), req["questions"]
            except (json.JSONDecodeError, KeyError, TypeError):
                return self._json({"error": "state와 questions가 필요해"}, 400)
            try:
                with lock:                          # one judgment at a time on the GPU
                    t0 = time.time()
                    res = answer(model, state, questions, tree)
                    res["ms"] = round((time.time() - t0) * 1000, 1)
                self._json(res)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--compile", action="store_true", help="CUDA graph로 한 질문을 더 빠르게 (윈도우에선 안 될 수도 있음)")
    a = p.parse_args()
    model, device = load(a.model, a.device, a.compile)
    tree = avoid_missing_flash(device)
    if tree in ("explicit", "split"):
        print(f"[판단 기관] 이 PyTorch엔 flash attention이 없어서 여러 질문을 "
              + ("다른 계산 경로로" if tree == "explicit" else "하나씩 나눠서") + " 처리한다")
    warm = {"q": {"type": "noul", "instructions": "인사인가?"},
            "s": {"type": "score", "instructions": "얼마나 반가운가?", "criteria": ["전혀", "조금", "보통", "꽤", "매우"]}}
    try:                                            # the first call with a new shape pays for kernel selection
        answer(model, "안녕", {"q": warm["q"]}, tree)
        answer(model, "안녕", warm, tree)           # several questions at once: the tree path
    except Exception as e:
        if tree == "explicit":                      # the gate didn't take: fall back to one question at a time
            tree = "split"
            print(f"[판단 기관] 여러 질문 한꺼번에가 안 돼서 하나씩 나눠 처리한다 ({e})")
            try:
                answer(model, "안녕", warm, tree)
            except Exception as e2:
                print(f"[판단 기관] 예열 실패: {e2}")
        else:
            print(f"[판단 기관] 예열 실패: {e}")
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(model, device, a.model, tree))
    print(f"[판단 기관] http://127.0.0.1:{a.port} 에서 대기 중. 끄려면 Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
