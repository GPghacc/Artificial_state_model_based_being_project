"""Model backends. The rest of the system never talks to a model directly.

Only MockBackend is functional without external services. OllamaBackend and
AnthropicBackend are thin adapters (stdlib urllib only) to be switched on later.
"""
from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from typing import Protocol


class Backend(Protocol):
    def generate(self, prompt: str, temperature: float = 0.7) -> str: ...


class MockBackend:
    """Deterministic-ish stand-in so the whole pipeline runs without a key.

    It "knows" simple arithmetic with a difficulty-dependent error rate, and
    answers anything else with a canned string. The point is not to be smart but
    to produce *variance across samples* so self-consistency cues mean something.
    """

    def __init__(self, seed: int = 0, base_error: float = 0.15):
        self.rng = random.Random(seed)
        self.base_error = base_error
        self.calls = 0

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        self.calls += 1
        m = re.search(r"(-?\d+)\s*([+\-*])\s*(-?\d+)", prompt)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            truth = {"+": a + b, "-": a - b, "*": a * b}[op]
            # bigger numbers -> harder -> more error; temperature widens the spread
            difficulty = min(1.0, (abs(a) + abs(b)) / 400.0)
            p_err = self.base_error + 0.6 * difficulty * temperature
            if self.rng.random() < p_err:
                return str(truth + self.rng.choice([-10, -1, 1, 2, 10]))
            return str(truth)
        # non-arithmetic: pretend to reason, low agreement between samples
        return self.rng.choice(["I think so.", "Probably not.", "It depends.", "Yes."])


class OllamaBackend:
    """Adapter for a local Ollama server (default http://localhost:11434).

    Uses /api/chat. The legacy /api/generate endpoint was removed in newer Ollama
    versions, so we fall back to it only if /api/chat is missing (very old servers).
    """

    def __init__(self, model: str = "llama3.1", host: str | None = None,
                 seed: int | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.seed = seed
        self.calls = 0

    def _sampling_seed(self) -> int | None:
        """A distinct but deterministic seed per call.

        Passing one fixed seed would make every sample in a cycle identical, so
        self-consistency would always read 1.0 and the cue would be meaningless.
        Deriving it from the call counter keeps samples different from each other
        while making the whole run reproducible.
        """
        if self.seed is None:
            return None
        return (self.seed * 100003 + self.calls) % (2 ** 31 - 1)

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"ollama {path} -> HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach ollama at {self.host} ({e.reason}). "
                "Is `ollama serve` running?") from None

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        self.calls += 1
        opts: dict = {"temperature": temperature}
        sseed = self._sampling_seed()
        if sseed is not None:
            opts["seed"] = sseed
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": opts,
        }
        try:
            data = self._post("/api/chat", payload)
        except RuntimeError as e:
            if "HTTP 404" not in str(e):
                raise
            # very old server: legacy endpoint
            data = self._post("/api/generate", {
                "model": self.model, "prompt": prompt, "stream": False,
                "options": opts})
            return data.get("response", "")
        msg = data.get("message") or {}
        return msg.get("content", "")


class AnthropicBackend:
    """Adapter for the Anthropic Messages API. Needs ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 512):
        self.model = model
        self.max_tokens = max_tokens
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"Content-Type": "application/json", "x-api-key": self.key,
                     "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return "".join(block.get("text", "") for block in data.get("content", []))


def make_backend(name: str = "mock", **kw) -> Backend:
    return {"mock": MockBackend, "ollama": OllamaBackend, "anthropic": AnthropicBackend}[name](**kw)
