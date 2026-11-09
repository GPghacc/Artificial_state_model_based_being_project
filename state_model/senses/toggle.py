"""현의 존재 신호: presence.txt에 'watching' 또는 'sleep'을 적어 토글한다.
들었는지 확인할 수 있게 바뀔 때마다 콘솔에 표시한다."""
import os
from .base import Sense


class PresenceSense(Sense):
    name, zone = "presence", "audio"

    def __init__(self, path, every=15):
        self.path = path
        self.every = every
        self.count = 0
        self.state = "sleep"
        self._read()

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                s = f.read().strip().lower()
        except FileNotFoundError:
            s = "sleep"
        s = "watching" if s.startswith("watch") else "sleep"
        if s != self.state:
            print(f"[presence] {self.state} → {s} (감지함)")
        self.state = s

    def level(self, t, ctx):
        self.count += 1
        if self.count % self.every == 0:
            self._read()
        return 0.5 if self.state == "watching" else 0.0
