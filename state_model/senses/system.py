"""실제 신호: 시계, 컴퓨터 부하, 내부 감각."""
import math
import os
import time
from .base import Sense

try:
    import psutil
except ImportError:  # 없으면 load average로 대체
    psutil = None


class ClockSense(Sense):
    """하루 중 시각을 천천히 변하는 세기로. 시간 감각의 바탕."""
    name, zone = "clock", "interior"

    def level(self, t, ctx):
        lt = time.localtime()
        frac = (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec) / 86400.0
        return 0.15 + 0.1 * (0.5 + 0.5 * math.sin(2 * math.pi * (frac - 0.25)))


class LoadSense(Sense):
    """CPU 부하 = 몸의 피로 같은 감각."""
    name, zone = "cpu", "body"

    def __init__(self, every=15):
        self.every = every
        self.count = 0
        self.value = 0.0
        if psutil:
            psutil.cpu_percent(None)

    def level(self, t, ctx):
        self.count += 1
        if self.count % self.every == 0:
            if psutil:
                self.value = psutil.cpu_percent(None) / 100.0
            elif hasattr(os, "getloadavg"):
                self.value = min(1.0, os.getloadavg()[0] / (os.cpu_count() or 1))
        return self.value


class InteriorSense(Sense):
    """지난 틱의 장 전체 에너지를 다시 느끼는 내부 감각 (되먹임)."""
    name, zone = "interior", "interior"

    def __init__(self, scale=0.12):
        self.scale = scale

    def level(self, t, ctx):
        r = ctx.get("readout")
        if not r:
            return 0.0
        return min(1.0, r["total"] / self.scale)
