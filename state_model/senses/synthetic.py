"""가짜 감각: 아직 장치가 없는 구역(시각, 청각)을 채우는 자리표시자.
가끔 세기가 바뀌는 단순한 과정이다."""
import numpy as np
from .base import Sense


class SyntheticSense(Sense):
    def __init__(self, name, zone, seed, switch_prob=0.004, max_level=0.6):
        self.name, self.zone = name, zone
        self.rng = np.random.default_rng(seed)
        self.switch_prob = switch_prob
        self.max_level = max_level
        self.current = 0.0

    def level(self, t, ctx):
        if self.rng.random() < self.switch_prob:
            self.current = 0.0 if self.current > 0 else float(self.rng.uniform(0.2, self.max_level))
        return self.current
