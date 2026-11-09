"""추출: 장에서 의미 있는 값을 읽는다. v0는 에너지 계열만."""
import numpy as np


class Readout:
    def __init__(self, zones, attention, ema_sec=10.0, tick_hz=30):
        self.zones = zones
        self.attention = attention
        self.alpha = 1.0 / (ema_sec * tick_hz)
        self.slow_energy = None
        self._zone_norm = zones.M.sum(axis=0) + 1e-9

    def read(self, field):
        en = field.u ** 2 + field.v ** 2
        total = float(en.mean())
        per_zone = (en @ self.zones.M) / self._zone_norm
        aw = self.attention.weight()
        attended = float((en * aw).sum() / (aw.sum() + 1e-9))
        if self.slow_energy is None:
            self.slow_energy = total
        self.slow_energy += self.alpha * (total - self.slow_energy)
        out = {"total": total, "attended": attended, "slow": self.slow_energy,
               "afterglow": max(0.0, self.slow_energy - total)}  # 최근보다 지금이 조용한 만큼 = 잔향 성분
        for name, val in zip(self.zones.names, per_zone):
            out["z_" + name] = float(val)
        return out
