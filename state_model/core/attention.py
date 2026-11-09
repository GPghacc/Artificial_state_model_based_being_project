"""주의: 칸마다의 증폭 비율 지도 gain (길이 N).

인터페이스는 지도 하나를 내놓는 것뿐이라, 나중에 원형 지점 대신
물결을 따라가는 지도나 특징 기반 지도로 바꿔도 다른 모듈은 그대로다.
v0는 원형 지점 몇 개로 지도를 만든다. 예산은 고정(지점 비중의 합 = 1).
"""
import numpy as np


class Attention:
    def __init__(self, w, h, cfg):
        self.w, self.h = w, h
        self.floor = cfg["floor"]
        self.sigma = cfg["sigma"]
        self.peak = cfg["peak"]
        ys, xs = np.mgrid[0:h, 0:w]
        self._xs = xs.ravel().astype(np.float32)
        self._ys = ys.ravel().astype(np.float32)
        self.spots = [dict(s) for s in cfg["spots"]]
        self._gain = None
        self._spot_map = None
        self._rebuild()

    def set(self, spots):
        self.spots = [dict(s) for s in spots]
        self._rebuild()

    def _rebuild(self):
        total = sum(s["w"] for s in self.spots) or 1.0
        inv = 1.0 / (2 * self.sigma ** 2)
        spot_map = np.zeros(self.w * self.h, dtype=np.float32)
        for s in self.spots:
            d2 = (self._xs - s["x"]) ** 2 + (self._ys - s["y"]) ** 2
            spot_map += (s["w"] / total) * np.exp(-d2 * inv)
        self._spot_map = spot_map
        self._gain = self.floor + self.peak * spot_map

    def gain(self):
        return self._gain

    def weight(self):
        """추출용: 주의가 실린 정도 (0~1 근처)."""
        return self._spot_map

    def state(self):
        return {"spots": self.spots, "sigma": self.sigma}
