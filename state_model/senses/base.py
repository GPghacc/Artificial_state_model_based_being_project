"""감각 공통 인터페이스와, 감각 신호를 장의 입력(drive)으로 바꾸는 인코더.

감각은 0~1 사이 세기(level) 하나만 내놓는다. 인코더는 그 세기를
해당 구역 안 무작위 칸에 떨어지는 작은 사건들로 바꾼다 (세기 = 사건 빈도).
세기가 갑자기 변하면 사건이 몰려서 생긴다. 배경 자극은 모든 칸에 항상 더해진다.
"""
import numpy as np


class Sense:
    name = "sense"
    zone = None

    def level(self, t, ctx):
        """t: 실행 후 경과 초, ctx: 지난 틱 추출값 등. 0~1 반환."""
        raise NotImplementedError


class Encoder:
    def __init__(self, field, zones, cfg_senses, noise_floor, seed):
        self.n = field.n
        self.w, self.h = field.w, field.h
        self.zones = zones
        self.gain_ev = cfg_senses["event_gain"]
        self.rate = cfg_senses["event_rate"]
        self.burst = cfg_senses["change_burst"]
        self.noise_floor = noise_floor
        self.rng_noise = np.random.default_rng(seed)
        self.rng_events = np.random.default_rng(seed + 1)
        self.prev = {}
        r = np.arange(-3, 4)
        ky, kx = np.meshgrid(r, r, indexing="ij")
        self.kernel = np.exp(-(kx ** 2 + ky ** 2) / 5.0).astype(np.float32)
        self.k_off = (ky.ravel(), kx.ravel(), self.kernel.ravel())
        self.pdf = {}
        for z in zones.names:
            m = zones.membership(z).astype(np.float64)
            self.pdf[z] = m / m.sum()

    def sample_events(self, levels):
        """levels: {sense_name: (zone, level)} → [(cell, amp), ...]"""
        events = []
        for name, (zone, lv) in levels.items():
            change = abs(lv - self.prev.get(name, lv))
            self.prev[name] = lv
            lam = self.rate * lv + self.burst * change
            k = self.rng_events.poisson(lam)
            if k == 0:
                continue
            cells = self.rng_events.choice(self.n, size=k, p=self.pdf[zone])
            signs = self.rng_events.choice((-1.0, 1.0), size=k)
            for c, s in zip(cells, signs):
                events.append((int(c), float(s * self.gain_ev)))
        return events

    def build_drive(self, events, gain):
        drive = (self.rng_noise.standard_normal(self.n).astype(np.float32) * self.noise_floor)
        if events:
            grid = drive.reshape(self.h, self.w)
            dy, dx, kv = self.k_off
            for cell, amp in events:
                y, x = divmod(cell, self.w)
                yy, xx = y + dy, x + dx
                ok = (yy >= 0) & (yy < self.h) & (xx >= 0) & (xx < self.w)
                np.add.at(grid, (yy[ok], xx[ok]), amp * kv[ok])
        return drive * gain
