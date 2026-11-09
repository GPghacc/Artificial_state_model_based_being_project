"""설정값(본능): 상태값이 원하는 범위에서 얼마나 벗어났는지. v0는 계산과 기록만."""


class Drives:
    def __init__(self, cfg):
        self.ranges = {k: tuple(v) for k, v in cfg.items()}

    def deviation(self, readout):
        out = {}
        for key, (lo, hi) in self.ranges.items():
            x = readout.get(key.replace("total_energy", "total"), 0.0)
            out["dev_" + key] = (lo - x) / lo if x < lo else ((x - hi) / hi if x > hi else 0.0)
        return out
