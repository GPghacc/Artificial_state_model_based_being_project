"""구역 맵: 칸마다 여러 구역에 대한 소속 비중 (N × K). 겹침 허용.

배치는 평면화한 대뇌피질을 크게 뭉뚱그린 임시 지도다. 왼쪽이 뒤(후두), 오른쪽이 앞(전두).
좌표는 격자 크기에 대한 비율이라 격자를 바꿔도 그대로 쓸 수 있다.
"""
import numpy as np

# 이름, 중심(x, y 비율), 반지름(x, y 비율), 종류
LAYOUT = [
    ("vision",    (0.12, 0.52), (0.09, 0.38), "sense"),
    ("body",      (0.45, 0.30), (0.09, 0.19), "sense"),
    ("action",    (0.60, 0.30), (0.08, 0.19), "sense"),
    ("audio",     (0.46, 0.85), (0.10, 0.13), "sense"),
    ("interior",  (0.61, 0.64), (0.08, 0.11), "sense"),
    ("parietal",  (0.29, 0.33), (0.09, 0.15), "assoc"),
    ("temporal",  (0.29, 0.79), (0.09, 0.14), "assoc"),
    ("prefrontal",(0.85, 0.52), (0.10, 0.38), "assoc"),
]


class Zones:
    def __init__(self, w, h, layout=LAYOUT):
        self.w, self.h = w, h
        self.names = [z[0] for z in layout]
        self.kinds = {z[0]: z[3] for z in layout}
        ys, xs = np.mgrid[0:h, 0:w]
        xs = (xs.ravel() + 0.5) / w
        ys = (ys.ravel() + 0.5) / h
        m = []
        for _, (cx, cy), (rx, ry), _ in layout:
            d2 = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
            m.append(np.exp(-d2 * 1.2))       # 경계 없이 부드럽게 줄어드는 소속
        M = np.stack(m, axis=1).astype(np.float32)
        s = M.sum(axis=1, keepdims=True)
        M = np.where(s > 1, M / s, M)          # 합 1 이하
        self.M = M                              # N × K
        self.index = {name: i for i, name in enumerate(self.names)}

    def membership(self, zone):
        return self.M[:, self.index[zone]]

    def association_weight(self):
        """어느 구역에도 거의 속하지 않는 정도 = 연합 영역 성격."""
        return 1.0 - self.M.sum(axis=1)
