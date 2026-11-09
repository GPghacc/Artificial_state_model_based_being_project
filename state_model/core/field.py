"""매질: 감쇠 결합 진동자 장. 상태는 u(물결 높이)와 v(변화 속도)뿐이다."""
import numpy as np
import scipy.sparse as sp


def neighbor_matrix(w, h):
    """격자 상하좌우 이웃을 잇는 연결 행렬 W. 나중에 장거리 연결은 여기에 더한다."""
    n = w * h
    idx = np.arange(n).reshape(h, w)
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = idx[: h - dy, : w - dx].ravel()
        b = idx[dy:, dx:].ravel()
        rows += [a, b]
        cols += [b, a]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


class Field:
    def __init__(self, cfg_grid, cfg_field):
        self.w, self.h = cfg_grid["width"], cfg_grid["height"]
        self.n = self.w * self.h
        self.p = dict(cfg_field)
        self.W = neighbor_matrix(self.w, self.h)
        self.deg = np.asarray(self.W.sum(axis=1)).ravel().astype(np.float32)
        self.u = np.zeros(self.n, dtype=np.float32)
        self.v = np.zeros(self.n, dtype=np.float32)
        self.tick_count = 0

    def tick(self, drive):
        p = self.p
        coupling = self.W @ self.u - self.deg * self.u   # 이웃과의 차이 = 퍼짐
        d_eff = 2 * p["damp"] * (1 - p["excite"])
        self.v += p["c2"] * coupling - p["w2"] * self.u - d_eff * self.v + drive
        s = p["sat"]
        self.u = (s * np.tanh((self.u + self.v) / s)).astype(np.float32)
        self.tick_count += 1

    def snapshot(self):
        return {"u": self.u.copy(), "v": self.v.copy(), "tick": self.tick_count,
                "W_data": self.W.data, "W_indices": self.W.indices, "W_indptr": self.W.indptr}

    def restore(self, s):
        self.u = s["u"].astype(np.float32)
        self.v = s["v"].astype(np.float32)
        self.tick_count = int(s["tick"])
        self.W = sp.csr_matrix((s["W_data"], s["W_indices"], s["W_indptr"]), shape=(self.n, self.n))
        self.deg = np.asarray(self.W.sum(axis=1)).ravel().astype(np.float32)
