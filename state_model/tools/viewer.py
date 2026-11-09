"""실시간 뷰어: 왼쪽은 장 그래픽(구역 윤곽·주의 영역 포함), 오른쪽은 추출값 그래프와 수치."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
    if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

ZONE_KO = {"vision": "시각", "body": "몸 감각", "action": "행동", "audio": "청각",
           "interior": "내부 감각", "parietal": "두정 연합", "temporal": "측두 연합", "prefrontal": "전전두"}


class Viewer:
    def __init__(self, being):
        self.b = being
        f = being.field
        self.w, self.h = f.w, f.h
        self.fig = plt.figure(figsize=(13, 6.5))
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
        self.ax_f = self.fig.add_subplot(gs[:, 0])
        self.ax_z = self.fig.add_subplot(gs[0:2, 1])
        self.ax_t = self.fig.add_subplot(gs[2, 1])
        self.img = self.ax_f.imshow(np.zeros((self.h, self.w)), cmap="RdBu_r", vmin=-1, vmax=1,
                                    interpolation="bilinear")
        self.ax_f.set_xticks([]); self.ax_f.set_yticks([])
        Z = being.zones
        for i, name in enumerate(Z.names):
            m = Z.M[:, i].reshape(self.h, self.w)
            self.ax_f.contour(m, levels=[0.35], colors="0.35", linewidths=0.6)
            cy, cx = np.unravel_index(np.argmax(m), m.shape)
            self.ax_f.text(cx, cy, ZONE_KO.get(name, name), ha="center", va="center", fontsize=8, color="0.25")
        self.attn_artists = []
        self.title = self.ax_f.set_title("")
        self.zone_lines = {}
        for name in Z.names:
            (ln,) = self.ax_z.plot([], [], lw=1.2, label=ZONE_KO.get(name, name))
            self.zone_lines[name] = ln
        self.ax_z.legend(fontsize=8, ncol=2, loc="upper left")
        self.ax_z.set_title("구역별 에너지 (최근 30초)", fontsize=10)
        (self.l_total,) = self.ax_t.plot([], [], lw=1.4, color="k", label="전체")
        (self.l_attn,) = self.ax_t.plot([], [], lw=1.2, ls="--", color="tab:green", label="주의 영역")
        self.ax_t.legend(fontsize=8, loc="upper left")
        self.fig.tight_layout(rect=(0, 0.06, 1, 1))
        self.info = self.fig.text(0.99, 0.015, "", ha="right", va="bottom", fontsize=8, family="monospace")

    def _draw_attention(self):
        for a in self.attn_artists:
            a.remove()
        self.attn_artists = []
        m = self.b.attention.weight().reshape(self.h, self.w)
        cs = self.ax_f.contour(m, levels=[0.3], colors="tab:green", linewidths=1.2, linestyles="--")
        self.attn_artists.append(cs)

    def update(self, _):
        b = self.b
        with b.lock:
            u = b.field.u.reshape(self.h, self.w).copy()
            hist = list(b.history)
            tick = b.field.tick_count
        self.img.set_data(u)
        self._draw_attention()
        if hist:
            ticks = np.array([h[0] for h in hist], dtype=float)
            x = (ticks - ticks[-1]) / b.cfg["time"]["tick_hz"]
            for name, ln in self.zone_lines.items():
                ln.set_data(x, [h[1]["z_" + name] for h in hist])
            self.l_total.set_data(x, [h[1]["total"] for h in hist])
            self.l_attn.set_data(x, [h[1]["attended"] for h in hist])
            for ax in (self.ax_z, self.ax_t):
                ax.relim(); ax.autoscale_view(); ax.set_xlim(-30, 0)
            r, levels = hist[-1][1], hist[-1][2]
            lv = "  ".join(f"{k}:{v[1]:.2f}" for k, v in levels.items())
            self.info.set_text(f"total {r['total']:.2e}  attn {r['attended']:.2e}  afterglow {r['afterglow']:.2e}\n{lv}")
        sec = tick / b.cfg["time"]["tick_hz"]
        self.title.set_text(f"상태장 · tick {tick} · {sec/60:.1f}분")
        return []

    def show(self):
        self.anim = FuncAnimation(self.fig, self.update, interval=100, cache_frame_data=False)
        plt.show()
