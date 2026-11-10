"""장의 역할에 맞춘 시험.

장의 역할: 정신적 에너지를 느끼는 감각 + 자극 패턴의 재현.
쓰임새: 에너지가 몰린 곳에 연산(주의)을 더 주기 위한 지도. 출처를 정확히 가려낼 필요는 없다.

여기서는 매질 자체의 성질만 잰다. 주의는 꺼 둔다(모든 칸 증폭 1).
자극은 구역과 상관없이 격자의 원하는 자리에 '몰린 사건들'로 직접 넣는다.

  A 위치   : 자극이 몰린 곳에 에너지 봉우리가 생기는가
  B 섞임   : 가까운 두 자극이 하나의 봉우리로 합쳐지는가
  C 에너지 : 전체 에너지가 자극의 양을 재는 눈금이 되는가
  D 잔향   : 자극이 끊긴 뒤 봉우리가 얼마나 남는가
  E 따라감 : 몰린 곳이 옮겨가면 봉우리가 얼마나 빨리 따라가는가

실행:  python experiments/field_role_tests.py        (약 1~2분)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from core.field import Field      # noqa: E402
from core.zones import Zones      # noqa: E402
from senses.base import Encoder   # noqa: E402

WARMUP = 40
MARGIN = 16


class Harness:
    def __init__(self, cfg, seed):
        g = cfg["grid"]
        self.w, self.h = g["width"], g["height"]
        self.field = Field(g, cfg["field"])
        self.enc = Encoder(self.field, Zones(self.w, self.h), cfg["senses"],
                           cfg["field"]["noise_floor"], seed)
        self.gain = np.ones(self.field.n, dtype=np.float32)   # 주의 끔
        self.amp = cfg["senses"]["event_gain"]
        self.rng = np.random.default_rng(seed + 100)
        self.reset()

    def reset(self):
        self.field.u[:] = 0
        self.field.v[:] = 0
        self.ema = np.zeros(self.field.n)
        self.run([], WARMUP)

    def step(self, clusters):
        """clusters: [(cx, cy, 퍼짐(칸), 틱당 평균 사건 수)]"""
        events = []
        for cx, cy, spread, rate in clusters:
            k = self.rng.poisson(rate)
            if k == 0:
                continue
            xs = np.clip(np.rint(self.rng.normal(cx, spread, k)), 0, self.w - 1).astype(int)
            ys = np.clip(np.rint(self.rng.normal(cy, spread, k)), 0, self.h - 1).astype(int)
            sg = self.rng.choice((-1.0, 1.0), k)
            events += [(int(y * self.w + x), float(s * self.amp)) for x, y, s in zip(xs, ys, sg)]
        self.field.tick(self.enc.build_drive(events, self.gain))
        en = self.field.u.astype(np.float64) ** 2 + self.field.v.astype(np.float64) ** 2
        self.ema += 0.1 * (en - self.ema)          # 물결의 오르내림을 평평하게 (약 10틱 평균)

    def run(self, clusters, ticks):
        for _ in range(ticks):
            self.step(clusters)

    def emap(self):
        """에너지 지도: 시간·공간으로 살짝 뭉갠 에너지."""
        return gaussian_filter(self.ema.reshape(self.h, self.w), 3.0)

    def peak(self):
        m = self.emap()
        y, x = np.unravel_index(m.argmax(), m.shape)
        return x, y, m.max() / (np.median(m) + 1e-12), m

    def total(self):
        return float(self.ema.mean())

    def rand_point(self):
        return (self.rng.uniform(MARGIN, self.w - MARGIN), self.rng.uniform(MARGIN, self.h - MARGIN))


def noise_contrast(hz, n=30):
    """자극이 없을 때의 봉우리 대비. 이보다 뚜렷해야 '봉우리가 있다'고 본다."""
    c = []
    for _ in range(n):
        hz.reset()
        hz.run([], 60)
        c.append(hz.peak()[2])
    return float(np.percentile(c, 95))


def test_location(hz, n):
    rows = []
    for spread in (2, 6, 12):
        for rate in (0.25, 1.0, 4.0):
            d, c = [], []
            for _ in range(n):
                hz.reset()
                cx, cy = hz.rand_point()
                hz.run([(cx, cy, spread, rate)], 60)
                x, y, con, _ = hz.peak()
                d.append(np.hypot(x - cx, y - cy))
                c.append(con)
            rows.append({"spread": spread, "rate": rate, "dist": float(np.mean(d)),
                         "within8": float(np.mean(np.array(d) <= 8)), "contrast": float(np.median(c))})
    return rows


def test_mixing(hz, n):
    rows = []
    for sep in (6, 12, 20, 32, 48):
        ratio, mid_peak = [], []
        for _ in range(n):
            hz.reset()
            cx = hz.rng.uniform(MARGIN + sep / 2, hz.w - MARGIN - sep / 2)
            cy = hz.rng.uniform(MARGIN, hz.h - MARGIN)
            a, b = (cx - sep / 2, cy), (cx + sep / 2, cy)
            hz.run([(a[0], a[1], 3, 1.0), (b[0], b[1], 3, 1.0)], 80)
            x, y, _, m = hz.peak()
            at = lambda p: m[int(round(p[1])), int(round(p[0]))]   # noqa: E731
            ratio.append(at((cx, cy)) / ((at(a) + at(b)) / 2))
            mid_peak.append(abs(x - cx) <= sep / 4 + 2)
        rows.append({"sep": sep, "valley": float(np.median(ratio)), "peak_mid": float(np.mean(mid_peak))})
    return rows


def test_gauge(hz, n):
    rows = []
    for rate in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        e = []
        for _ in range(n):
            hz.reset()
            cx, cy = hz.rand_point()
            hz.run([(cx, cy, 6, rate)], 200)
            e.append(hz.total())
        rows.append({"rate": rate, "energy": float(np.mean(e)), "cv": float(np.std(e) / np.mean(e))})
    return rows


def test_afterglow(hz, n, noise_c):
    delays = (0, 10, 30, 60, 120, 240)
    acc = {d: {"dist": [], "con": []} for d in delays}
    for _ in range(n):
        hz.reset()
        cx, cy = hz.rand_point()
        hz.run([(cx, cy, 4, 2.0)], 60)
        t = 0
        for d in delays:
            hz.run([], d - t)
            t = d
            x, y, con, _ = hz.peak()
            acc[d]["dist"].append(np.hypot(x - cx, y - cy))
            acc[d]["con"].append(con)
    return [{"delay": d, "within8": float(np.mean(np.array(acc[d]["dist"]) <= 8)),
             "contrast": float(np.median(acc[d]["con"])),
             "visible": float(np.mean(np.array(acc[d]["con"]) > noise_c))} for d in delays]


def test_tracking(hz, n):
    rows = []
    for rate in (1.0, 4.0):
        lag = []
        for _ in range(n):
            hz.reset()
            while True:
                p1, p2 = hz.rand_point(), hz.rand_point()
                if np.hypot(p1[0] - p2[0], p1[1] - p2[1]) >= 40:
                    break
            hz.run([(p1[0], p1[1], 4, rate)], 60)
            t = None
            for k in range(1, 301):
                hz.step([(p2[0], p2[1], 4, rate)])
                x, y, _, _ = hz.peak()
                if np.hypot(x - p2[0], y - p2[1]) <= 8:
                    t = k
                    break
            lag.append(t if t is not None else 300)
        rows.append({"rate": rate, "median": float(np.median(lag)), "p90": float(np.percentile(lag, 90))})
    return rows


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="예: field.c2=0.12")
    ap.add_argument("--tag", default="v0")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for s in args.set:
        key, val = s.split("=")
        sec, name = key.split(".")
        cfg[sec][name] = float(val)

    hz = Harness(cfg, cfg["seed"] + args.seed)
    t0 = time.time()
    n = args.trials
    noise_c = noise_contrast(hz)
    loc = test_location(hz, n); print("  A 위치 끝", flush=True)          # noqa: E702
    mix = test_mixing(hz, n); print("  B 섞임 끝", flush=True)            # noqa: E702
    gauge = test_gauge(hz, max(8, n // 4)); print("  C 에너지 끝", flush=True)  # noqa: E702
    glow = test_afterglow(hz, n, noise_c); print("  D 잔향 끝", flush=True)     # noqa: E702
    track = test_tracking(hz, n); print("  E 따라감 끝", flush=True)      # noqa: E702

    out = [f"# 장의 역할 시험 ({args.tag})", "",
           f"시행 {n}회, 설정 변경: {', '.join(args.set) or '없음'}, 걸린 시간 {time.time() - t0:.0f}초.",
           f"격자 {hz.w}×{hz.h}, 초당 30틱. 대비 = 봉우리 에너지 ÷ 지도 중간값. "
           f"자극이 없을 때 대비는 {noise_c:.1f} 이하.", "",
           "## A 위치: 자극이 몰린 곳에 봉우리가 생기는가", "",
           md_table(["퍼짐(칸)", "틱당 사건", "봉우리와의 거리(칸)", "8칸 안에 든 비율", "대비"],
                    [[str(r["spread"]), f"{r['rate']:g}", f"{r['dist']:.1f}", f"{r['within8'] * 100:.0f}%",
                      f"{r['contrast']:.0f}"] for r in loc]), "",
           "## B 섞임: 두 자극이 하나로 합쳐지는가", "",
           "골 깊이 = 두 자극 가운데 지점의 에너지 ÷ 두 자극 자리의 에너지. 1에 가까우면 하나로 합쳐진 것.", "",
           md_table(["두 자극 사이 거리(칸)", "골 깊이", "봉우리가 가운데에 선 비율"],
                    [[str(r["sep"]), f"{r['valley']:.2f}", f"{r['peak_mid'] * 100:.0f}%"] for r in mix]), "",
           "## C 에너지: 전체 에너지가 자극의 양을 재는가", "",
           md_table(["틱당 사건", "전체 에너지", "시행 간 흔들림"],
                    [[f"{r['rate']:g}", f"{r['energy']:.4f}", f"{r['cv'] * 100:.0f}%"] for r in gauge]), "",
           "## D 잔향: 자극이 끊긴 뒤 봉우리가 남는가", "",
           md_table(["끊긴 뒤(틱)", "초", "제자리에 있는 비율", "대비", "아직 보이는 비율"],
                    [[str(r["delay"]), f"{r['delay'] / 30:.1f}", f"{r['within8'] * 100:.0f}%",
                      f"{r['contrast']:.1f}", f"{r['visible'] * 100:.0f}%"] for r in glow]), "",
           "## E 따라감: 몰린 곳이 옮겨가면 봉우리가 언제 따라가는가", "",
           md_table(["틱당 사건", "걸린 틱(중간값)", "초", "느린 10%"],
                    [[f"{r['rate']:g}", f"{r['median']:.0f}", f"{r['median'] / 30:.1f}", f"{r['p90']:.0f}틱"]
                     for r in track]), ""]
    text = "\n".join(out)
    print(text)

    res = os.path.join(HERE, "results")
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, f"field_role_{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(res, f"field_role_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"noise_contrast": noise_c, "location": loc, "mixing": mix, "gauge": gauge,
                   "afterglow": glow, "tracking": track, "set": args.set}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
