"""기준선 비교 시험: 장(매질)이 단순한 감쇠 벡터보다 뭘 더 담는가.

같은 자극을 장과 기준선에 똑같이 넣고, 일정 시간 뒤의 상태만 보고 문제를 맞히게 한다.
읽는 쪽은 모두 같은 선형 읽기(릿지 회귀)라서, 차이는 상태가 담은 정보에서만 나온다.

과제
  detect : 자극 A가 있었는가            (점검용. 이것도 못 하면 읽기가 고장)
  order  : A와 B 중 어느 쪽이 먼저 왔는가  (세기는 무작위 → 세기만으로는 못 맞힘)
  elapsed: 자극이 몇 틱 전에 왔는가       (세기는 무작위)
  xor    : A와 B 중 정확히 하나만 왔는가   (두 자극이 서로 영향을 줘야 풀림 = 비선형성 시험)
           이 과제만 로그 없이 날값으로 읽는다. 로그를 씌우면 읽는 쪽이 대신 풀어버려서
           매질 안에서 자극끼리 영향을 주는지 알 수 없게 된다.

비교 대상
  field_zone : 장의 구역별 에너지 (지금 readout.py가 읽는 것)
  field_grid : 장의 에너지를 16×10 격자로 나눈 것 (공간 모양까지 봄)
  leaky1     : 감각 구역마다 감쇠 값 하나 + 합계 (기분 벡터)
  leaky3     : 감각 구역마다 빠름·중간·느림 감쇠 값 셋 + 합계

실행:  python experiments/baseline_benchmark.py            (약 2~4분)
       python experiments/baseline_benchmark.py --trials 200   (빠르게)
       python experiments/baseline_benchmark.py --set field.excite=0.9
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from core.field import Field          # noqa: E402
from core.zones import Zones          # noqa: E402
from core.attention import Attention  # noqa: E402
from senses.base import Encoder       # noqa: E402

PULSE = 6            # 자극 길이 (틱)
WARMUP = 40          # 배경 자극만 흐르는 준비 시간
LEAKY_TAUS = (15.0, 50.0, 150.0)
A_ZONE, B_ZONE = "vision", "audio"


class Rig:
    """장 + 인코더 + 기준선을 같은 자극으로 한 틱씩 돌린다."""

    def __init__(self, cfg):
        self.cfg = cfg
        g = cfg["grid"]
        self.field = Field(g, cfg["field"])
        self.zones = Zones(g["width"], g["height"])
        self.attn = Attention(g["width"], g["height"], cfg["attention"])
        self.enc = Encoder(self.field, self.zones, cfg["senses"],
                           cfg["field"]["noise_floor"], cfg["seed"])
        self.senses = [z for z in self.zones.names if self.zones.kinds[z] == "sense"]
        self.zone_norm = self.zones.M.sum(axis=0) + 1e-9
        w, h = g["width"], g["height"]
        self.bw, self.bh = 16, 10
        ys, xs = np.mgrid[0:h, 0:w]
        self.block = ((ys * self.bh // h) * self.bw + (xs * self.bw // w)).ravel()
        self.block_n = np.bincount(self.block, minlength=self.bw * self.bh)
        self.decay = np.array([1.0 - 1.0 / t for t in LEAKY_TAUS])
        self.reset()

    def reset(self):
        self.field.u[:] = 0
        self.field.v[:] = 0
        self.enc.prev = {}
        self.leaky = np.zeros((len(self.senses), len(LEAKY_TAUS)))

    def step(self, levels):
        """levels: {구역 이름: 세기}. 없는 구역은 0."""
        events = []
        hit = np.zeros(len(self.senses))
        for i, z in enumerate(self.senses):
            ev = self.enc.sample_events({z: (z, levels.get(z, 0.0))})
            hit[i] = len(ev)
            events += ev
        self.field.tick(self.enc.build_drive(events, self.attn.gain()))
        # 기준선은 장이 실제로 맞은 사건 수를 그대로 받는다 (같은 잡음, 같은 정보)
        self.leaky = self.leaky * self.decay + hit[:, None]

    def features(self):
        en = self.field.u.astype(np.float64) ** 2 + self.field.v.astype(np.float64) ** 2
        zone = np.append((en @ self.zones.M) / self.zone_norm, en.mean())
        grid = np.bincount(self.block, weights=en, minlength=self.bw * self.bh) / self.block_n
        tot = self.leaky.sum(axis=0)          # 기준선에도 장의 '전체 에너지'에 해당하는 합계를 준다
        return {
            "field_zone": zone,
            "field_grid": grid,
            "leaky1": np.append(self.leaky[:, 1], tot[1]),
            "leaky3": np.append(self.leaky.ravel(), tot),
        }


def run_trial(rig, pulses, read_ticks):
    """pulses: [(구역, 시작 틱, 세기)], read_ticks: 상태를 읽을 틱들 → [features, ...]"""
    rig.reset()
    out, read = [], set(read_ticks)
    for t in range(max(read_ticks) + 1):
        lv = {z: a for z, t0, a in pulses if t0 <= t < t0 + PULSE}
        rig.step(lv)
        if t in read:
            out.append(rig.features())
    return out


# ---------- 읽기: 릿지 회귀 (모든 비교 대상에 똑같이 적용) ----------

def _fit(X, y, lam):
    A = X.T @ X + lam * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def ridge_eval(X, y, kind, rng):
    """kind: 'cls'(±1 맞히기 → 정확도) 또는 'reg'(값 맞히기 → R²). 5번 나눠 평균."""
    n = len(X)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    scores = []
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
        Xte = np.c_[(X[te] - mu) / sd, np.ones(len(te))]
        cut = int(len(tr) * 0.8)
        best, best_s = 1e-2, -np.inf
        for lam in (1e-4, 1e-2, 1.0):          # 규제 세기는 훈련 자료 안에서 고른다
            w = _fit(Xtr[:cut], y[tr][:cut], lam)
            s = -np.mean((Xtr[cut:] @ w - y[tr][cut:]) ** 2)
            if s > best_s:
                best, best_s = lam, s
        pred = Xte @ _fit(Xtr, y[tr], best)
        if kind == "cls":
            scores.append(np.mean(np.sign(pred) == y[te]))
        else:
            scores.append(1 - np.sum((pred - y[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    return float(np.mean(scores))


def score_all(feats, y, kind, rng, log=True):
    """log=True: 값의 비율을 볼 수 있게 로그를 씌워 읽는다.
    log=False: 날값 그대로 읽는다. 읽기 쪽의 비선형성을 빼야 하는 XOR에서 쓴다."""
    out = {}
    for name in feats[0]:
        X = np.array([f[name] for f in feats])
        out[name] = ridge_eval(np.log(X + 1e-6) if log else X, y, kind, rng)
    return out


# ---------- 과제 ----------

def task_order(rig, n, rng, delays):
    per_delay = {d: [] for d in delays}
    y = []
    for _ in range(n):
        while True:
            ta, tb = rng.integers(WARMUP, WARMUP + 60, size=2)
            if abs(ta - tb) >= 8:
                break
        a, b = rng.uniform(0.4, 1.0, size=2)
        end = max(ta, tb) + PULSE
        fs = run_trial(rig, [(A_ZONE, ta, a), (B_ZONE, tb, b)], [end + d for d in delays])
        for d, f in zip(delays, fs):
            per_delay[d].append(f)
        y.append(1.0 if ta < tb else -1.0)
    y = np.array(y)
    return {d: score_all(per_delay[d], y, "cls", rng) for d in delays}


def task_elapsed(rig, n, rng):
    feats, y = [], []
    read = WARMUP + 170
    for _ in range(n):
        t0 = int(rng.integers(WARMUP, WARMUP + 150))
        a = rng.uniform(0.4, 1.0)
        feats.append(run_trial(rig, [(A_ZONE, t0, a)], [read])[0])
        y.append(float(read - t0))
    return score_all(feats, np.array(y), "reg", rng)


def task_xor(rig, n, rng, delays):
    per_delay = {d: [] for d in delays}
    y_xor, y_a = [], []
    t0 = WARMUP + 20
    for _ in range(n):
        pa, pb = rng.random(2) < 0.5
        pulses = ([(A_ZONE, t0, 0.8)] if pa else []) + ([(B_ZONE, t0, 0.8)] if pb else [])
        fs = run_trial(rig, pulses, [t0 + PULSE + d for d in delays])
        for d, f in zip(delays, fs):
            per_delay[d].append(f)
        y_xor.append(1.0 if pa != pb else -1.0)
        y_a.append(1.0 if pa else -1.0)
    y_xor, y_a = np.array(y_xor), np.array(y_a)
    return ({d: score_all(per_delay[d], y_a, "cls", rng) for d in delays},
            {d: score_all(per_delay[d], y_xor, "cls", rng, log=False) for d in delays})


# ---------- 실행 ----------

MODELS = ("field_zone", "field_grid", "leaky1", "leaky3")
LABEL = {"field_zone": "장 (구역 에너지)", "field_grid": "장 (격자 에너지)",
         "leaky1": "기준선 (감쇠 1종)", "leaky3": "기준선 (감쇠 3종)"}


def table(title, rows, unit):
    lines = [f"\n### {title}", "", "| 조건 | " + " | ".join(LABEL[m] for m in MODELS) + " |",
             "|---|" + "---|" * len(MODELS)]
    for name, sc in rows:
        cells = [f"{sc[m] * 100:.0f}%" if unit == "acc" else f"{sc[m]:.2f}" for m in MODELS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="예: field.excite=0.9")
    ap.add_argument("--tag", default="v0")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for s in args.set:
        key, val = s.split("=")
        sec, name = key.split(".")
        cfg[sec][name] = float(val)

    rng = np.random.default_rng(args.seed)
    rig = Rig(cfg)
    t_start = time.time()
    delays = (5, 20, 60, 150)

    print(f"trials={args.trials}  설정 변경={args.set or '없음'}", flush=True)
    detect, xor = task_xor(rig, args.trials, rng, delays[:3])
    print("  detect, xor 끝", flush=True)
    order = task_order(rig, args.trials, rng, delays)
    print("  order 끝", flush=True)
    elapsed = task_elapsed(rig, args.trials, rng)
    print("  elapsed 끝", flush=True)

    out = [f"# 기준선 비교 결과 ({args.tag})", "",
           f"시행 {args.trials}회, 설정 변경: {', '.join(args.set) or '없음'}, "
           f"걸린 시간 {time.time() - t_start:.0f}초", "",
           "정확도 과제는 찍으면 50%다. 경과 시간은 R²(1이면 완벽, 0이면 평균만 말하는 수준)."]
    out.append(table("점검: A가 있었는가 (정확도)",
                     [(f"{d}틱 뒤", detect[d]) for d in delays[:3]], "acc"))
    out.append(table("순서: 어느 쪽이 먼저 왔는가 (정확도)",
                     [(f"{d}틱 뒤", order[d]) for d in delays], "acc"))
    out.append(table("경과 시간: 몇 틱 전에 왔는가 (R²)", [("20~170틱 전", elapsed)], "r2"))
    out.append(table("XOR: 정확히 하나만 왔는가 (정확도, 날값 읽기)",
                     [(f"{d}틱 뒤", xor[d]) for d in delays[:3]], "acc"))
    text = "\n".join(out) + "\n"
    print(text)

    res_dir = os.path.join(HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, f"baseline_{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(res_dir, f"baseline_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"trials": args.trials, "set": args.set, "detect": detect, "order": order,
                   "elapsed": elapsed, "xor": xor}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
