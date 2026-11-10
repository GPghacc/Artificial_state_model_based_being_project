"""자극 개수를 늘리면 장과 기준선이 어떻게 달라지는가.

실험 1 (구역 수): 자극을 받는 감각 구역을 2개에서 5개까지 늘린다.
  order  : 시각과 청각 중 어느 쪽이 먼저 왔는가 (나머지 구역은 방해 자극)
  parity : 켜진 구역 수가 홀수인가 (XOR을 여러 개로 넓힌 것, 날값 읽기)

실험 2 (반복 횟수): 시각과 청각에 자극을 각각 1번에서 4번까지 준다.
  last   : 마지막 자극이 어느 쪽이었는가
  first  : 첫 자극이 어느 쪽이었는가

실행:  python experiments/stimulus_count.py            (약 3분)
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
sys.path.insert(0, HERE)

from baseline_benchmark import (Rig, run_trial, score_all, PULSE, WARMUP,  # noqa: E402
                                MODELS, LABEL, A_ZONE, B_ZONE)

DISTRACTORS = ("body", "action", "interior")
READ_DELAY = 20


def exp_zones(rig, n, rng, k):
    """k개 구역에 자극. 시각·청각의 순서 맞히기 + 켜진 구역 수의 홀짝."""
    zones = (A_ZONE, B_ZONE) + DISTRACTORS[: k - 2]
    feats, y = [], []
    for _ in range(n):
        while True:
            ts = rng.integers(WARMUP, WARMUP + 60, size=k)
            if abs(ts[0] - ts[1]) >= 8:
                break
        amps = rng.uniform(0.4, 1.0, size=k)
        pulses = [(z, int(t), float(a)) for z, t, a in zip(zones, ts, amps)]
        feats.append(run_trial(rig, pulses, [int(ts.max()) + PULSE + READ_DELAY])[0])
        y.append(1.0 if ts[0] < ts[1] else -1.0)
    order = score_all(feats, np.array(y), "cls", rng)

    feats, y = [], []
    t0 = WARMUP + 20
    for _ in range(n):
        on = rng.random(k) < 0.5
        pulses = [(z, t0, 0.8) for z, o in zip(zones, on) if o]
        feats.append(run_trial(rig, pulses, [t0 + PULSE + READ_DELAY])[0])
        y.append(1.0 if on.sum() % 2 == 1 else -1.0)
    parity = score_all(feats, np.array(y), "cls", rng, log=False)
    return order, parity


def exp_repeats(rig, n, rng, m):
    """시각·청각에 각각 m번씩 자극. 마지막과 첫 자극이 어느 쪽이었는지 맞히기."""
    feats, y_last, y_first = [], [], []
    for _ in range(n):
        while True:
            ts = np.sort(rng.choice(np.arange(WARMUP, WARMUP + 150), size=2 * m, replace=False))
            if np.all(np.diff(ts) >= 8):
                break
        who = np.array([0] * m + [1] * m)
        rng.shuffle(who)
        amps = rng.uniform(0.4, 1.0, size=2 * m)
        pulses = [((A_ZONE, B_ZONE)[w], int(t), float(a)) for w, t, a in zip(who, ts, amps)]
        feats.append(run_trial(rig, pulses, [int(ts[-1]) + PULSE + READ_DELAY])[0])
        y_last.append(1.0 if who[-1] == 0 else -1.0)
        y_first.append(1.0 if who[0] == 0 else -1.0)
    return (score_all(feats, np.array(y_last), "cls", rng),
            score_all(feats, np.array(y_first), "cls", rng))


def table(title, rows):
    lines = [f"\n### {title}", "", "| 조건 | " + " | ".join(LABEL[m] for m in MODELS) + " |",
             "|---|" + "---|" * len(MODELS)]
    for name, sc in rows:
        lines.append(f"| {name} | " + " | ".join(f"{sc[m] * 100:.0f}%" for m in MODELS) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v0")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(args.seed)
    rig = Rig(cfg)
    t_start = time.time()

    zones, repeats = {}, {}
    for k in (2, 3, 4, 5):
        zones[k] = exp_zones(rig, args.trials, rng, k)
        print(f"  구역 {k}개 끝", flush=True)
    for m in (1, 2, 3, 4):
        repeats[m] = exp_repeats(rig, args.trials, rng, m)
        print(f"  반복 {m}번 끝", flush=True)

    out = [f"# 자극 개수 실험 ({args.tag})", "",
           f"시행 {args.trials}회, 마지막 자극 {READ_DELAY}틱 뒤에 읽음, "
           f"걸린 시간 {time.time() - t_start:.0f}초. 찍으면 50%."]
    out.append(table("구역 수: 시각과 청각 중 어느 쪽이 먼저 왔는가",
                     [(f"구역 {k}개", zones[k][0]) for k in zones]))
    out.append(table("구역 수: 켜진 구역이 홀수 개인가 (날값 읽기)",
                     [(f"구역 {k}개", zones[k][1]) for k in zones]))
    out.append(table("반복: 마지막 자극이 어느 쪽이었는가",
                     [(f"각 {m}번", repeats[m][0]) for m in repeats]))
    out.append(table("반복: 첫 자극이 어느 쪽이었는가",
                     [(f"각 {m}번", repeats[m][1]) for m in repeats]))
    text = "\n".join(out) + "\n"
    print(text)

    res_dir = os.path.join(HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, f"stimulus_count_{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(res_dir, f"stimulus_count_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"trials": args.trials,
                   "zones": {k: {"order": v[0], "parity": v[1]} for k, v in zones.items()},
                   "repeats": {m: {"last": v[0], "first": v[1]} for m, v in repeats.items()}},
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
