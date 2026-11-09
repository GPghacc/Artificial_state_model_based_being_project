"""같은 경험 다시 흘리기: --fresh로 시작한 실행의 사건 로그를 새 장에 다시 넣고,
기록된 추출값과 비교한다. 매질 규칙을 바꾼 뒤 전후를 비교하는 데 쓴다.

  python tools/replay.py logs/<run_id>_events.jsonl
  python tools/replay.py logs/<run_id>_events.jsonl --config other.yaml   # 다른 설정으로 같은 경험
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from core.field import Field
from core.zones import Zones
from core.attention import Attention
from core.readout import Readout
from senses.base import Encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = cfg["grid"]
    field = Field(g, cfg["field"])
    zones = Zones(g["width"], g["height"])
    attn = Attention(g["width"], g["height"], cfg["attention"])
    readout = Readout(zones, attn, tick_hz=cfg["time"]["tick_hz"])
    enc = Encoder(field, zones, cfg["senses"], cfg["field"]["noise_floor"], cfg["seed"])

    rows = []
    with open(a.events, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            drive = enc.build_drive([tuple(e) for e in rec["ev"]], attn.gain())
            field.tick(drive)
            rows.append(readout.read(field)["total"])

    logged_path = a.events.replace("_events.jsonl", "_readout.csv")
    if os.path.exists(logged_path):
        with open(logged_path, encoding="utf-8") as f:
            logged = [float(r["total"]) for r in csv.DictReader(f)]
        n = min(len(logged), len(rows))
        diff = np.max(np.abs(np.array(rows[:n]) - np.array(logged[:n]))) if n else float("nan")
        print(f"[replay] {n} ticks · 기록과의 최대 차이 {diff:.3e}")
    else:
        print(f"[replay] {len(rows)} ticks · 마지막 전체 에너지 {rows[-1]:.3e}")


if __name__ == "__main__":
    main()
