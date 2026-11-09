"""상태모델 v0 실행. 실시간 30틱/초로 리셋 없이 계속 돈다.

  python run.py                 # 이전 스냅샷에서 이어서 실행
  python run.py --fresh         # 새로 시작
  python run.py --view          # 실시간 뷰어와 함께
  python run.py --seconds 60    # 60초만 돌고 종료 (테스트용)
  python run.py --fast          # 실시간 대기 없이 최대 속도 (테스트용)

종료(Ctrl+C) 시 스냅샷을 저장한다.
"""
import argparse
import os
import sys
import threading
import time
from collections import deque

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core.field import Field
from core.zones import Zones
from core.attention import Attention
from core.readout import Readout
from senses.base import Encoder
from senses.synthetic import SyntheticSense
from senses.system import ClockSense, LoadSense, InteriorSense
from senses.toggle import PresenceSense
from mind.controller import Controller
from mind.drives import Drives
from mind.plasticity import Plasticity
from store.logger import Logger
from store import snapshot


class Being:
    """모든 모듈을 한 틱 루프로 묶는다. 순서는 v0부터 고정."""

    def __init__(self, cfg, fresh=False):
        self.cfg = cfg
        g = cfg["grid"]
        self.field = Field(g, cfg["field"])
        self.zones = Zones(g["width"], g["height"])
        self.attention = Attention(g["width"], g["height"], cfg["attention"])
        hz = cfg["time"]["tick_hz"]
        self.readout = Readout(self.zones, self.attention, tick_hz=hz)
        self.encoder = Encoder(self.field, self.zones, cfg["senses"], cfg["field"]["noise_floor"], cfg["seed"])
        tf = os.path.join(HERE, cfg["senses"]["toggle_file"])
        self.senses = [
            SyntheticSense("vision_fake", "vision", cfg["seed"] + 10),
            SyntheticSense("audio_fake", "audio", cfg["seed"] + 11, max_level=0.4),
            ClockSense(), LoadSense(), InteriorSense(), PresenceSense(tf),
        ]
        self.controller = Controller()
        self.drives = Drives(cfg["drives"])
        self.plasticity = Plasticity()
        self.snap_dir = os.path.join(HERE, cfg["snapshot_dir"])
        self.restored_from = None
        if not fresh:
            s = snapshot.load_latest(self.snap_dir)
            if s is not None:
                self.field.restore(s)
                self.attention.set(s["attention"]["spots"])
                self.restored_from = s["_path"]
        self.logger = Logger(os.path.join(HERE, cfg["log_dir"]))
        self.last = {}
        self.history = deque(maxlen=hz * 30)   # 뷰어용 최근 30초
        self.t0 = time.time()
        self.lock = threading.Lock()

    def step(self):
        t = time.time() - self.t0
        ctx = {"readout": self.last}
        # 1. 감각 읽기
        levels = {s.name: (s.zone, s.level(t, ctx)) for s in self.senses}
        events = self.encoder.sample_events(levels)
        # 2. 입력 만들기 (배경 자극 + 사건) × 주의
        drive = self.encoder.build_drive(events, self.attention.gain())
        with self.lock:
            # 3. 매질 진행
            self.field.tick(drive)
            # 4. 추출
            r = self.readout.read(self.field)
        tick = self.field.tick_count
        # 5. 판단 (느린 주기)
        if tick % self.cfg["time"]["decide_every"] == 0:
            spots = self.controller.decide(r, self.attention)
            if spots is not None:
                self.attention.set(spots)
        # 6. 설정값 거리
        d = self.drives.deviation(r)
        # 7. 매질 학습 (가장 느린 주기)
        if self.plasticity.enabled and tick % self.cfg["time"]["plasticity_every"] == 0:
            self.plasticity.update(self.field)
        # 8. 기록
        self.logger.log(tick, r, d, events)
        self.last = r
        self.history.append((tick, r, levels))
        return r

    def save(self):
        with self.lock:
            p = snapshot.save(self.snap_dir, self.field, self.attention)
        self.logger.flush()
        return p


def run_loop(being, seconds=None, fast=False, stop_flag=None):
    hz = being.cfg["time"]["tick_hz"]
    dt = 1.0 / hz
    snap_every = being.cfg["time"]["snapshot_every_sec"]
    last_snap = time.time()
    next_t = time.perf_counter()
    start = time.time()
    behind = 0
    ran = 0
    while True:
        if stop_flag is not None and stop_flag.is_set():
            break
        being.step()
        ran += 1
        if seconds is not None:
            done = ran >= seconds * hz if fast else time.time() - start >= seconds
            if done:
                break
        if time.time() - last_snap > snap_every:
            being.save()
            last_snap = time.time()
        if not fast:
            next_t += dt
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                behind += 1
                if behind % (hz * 10) == 0:
                    print(f"[경고] 실시간을 따라가지 못하는 중 ({-wait*1000:.1f}ms 늦음)")
                next_t = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    being = Being(cfg, fresh=a.fresh)
    print(f"[시작] run {being.logger.run_id}" + (f" · 복원: {os.path.basename(being.restored_from)}" if being.restored_from else " · 새로 시작"))
    try:
        if a.view:
            from tools.viewer import Viewer
            stop = threading.Event()
            th = threading.Thread(target=run_loop, args=(being, a.seconds, a.fast, stop), daemon=True)
            th.start()
            Viewer(being).show()
            stop.set()
            th.join(timeout=2)
        else:
            run_loop(being, a.seconds, a.fast)
    except KeyboardInterrupt:
        pass
    finally:
        p = being.save()
        being.logger.close()
        print(f"[종료] tick {being.field.tick_count} · 스냅샷 {os.path.basename(p)}")


if __name__ == "__main__":
    main()
