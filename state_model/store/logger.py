"""기록: 매 틱 추출값(CSV)과 감각 사건(JSONL). 사건 로그가 있으면 같은 경험을 다시 흘릴 수 있다."""
import csv
import json
import os
import time


class Logger:
    def __init__(self, log_dir, run_id=None):
        os.makedirs(log_dir, exist_ok=True)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.readout_path = os.path.join(log_dir, f"{self.run_id}_readout.csv")
        self.events_path = os.path.join(log_dir, f"{self.run_id}_events.jsonl")
        self._rf = open(self.readout_path, "w", newline="", encoding="utf-8")
        self._ef = open(self.events_path, "w", encoding="utf-8")
        self._writer = None
        self._last_flush = time.time()

    def log(self, tick, readout, drives, events):
        row = {"tick": tick, "wall": round(time.time(), 3), **readout, **drives}
        if self._writer is None:
            self._writer = csv.DictWriter(self._rf, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._ef.write(json.dumps({"tick": tick, "ev": events}) + "\n")
        if time.time() - self._last_flush > 1.0:
            self.flush()

    def flush(self):
        self._rf.flush()
        self._ef.flush()
        self._last_flush = time.time()

    def close(self):
        self.flush()
        self._rf.close()
        self._ef.close()
