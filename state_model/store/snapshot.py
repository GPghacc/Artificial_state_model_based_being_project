"""스냅샷: 장 상태 + 연결 + 주의를 .npz 하나로. 복원하면 끊김 없이 이어서 돈다."""
import glob
import json
import os
import numpy as np


def save(snapshot_dir, field, attention):
    os.makedirs(snapshot_dir, exist_ok=True)
    s = field.snapshot()
    path = os.path.join(snapshot_dir, f"tick_{s['tick']:010d}.npz")
    np.savez_compressed(path, **s, attention=json.dumps(attention.state()))
    latest = os.path.join(snapshot_dir, "latest.txt")
    with open(latest, "w") as f:
        f.write(os.path.basename(path))
    return path


def load_latest(snapshot_dir):
    latest = os.path.join(snapshot_dir, "latest.txt")
    if not os.path.exists(latest):
        files = sorted(glob.glob(os.path.join(snapshot_dir, "tick_*.npz")))
        if not files:
            return None
        path = files[-1]
    else:
        with open(latest) as f:
            path = os.path.join(snapshot_dir, f.read().strip())
    d = np.load(path, allow_pickle=False)
    s = {k: d[k] for k in d.files}
    s["attention"] = json.loads(str(s["attention"]))
    s["_path"] = path
    return s
