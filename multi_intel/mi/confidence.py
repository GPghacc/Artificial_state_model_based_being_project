"""Confidence estimator. Psychological seat: the meta level in Nelson & Narens;
Koriat's cue-utilization. It never inspects the module's content, only cues.

One small online logistic model per module: P(success | cues). It starts from
hand-set priors (so it works before any outcome exists) and updates only when an
Outcome is SUCCESS or FAILURE. UNKNOWN outcomes teach it nothing - that is the
honest cost of this design, not a bug.

Because it learns from cues alone, it will inherit human-like illusions: whatever
correlates with success in the past (fluency, agreement) is trusted even when the
correlation breaks. That error is accepted as part of the function.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .types import Cues


class ConfidenceEstimator:
    def __init__(self, cue_names: dict[str, tuple[str, ...]], lr: float = 0.15,
                 prior_weight: float = 2.0, prior_bias: float = -1.0):
        self.lr = lr
        self.w: dict[str, dict[str, float]] = {}
        self.b: dict[str, float] = {}
        self.n_updates: dict[str, int] = {}
        for mod, names in cue_names.items():
            self.w[mod] = {n: prior_weight for n in names}
            self.b[mod] = prior_bias
            self.n_updates[mod] = 0

    def predict(self, module: str, cues: Cues) -> float:
        z = self.b[module] + sum(self.w[module].get(k, 0.0) * v for k, v in cues.values.items())
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, module: str, cues: Cues, success: bool) -> None:
        p = self.predict(module, cues)
        err = (1.0 if success else 0.0) - p
        for k, v in cues.values.items():
            if k in self.w[module]:
                self.w[module][k] += self.lr * err * v
        self.b[module] += self.lr * err
        self.n_updates[module] += 1

    # persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"w": self.w, "b": self.b, "n": self.n_updates}, indent=1))

    def load(self, path: str | Path) -> None:
        d = json.loads(Path(path).read_text())
        self.w, self.b, self.n_updates = d["w"], d["b"], d["n"]
