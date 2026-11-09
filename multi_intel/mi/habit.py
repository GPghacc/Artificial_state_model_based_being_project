"""Habit layer. Psychological seat: dorsal striatum / procedural memory; Minsky's
K-lines. It remembers, per surface signature, which module tended to succeed, and
turns that into a prior that shortcuts deliberation.

It is the only place where learning flows *across* modules, so it is also the
seat where anything g-like could emerge (mutualism, not a shared engine).

Its known failure is structural conservatism: called modules get called more.
`openness` is the exploration parameter (DeYoung's cybernetic reading of the
Big-Five trait): with that probability the prior is ignored for one cycle so an
unused module gets a chance to show its cues.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .types import Signature


class HabitLayer:
    def __init__(self, openness: float = 0.1, smoothing: float = 1.0, seed: int = 0,
                 enabled: bool = True):
        self.openness = openness
        self.smoothing = smoothing
        self.enabled = enabled
        self.rng = random.Random(seed)
        # signature key -> module -> [successes, trials]
        self.table: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.explored_last = False

    def prior(self, sig: Signature, module: str) -> float:
        """Laplace-smoothed success rate; 0.5 when nothing is known."""
        if not self.enabled:
            return 0.5
        s, n = self.table[sig.key()][module]
        return (s + self.smoothing) / (n + 2 * self.smoothing)

    def weight(self, sig: Signature, module: str) -> float:
        """How much to trust the prior: grows with evidence, capped."""
        if not self.enabled:
            return 0.0
        _, n = self.table[sig.key()][module]
        return min(0.6, n / (n + 5.0))

    def begin_cycle(self) -> bool:
        """Decide whether this cycle explores (ignores priors)."""
        self.explored_last = self.enabled and self.rng.random() < self.openness
        return self.explored_last

    def record(self, sig: Signature, module: str, success: bool) -> None:
        cell = self.table[sig.key()][module]
        cell[1] += 1
        if success:
            cell[0] += 1

    # persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        flat = {json.dumps(list(k)): dict(v) for k, v in self.table.items()}
        Path(path).write_text(json.dumps(flat, indent=1))

    def load(self, path: str | Path) -> None:
        flat = json.loads(Path(path).read_text())
        for k, v in flat.items():
            self.table[tuple(json.loads(k))] = defaultdict(lambda: [0, 0], {m: list(c) for m, c in v.items()})
