"""Episodic workspace. Psychological seat: Baddeley's episodic buffer / Tulving's
episodic memory / the broadcast step of Global Workspace Theory.

For now it does one thing: bind each cycle into a single Episode and keep the log.
This is deliberately thin. It is the seat where "one entity" would live if this
system were ever to be more than a toolbox - which is a later decision, not this
prototype's.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .types import Episode, Outcome


class Workspace:
    def __init__(self):
        self.episodes: list[Episode] = []

    def bind(self, ep: Episode) -> None:
        self.episodes.append(ep)

    def recent(self, n: int = 5) -> list[Episode]:
        return self.episodes[-n:]

    def stats(self) -> dict:
        n = len(self.episodes)
        if not n:
            return {"episodes": 0}
        known = [e for e in self.episodes if e.outcome != Outcome.UNKNOWN]
        acc = sum(e.outcome == Outcome.SUCCESS for e in known) / len(known) if known else None
        return {
            "episodes": n,
            "with_ground_truth": len(known),
            "accuracy": acc,
            "referee_interventions": sum(e.referee_intervened for e in self.episodes),
            "chosen_counts": _count(e.chosen for e in self.episodes),
        }

    def dump(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self.episodes:
                d = asdict(e)
                d["outcome"] = e.outcome.value
                d["signature"] = list(e.signature.key())
                f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")


def _count(it):
    out: dict[str, int] = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out
