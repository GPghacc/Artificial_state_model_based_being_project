"""Referee. Psychological seat: Norman & Shallice's supervisory attentional system;
Miyake's inhibition / shifting / updating. It does NOT understand content. It sees
only bids (module, confidence) and holds the goal.

Default path is contention scheduling: the strongest bid wins with no intervention.
The referee steps in only on three conditions:
  conflict  - the top two bids are too close to call
  novelty   - nobody claims the stimulus with any confidence
  goal      - the goal forbids the winner or prefers a verifiable module

Interventions available: inhibit (drop a module), compare (run the top two and
prefer the verified one), defer (pick the lower-cost one when tied).

Everything the referee holds - goal, thresholds - is the seat where a value layer
(Luna's emotional weighting) would plug in later. Nothing here is emotional yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .types import Bid, Result, Stimulus


@dataclass
class Goal:
    prefer_verified: bool = True
    forbid: set[str] = field(default_factory=set)
    conflict_margin: float = 0.08
    novelty_floor: float = 0.35


@dataclass
class Decision:
    chosen: str
    intervened: bool
    note: str
    compare: list[str] = field(default_factory=list)   # modules to run and compare, if any


class Referee:
    def __init__(self, goal: Goal | None = None):
        self.goal = goal or Goal()

    def decide(self, bids: list[Bid]) -> Decision:
        live = [b for b in bids if b.module not in self.goal.forbid]
        inhibited = [b.module for b in bids if b.module in self.goal.forbid]
        if not live:
            return Decision(chosen="", intervened=True, note="all candidates forbidden")
        live.sort(key=lambda b: b.confidence, reverse=True)
        top = live[0]
        note_parts = [f"inhibited {inhibited}"] if inhibited else []

        if top.confidence < self.goal.novelty_floor:
            # nobody wants it: run the top two and let verification decide
            cmp = [b.module for b in live[:2]]
            return Decision(chosen=top.module, intervened=True,
                            note="; ".join(note_parts + [f"novelty: top={top.confidence:.2f} < floor"]),
                            compare=cmp)

        if len(live) > 1 and (top.confidence - live[1].confidence) < self.goal.conflict_margin:
            cmp = [b.module for b in live[:2]]
            return Decision(chosen=top.module, intervened=True,
                            note="; ".join(note_parts + [f"conflict: {live[0].module} vs {live[1].module}"]),
                            compare=cmp)

        return Decision(chosen=top.module, intervened=bool(inhibited),
                        note="; ".join(note_parts) or "contention scheduling")

    def resolve_compare(self, results: dict[str, Result], bids: dict[str, Bid]) -> tuple[str, str]:
        """After running compared modules: verified beats unverified; else higher bid."""
        if self.goal.prefer_verified:
            verified = [m for m, r in results.items() if r.verified is True]
            if verified:
                return verified[0], "compare: verified result preferred"
            failed = {m for m, r in results.items() if r.verified is False}
            live = [m for m in results if m not in failed]
            if live:
                best = max(live, key=lambda m: bids[m].confidence)
                why = f"compare: {sorted(failed)} failed verification, " if failed else "compare: nothing verifiable, "
                return best, why + "kept top bid"
        best = max(results, key=lambda m: bids[m].confidence)
        return best, "compare: no verification available, kept top bid"
