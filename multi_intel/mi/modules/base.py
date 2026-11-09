"""Module interface.

A module is a *tool*, not a being. It must do two things:
  cues(stimulus)  -> cheap signals correlated with its own success (its metacognitive cues)
  run(stimulus)   -> actually do the work

The split matters: cues() is what the bidding phase calls on every candidate, so it
must be cheap. run() is called only for the winner (or the top few if the referee
decides to compare).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Cues, Result, Stimulus


class Module(ABC):
    name: str = "base"
    # names of the cues this module emits; the confidence estimator keys weights on these
    cue_names: tuple[str, ...] = ()

    @abstractmethod
    def cues(self, s: Stimulus) -> Cues: ...

    @abstractmethod
    def run(self, s: Stimulus) -> Result: ...

    def judge(self, s: Stimulus, r: Result) -> bool | None:
        """Compare against ground truth when the stimulus carries it. None = unknown.

        A task may supply its own grader in ``meta["grader"]``. This matters when
        modules answer in different formats: a retrieval module returns a document
        id while a reasoning module returns prose. Grading both by string equality
        would score a correct prose answer as wrong and teach the confidence
        estimator from an artifact of the scoring, not from real failure.
        """
        if s.expected is None:
            return None
        grader = s.meta.get("grader")
        if grader is not None:
            return bool(grader(r.answer, s.expected))
        return _norm(r.answer) == _norm(s.expected)


def _norm(x) -> str:
    return str(x).strip().lower()
