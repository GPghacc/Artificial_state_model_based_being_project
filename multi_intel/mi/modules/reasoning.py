"""Reasoning module. Psychological seat: fluid reasoning (Gf) / Type 2 processing.

This is the module whose cues are the *bad* kind. A language model's self-report of
confidence is poorly calibrated, so we never ask for it. Instead we use
self-consistency: sample k answers at nonzero temperature and measure agreement.
Convergence is a Koriat-style cue that is cheap relative to the work and far
better calibrated than introspection. It still inherits fluency illusions.
"""
from __future__ import annotations

import re
from collections import Counter

from ..backend import Backend
from ..types import Cues, Result, Stimulus
from .base import Module


class ReasoningModule(Module):
    name = "reasoning"
    cue_names = ("agreement", "decisiveness")

    def __init__(self, backend: Backend, k: int = 3, temperature: float = 0.8):
        self.backend = backend
        self.k = k
        self.temperature = temperature
        self._cache: dict[str, list[str]] = {}

    def _samples(self, s: Stimulus) -> list[str]:
        # cue computation and run share the same samples; do not pay twice
        if s.text not in self._cache:
            prompt = ("Reply with ONLY the final answer. No explanation, no units, "
                      "no punctuation, no markdown.\n\n" + s.text)
            self._cache[s.text] = [_normalize(self.backend.generate(prompt, self.temperature))
                                   for _ in range(self.k)]
        return self._cache[s.text]

    def cues(self, s: Stimulus) -> Cues:
        samples = self._samples(s)
        counts = Counter(samples)
        top, n_top = counts.most_common(1)[0]
        agreement = n_top / len(samples)
        # decisiveness: short, answer-shaped output vs hedging prose
        decisiveness = 1.0 if len(top) <= 12 else max(0.0, 1.0 - len(top) / 200.0)
        return Cues(values={"agreement": agreement, "decisiveness": decisiveness}, cost=float(self.k))

    def run(self, s: Stimulus) -> Result:
        samples = self._samples(s)
        answer = Counter(samples).most_common(1)[0][0]
        self._cache.pop(s.text, None)
        return Result(module=self.name, answer=answer, detail={"samples": samples})


def _normalize(raw: str) -> str:
    """Terse-answer normalization.

    Instruct-tuned models wrap answers in prose and punctuation even when told not
    to. Without this, "168" and "168." count as disagreement and the self-consistency
    cue measures formatting noise instead of convergence. This normalizes form only;
    it never changes which answer was given.
    """
    t = (raw or "").strip()
    t = re.sub(r"^```.*?\n|```$", "", t, flags=re.S).strip()   # code fences
    t = t.replace("*", "").replace("`", "").strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        t = lines[-1]                                          # final line carries the answer
    t = re.sub(r"^(the\s+)?answer\s*(is)?\s*[:\-]?\s*", "", t, flags=re.I)
    t = t.strip().rstrip(".!。").strip()
    m = re.fullmatch(r"[^\d\-]*(-?\d[\d,]*)[^\d]*", t)      # bare number buried in a phrase
    if m:
        t = m.group(1).replace(",", "")
    return t
