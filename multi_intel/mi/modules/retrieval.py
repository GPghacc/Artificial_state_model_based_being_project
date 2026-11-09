"""Retrieval module. Psychological seat: semantic memory + retrieval fluency (CHC Gr).

Its cues are the *good* kind: similarity scores are an external, cheap signal that
is already produced by the work itself. Retrieval fluency ("how fast and how much
comes back") is exactly Koriat's accessibility cue.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..types import Cues, Result, Stimulus
from .base import Module

_TOKEN = re.compile(r"[가-힣]+|[a-z0-9]+")


def _tokens(t: str) -> list[str]:
    toks = _TOKEN.findall(t.lower())
    # crude Korean handling: also add 2-grams of syllables so particles hurt less
    out = []
    for tok in toks:
        out.append(tok)
        if re.match(r"[가-힣]{3,}", tok):
            out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
    return out


def _vec(t: str) -> Counter:
    return Counter(_tokens(t))


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a if k in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class RetrievalModule(Module):
    name = "retrieval"
    cue_names = ("top_sim", "margin", "hits")

    def __init__(self, corpus: list[tuple[str, str]]):
        self.corpus = [(doc_id, text, _vec(text)) for doc_id, text in corpus]

    def _rank(self, s: Stimulus):
        q = _vec(s.text)
        scored = sorted(((_cos(q, v), doc_id) for doc_id, _, v in self.corpus), reverse=True)
        return scored

    def cues(self, s: Stimulus) -> Cues:
        scored = self._rank(s)
        top = scored[0][0] if scored else 0.0
        second = scored[1][0] if len(scored) > 1 else 0.0
        hits = sum(1 for sc, _ in scored if sc > 0.2)
        return Cues(values={
            "top_sim": top,
            "margin": max(0.0, top - second),
            "hits": min(1.0, hits / 3.0),
        }, cost=0.01)

    def run(self, s: Stimulus) -> Result:
        scored = self._rank(s)
        best = scored[0][1] if scored else None
        return Result(module=self.name, answer=best, detail={"ranked": scored[:3]})
