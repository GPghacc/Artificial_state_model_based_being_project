"""Cheap surface classifier. Psychological seat: Posner's orienting network /
sensory cortex. It sees *modality*, never *task*. Its only job is to produce a
Signature and a list of candidate modules worth asking for bids.

Deliberately dumb. Anything that requires understanding the content belongs to
the modules (their cue computation) and not here.
"""
from __future__ import annotations

import re

from .types import Signature, Stimulus

_CODE_HINTS = re.compile(r"(def |class |import |print\(|=>|;\s*$|```|\breturn\b|\bfor\b.*\bin\b.*:)", re.M)
_MATH_HINTS = re.compile(r"\d+\s*[+\-*/^]\s*\d+|\b(sum|product|계산|더하|곱하|빼)\b", re.I)
_HANGUL = re.compile(r"[가-힣]")


def signature_of(s: Stimulus) -> Signature:
    t = s.text
    n = len(t)
    ko = bool(_HANGUL.search(t))
    en = bool(re.search(r"[A-Za-z]{3,}", t))
    return Signature(
        has_code=bool(_CODE_HINTS.search(t)),
        has_math=bool(_MATH_HINTS.search(t)),
        is_question=t.strip().endswith("?") or bool(re.search(r"(뭐|무엇|왜|어떻게|what|why|how)\b", t, re.I)),
        length_bucket="short" if n < 40 else "medium" if n < 200 else "long",
        lang="mixed" if ko and en else "ko" if ko else "en",
    )


def candidates_for(sig: Signature, all_modules: list[str]) -> list[str]:
    """Prune by modality only. When nothing is excluded, everyone bids."""
    out = list(all_modules)
    if not sig.has_code and "code" in out:
        # code module still allowed for math (it can compute); excluded only for pure prose
        if not sig.has_math:
            out.remove("code")
    return out
