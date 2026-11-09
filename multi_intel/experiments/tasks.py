"""Labeled task generator for the calibration experiment.

Three families, chosen so cue quality differs on purpose:
  arithmetic  - verifiable; code module should dominate, reasoning is a fallible rival
  lookup      - retrieval has ground truth (doc id); similarity cues are external
  open        - no ground truth at all -> Outcome.UNKNOWN; tests that nothing learns from noise
"""
from __future__ import annotations

import random

from mi.modules.retrieval import _cos, _vec
from mi.types import Stimulus

CORPUS = [
    ("doc_gf", "유동추론은 새로운 문제를 귀납과 연역으로 푸는 능력이다. fluid reasoning solves novel problems"),
    ("doc_gc", "결정지능은 문화에 축적된 언어와 지식이다. crystallized knowledge vocabulary facts"),
    ("doc_wm", "작업기억은 정보를 붙잡고 조작하는 용량이다. working memory holds and manipulates information"),
    ("doc_ef", "실행기능은 억제 전환 갱신으로 나뉜다. executive function inhibition shifting updating"),
    ("doc_ep", "일화기억은 언제 어디서 겪은 사건의 기억이다. episodic memory remembers events in context"),
    ("doc_sas", "감독주의체계는 예외 상황에서만 개입한다. supervisory attentional system intervenes on novelty"),
    ("doc_kl", "K-라인은 과거의 유용한 에이전트 조합을 다시 켠다. K-lines reactivate useful agent sets"),
    ("doc_gw", "전역 작업공간은 선택된 내용을 모든 프로세서에 방송한다. global workspace broadcasts"),
]

_LOOKUP_QUERIES = [
    ("doc_gf", "새로운 문제를 추론으로 푸는 능력은?"),
    ("doc_gf", "what solves novel problems by reasoning?"),
    ("doc_gc", "축적된 언어 지식을 뭐라고 하나?"),
    ("doc_wm", "정보를 붙잡고 조작하는 용량은?"),
    ("doc_wm", "which memory holds and manipulates information?"),
    ("doc_ef", "억제 전환 갱신은 무엇의 세 요소인가?"),
    ("doc_ep", "언제 어디서 겪은 사건의 기억은?"),
    ("doc_sas", "예외 상황에서만 개입하는 체계는?"),
    ("doc_kl", "과거의 에이전트 조합을 다시 켜는 것은?"),
    ("doc_gw", "선택된 내용을 모든 프로세서에 방송하는 것은?"),
    ("doc_gw", "what broadcasts to every processor?"),
]

_OPEN = [
    "이 설계가 루나보다 나은가?",
    "심판이 내용을 몰라도 되는 이유는 무엇인가?",
    "is a toolbox an entity?",
    "습관 층이 보수적으로 굳는 걸 막을 방법은?",
]


_DOC_VECS = {doc_id: _vec(text) for doc_id, text in CORPUS}


def grade_lookup(answer, expected) -> bool:
    """Judge a lookup answer at the level of the concept, not the string.

    Accepts the document id directly, or prose that matches that document better
    than any other. Both modules face the same yardstick.
    """
    if answer is None:
        return False
    a = str(answer).strip()
    if a in _DOC_VECS:
        return a == expected
    v = _vec(a)
    scored = sorted(((_cos(v, dv), d) for d, dv in _DOC_VECS.items()), reverse=True)
    if not scored or scored[0][0] < 0.05:
        return False
    return scored[0][1] == expected


def make_tasks(n: int, seed: int = 0, mix=(0.5, 0.3, 0.2)) -> list[Stimulus]:
    rng = random.Random(seed)
    out: list[Stimulus] = []
    for _ in range(n):
        r = rng.random()
        if r < mix[0]:
            a, b = rng.randint(1, 300), rng.randint(1, 300)
            op = rng.choice(["+", "-", "*"])
            expected = {"+": a + b, "-": a - b, "*": a * b}[op]
            phrasing = rng.choice([f"{a} {op} {b} = ?", f"계산해줘: {a} {op} {b}", f"what is {a} {op} {b}?"])
            out.append(Stimulus(text=phrasing, expected=expected, meta={"family": "arithmetic"}))
        elif r < mix[0] + mix[1]:
            doc, q = rng.choice(_LOOKUP_QUERIES)
            out.append(Stimulus(text=q, expected=doc,
                                meta={"family": "lookup", "grader": grade_lookup}))
        else:
            out.append(Stimulus(text=rng.choice(_OPEN), expected=None, meta={"family": "open"}))
    return out
