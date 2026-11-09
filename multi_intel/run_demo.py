"""Walk a few stimuli through one cycle each and print what every layer did.

  python run_demo.py            # mock backend
  python run_demo.py ollama llama3.1   # local Ollama (needs a running server)
  python run_demo.py anthropic  # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import sys

from mi import CodeModule, MultiIntelSystem, ReasoningModule, RetrievalModule, Stimulus, make_backend
from experiments.tasks import CORPUS, grade_lookup

backend_name = sys.argv[1] if len(sys.argv) > 1 else "mock"
model = sys.argv[2] if len(sys.argv) > 2 else None
backend = make_backend(backend_name, model=model) if model else make_backend(backend_name)
system = MultiIntelSystem([RetrievalModule(CORPUS), CodeModule(), ReasoningModule(backend, k=3)])

stimuli = [
    Stimulus("계산해줘: 123 + 45", expected=168),
    Stimulus("작업기억이 뭐야?", expected="doc_wm", meta={"grader": grade_lookup}),
    Stimulus("287 * 199 = ?", expected=57113),
    Stimulus("이 설계가 루나보다 나은가?"),          # no ground truth -> UNKNOWN
    Stimulus("what is 12 - 7?", expected=5),
    Stimulus("예외 상황에서만 개입하는 체계는?", expected="doc_sas", meta={"grader": grade_lookup}),
]

for s in stimuli:
    ep = system.cycle(s)
    print("-" * 70)
    print(f"stimulus : {s.text}")
    print(f"signature: {ep.signature}")
    print(f"candidates: {ep.candidates}")
    for b in sorted(ep.bids, key=lambda b: -b.confidence):
        cues = ", ".join(f"{k}={v:.2f}" for k, v in b.cues.values.items())
        print(f"  bid {b.module:<10} conf={b.confidence:.2f} (cue={b.cue_confidence:.2f}, prior={b.habit_prior:.2f})  [{cues}]")
    print(f"referee  : {'INTERVENED' if ep.referee_intervened else 'let it run'} — {ep.referee_note}")
    print(f"chosen   : {ep.chosen} -> {ep.result.answer!r}  verified={ep.result.verified}")
    print(f"outcome  : {ep.outcome.value}")

print("=" * 70)
print("workspace:", system.workspace.stats())
