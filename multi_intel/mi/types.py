"""Core data types shared by every component.

Psychological mapping (see DESIGN.md):
  Stimulus   - the raw input arriving at the senses
  Signature  - surface features extracted by the cheap classifier (Posner's orienting network)
  Cues       - metacognitive cues a module can compute cheaply (Koriat's cue-utilization)
  Bid        - a module's claim "this is mine" with an estimated confidence
  Result     - what a module actually produced when run
  Outcome    - ground truth about whether the result was right (may be UNKNOWN)
  Episode    - one full cycle bound together (Baddeley's episodic buffer / global workspace)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class Outcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"  # no ground truth available -> no learning happens


@dataclass
class Stimulus:
    text: str
    expected: Optional[Any] = None       # ground truth, if the caller has it (experiments do)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Signature:
    """Surface features only. Deliberately cheap and shallow: modality-level, not task-level."""
    has_code: bool
    has_math: bool
    is_question: bool
    length_bucket: str          # "short" | "medium" | "long"
    lang: str                   # "ko" | "en" | "mixed"

    def key(self) -> tuple:
        return (self.has_code, self.has_math, self.is_question, self.length_bucket, self.lang)


@dataclass
class Cues:
    """Cheap, module-computed signals correlated with success. Values should be in [0, 1]."""
    values: dict[str, float]
    cost: float = 0.0           # how expensive it was to compute these cues (for later budgeting)


@dataclass
class Bid:
    module: str
    cues: Cues
    cue_confidence: float       # estimator output from cues alone
    habit_prior: float          # habit layer's prior from call history
    confidence: float           # combined score the referee sees


@dataclass
class Result:
    module: str
    answer: Any
    verified: Optional[bool] = None   # module-internal verification, when the module can do it
    detail: dict = field(default_factory=dict)


@dataclass
class Episode:
    stimulus: Stimulus
    signature: Signature
    candidates: list[str]
    bids: list[Bid]
    referee_intervened: bool
    referee_note: str
    chosen: str
    result: Result
    outcome: Outcome
    t: float = field(default_factory=time.time)
