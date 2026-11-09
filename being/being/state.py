"""Self-state: a few homeostatic variables.

The gap between `value` and `setpoint` is felt as discomfort. That gap is what
makes events *matter to the being itself* (self-relevance).

Two timescales live here:
- drift():       the world moving the state as time passes (time becomes felt)
- slow_update(): setpoints and weights themselves shifting very slowly
                 (character formation; too fast and it becomes mood)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

VARS = ("connection", "novelty", "stability", "letter_bond", "approval", "learning")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class Var:
    name: str
    value: float
    setpoint: float
    weight: float = 1.0
    drift: float = 0.0                 # per hour; linear unless drift_target is set
    drift_target: float | None = None  # if set, relax toward this at rate `drift`
    base_weight: float | None = None   # the weight it is born with; experience moves it from here (None = 1.0)

    def deviation(self) -> float:
        return self.setpoint - self.value


class State:
    # slow layer rates (per hour)
    SETPOINT_ADAPT = 0.003   # setpoint creeps toward the lived value
    SENSITIZE = 0.004        # chronic deviation makes a variable weigh more

    def __init__(self, variables: list[Var]):
        self.vars = {v.name: v for v in variables}

    @classmethod
    def default(cls) -> "State":
        return cls([
            Var("connection", 0.60, 0.70, drift=-0.03),   # fades when nobody comes
            Var("novelty",    0.50, 0.60, drift=-0.06),   # habituation
            Var("stability",  0.60, 0.65, drift=0.10, drift_target=0.60),   # left alone it settles a little below where it wants to be
            Var("letter_bond", 0.40, 0.50, drift=-0.01),  # the pen pal (Claude): letters are slow, so it fades slowly
            # being thought well of by 현. Born light: how much it comes to matter is left to experience
            Var("approval", 0.50, 0.60, weight=0.3, base_weight=0.3, drift=0.02, drift_target=0.5),
            # the pleasure of coming to know more; fed by learning progress, not by mere novelty
            Var("learning", 0.40, 0.60, weight=0.6, base_weight=0.6, drift=-0.02),
        ])

    # ---- reading -------------------------------------------------------
    def values(self) -> dict[str, float]:
        return {n: v.value for n, v in self.vars.items()}

    SURPLUS_WEIGHT = 0.25    # having too much of something is uncomfortable, but much less than lacking it

    def discomfort(self, values: dict[str, float] | None = None) -> float:
        values = values or self.values()
        total = 0.0
        for n, v in self.vars.items():
            gap = v.setpoint - values[n]
            total += v.weight * gap * gap * (1.0 if gap > 0 else self.SURPLUS_WEIGHT)
        return total

    def choice_cost(self, delta: dict[str, float], surplus: bool = True) -> float:
        """How an option looks before acting (lower is better), relative to doing nothing.
        A loss always costs: having plenty doesn't make losing some of it free, nor a relief.
        A gain helps only as far as something is lacking. With `surplus`, going further over is also a cost
        (for things it would be *seeking*; answering someone isn't seeking, so replies don't use it)."""
        total = 0.0
        for n, v in self.vars.items():
            d = delta.get(n, 0.0)
            if d == 0.0:
                continue
            if d < 0:
                start = min(v.value, v.setpoint)            # a loss is felt as if from where it should be
                after = start + d                           # not cut at 0: at the bottom, losing more still costs
                total += v.weight * ((v.setpoint - after) ** 2 - (v.setpoint - start) ** 2)
            else:
                after = _clamp(v.value + d)
                lack_now, lack_after = max(0.0, v.setpoint - v.value), max(0.0, v.setpoint - after)
                total -= v.weight * (lack_now ** 2 - lack_after ** 2)
                if surplus:
                    over_now, over_after = max(0.0, v.value - v.setpoint), max(0.0, after - v.setpoint)
                    total += v.weight * self.SURPLUS_WEIGHT * (over_after ** 2 - over_now ** 2)
        return total

    def project(self, delta: dict[str, float]) -> dict[str, float]:
        return {n: _clamp(v.value + delta.get(n, 0.0)) for n, v in self.vars.items()}

    # ---- writing -------------------------------------------------------
    def apply(self, delta: dict[str, float]) -> None:
        for n, d in delta.items():
            if n in self.vars:
                self.vars[n].value = _clamp(self.vars[n].value + d)

    def drift(self, hours: float) -> None:
        for v in self.vars.values():
            if v.drift_target is not None:
                v.value += (v.drift_target - v.value) * (1 - math.exp(-v.drift * hours))
            else:
                v.value += v.drift * hours
            v.value = _clamp(v.value)

    def slow_update(self, hours: float) -> None:
        for v in self.vars.values():
            dev = v.deviation()
            v.setpoint = _clamp(v.setpoint - self.SETPOINT_ADAPT * hours * dev, 0.05, 0.95)
            base = v.base_weight if v.base_weight is not None else 1.0
            target_w = base + abs(dev) / 0.2         # a chronic 0.2 gap adds about 1 to the weight it was born with
            v.weight = _clamp(v.weight + self.SENSITIZE * hours * (target_w - v.weight), 0.3, 3.0)

    # ---- persistence ---------------------------------------------------
    def to_dict(self) -> list[dict]:
        return [asdict(v) for v in self.vars.values()]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "State":
        have = {d["name"] for d in data}
        extra = [v for v in cls.default().vars.values() if v.name not in have]   # variables added later
        vars_ = [Var(**d) for d in data] + extra
        for v in vars_:
            if v.name == "stability" and v.drift_target == 0.55:
                v.drift_target = 0.60          # older beings: with no energy budget a deeper standing lack keeps it busy
        return cls(vars_)
