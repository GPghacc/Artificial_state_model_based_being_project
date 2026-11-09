"""Wiring. One cycle:

  stimulus
    -> classifier      : signature + candidate modules          (cheap, modality only)
    -> each candidate  : cues                                    (module-side, cheap)
    -> estimator       : cue_confidence per module               (meta level)
    -> habit layer     : prior per module, combined into a bid   (striatum / K-lines)
    -> referee         : contention scheduling, or intervene     (SAS)
    -> winner.run()    : result                                  (the actual work)
    -> outcome         : ground truth if available, else UNKNOWN
    -> learn           : estimator + habit update only on known outcomes
    -> workspace       : bind everything into one Episode
"""
from __future__ import annotations

from .classifier import candidates_for, signature_of
from .confidence import ConfidenceEstimator
from .habit import HabitLayer
from .modules import Module
from .referee import Decision, Referee
from .types import Bid, Episode, Outcome, Result, Stimulus
from .workspace import Workspace


class MultiIntelSystem:
    def __init__(self, modules: list[Module], referee: Referee | None = None,
                 habit: HabitLayer | None = None, estimator: ConfidenceEstimator | None = None):
        self.modules = {m.name: m for m in modules}
        self.referee = referee or Referee()
        self.habit = habit or HabitLayer()
        self.estimator = estimator or ConfidenceEstimator({m.name: m.cue_names for m in modules})
        self.workspace = Workspace()

    # -- one cycle ------------------------------------------------------
    def cycle(self, s: Stimulus) -> Episode:
        sig = signature_of(s)
        cands = candidates_for(sig, list(self.modules))
        exploring = self.habit.begin_cycle()

        bids: list[Bid] = []
        for name in cands:
            cues = self.modules[name].cues(s)
            cue_conf = self.estimator.predict(name, cues)
            prior = self.habit.prior(sig, name)
            w = 0.0 if exploring else self.habit.weight(sig, name)
            combined = (1 - w) * cue_conf + w * prior
            bids.append(Bid(module=name, cues=cues, cue_confidence=cue_conf,
                            habit_prior=prior, confidence=combined))

        decision: Decision = self.referee.decide(bids)
        by_name = {b.module: b for b in bids}
        note = decision.note
        if decision.compare:
            results = {m: self.modules[m].run(s) for m in decision.compare}
            chosen, why = self.referee.resolve_compare(results, by_name)
            note = f"{note} -> {why}"
            result = results[chosen]
            # losers of a comparison still yield learning signal
            for m, r in results.items():
                if m != chosen:
                    self._learn(sig, m, by_name[m], self._outcome(s, m, r))
        else:
            chosen = decision.chosen
            result = self.modules[chosen].run(s)

        outcome = self._outcome(s, chosen, result)
        self._learn(sig, chosen, by_name[chosen], outcome)

        ep = Episode(stimulus=s, signature=sig, candidates=cands, bids=bids,
                     referee_intervened=decision.intervened, referee_note=note,
                     chosen=chosen, result=result, outcome=outcome)
        self.workspace.bind(ep)
        return ep

    # -- helpers --------------------------------------------------------
    def _outcome(self, s: Stimulus, module: str, r: Result) -> Outcome:
        judged = self.modules[module].judge(s, r)
        if judged is None:
            # no ground truth: the only thing we can trust is a module's own verification
            if r.verified is False:
                return Outcome.FAILURE
            return Outcome.UNKNOWN
        return Outcome.SUCCESS if judged else Outcome.FAILURE

    def _learn(self, sig, module: str, bid: Bid, outcome: Outcome) -> None:
        if outcome == Outcome.UNKNOWN:
            return
        ok = outcome == Outcome.SUCCESS
        self.estimator.update(module, bid.cues, ok)
        self.habit.record(sig, module, ok)
