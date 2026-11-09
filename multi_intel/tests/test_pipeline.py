"""python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mi import (CodeModule, HabitLayer, MultiIntelSystem, Outcome, ReasoningModule,  # noqa: E402
                RetrievalModule, Stimulus, make_backend)
from mi.classifier import signature_of  # noqa: E402
from experiments.tasks import CORPUS  # noqa: E402


def build(habit_on=True):
    return MultiIntelSystem([RetrievalModule(CORPUS), CodeModule(), ReasoningModule(make_backend("mock"), k=3)],
                            habit=HabitLayer(openness=0.0, enabled=habit_on))


class PipelineTests(unittest.TestCase):
    def test_arithmetic_goes_to_verified_code(self):
        ep = build().cycle(Stimulus("계산해줘: 41 * 7", expected=287))
        self.assertEqual(ep.chosen, "code")
        self.assertEqual(ep.outcome, Outcome.SUCCESS)
        self.assertTrue(ep.result.verified)

    def test_lookup_has_ground_truth(self):
        ep = build().cycle(Stimulus("예외 상황에서만 개입하는 체계는?", expected="doc_sas"))
        self.assertIn(ep.outcome, (Outcome.SUCCESS, Outcome.FAILURE))

    def test_open_question_is_unknown_and_teaches_nothing(self):
        s = build()
        before = dict(s.estimator.n_updates)
        ep = s.cycle(Stimulus("is a toolbox an entity?"))
        self.assertEqual(ep.outcome, Outcome.UNKNOWN)
        self.assertEqual(before, s.estimator.n_updates)

    def test_habit_prior_moves_with_outcomes(self):
        s = build()
        st = Stimulus("계산해줘: 10 + 10", expected=20)
        sig = signature_of(st)
        p0 = s.habit.prior(sig, "code")
        for _ in range(5):
            s.cycle(st)
        self.assertGreater(s.habit.prior(sig, "code"), p0)

    def test_classifier_excludes_code_for_prose(self):
        ep = build().cycle(Stimulus("작업기억이 뭐야?", expected="doc_wm"))
        self.assertNotIn("code", ep.candidates)


if __name__ == "__main__":
    unittest.main()
