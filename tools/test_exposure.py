import copy
import random
import unittest
from agent_core.exposure import Curve, choose, optimize
from agent_core.geometry import Moon, local_sidereal_deg, radec_to_altaz
from agent_core.planner import Planner
from agent_core.state import SurveyState
from test_strategy import payload


class ExposureTests(unittest.TestCase):
    def curve(self, i, k, weight=1, previous=0, penalty=0, goal=.6, up=3600):
        return Curve(i, k, up, weight, previous, penalty, goal, 1, 60, 90)

    def test_long_exposure_winner_can_lose_short_exposure_fibre(self):
        state = SurveyState(payload())
        slow = self.curve(0, .0002, 10)
        fast = self.curve(1, .01)
        self.assertGreater(slow.gain(3600, 1.2), fast.gain(3600, 1.2))
        result = optimize({0: [slow, fast]}, ("DARK",), state.scoring, 1, 60, 3600)
        self.assertEqual(result[3][0].i, 1)
        self.assertEqual(result[1], 100)

    def test_envelopes_match_integer_brute_force(self):
        rng = random.Random(173)
        scoring = SurveyState(payload()).scoring
        for case in range(32):
            cells = {fiber: [self.curve(fiber * 4 + j, rng.uniform(.001, .04), rng.uniform(.1, 5),
                                      rng.uniform(0, 1), 50 if rng.random() < .15 else 0,
                                      rng.uniform(.3, .9), rng.uniform(25, 160)) for j in range(4)]
                     for fiber in range(3)}
            trial = optimize(cells, ("DARK",), scoring, 1, 10, 160)
            multipliers = {f: [1.2] * len(curves) for f, curves in cells.items()}
            brute = max(choose(cells, multipliers, t)[0] / t for t in range(10, 161))
            self.assertAlmostEqual(trial[0], brute, places=9, msg=str(case))

    def test_repeated_exposure_only_gains_above_best_score(self):
        scoring = SurveyState(payload()).scoring
        target = self.curve(0, .001, 1, previous=.9)
        result = optimize({0: [target]}, ("DARK",), scoring, 1, 60, 3600)
        self.assertEqual(result[1], 1000)
        self.assertAlmostEqual(result[0], .3 / 1000)

    def test_threshold_and_setting_are_integer_safe(self):
        scoring = SurveyState(payload()).scoring
        target = self.curve(0, .007, penalty=50, goal=.5, up=71.9)
        result = optimize({0: [target]}, ("DARK",), scoring, 1, 60, 100)
        self.assertLessEqual(result[1], 71)
        self.assertLess(result[0], 1)
        target = self.curve(0, .007, penalty=50, goal=.5, up=72.9)
        result = optimize({0: [target]}, ("DARK",), scoring, 1, 60, 100)
        self.assertEqual(result[1], 72)
        self.assertGreater(result[0], .6)

    def test_forced_request_requires_assignment_and_factor(self):
        scoring = SurveyState(payload()).scoring
        cells = {0: [self.curve(0, .001), self.curve(1, .1, 10)]}
        result = optimize(cells, ("DARK",), scoring, 1, 60, 900, forced=(0, .5))
        self.assertEqual(result[3][0].i, 0)
        self.assertGreaterEqual(result[1] * result[3][0].k, .5)
        self.assertIsNone(optimize(cells, ("DARK",), scoring, 1, 60, 300, forced=(0, .5)))

    def test_completed_science_request_uses_shortest_threshold_exposure(self):
        scoring = SurveyState(payload()).scoring
        cells = {0: [self.curve(0, .001, previous=1.2)]}
        result = optimize(cells, ("DARK",), scoring, 1, 60, 3600, forced=(0, .5))
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 500)

    def test_joint_trial_is_pure(self):
        state = SurveyState(payload())
        planner = Planner(state)
        tracked = ("pending", "pending_start", "pending_duration", "best_score", "factor", "ledger", "active")
        before = {k: copy.deepcopy(getattr(state, k)) for k in tracked}
        now = state.survey_start
        lst = local_sidereal_deg(now, state.lon)
        altaz = lambda i: radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
        alt, az = altaz(0)
        result = planner._evaluate_joint(now, lst, alt, az, {0: [(1, 0, .2)]}, 300,
                                         Moon(now, lst, state.lat), altaz)
        self.assertIsNotNone(result)
        self.assertLessEqual(result.action["duration_seconds"], 300)
        self.assertEqual(before, {k: getattr(state, k) for k in tracked})

    def test_expired_budget_returns_without_using_state(self):
        scoring = SurveyState(payload()).scoring
        self.assertIsNone(optimize({0: [self.curve(0, .001)]}, ("DARK",), scoring, 1, 60, 3600, deadline=0))

    def test_second_step_uses_prefix_gain_and_elapsed_time(self):
        scoring = SurveyState(payload()).scoring
        cells = {0: [self.curve(0, .003, 1, previous=.4), self.curve(1, .001, 2)]}
        result = optimize(cells, ("DARK",), scoring, 1, 60, 1500,
                          prefix_gain=8, prefix_seconds=700, discount=.9)
        multipliers = {0: [1.2, 1.2]}
        brute = max((8 + .9 * choose(cells, multipliers, t)[0]) / (700 + t) for t in range(60, 1501))
        self.assertAlmostEqual(result[0], brute, places=10)

    def test_shadow_changes_only_overlay_and_keeps_ambiguous_completion(self):
        state = SurveyState(payload())
        planner = Planner(state)
        now = state.survey_start
        lst = local_sidereal_deg(now, state.lon)
        altaz = lambda i: radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
        alt, az = altaz(0)
        exposure = planner._evaluate_pointing(now, lst, alt, az, {0: (1, 0, .2)}, 3600, Moon(now, lst, state.lat), altaz)
        before = copy.deepcopy(state.best_score), copy.deepcopy(state.factor), copy.deepcopy(state.ledger)
        overlay, gain = planner._shadow_progress(exposure, .8)
        self.assertGreater(gain, 0)
        self.assertTrue(overlay)
        self.assertEqual(before, (state.best_score, state.factor, state.ledger))
        for i, (score, lower) in overlay.items():
            self.assertLessEqual(lower, state.factor_bounds(i, score, exposure.action["program"])[0] + 1e-12)

    def test_lookahead_deadline_keeps_current_winner(self):
        state = SurveyState(payload())
        planner = Planner(state)
        self.assertEqual(planner._lookahead([(0, None), (0, None)], [0, 1], 1,
                         state.survey_start, state.nights[0][1], 0), 1)


if __name__ == "__main__":
    unittest.main()
