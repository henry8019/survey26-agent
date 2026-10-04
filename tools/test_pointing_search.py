import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_core.geometry import Moon, local_sidereal_deg, radec_to_altaz
from agent_core.planner import Planner
from agent_core.state import PendingPrediction, SurveyState
from agent_core.validation import validate_action
from test_strategy import payload


class PointingSearchTests(unittest.TestCase):
    def test_smaller_field_with_higher_rate_beats_larger_slow_field(self):
        init = payload()
        ra = init["targets"]["rows"][0][1]
        init["targets"]["rows"] = [["slow", ra - 15, -10, .1, 10, False],
                                    ["fast", ra + 15, -10, 1, 5, False]]
        state = SurveyState(init)
        planner = Planner(state)
        action = planner.plan(state.survey_start, state.nights[0][1], 0, 0)
        validate_action(action, state)
        self.assertIn("fast", action["assignments"].values())
        self.assertNotIn("slow", action["assignments"].values())
        self.assertEqual(set(state.pending), {"fast"})

    def test_candidate_trial_leaves_feedback_and_progress_untouched(self):
        state = SurveyState(payload())
        planner = Planner(state)
        state.pending = {"previous": PendingPrediction(.2, .3, 60, 90, True)}
        state.pending_start, state.pending_duration = state.survey_start, 120
        state.pending_action_index, state.pending_program, state.pending_night = 17, "BACKUP", 1
        tracked = ("pending", "pending_start", "pending_duration", "pending_action_index", "pending_program",
                   "pending_night", "best_score", "factor", "factor_upper", "ledger", "active")
        before = {k: copy.deepcopy(getattr(state, k)) for k in tracked}
        now = state.survey_start
        lst = local_sidereal_deg(now, state.lon)
        altaz = lambda i: radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
        alt, az = altaz(0)
        result = planner._evaluate_pointing(now, lst, alt, az, {0: (1, 0, .2)}, 3600, Moon(now, lst, state.lat), altaz)
        self.assertIsNotNone(result)
        self.assertEqual(before, {k: getattr(state, k) for k in tracked})
        action = planner._commit_exposure(result, now, 0)
        self.assertEqual(action, result.action)
        self.assertEqual(set(state.pending), set(action["assignments"].values()))

    def test_required_threshold_reward_is_included_in_field_rate(self):
        init = payload()
        ra = init["targets"]["rows"][0][1]
        init["targets"]["rows"] = [["required", ra - 15, -10, .5, .1, True],
                                    ["science", ra + 15, -10, 1, 5, False]]
        state = SurveyState(init)
        planner = Planner(state)
        action = planner.plan(state.survey_start, state.nights[0][1], 0, 0)
        self.assertIn("required", action["assignments"].values())
        self.assertGreaterEqual(planner._required_completion_predictions(action, state.survey_start)[0],
                                state.scoring.required_threshold)

    def test_higher_science_rate_cannot_discard_original_required_completion(self):
        init = payload()
        ra = init["targets"]["rows"][0][1]
        init["targets"]["rows"] = [["required", ra - 15, -10, .15, .1, True],
                                    ["fast", ra + 15, -10, 100, 2, False]]
        state = SurveyState(init)
        planner = Planner(state)
        action = planner.plan(state.survey_start, state.nights[0][1], 0, 0)
        self.assertIn("required", action["assignments"].values())
        self.assertEqual(set(state.pending), {"required"})

    def test_quality_anomaly_preserves_original_pointing_policy(self):
        init = payload()
        ra = init["targets"]["rows"][0][1]
        init["targets"]["rows"] = [["slow", ra - 15, -10, .1, 10, False],
                                    ["fast", ra + 15, -10, 1, 5, False]]
        state = SurveyState(init)
        planner = Planner(state)
        with patch.object(state, "fault_evidence", return_value=SimpleNamespace(drop=.64)):
            action = planner.plan(state.survey_start, state.nights[0][1], 0, 0)
        self.assertIn("slow", action["assignments"].values())
        self.assertEqual(set(state.pending), {"slow"})


if __name__ == "__main__":
    unittest.main()
