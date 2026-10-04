import copy
import math
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".local" / "runner"))
from agent_core.geometry import FiberGrid, format_utc, local_sidereal_deg, max_hour_angle_deg, parse_utc
from agent_core.planner import Planner
from agent_core.state import ExposureRecord, PendingPrediction, SurveyState
from agent_core.validation import ActionRejected, fallback_action, validate_action
from challenge.v4_fiber_map import FiberGrid as OfficialGrid


def payload(n=16, gap=0, threshold=0.5, penalty=50):
    now = parse_utc("2026-11-01T00:00:00Z")
    side = math.isqrt(n)
    glass = math.sqrt(0.4)
    return {"site": {"latitude_deg": -24.6157, "longitude_deg": -70.3976},
            "survey": {"start_utc": format_utc(now), "end_utc": format_utc(now + timedelta(days=3)), "slot_seconds": 900,
                       "nights": [{"observing_start_utc": format_utc(now + timedelta(days=d)),
                                   "observing_end_utc": format_utc(now + timedelta(days=d, hours=8))} for d in range(3)]},
            "instrument": {"grid_side": side, "n_fibers": n, "glass_side_deg": glass, "pitch_deg": glass + gap,
                           "fov_side_deg": side * glass + (side - 1) * gap,
                           "exposure": {"min_duration_seconds": 60, "max_duration_seconds": 3600}},
            "scoring": {"required": {"observed_factor_threshold": threshold, "penalty_per_missing": penalty}},
            "targets": {"columns": ["target_id", "ra_deg", "dec_deg", "feature_flux", "science_weight", "required"],
                        "rows": [["a", local_sidereal_deg(now, -70.3976), -10, 1, 1, True],
                                 ["b", local_sidereal_deg(now, -70.3976) + 1, -11, 1, 1, False]]}}


class StrategyTests(unittest.TestCase):
    def test_linear_marginal_gain_uses_best_score(self):
        state = SurveyState(payload())
        state.required[0] = False
        state.best_score[0] = 0.4
        planner = Planner(state)
        self.assertAlmostEqual(planner._gain(0, 0.5, "DARK", "DARK"), 0.2)
        self.assertEqual(planner._gain(0, 0.2, "DARK", "DARK"), 0)

    def test_custom_required_threshold_and_penalty(self):
        state = SurveyState(payload(threshold=0.7, penalty=73))
        planner = Planner(state)
        self.assertLess(planner._gain(0, 0.6), 73)
        self.assertGreater(planner._gain(0, 0.9), 73)

    def test_unusual_program_multipliers_use_current_card(self):
        init = payload()
        init["scoring"]["program"] = {"mismatch_multiplier": 1.3}
        state = SurveyState(init)
        state.required[0] = False
        planner = Planner(state)
        self.assertAlmostEqual(planner._gain(0, 1), 1.3)

    def test_ambiguous_factor_is_not_exact(self):
        state = SurveyState(payload(threshold=0.55))
        lo, hi = state.factor_bounds(0, 0.6, "DARK")
        self.assertLess(lo, .5)
        self.assertGreater(hi, .6)
        lo, hi = state.factor_bounds(0, 1.2, "DARK")
        self.assertGreater(lo, .999999)
        self.assertEqual(hi, 1)

    def test_repeated_scores_do_not_accumulate(self):
        state = SurveyState(payload())
        for action in range(2):
            state.pending = {"a": PendingPrediction(1, 1, 70, 90, True)}
            state.pending_action_index = action
            state.pending_program = "DARK"
            state.pending_duration = 300
            state.on_result({"action": "observe", "hits": [{"target_id": "a", "score": 0.36}]}, 1 + action)
        self.assertAlmostEqual(state.best_score[0], 0.36)
        self.assertAlmostEqual(state.factor[0], 0.3, places=6)

    def test_zero_score_keeps_rounding_range_and_valid_exposure(self):
        state = SurveyState(payload())
        lo, hi = state.factor_bounds(0, 0, "DARK")
        self.assertEqual(lo, 0)
        self.assertGreater(hi, 0)
        state.pending = {"a": PendingPrediction(1, 1, 70, 90, True)}
        state.pending_action_index = 0
        state.pending_duration = 60
        state.on_result({"action": "observe", "hits": [{"target_id": "a", "score": 0}]}, 1)
        self.assertEqual(len(state.ledger), 1)
        self.assertEqual(state.factor[0], 0)

    def test_resync_removes_only_invalidated_actions(self):
        state = SurveyState(payload())
        state.ledger = [ExposureRecord(0, "a", 0.6, .5, .6, .5, "", ""),
                        ExposureRecord(1, "a", 0.3, .25, .3, .25, "", ""),
                        ExposureRecord(2, "b", 0.72, .6, .72, .6, "", "")]
        state._resync({"invalidated_window": {"action_index_start": 0, "action_index_end_exclusive": 1},
                       "best_scores": [{"target_id": "a", "best_score": .3}, {"target_id": "b", "best_score": .72}]})
        self.assertEqual([r.action_index for r in state.ledger], [1, 2])
        self.assertEqual(state.best_score, [.3, .72])
        self.assertEqual(state.factor, [.25, .6])

    def test_last_result_survives_same_round_resync(self):
        state = SurveyState(payload())
        state.pending = {"b": PendingPrediction(1, 1, 70, 90, True)}
        state.pending_action_index = 2
        state.pending_duration = 300
        planner = Planner(state)
        now = state.survey_start + timedelta(seconds=300)
        message = {"now_utc": format_utc(now), "wallclock": {"remaining_seconds": 0}, "observe_action_index": 3,
                   "last_result": {"action": "observe", "hits": [{"target_id": "b", "score": .72}]},
                   "new_messages": [{"record_type": "state_resync", "invalidated_window": {
                       "action_index_start": 0, "action_index_end_exclusive": 1},
                       "best_scores": [{"target_id": "b", "best_score": .72}]}]}
        with patch.object(planner, "plan", return_value=None):
            planner.decide(message)
        self.assertEqual(len(state.ledger), 1)
        self.assertEqual(state.ledger[0].action_index, 2)

    def test_truncated_duration_uses_actual_time(self):
        state = SurveyState(payload())
        state.pending = {"a": PendingPrediction(1, 1, 70, 90, True)}
        state.pending_action_index = 0
        state.pending_duration = 900
        state.pending_start = state.survey_start
        state.on_result({"action": "observe", "hits": [{"target_id": "a", "score": .3}]}, 200 / 3600)
        self.assertAlmostEqual(state.pending_duration, 200)

    def test_grid_matches_official_geometry(self):
        for n in (9, 16, 25, 100):
            for gap in (0, .05):
                grid = FiberGrid(payload(n, gap)["instrument"])
                official = OfficialGrid(.4, gap, n)
                for fiber in range(n):
                    point = grid.fiber_center(fiber)
                    self.assertEqual(grid.classify(*point)[0], official.classify_offset(*point)[0])
                point = (0, grid.fov / 2)
                expected, region = official.classify_offset(*point)
                self.assertEqual(grid.classify(*point)[0], expected if region == "glass" else None)

    def test_polar_visibility(self):
        self.assertEqual(max_hour_angle_deg(80, 90, 30), 180)
        self.assertEqual(max_hour_angle_deg(-80, 90, 30), 0)

    def test_polar_neighbours_cross_all_right_ascensions(self):
        init = payload()
        init["site"]["latitude_deg"] = 80
        init["targets"]["rows"][0][1:3] = [0, 89]
        init["targets"]["rows"][1][1:3] = [180, 89]
        state = SurveyState(init)
        self.assertEqual(set(state.neighbours(0, 89, 3)), {0, 1})

    def test_fallback_obeys_unusual_exposure_limits(self):
        init = payload()
        init["instrument"]["exposure"] = {"min_duration_seconds": 100, "max_duration_seconds": 200}
        state = SurveyState(init)
        self.assertEqual(validate_action(fallback_action("test", state), state)["duration_seconds"], 200)
        with self.assertRaises(ActionRejected):
            validate_action({"action": "wait", "duration_seconds": 100.5}, state)

    def test_urgency_does_not_promote_an_unreachable_required_threshold(self):
        from agent_core.geometry import Moon
        from experiments.stage3.state import SurveyState as ExperimentalState
        from experiments.stage3.planner import Planner as ExperimentalPlanner
        state = ExperimentalState(payload())
        planner = ExperimentalPlanner(state)
        now = state.survey_start
        moon = Moon(now, local_sidereal_deg(now, state.lon), state.lat)
        state.scale = .01
        self.assertEqual(planner._required_window_boost(0, 70, moon, 100, 100, 1), 1)
        state.scale = 1
        self.assertGreater(planner._required_window_boost(0, 70, moon, 500, 500, 1), 1)

    def test_plans_use_card_grid_site_and_exposure_limits(self):
        for n in (9, 16, 25, 100):
            init = payload(n, .08)
            init["site"] = {"latitude_deg": 35, "longitude_deg": 115}
            init["instrument"]["exposure"] = {"min_duration_seconds": 100, "max_duration_seconds": 180}
            now = parse_utc(init["survey"]["start_utc"])
            init["targets"]["rows"][0][1:3] = [local_sidereal_deg(now, 115), 45]
            init["targets"]["rows"][1][1:3] = [local_sidereal_deg(now, 115) + .5, 45.5]
            state = SurveyState(init)
            planner = Planner(state)
            action = planner.plan(now, state.nights[0][1], 0, 0)
            self.assertIsNotNone(action)
            checked = validate_action(action, state)
            self.assertTrue(100 <= checked["duration_seconds"] <= 180)
            self.assertTrue(all(int(f) < n for f in checked["assignments"]))


if __name__ == "__main__":
    unittest.main()
