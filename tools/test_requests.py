import sys
import unittest
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_core.geometry import format_utc
from agent_core.requests import RequestPlanner
from agent_core.state import ExposureRecord, PendingPrediction, SurveyState
from unittest.mock import patch
from test_advice import Trace
from test_strategy import payload
from agent_core.planner import Planner


class RequestTests(unittest.TestCase):
    def scheduler(self):
        state = SurveyState(payload())
        state.ledger = []
        request = {"request_id": "r", "target_ids": ["a", "b"], "minimum_completed": 2,
                   "completion_factor_threshold": .5, "completion_reward": 100,
                   "issued_at_utc": format_utc(state.survey_start + timedelta(seconds=60)),
                   "deadline_utc": format_utc(state.survey_start + timedelta(hours=2)), "completed_target_ids": []}
        scheduler = RequestPlanner(state, Trace())
        scheduler.sync([request])
        return scheduler, request

    def test_prepublication_and_late_exposures_do_not_count(self):
        scheduler, request = self.scheduler()
        state = scheduler.state
        for start, end in ((0, 300), (7000, 7500)):
            state.ledger.append(ExposureRecord(0, "a", 1, .9, 1, 1,
                                               format_utc(state.survey_start + timedelta(seconds=start)),
                                               format_utc(state.survey_start + timedelta(seconds=end))))
        self.assertEqual(scheduler.completed(request), set())

    def test_data_loss_rebuilds_progress_from_current_snapshot(self):
        scheduler, request = self.scheduler()
        request["completed_target_ids"] = ["a"]
        scheduler.sync([request])
        self.assertEqual(scheduler.completed(scheduler.requests["r"]), {"a"})
        scheduler.sync([{**request, "completed_target_ids": []}])
        self.assertEqual(scheduler.completed(scheduler.requests["r"]), set())

    def test_whole_bundle_reward_exceeds_opportunity_cost(self):
        scheduler, request = self.scheduler()
        self.assertIsNotNone(scheduler.choose(scheduler.state.survey_start, .001))
        self.assertIsNone(scheduler.choose(scheduler.state.survey_start, 10))

    def test_unreachable_group_is_rejected(self):
        scheduler, request = self.scheduler()
        request["deadline_utc"] = request["issued_at_utc"]
        scheduler.sync([request])
        self.assertIsNone(scheduler.choose(scheduler.state.survey_start, .001))

    def test_dedicated_target_is_not_cut_off_by_normal_anchor_pool(self):
        init = payload()
        a, b = init["targets"]["rows"]
        init["targets"]["rows"] = [[f"required{j}", *a[1:]] for j in range(320)] + [["request", *b[1:]]]
        state = SurveyState(init)
        planner = Planner(state)
        planner._forced_request = {"target": 320, "threshold": .5, "deadline": state.nights[0][1]}
        action = planner.plan(state.survey_start, state.nights[0][1], 0, 0)
        self.assertIsNotNone(action)
        self.assertIn("request", action["assignments"].values())

    def test_requests_preserve_required_completion_and_restore_feedback_prediction(self):
        state = SurveyState(payload())
        planner = Planner(state)
        now = state.survey_start
        normal = {"action": "observe", "pointing": {"alt_deg": 70, "az_deg": 90},
                  "assignments": {"0": "a"}, "program": "DARK", "duration_seconds": 600}
        requested = {**normal, "assignments": {"0": "b"}, "program": "BRIGHT"}
        def plan(*args):
            action = normal if plan.calls == 0 else requested
            plan.calls += 1
            state.pending = {next(iter(action["assignments"].values())): PendingPrediction(1, 1, 70, 90, True)}
            state.pending_start, state.pending_duration = now, 600
            state.pending_night, state.pending_action_index = 0, 0
            state.pending_program = action["program"]
            return action
        plan.calls = 0
        request = {"request_id": "r", "target_ids": ["b"], "minimum_completed": 1,
                   "remaining_count": 1, "completed_target_ids": [], "completion_factor_threshold": .5,
                   "completion_reward": 100, "issued_at_utc": format_utc(now), "deadline_utc": format_utc(state.nights[0][1])}
        bundle = {"request_id": "r", "target": 1, "start": now, "threshold": .5, "deadline": state.nights[0][1]}
        with patch.object(planner, "plan", side_effect=plan), patch.object(planner.requests, "choose", return_value=bundle):
            action = planner.decide({"now_utc": format_utc(now), "observe_action_index": 0,
                                      "wallclock": {"remaining_seconds": 0}, "active_requests": [request]})
        self.assertEqual(action["assignments"], normal["assignments"])
        self.assertEqual(set(state.pending), {"a"})
        self.assertEqual(state.pending_program, "DARK")
        state.on_result({"action": "observe", "hits": [{"target_id": "a", "score": 1}]}, 600/3600)
        self.assertEqual(state.best_score[0], 1)

    def test_request_remains_actionable_after_science_is_complete(self):
        scheduler, request = self.scheduler()
        state = scheduler.state
        state.best_score, state.factor, state.active = [1.2, 1.2], [1, 1], []
        request["issued_at_utc"] = format_utc(state.survey_start)
        planner = Planner(state)
        action = planner.decide({"now_utc": format_utc(state.survey_start), "observe_action_index": 0,
                                 "wallclock": {"remaining_seconds": 0}, "active_requests": [request]})
        self.assertEqual(action["action"], "observe")
        self.assertTrue(set(action["assignments"].values()) & {"a", "b"})

if __name__ == "__main__":
    unittest.main()
