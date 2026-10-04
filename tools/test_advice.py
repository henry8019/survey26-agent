import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_core.advice import AdviceController
from agent_core.state import SurveyState
from test_strategy import payload


class Trace:
    def __init__(self):
        self.rows = []
    def write(self, value):
        self.rows.append(value)


class Client:
    calls_made = 0
    max_retries = 2
    def __init__(self, answers):
        self.answers = iter(answers)
    def ask_json(self, *args, **kwargs):
        self.calls_made += 1
        return next(self.answers, None)


class AdviceTests(unittest.TestCase):
    def controller(self, answers):
        state = SurveyState(payload())
        state.night_dates = ["2026-11-01", "2026-11-02", "2026-11-03"]
        return AdviceController(state, Client(answers), Trace())

    def message(self):
        return {"wallclock": {"remaining_seconds": 900}, "latest_bulletin": {
            "slot_id": "slot-1", "notices": [{"event_kind": "rain", "direction": "N"}]}}

    def test_slot_advice_expires_on_clear_bulletin(self):
        controller = self.controller([{"events": [{"source_index": 0, "severity": "closed"}]}, {"priority": "required"}])
        controller.update(self.message(), 0, controller.state.survey_start)
        self.assertEqual(controller.state.extra_avoid, {"N"})
        controller.update({"latest_bulletin": {"slot_id": "slot-2", "notices": []}}, 0, controller.state.survey_start)
        self.assertEqual(controller.state.extra_avoid, set())

    def test_model_cannot_invent_directions_or_source(self):
        for answer in ({"events": [{"source_index": 99, "severity": "closed"}]},
                       {"events": [], "action": "report"}, {"events": [{"source_index": 0, "severity": "invented"}]},
                       {"events": [{"source_index": 0, "severity": []}]}, {"events": [], "urgent_requests": [{}]}):
            controller = self.controller([answer, {"priority": "required"}])
            controller.update(self.message(), 0, controller.state.survey_start)
            self.assertEqual(controller.state.extra_avoid, set())
            self.assertNotIn("message_understanding", controller.applied_stages)

    def test_all_sky_degradation_is_not_eight_direction_avoidance(self):
        controller = self.controller([{"events": [{"source_index": 0, "severity": "degraded"}]}, {"priority": "survey"}])
        message = self.message()
        message["latest_bulletin"]["notices"] = [{"event_kind": "haze", "direction": "ALL"}]
        controller.update(message, 0, controller.state.survey_start)
        self.assertFalse(controller.state.extra_avoid)
        self.assertEqual(controller.state.duration_scale, 1)

    def test_forecast_uses_public_night_date(self):
        controller = self.controller([{"events": []}, {"priority": "required"}])
        message = {"latest_forecast": {"notices": [{"event_kind": "storm", "direction": "E", "nights": ["2026-11-02"]}]}}
        controller.update(message, 0, controller.state.survey_start)
        self.assertFalse(controller.state.extra_avoid)

    def test_invalid_replan_falls_back(self):
        controller = self.controller([{"events": []}, {"priority": "diagnostic", "action": "report"}])
        controller.update({}, 0, controller.state.survey_start)
        self.assertEqual(controller.priority, "required")
        self.assertNotIn("plan_adaptation", controller.applied_stages)

    def test_two_distinct_model_stages_are_recorded(self):
        controller = self.controller([{"events": []}, {"priority": "required", "reason": "required targets outstanding"}])
        controller.update({}, 0, controller.state.survey_start)
        self.assertEqual(controller.applied_stages, {"message_understanding", "plan_adaptation"})
        self.assertEqual(controller.required_multiplier(), 1.15)

    def test_stage_attempt_limit(self):
        controller = self.controller([])
        controller.attempts["message_understanding"] = 24
        self.assertIsNone(controller._ask("message_understanding", "", {}, 900))
        self.assertEqual(controller.client.calls_made, 0)

    def test_priority_expires_at_night_boundary(self):
        controller = self.controller([{"events": []}, {"priority": "required"}])
        controller.update({}, 0, controller.state.survey_start)
        self.assertEqual(controller.required_multiplier(), 1.15)
        controller.update({}, 1, controller.state.survey_start)
        self.assertEqual(controller.required_multiplier(), 1)

    def test_forecast_has_lower_decision_weight_than_current_bulletin(self):
        controller = self.controller([{"events": [{"source_index": 0, "severity": "closed"}]}, {"priority": "required"}])
        message = {"latest_forecast": {"notices": [{"event_kind": "storm", "direction": "E", "nights": ["2026-11-01"]}]}}
        controller.update(message, 0, controller.state.survey_start)
        self.assertEqual(controller.state.model_direction_factors, {"E": .8})

    def test_second_stage_subtracts_first_stage_wall_time(self):
        from unittest.mock import patch
        controller = self.controller([])
        with patch("agent_core.advice.time.monotonic", side_effect=[10, 18]), patch.object(controller, "_ask", side_effect=[{"events": []}, {"priority": "required"}]) as ask:
            controller.update({"wallclock": {"remaining_seconds": 70}}, 0, controller.state.survey_start)
        self.assertEqual(ask.call_args_list[0].args[-1], 70)
        self.assertEqual(ask.call_args_list[1].args[-1], 62)

    def test_all_sky_altitude_preference_expires_with_source(self):
        controller = self.controller([{"events": [{"source_index": 0, "severity": "degraded"}]}, {"priority": "survey"}])
        message = self.message()
        message["latest_bulletin"]["notices"] = [{"event_kind": "haze", "direction": "ALL"}]
        controller.update(message, 0, controller.state.survey_start)
        self.assertEqual(controller.state.model_altitude_risk, .65)
        controller.update({"latest_bulletin": {"slot_id": "clear", "notices": []}}, 0, controller.state.survey_start)
        self.assertEqual(controller.state.model_altitude_risk, 0)

    def test_calendar_date_fallback_uses_site_offset(self):
        init = payload()
        init["site"]["utc_offset_hours"] = 8
        init["survey"]["nights"][0]["observing_start_utc"] = "2026-11-01T11:00:00Z"
        self.assertEqual(SurveyState(init).night_dates[0], "2026-11-01")

    def test_distinct_requests_in_same_night_both_replan(self):
        from datetime import timedelta
        c = self.controller([{"events": []}, {"priority": "required"},
                             {"priority": "required"}, {"priority": "required"}])
        now = c.state.survey_start
        c.update({}, 0, now)
        for index in (1, 2):
            c.update({"new_messages": [{"record_type": "observation_request", "request_id": str(index)}]},
                     0, now + timedelta(seconds=2 * index * c.state.slot_seconds))
        adjustments = [r for r in c.trace.rows if r.get("stage") == "plan_adaptation"]
        self.assertEqual(len(adjustments), 3)
        self.assertNotEqual(adjustments[1]["event_ids"], adjustments[2]["event_ids"])

    def test_cooldown_queues_revision_until_later_snapshot(self):
        from datetime import timedelta
        c = self.controller([{"events": []}, {"priority": "required"}, {"priority": "required"}])
        now = c.state.survey_start
        c.update({}, 0, now)
        event = {"record_type": "state_resync", "invalidated_window": {"action_index_start": 1, "action_index_end_exclusive": 3}}
        c.update({"new_messages": [event]}, 0, now + timedelta(seconds=1))
        self.assertTrue(c.pending_events)
        c.update({}, 0, now + timedelta(seconds=2 * c.state.slot_seconds))
        self.assertFalse(c.pending_events)
        self.assertEqual(c.trace.rows[-1]["trigger"], "state_resync")
        self.assertEqual(len(c.trace.rows[-1]["event_ids"]), 1)
        calls = c.client.calls_made
        c.update({"new_messages": [event]}, 0, now + timedelta(seconds=4 * c.state.slot_seconds))
        self.assertEqual(c.client.calls_made, calls)

    def test_corrected_request_result_is_distinct_event(self):
        from datetime import timedelta
        c = self.controller([{"events": []}, {"priority": "required"},
                             {"priority": "required"}, {"priority": "required"}])
        now = c.state.survey_start
        c.update({}, 0, now)
        for index, revised in enumerate((False, True), 1):
            c.update({"new_messages": [{"record_type": "observation_request_result", "request_id": "same", "revised": revised}]},
                     0, now + timedelta(seconds=2 * index * c.state.slot_seconds))
        self.assertEqual(len([r for r in c.trace.rows if r.get("stage") == "plan_adaptation"]), 3)


if __name__ == "__main__":
    unittest.main()
