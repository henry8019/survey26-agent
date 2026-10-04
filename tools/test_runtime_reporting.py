import unittest
from unittest.mock import patch
from datetime import timedelta
from agent_core.planner import Planner
from agent_core.geometry import local_sidereal_deg, SIDEREAL_DEG_PER_SECOND, wrap180
from agent_core.validation import validate_action
from agent_core.state import FaultEvidence, PendingPrediction, SurveyState
from test_strategy import payload


class ReportingTests(unittest.TestCase):
    def planner(self, **reporting):
        init = payload()
        init["scoring"]["reporting"] = {"correct_reward": 100, "false_penalty": -150,
                                      "false_report_free_allowance": 2, "max_consecutive_reports": 5, **reporting}
        return Planner(SurveyState(init))

    def evidence(self, checks=8, matched=8):
        return FaultEvidence(.3, 1, .3, 60, 2, 60, checks, matched)

    def confirm(self, planner, evidence):
        with patch.object(planner.state, "fault_evidence", return_value=evidence):
            results = [planner._maybe_report(hour, {}) for hour in (0, 6, 12)]
        return results[-1]

    def test_requires_band_evidence_before_report(self):
        planner = self.planner(false_report_free_allowance=0)
        self.assertIsNone(self.confirm(planner, self.evidence(checks=0, matched=0)))
        self.assertEqual(planner.state.force_program, "DARK")
        self.assertFalse(planner.suspicion_hours)

    def test_weather_and_program_mismatch_do_not_report(self):
        planner = self.planner(false_report_free_allowance=0)
        planner.state.notices = {"haze|ALL"}
        self.assertIsNone(self.confirm(planner, self.evidence()))
        planner.state.notices.clear()
        self.assertIsNone(self.confirm(planner, self.evidence(matched=0)))

    def test_free_report_does_not_require_costly_band_probe(self):
        planner = self.planner()
        with patch.object(planner.state, "fault_evidence", return_value=self.evidence(checks=0, matched=0)):
            self.assertEqual(planner._maybe_report(0, {})["action"], "report")
        self.assertIsNone(planner.state.force_program)

    def test_configured_allowance_and_reward_control_risk(self):
        planner = self.planner(false_penalty=-10000)
        self.assertEqual(self.confirm(planner, self.evidence())["action"], "report")
        planner = self.planner(false_penalty=-10000)
        planner.state.false_reports_since_correct = 2
        self.assertIsNone(self.confirm(planner, self.evidence()))

    def test_no_fixed_two_report_cap(self):
        planner = self.planner()
        planner.reports = 8
        self.assertEqual(self.confirm(planner, self.evidence())["action"], "report")
        planner = self.planner()
        planner.consecutive_reports = planner.state.max_consecutive_reports
        self.assertIsNone(self.confirm(planner, self.evidence()))

    def test_report_feedback_updates_allowance_without_double_message_count(self):
        state = self.planner().state
        state.on_result({"action": "report", "correct": False}, 1)
        state.on_messages([{"record_type": "report_result", "correct": False}], {})
        self.assertEqual(state.false_reports_since_correct, 1)
        state.clean_history = [(1, 0, .3)]
        state.on_result({"action": "report", "correct": True}, 2)
        self.assertEqual(state.false_reports_since_correct, 0)
        self.assertFalse(state.clean_history)

    def test_stale_clean_history_cannot_report_after_weather(self):
        planner = self.planner()
        planner.state.clean_history = [(-5, 0, .3)]
        self.assertIsNone(self.confirm(planner, self.evidence()))

    def test_many_targets_in_one_short_exposure_are_not_many_nights(self):
        state = self.planner().state
        state.clean_intervals = [(h, 0, .9, 1) for h in range(8)]
        state.clean_intervals += [(24, 1, .2, .3)] * 1000
        self.assertIsNone(state.fault_evidence())

    def test_short_exposure_cadence_detects_persistent_two_night_drop(self):
        state = self.planner().state
        state.clean_intervals = [(h / 60, 0, .9, 1) for h in range(8)]
        for night in (1, 2):
            for exposure in range(8):
                state.clean_intervals += [(24 * night + exposure / 60, night, .2, .3)] * 16
        evidence = state.fault_evidence()
        self.assertIsNotNone(evidence)
        self.assertAlmostEqual(evidence.drop, .333, places=3)
        self.assertEqual(evidence.recent_samples, 16)

    def test_factor_interpretation_ambiguity_cannot_manufacture_drop(self):
        state = self.planner().state
        state.clean_intervals = [(h / 60, 0, 1, 1.2) for h in range(8)]
        for night in (1, 2):
            state.clean_intervals += [(24 * night + h / 60, night, 1, 1.2) for h in range(8)]
        self.assertGreaterEqual(state.fault_evidence().drop, 1)

    def probe_setup(self):
        init = payload()
        target = init["targets"]["rows"][0]
        init["targets"]["rows"] = [[str(i), target[1] + .7 * i, target[2], 1, 1, False]
                                   for i in range(3)]
        planner = Planner(SurveyState(init))
        state = planner.state
        now, end = state.nights[0]
        action = planner.plan(now, end, 0, 0)
        planner._short_probe_evidence = self.evidence(checks=0, matched=0)
        return planner, action, now, end

    def test_one_night_drop_only_requests_evidence_not_report(self):
        planner = self.planner()
        state = planner.state
        state.clean_intervals = [(h / 60, 0, .9, 1) for h in range(8)] + [(24, 1, .2, .3)]
        state.clean_history = [(24, 1, .25)]
        self.assertIsNone(state.fault_evidence())
        self.assertAlmostEqual(state.fault_evidence(1, 1).drop, .333, places=3)
        self.assertIsNone(planner._maybe_report(24, {}))
        self.assertIsNotNone(planner._short_probe_evidence)

    def test_short_probe_is_legal_and_commits_its_actual_duration(self):
        planner, action, now, end = self.probe_setup()
        state = planner.state
        scores, factors = state.best_score[:], state.factor[:]
        probe = planner._short_probe(action, now, end, 0, 0)
        self.assertEqual(probe["duration_seconds"], state.min_exposure)
        self.assertEqual(probe["program"], "DARK")
        validate_action(probe, state)
        self.assertEqual(state.pending_duration, probe["duration_seconds"])
        self.assertEqual(state.pending_program, probe["program"])
        self.assertEqual((state.best_score, state.factor), (scores, factors))

    def test_short_probe_skips_weather_requests_and_unclean_geometry(self):
        for blocker in ("weather", "request", "geometry"):
            with self.subTest(blocker=blocker):
                planner, action, now, end = self.probe_setup()
                if blocker == "weather":
                    planner.state.notices = {"haze|ALL"}
                elif blocker == "request":
                    planner.requests.requests = {"r": {}}
                else:
                    planner.state.pending = {k: v._replace(clean=False) for k, v in planner.state.pending.items()}
                self.assertIs(planner._short_probe(action, now, end, 0, 0), action)

    def test_short_probe_preserves_last_required_visibility_and_night_end(self):
        planner, action, now, end = self.probe_setup()
        state = planner.state
        i = state.index_of["0"]
        state.required[i] = True
        ha = wrap180(local_sidereal_deg(now, state.lon) - state.ra[i])
        state.hmax[i] = ha + (action["duration_seconds"] + state.min_exposure / 2) * SIDEREAL_DEG_PER_SECOND
        self.assertIs(planner._short_probe(action, now, end, 0, 0), action)
        state.hmax[i] = 180
        state.required[i] = False
        end = now + timedelta(seconds=action["duration_seconds"] + state.min_exposure / 2)
        self.assertIs(planner._short_probe(action, now, end, 0, 0), action)

    def test_short_probe_does_not_displace_any_planned_required_completion(self):
        planner, action, now, end = self.probe_setup()
        planner.state.required[0] = True
        self.assertTrue(planner._required_completion_predictions(action, now))
        self.assertIs(planner._short_probe(action, now, end, 0, 0), action)

    def test_ambiguous_saturation_is_not_an_exact_quality_sample(self):
        init = payload()
        init["scoring"]["q0"] = .68
        target = init["targets"]["rows"][0]
        init["targets"]["rows"] = [[str(i), target[1] + .7 * i, target[2], 1, 1, False] for i in range(3)]
        state = SurveyState(init)
        # A DARK sky can saturate a BRIGHT exposure. Score 1.0 can
        # equally mean a partial BRIGHT match: selecting .893 as an exact
        # completion factor would invent a throughput reduction.
        for action in range(4):
            state.pending = {t: PendingPrediction(1 / .68, 1 / (.68 * .95), 80, 90, True) for t in state.ids}
            state.pending_program = "BRIGHT"
            state.pending_duration = 900
            state.pending_night = 0
            state.pending_action_index = action
            state.on_result({"action": "observe", "hits": [{"target_id": t, "score": 1.0} for t in state.ids]}, action / 3)
        self.assertAlmostEqual(state.factor[0], 1 / 1.12, places=5)
        self.assertEqual(state.factor_upper[0], 1)
        self.assertEqual(state.best_score[0], 1)
        self.assertFalse(state._samples)
        self.assertFalse(state.clean_history)
        self.assertEqual(state.scale, 1)
        self.assertTrue(state.calibration_due(1.5, 0))
        state.forget_quality_history()
        self.assertFalse(state.calibration_due(1.5, 0))

    def test_stale_censored_nights_and_fresh_samples_do_not_trigger_calibration(self):
        planner = self.planner()
        state = planner.state
        state.censored_exposures.extend((hour, 0) for hour in (0, .1, .2, .3))
        self.assertFalse(state.calibration_due(1, 1))
        self.assertFalse(state.calibration_due(3, 0))
        state.clean_intervals = [(.5, 0, .2, .3), (.6, 0, .2, .3)]
        self.assertFalse(state.calibration_due(1, 0))

    def test_censored_feedback_can_trigger_bounded_calibration_without_fault_claim(self):
        planner, action, now, end = self.probe_setup()
        planner._short_probe_evidence = None
        state = planner.state
        state.censored_exposures.extend((hour, 0) for hour in (0, .1, .2, .3))
        now += timedelta(hours=.5)
        action = planner.plan(now, end, 0, .5)
        probe = planner._short_probe(action, now, end, 0, .5)
        self.assertEqual(probe["duration_seconds"], state.min_exposure)
        validate_action(probe, state)
        self.assertEqual(planner.reports, 0)
        # Calibration can resume after two hours, never every public slot.
        self.assertIs(planner._short_probe(action, now, end, 0, 1), action)

    def test_terminal_report_is_allowed_only_without_a_possible_score_loss(self):
        finish = {"action": "finish"}
        planner = self.planner()
        self.assertEqual(planner._terminal_action({}, finish)["action"], "report")
        planner = self.planner(false_report_free_allowance=0)
        self.assertIs(planner._terminal_action({}, finish), finish)
        planner = self.planner(false_report_free_allowance=0, false_penalty=0)
        self.assertEqual(planner._terminal_action({}, finish)["action"], "report")
        planner = self.planner(correct_reward=-1)
        self.assertIs(planner._terminal_action({}, finish), finish)

    def test_terminal_diagnostic_stops_after_false_and_respects_report_cap(self):
        planner = self.planner(max_consecutive_reports=2)
        finish = {"action": "finish"}
        report = planner._terminal_action({}, finish)
        planner.note_action(report)
        self.assertEqual(planner._terminal_action({"last_result": {"action": "report", "correct": True}}, finish)["action"], "report")
        planner.note_action(report)
        self.assertIs(planner._terminal_action({"last_result": {"action": "report", "correct": True}}, finish), finish)
        planner = self.planner()
        planner._terminal_action({}, finish)
        self.assertIs(planner._terminal_action({"last_result": {"action": "report", "correct": False}}, finish), finish)
        self.assertIs(planner._terminal_action({}, finish), finish)

    def test_terminal_diagnostic_preserves_wall_budget_and_future_observing_nights(self):
        planner = self.planner()
        finish = {"action": "finish"}
        self.assertIs(planner._terminal_action({"wallclock": {"remaining_seconds": 1}}, finish), finish)
        state = planner.state
        from agent_core.geometry import format_utc
        payload_ = {"now_utc": format_utc(state.nights[0][1] - timedelta(seconds=1))}
        with patch.object(planner.advice, "update"):
            self.assertEqual(planner.decide(payload_)["action"], "wait")
        self.assertFalse(planner._terminal_report_attempted)

    def test_last_night_uses_legal_terminal_report_then_finishes_waiting(self):
        planner = self.planner()
        state = planner.state
        from agent_core.geometry import format_utc
        payload_ = {"now_utc": format_utc(state.nights[-1][1] - timedelta(seconds=1))}
        with patch.object(planner.advice, "update"):
            report = planner.decide(payload_)
            validate_action(report, state)
            planner.note_action(report)
            action = planner.decide({**payload_, "last_result": {"action": "report", "correct": False}})
        self.assertEqual(action["action"], "wait")
        self.assertEqual(action["until_utc"], format_utc(state.survey_end))

    def test_last_night_audit_runs_before_the_exposure_can_end_the_survey(self):
        planner = self.planner()
        state = planner.state
        from agent_core.geometry import format_utc
        payload_ = {"now_utc": format_utc(state.nights[-1][0]), "wallclock": {"remaining_seconds": 300}}
        with patch.object(planner.advice, "update"), patch.object(planner, "plan", return_value=None) as plan:
            report = planner.decide(payload_)
            validate_action(report, state)
            self.assertEqual(report["action"], "report")
            plan.assert_not_called()
            planner.note_action(report)
            next_action = planner.decide({**payload_, "last_result": {"action": "report", "correct": False}})
        self.assertEqual(next_action["action"], "wait")
        self.assertEqual(planner.reports, 1)
        self.assertFalse(state.ledger)

    def test_last_night_audit_cannot_use_exhausted_free_allowance(self):
        planner = self.planner()
        state = planner.state
        state.false_reports_since_correct = state.false_report_free_allowance
        from agent_core.geometry import format_utc
        payload_ = {"now_utc": format_utc(state.nights[-1][0])}
        with patch.object(planner.advice, "update"), patch.object(planner, "plan", return_value=None):
            self.assertEqual(planner.decide(payload_)["action"], "wait")
        self.assertEqual(planner.reports, 0)

    def test_short_probe_is_bounded_and_does_not_repeat_existing_evidence(self):
        planner, action, now, end = self.probe_setup()
        planner.state.clean_intervals = [(h / 60, 0, .2, .3) for h in range(4)]
        self.assertIs(planner._short_probe(action, now, end, 0, 0), action)
        planner.state.clean_intervals.clear()
        planner._short_probe_counts[0] = 4
        self.assertIs(planner._short_probe(action, now, end, 0, 0), action)
        planner._short_probe_counts.clear()
        planner._last_short_probe_hours = 0
        self.assertIs(planner._short_probe(action, now, end, 0, .1), action)

    def test_dark_evidence_exposure_accounts_for_multiplier_ambiguity(self):
        init = payload()
        target = init["targets"]["rows"][0]
        init["targets"]["rows"] = [[str(j), target[1] + .7 * j, target[2], flux, weight, False]
                                   for j, (flux, weight) in enumerate(((3, 10), (.35, 1), (.3, 1)))]
        state = SurveyState(init)
        state.force_program = "DARK"
        action = Planner(state).plan(state.survey_start, state.nights[0][1], 0, 0)
        multipliers = [*state.scoring.program_multipliers.values(), state.scoring.mismatch_multiplier]
        ambiguity = max(multipliers) / min(multipliers)
        robust_saturated = sum(state.scoring.completion_factor(state.flux[state.index_of[target]], action["duration_seconds"],
                              prediction.model * state.scale * .9 / ambiguity) >= 1
                              for target, prediction in state.pending.items())
        self.assertGreaterEqual(robust_saturated, 3)


if __name__ == "__main__":
    unittest.main()
