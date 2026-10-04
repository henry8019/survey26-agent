import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stage3.planner import Planner
from experiments.stage3.state import FaultEvidence, SurveyState
from test_strategy import payload
from unittest.mock import patch
from experiments.stage3.diagnostics import verify_pair
from experiments.stage3.state import PendingPrediction


class DiagnosisTests(unittest.TestCase):
    def test_program_ambiguity_alone_is_not_a_fault(self):
        state = SurveyState(payload())
        for j in range(80):
            # The plausible intervals overlap; selecting opposite endpoints
            # would wrongly claim a drop from 1.2 to 1.0.
            state.quality_evidence.append((j, j % 3, 1, 1.2, j // 8))
        self.assertGreaterEqual(state.fault_evidence().drop, 1)

    def test_clean_persistent_drop_reports_but_weather_does_not(self):
        state = SurveyState(payload())
        planner = Planner(state)
        evidence = FaultEvidence(.3, .9, .333, 24, 0, 50, 0, 0, 3, 3)
        with patch.object(state, "fault_evidence", return_value=evidence):
            state.notices = {"haze|ALL"}
            self.assertIsNone(planner._maybe_report(10, {}))
            state.notices.clear()
            planner.verified_probe_fields = {(1, 2), (3, 2)}
            state.notices = {"haze|E"}
            self.assertEqual(planner._maybe_report(10, {})["action"], "report")

    def test_insufficient_evidence_sets_short_probe(self):
        state = SurveyState(payload())
        planner = Planner(state)
        with patch.object(state, "fault_evidence", return_value=FaultEvidence(.3, .9, .333, 24, 0, 50, 0, 0, 2, 2)):
            self.assertIsNone(planner._maybe_report(10, {}))
        self.assertEqual(planner.diagnostic_ceiling, 120)

    def test_repair_resets_false_allowance_and_quality_history(self):
        state = SurveyState(payload())
        state.on_result({"action": "report", "correct": False}, 1)
        self.assertEqual(state.false_reports_since_repair, 1)
        state.quality_evidence.append((0, 0, .1, .1, 0))
        state.on_result({"action": "report", "correct": True}, 2)
        self.assertEqual(state.false_reports_since_repair, 0)
        self.assertFalse(state.quality_evidence)

    def test_unprofitable_penalized_report_is_rejected(self):
        state = SurveyState(payload())
        state.false_report_penalty = 10000
        planner = Planner(state)
        planner.verified_probe_fields = {(1, 2), (3, 2)}
        with patch.object(state, "fault_evidence", return_value=FaultEvidence(.55, 1, .55, 24, 0, 50, 0, 0, 3, 3)):
            self.assertIsNone(planner._maybe_report(10, {}))

    def test_paired_programs_prove_efficiency_drop_and_reject_weather(self):
        init = payload()
        init["targets"]["rows"] = [[str(j), 10 + j, -10, 1, 1, False] for j in range(4)]
        state = SurveyState(init)
        predictions = {t: PendingPrediction(.9, .9, 70, 90, True) for t in state.ids}
        # Same band quality .8, but instrument efficiency .3, gives q_eff .24.
        first = {"program": "DARK", "duration": 120, "models": {t: .9 for t in state.ids},
                 "hits": {t: .24 * 120 / 450 * 1.2 for t in state.ids}}
        backup = {t: .24 * 120 / 450 for t in state.ids}
        self.assertIsNotNone(verify_pair(state, first, backup, predictions, 120, .8))
        # Weather alone drops the actual band to BRIGHT: both declarations
        # mismatch and their score ratio cannot certify DARK sky quality.
        first["hits"] = {t: .24 * 120 / 450 for t in state.ids}
        self.assertIsNone(verify_pair(state, first, backup, predictions, 120, .8))

    def test_inconclusive_short_probes_have_a_nightly_limit(self):
        state = SurveyState(payload())
        planner = Planner(state)
        planner.probe_night, planner.probes_this_night = 0, 2
        with patch.object(state, "fault_evidence", return_value=FaultEvidence(.3, .9, .333, 24, 0, 50, 0, 0, 2, 2)):
            self.assertIsNone(planner._maybe_report(1, {}))
        self.assertIsNone(planner.diagnostic_ceiling)
        self.assertEqual(state.force_program, "DARK")

    def test_scientific_dark_check_reaches_saturation_for_band_evidence(self):
        init = payload()
        a, b = init["targets"]["rows"]
        init["targets"]["rows"] = [[str(j), a[1] + .7*j, a[2], flux, weight, False]
                                    for j, (flux, weight) in enumerate([(3, 10), (.35, 1), (.3, 1)])]
        state = SurveyState(init)
        state.force_program = "DARK"
        planner = Planner(state)
        action = planner.plan(state.survey_start, state.nights[0][1], 0, 0)
        self.assertIsNotNone(action)
        saturation = sum(state.scoring.completion_factor(state.flux[state.index_of[t]], action["duration_seconds"],
                                                        prediction.model * state.scale * .9) >= 1
                         for t, prediction in state.pending.items())
        self.assertGreaterEqual(saturation, 3)

    def test_repeated_dark_mismatch_keeps_science_program_choice_open(self):
        state = SurveyState(payload())
        planner = Planner(state)
        planner.probe_night, planner.probes_this_night = 0, 2
        with patch.object(state, "fault_evidence", return_value=FaultEvidence(.3, .9, .333, 24, 0, 50, 6, 1, 2, 2)):
            self.assertIsNone(planner._maybe_report(1, {}))
        self.assertIsNone(state.force_program)
        self.assertIsNone(planner.diagnostic_ceiling)

    def test_conservative_reference_still_detects_a_proven_small_drop(self):
        init = payload()
        init["targets"]["rows"] = [[str(j), 10+j, -10, 1, 1, False] for j in range(4)]
        state = SurveyState(init)
        predictions = {t: PendingPrediction(.9, .9, 70, 90, True) for t in state.ids}
        # A DARK-band probe constrains the instrument to <= .47, below the
        # already conservative earlier bound .543 by more than five percent.
        q_eff = .47 * .65
        first = {"program": "DARK", "duration": 120, "models": {t: .9 for t in state.ids},
                 "hits": {t: q_eff * 120/450 * 1.2 for t in state.ids}}
        backup = {t: q_eff * 120/450 for t in state.ids}
        self.assertIsNotNone(verify_pair(state, first, backup, predictions, 120, .543))

    def test_equal_program_bonuses_cannot_prove_a_band(self):
        state = SurveyState(payload())
        state.scoring.program_multipliers["DARK"] = state.scoring.mismatch_multiplier
        self.assertIsNone(verify_pair(state, {"program": "DARK"}, {}, {}, 120, .9))

    def test_completed_targets_can_supply_diagnostic_evidence(self):
        state = SurveyState(payload())
        state.best_score = [1.2, 1.2]
        state.factor = [1, 1]
        state.active = []
        state.force_program = "DARK"
        planner = Planner(state)
        planner.diagnostic_ceiling = 120
        action = planner.plan(state.survey_start, state.nights[0][1], 0, 0)
        self.assertIsNotNone(action)
        self.assertTrue(action["duration_seconds"] <= 120)
        self.assertEqual(state.factor, [1, 1])

if __name__ == "__main__":
    unittest.main()

