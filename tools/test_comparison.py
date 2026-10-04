import copy
import unittest
from compare_versions import compare

class ComparisonTests(unittest.TestCase):
    def groups(self):
        return {card: [{"card": card, "total": 100, "required_missing": 1,
                       "termination_reason": "survey_complete", "model_mode": "configured-api",
                       "source_sha256": "same"} for _ in range(3)] for card in ("L1", "L2", "L3", "L4")}

    def test_median_gate_rejects_protocol_failure_mock_or_changed_source(self):
        baseline = self.groups()
        for key, value in [("validation_errors", 1), ("planner_errors", 1),
                           ("model_mode", "mock-neutral"), ("source_sha256", "different"),
                           ("termination_reason", "agent_error")]:
            candidate = self.groups()
            candidate["L1"][0][key] = value
            self.assertFalse(compare(baseline, candidate)["accepted"])

    def test_extra_missing_required_targets_are_not_hidden_by_score_gain(self):
        baseline, candidate = self.groups(), self.groups()
        for r in candidate["L1"]:
            r.update(total=150, required_missing=2)
        self.assertFalse(compare(baseline, candidate)["accepted"])

    def test_declining_card_requires_three_samples(self):
        baseline, candidate = self.groups(), self.groups()
        for r in candidate["L1"]:
            r["total"] = 99
        for r in candidate["L2"]:
            r["total"] = 110
        one = lambda groups: {c: rows[:1] for c, rows in groups.items()}
        self.assertTrue(compare(one(baseline), one(candidate))["needs_paired_repeats"])
        self.assertFalse(compare(baseline, candidate)["accepted"])  # Minimum score dropped.
