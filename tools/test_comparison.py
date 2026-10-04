import copy
import json
import tempfile
import unittest
from pathlib import Path
from compare_versions import compare, compare_evaluations

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

    def test_whole_evaluations_can_reject_card_median_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def evaluations(prefix, scores):
                paths = []
                for number, values in enumerate(scores):
                    path = root / f"{prefix}-{number}"
                    path.mkdir()
                    runs = [{"card": card, "total": value, "required_missing": 0,
                             "termination_reason": "survey_complete", "model_mode": "configured-api",
                             "source_sha256": prefix, "model_attempts": 2,
                             "model_stages_applied": ["message_understanding", "plan_adaptation"]}
                            for card, value in zip(("L1", "L2", "L3", "L4"), values)]
                    (path / "summary.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
                    paths.append(path)
                return paths
            baseline = evaluations("baseline", [[100, 100, 100, 100]] * 3)
            candidate = evaluations("candidate", [[110, 110, 0, 0], [110, 0, 110, 110], [0, 110, 110, 110]])
            result = compare_evaluations(baseline, candidate)
            self.assertTrue(result["per_card_accepted"])
            self.assertFalse(result["whole_evaluations"]["accepted"])
            self.assertFalse(result["accepted"])

    def test_configured_api_without_actual_two_stages_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("baseline", "candidate"):
                path = Path(directory) / name
                path.mkdir()
                runs = [{**rows[0], "model_attempts": 0, "model_stages_applied": []}
                        for rows in self.groups().values()]
                (path / "summary.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
                paths.append(path)
            result = compare_evaluations(paths[:1], paths[1:])
            self.assertFalse(result["two_model_stages_verified"])
            self.assertFalse(result["accepted"])
