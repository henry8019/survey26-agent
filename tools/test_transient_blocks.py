import unittest
from agent_core.planner import Planner
from agent_core.state import PendingPrediction, SurveyState
from test_strategy import payload


class TransientBlockTests(unittest.TestCase):
    def state_with_mixed_hits(self):
        state = SurveyState(payload())
        state.pending = {target: PendingPrediction(1, 1, 70, 90, True) for target in ("a", "b")}
        state.pending_duration = 300
        state.on_result({"action": "observe", "hits": [{"target_id": "a", "score": 0},
                                                        {"target_id": "b", "score": .4}]}, 1)
        return state

    def test_partial_zero_is_temporary_even_without_new_observation(self):
        state = self.state_with_mixed_hits()
        self.assertEqual(Planner(state)._direction_factor(70, 90), .2)
        state.on_result({"action": "wait"}, 3)
        self.assertFalse(state.blocked)
        self.assertEqual(Planner(state)._direction_factor(70, 90), 1)

    def test_changed_bulletin_clears_old_inference(self):
        state = self.state_with_mixed_hits()
        state.notices = {"cloud|N"}
        state.on_messages([], {"notices": []})
        self.assertFalse(state.blocked)

    def test_correction_discards_transient_evidence(self):
        state = self.state_with_mixed_hits()
        state.on_messages([{"record_type": "state_resync", "best_scores": []}], {})
        self.assertFalse(state.blocked)


if __name__ == "__main__":
    unittest.main()
