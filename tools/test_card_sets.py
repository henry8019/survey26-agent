import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from card_sets import CARD_SETS, card_order, require_runnable
from compare_versions import compare


class CardSetTests(unittest.TestCase):
    def test_official_and_example_iterators_are_distinct(self):
        for cards in CARD_SETS.values():
            self.assertEqual(card_order(iter(cards)), cards)
        with self.assertRaises(ValueError):
            card_order(("alpha", "L2", "L3", "L4"))

    def test_incomplete_public_export_fails_before_model_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/v4_scenario.json").write_text(json.dumps({"products": {"slots": "../truth/v4_slots.csv"}}))
            with patch("card_sets.card_path", return_value=root), self.assertRaisesRegex(ValueError, "missing official products"):
                require_runnable("alpha")

    def test_different_card_hashes_cannot_be_compared(self):
        groups = {card: [{"card": card, "total": 100, "required_missing": 0, "termination_reason": "survey_complete",
                           "model_mode": "configured-api", "source_sha256": "a", "card_sha256": card}] for card in CARD_SETS["official"]}
        candidate = copy.deepcopy(groups)
        candidate["alpha"][0]["card_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "identical card files"):
            compare(groups, candidate)


if __name__ == "__main__":
    unittest.main()
