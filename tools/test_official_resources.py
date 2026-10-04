import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from card_sets import CARD_SETS
from prepare_official import prepare, verify_manifest


class OfficialResourceTests(unittest.TestCase):
    def files(self):
        return {"config/v4_scenario.json": json.dumps({"products": {"weather": "../truth/v4_weather_truth.csv"}}).encode(),
                "config/v4_score_config.json": b"{}", "config/v4_fiber_config.json": b"{}",
                "public/targets.csv": b"target_id\n", "public/footprint.csv": b"ra\n",
                "public/v4_night_calendar.csv": b"start\n"}

    def fetcher(self, files):
        def fetch(url, headers=None, data=None):
            if data is not None:
                directory = json.loads(data)["prefix"].split("/")[-1]
                return json.dumps([{"name": name.split("/")[1], "id": "public"}
                                   for name in files if name.startswith(directory + "/")]).encode()
            return files["/".join(url.split("/")[-2:])]
        return fetch

    def test_missing_products_and_hashes_are_preserved_without_network_refresh(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            destination = Path(directory)
            files = self.files()
            with patch("prepare_official.public_storage", return_value=("https://storage.example.org", {})), \
                 patch("prepare_official.fetch", side_effect=self.fetcher(files)):
                manifest = prepare(destination)
            for card in CARD_SETS["official"]:
                self.assertEqual(manifest["cards"][card]["missing_products"], ["truth/v4_weather_truth.csv"])
                self.assertFalse(manifest["cards"][card]["runnable"])
            with patch("prepare_official.public_storage", side_effect=AssertionError("network used")):
                self.assertEqual(prepare(destination), verify_manifest(destination))

    def test_refresh_adds_new_public_files_without_replacing_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            destination = Path(directory)
            files = self.files()
            with patch("prepare_official.public_storage", return_value=("https://storage.example.org", {})), \
                 patch("prepare_official.fetch", side_effect=self.fetcher(files)):
                prepare(destination)
                original = (destination / "alpha/config/v4_scenario.json").read_bytes()
                files["truth/v4_weather_truth.csv"] = b"public unit fixture\n"
                manifest = prepare(destination, refresh=True)
                self.assertTrue(all(c["runnable"] for c in manifest["cards"].values()))
                self.assertEqual((destination / "alpha/config/v4_scenario.json").read_bytes(), original)
                files["config/v4_scenario.json"] = b"changed"
                with self.assertRaisesRegex(ValueError, "existing resource differs"):
                    prepare(destination, refresh=True)
                self.assertEqual((destination / "alpha/config/v4_scenario.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
