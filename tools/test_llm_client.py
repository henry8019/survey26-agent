"""Regression checks for the K3 request rejected during real integration."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_core.llm_client import LLMClient


class K3RequestTests(unittest.TestCase):
    def setUp(self):
        self.credentials = patch.dict(os.environ, {"OPENAI_API_KEY": "local-test-only"})
        self.credentials.start()
        self.addCleanup(self.credentials.stop)

    def test_missing_credentials_skip_http_and_leave_program_fallback_available(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "KIMI_API_KEY": ""}):
            client = LLMClient()
        with patch.object(client, "_attempt") as request:
            self.assertIsNone(client.ask_json("json", {}, 900))
            request.assert_not_called()
        self.assertEqual(client.calls_made, 0)

    def test_timeout_and_attempt_budgets_fall_back(self):
        client = LLMClient()
        with patch.object(client, "_attempt", side_effect=TimeoutError):
            self.assertIsNone(client.ask_json("json", {}, 900))
        self.assertEqual(client.calls_made, 2)
        client.calls_made = 40
        with patch.object(client, "_attempt") as request:
            self.assertIsNone(client.ask_json("json", {}, 900))
            request.assert_not_called()
        client.calls_made = 0
        client.spent_seconds = 150
        with patch.object(client, "_attempt") as request:
            self.assertIsNone(client.ask_json("json", {}, 900))
            request.assert_not_called()

    def test_last_minute_is_reserved(self):
        client = LLMClient()
        with patch.object(client, "_attempt") as request:
            self.assertIsNone(client.ask_json("json", {}, 60))
            request.assert_not_called()

    def test_invalid_response_retries_twice(self):
        client = LLMClient()
        with patch.object(client, "_attempt", side_effect=ValueError("invalid JSON")):
            self.assertIsNone(client.ask_json("json", {}, 900))
        self.assertEqual(client.calls_made, 2)

    def request_body(self, model):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "local-test-only", "OPENAI_MODEL": model}, clear=True):
            client = LLMClient()
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"duration_scale": 1.0}'}}],
        }).encode()
        with patch("urllib.request.urlopen", return_value=response) as open_url:
            self.assertEqual(client._attempt("json", {}, 1), {"duration_scale": 1.0})
        request = open_url.call_args.args[0]
        return json.loads(request.data)

    def test_k3_uses_provider_temperature_and_no_reasoning(self):
        for model in ("k3", "k3-256k"):
            with self.subTest(model=model):
                body = self.request_body(model)
                self.assertNotIn("temperature", body)
                self.assertEqual(body["reasoning_effort"], "none")

    def test_other_providers_keep_original_request(self):
        body = self.request_body("other-model")
        self.assertEqual(body["temperature"], 0)
        self.assertNotIn("reasoning_effort", body)


if __name__ == "__main__":
    unittest.main()
