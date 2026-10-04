import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from test_strategy import payload
from agent_core.state import SurveyState
from agent_core.validation import validate_action

class ProtocolFallbackTests(unittest.TestCase):
    def test_missing_key_still_produces_one_legal_jsonl_response(self):
        root = Path(__file__).resolve().parents[1]
        init = payload()
        messages = [{"message_type": "initialize", "payload": init},
                    {"message_type": "decision_request", "decision_sequence": 1,
                     "payload": {"now_utc": init["survey"]["start_utc"], "wallclock": {"remaining_seconds": 0},
                                 "observe_action_index": 0}},
                    {"message_type": "finish", "payload": {}}]
        env = dict(os.environ)
        env.update(OPENAI_API_KEY="", KIMI_API_KEY="", AGENT_TRACE_PATH="", PYTHONIOENCODING="utf-8")
        run = subprocess.run([sys.executable, "-u", str(root / "agent.py")],
                             input="\n".join(json.dumps(m) for m in messages) + "\n",
                             capture_output=True, text=True, encoding="utf-8", timeout=5, cwd=root, env=env)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(run.stdout.splitlines()), 1)
        response = json.loads(run.stdout)
        self.assertEqual(response["message_type"], "decision_response")
        self.assertEqual(response["decision_sequence"], 1)
        action = {k:v for k,v in response.items() if k not in {"message_type", "decision_sequence", "protocol_version"}}
        validate_action(action, SurveyState(init))
