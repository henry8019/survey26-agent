"""Deadline and protocol checks for the local Windows pipe adapter."""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".local" / "runner"))
from windows_transport import WindowsTransport
from project_platform.transport import ExecutionError, GlobalDeadlineExpired, MAX_RESPONSE_BYTES


class TransportTests(unittest.TestCase):
    def transport(self, code):
        transport = WindowsTransport([sys.executable, "-u", "-c", code])
        self.addCleanup(transport.close, force=True)
        return transport

    def test_large_initial_message_and_response(self):
        transport = self.transport("import sys,json; v=json.loads(sys.stdin.readline()); print(json.dumps({'size':len(v['data'])}),flush=True)")
        deadline = time.monotonic() + 5
        transport.send({"data": "x" * 500000}, deadline)
        self.assertEqual(transport.receive(deadline), {"size": 500000})

    def test_receive_timeout_kills_child(self):
        transport = self.transport("import time; time.sleep(30)")
        with self.assertRaises(GlobalDeadlineExpired):
            transport.receive(time.monotonic() + 0.15)
        self.assertIsNone(transport.process)

    def test_blocked_write_timeout_kills_child(self):
        transport = self.transport("import time; time.sleep(30)")
        with self.assertRaises(GlobalDeadlineExpired):
            transport.send({"data": "x" * 500000}, time.monotonic() + 0.15)
        self.assertIsNone(transport.process)

    def test_oversize_response_is_rejected(self):
        transport = self.transport(f"print('x'*{MAX_RESPONSE_BYTES + 2},flush=True)")
        with self.assertRaises(ExecutionError):
            transport.receive(time.monotonic() + 5)

    def test_invalid_json_is_rejected(self):
        transport = self.transport("print('log on stdout',flush=True)")
        with self.assertRaises(ExecutionError):
            transport.receive(time.monotonic() + 5)

    def test_eof_is_rejected(self):
        transport = self.transport("pass")
        with self.assertRaises(ExecutionError):
            transport.receive(time.monotonic() + 5)


if __name__ == "__main__":
    unittest.main()
