"""Windows pipe I/O for local evaluation; the official scoring engine is unchanged."""
from __future__ import annotations

import json
import os
import subprocess
import threading

from project_platform.transport import (
    ExecutionError, GlobalDeadlineExpired, JsonlTransport,
    MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES,
)


class WindowsTransport(JsonlTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._condition = threading.Condition()
        self._stdout_error = None
        self._stdout_reader = None

    def start(self):
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            self.command, cwd=self.cwd, env=self.environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._reader = threading.Thread(target=self._drain_log, args=(self.process.stderr,), daemon=True)
        self._reader.start()
        self._stdout_reader = threading.Thread(target=self._drain_stdout, args=(self.process.stdout,), daemon=True)
        self._stdout_reader.start()

    def _drain_stdout(self, stream):
        try:
            while True:
                chunk = stream.read(65536)
                with self._condition:
                    if not chunk:
                        self._stdout_error = ExecutionError("Project exited without a complete response.")
                        self._condition.notify_all()
                        return
                    self._buffer.extend(chunk)
                    if len(self._buffer) > MAX_RESPONSE_BYTES + 1:
                        self._stdout_error = ExecutionError("Project response exceeds the size limit.")
                        self._condition.notify_all()
                        return
                    self._condition.notify_all()
        except OSError:
            with self._condition:
                self._stdout_error = ExecutionError("Project stdout pipe closed.")
                self._condition.notify_all()

    def send(self, message, deadline, *, limit=MAX_REQUEST_BYTES):
        self.start()
        data = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(data) > limit:
            raise ExecutionError("Public protocol request exceeds the size limit.")
        completed = threading.Event()
        errors = []
        stream = self.process.stdin

        def write():
            try:
                pending = memoryview(data)
                while pending:
                    count = os.write(stream.fileno(), pending)
                    if count <= 0:
                        raise OSError("pipe write failed")
                    pending = pending[count:]
            except (OSError, ValueError) as error:
                errors.append(error)
            finally:
                completed.set()

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        if not completed.wait(self._remaining(deadline)):
            self.close(force=True)
            writer.join(timeout=2)
            raise GlobalDeadlineExpired()
        if errors:
            raise ExecutionError("Project exited before reading a request.") from errors[0]

    def receive(self, deadline):
        self.start()
        with self._condition:
            while b"\n" not in self._buffer:
                if self._stdout_error is not None:
                    raise self._stdout_error
                # Wait without holding the condition while closing the process.
                import time
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            else:
                line, rest = self._buffer.split(b"\n", 1)
                self._buffer = bytearray(rest)
                if len(line) > MAX_RESPONSE_BYTES:
                    raise ExecutionError("Project response exceeds the size limit.")
                try:
                    payload = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                except (ValueError, UnicodeError) as error:
                    raise ExecutionError("Project stdout must contain JSON-Lines responses; send logs to stderr.") from error
                if not isinstance(payload, dict):
                    raise ExecutionError("Project response must be a JSON object.")
                return payload
        self.close(force=True)
        raise GlobalDeadlineExpired()

    def close(self, force=False):
        super().close(force=force)
        if self._stdout_reader is not None:
            self._stdout_reader.join(timeout=2)
