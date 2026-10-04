"""Run the unchanged official local engine with Python's current executable.

--mock-model uses a loopback HTTP fixture with fixed neutral advice. It checks
the protocol without real model calls; these scores are not a Kimi benchmark.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import hashlib
import os
import math
import shlex
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / ".local" / "runner"
from card_sets import CARD_SETS, card_path, require_runnable


class NeutralModel(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        type(self).calls += 1
        advice = {"avoid_directions": [], "duration_scale": 1.0}
        body = json.dumps({"choices": [{"message": {"content": json.dumps(advice)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", choices=[*sum(CARD_SETS.values(), ()), "all"], default="alpha")
    parser.add_argument("--card-set", choices=CARD_SETS, default="official")
    parser.add_argument("--wallclock", type=float, default=900)
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--project", type=Path, default=ROOT)
    args = parser.parse_args()
    for card in (CARD_SETS[args.card_set] if args.card == "all" else [args.card]):
        try:
            require_runnable(card)
        except ValueError as exc:
            parser.error(str(exc))
    project = args.project.resolve()
    if not math.isfinite(args.wallclock) or not 0 < args.wallclock <= 900:
        parser.error("--wallclock must be between 0 and 900 seconds")
    if not (RUNNER / "run_local.py").is_file():
        parser.error("official local runner is missing from .local/runner")
    output = (args.out or ROOT / "run_output" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    if output.exists():
        parser.error("--out already exists; choose a new directory")

    sys.path.insert(0, str(RUNNER))
    import run_local
    if sys.platform == "win32":
        from windows_transport import WindowsTransport
        run_local.JsonlTransport = WindowsTransport

    server = None
    worker = None
    original_environment = run_local.agent_environment
    credentials = run_local.load_dotenv(ROOT / ".env")
    def configured_environment(*values):
        env, keys = original_environment(*values)
        env.update({key: value for key, value in credentials.items() if key.startswith(("OPENAI_", "KIMI_"))})
        env["AGENT_TRACE_PATH"] = str(output / args.card / "model_trace.jsonl")
        return env, keys
    run_local.agent_environment = configured_environment
    if args.mock_model:
        server = ThreadingHTTPServer(("127.0.0.1", 0), NeutralModel)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base_url = f"http://127.0.0.1:{server.server_port}/v1"

        def mock_environment(*values):
            env, keys = configured_environment(*values)
            # Override AFTER loading .env, so the fixture never uses a real key.
            env.update(OPENAI_BASE_URL=base_url, OPENAI_API_KEY="local-test-only", OPENAI_MODEL="neutral-fixture")
            env.pop("KIMI_API_KEY", None)
            return env, keys

        run_local.agent_environment = mock_environment

    output.mkdir(parents=True)
    summaries = []
    failed = False
    command = shlex.join([sys.executable, "-u", str(project / "agent.py")])
    source_files = [project / "agent.py", project / "observer.project.json", *sorted((project / "agent_core").glob("*.py"))]
    source_hashes = {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    try:
        for card in (CARD_SETS[args.card_set] if args.card == "all" else [args.card]):
            if any(hashlib.sha256((project / name).read_bytes()).hexdigest() != digest for name, digest in source_hashes.items()):
                raise RuntimeError("evaluation source changed; freeze the project before evaluating")
            (output / card).mkdir(parents=True)
            def card_environment(*values):
                env, keys = (mock_environment if args.mock_model else configured_environment)(*values)
                env["AGENT_TRACE_PATH"] = str(output / card / "model_trace.jsonl")
                return env, keys
            run_local.agent_environment = card_environment
            print(f"Running {card}: {'MOCK MODEL' if args.mock_model else 'configured model'}, {args.wallclock:g}s limit", flush=True)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                status = run_local.main([
                    "--card", str(card_path(card)),
                    "--agent", command, "--agent-cwd", str(project),
                    "--wallclock", str(args.wallclock), "--out", str(output / card), "--quiet",
                ])
            summary = json.loads(captured.getvalue())
            summary["official_card_id"] = summary["card"]
            summary["card"] = card
            input_hashes = {p.relative_to(card_path(card)).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(card_path(card).rglob("*")) if p.is_file()}
            summary["card_sha256"] = hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()
            summary["card_source"] = ("https://create.gosim.org/survey26/platform/cards/" + card
                                      if card in CARD_SETS["official"] else "UPSTREAM.json example archive")
            summary["model_mode"] = "mock-neutral" if args.mock_model else "configured-api"
            summary["requested_wallclock_seconds"] = args.wallclock
            summary["source_sha256"] = fingerprint
            summary["model"] = "neutral-fixture" if args.mock_model else credentials.get("OPENAI_MODEL", credentials.get("KIMI_MODEL", "k3"))
            trace_path = output / card / "model_trace.jsonl"
            trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()] if trace_path.exists() else []
            calls = [r for r in trace if r.get("event") == "model_call"]
            summary["model_attempts"] = len(calls)
            summary["model_failed_attempts"] = sum(not r.get("ok") for r in calls)
            summary["model_seconds"] = round(sum(r.get("seconds", 0) for r in calls), 4)
            summary["model_stages"] = sorted({r.get("stage", "legacy") for r in calls if r.get("ok")})
            finishes = [r for r in trace if r.get("event") == "finish"]
            summary["model_stages_applied"] = finishes[-1].get("model_stages_applied", []) if finishes else []
            log_path = output / card / "agent.log"
            log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.exists() else []
            summary["planner_errors"] = sum("planner error (" in line or "failed to initialize" in line for line in log_lines)
            summary["validation_errors"] = sum("invalid action (" in line or "planner produced an invalid action (" in line for line in log_lines)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            failed |= status != 0
    finally:
        run_local.agent_environment = original_environment
        if server is not None:
            server.shutdown()
            server.server_close()
            worker.join()
        (output / "summary.json").write_text(json.dumps({
            "runs": summaries,
            "source_files": source_hashes,
            "mock_model_calls": NeutralModel.calls if args.mock_model else None,
            "note": "Mock scores are not Kimi benchmarks. alpha-delta are official practice; L1-L4 are example cards. Neither is the formal or hidden evaluation.",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
