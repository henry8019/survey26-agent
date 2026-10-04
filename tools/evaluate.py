"""Run a frozen candidate on four cards, with bounded local concurrency."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists")
    args.out.mkdir(parents=True)

    def run(card):
        completed = subprocess.run([sys.executable, str(ROOT / "tools/run_practice.py"), "--card", card,
                                    "--project", str(args.project.resolve()), "--out", str((args.out / card).resolve())],
                                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        summary = args.out / card / "summary.json"
        result = json.loads(summary.read_text(encoding="utf-8"))["runs"][0] if summary.exists() else {
            "card": card, "termination_reason": "evaluation_failed", "exit_code": completed.returncode}
        print(card, result.get("termination_reason"), result.get("total"), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(run, ("L1", "L2", "L3", "L4")))
    (args.out / "summary.json").write_text(json.dumps({"runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["termination_reason"] == "survey_complete" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
