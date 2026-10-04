"""Create a credential-free audit summary from immutable real evaluations."""
import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path
from compare_versions import load, compare, compare_evaluations, evaluation_medians

ROOT = Path(__file__).resolve().parents[1]

def describe(entry):
    project = ROOT / entry["project"]
    paths = [ROOT / p for p in entry["runs"]]
    groups = load(paths)
    files = [project / "agent.py", project / "observer.project.json", *sorted((project / "agent_core").glob("*.py"))]
    hashes = {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    tree = ast.parse((project / "agent_core/llm_client.py").read_text(encoding="utf-8"))
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults))
    config = {k: ast.literal_eval(v) for k, v in defaults.items() if k != "log"}
    config.update(model="k3", base_url="https://api.kimi.com/coding/v1", reasoning_effort="none", max_tokens=250,
                  temperature="provider default", wallclock_limit_seconds=900)
    all_runs = [r for rows in groups.values() for r in rows]
    trace_stats = []
    for r in all_runs:
        trace = (ROOT / r["outputs"]["agent.log"]).parent / "model_trace.jsonl"
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()] if trace.exists() else []
        finishes = [row for row in records if row.get("event") == "finish"]
        trace_stats.append({"card": r["card"], "file": str(trace.relative_to(ROOT)),
                            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest() if trace.exists() else None,
                            "stages_applied": finishes[-1].get("model_stages_applied", []) if finishes else [],
                            "advice_rejected": sum(row.get("event") == "model_advice_applied" and not row.get("accepted") for row in records),
                            "request_bundles": sum(row.get("event") == "request_bundle" for row in records),
                            "request_executions": sum(row.get("event") == "request_execution" for row in records),
                            "diagnostic_pairs": sum(row.get("event") == "diagnostic_pair" for row in records),
                            "pointing_selections": sum(row.get("event") == "pointing_selection" for row in records),
                            "pointing_switches": sum(row.get("event") == "pointing_selection" and row.get("selected_rank", 0) > 0 for row in records),
                            "required_protected_selections": sum(row.get("event") == "pointing_selection" and row.get("protected_required", 0) > 0 for row in records),
                            "quality_guard_selections": sum(row.get("event") == "pointing_selection" and bool(row.get("quality_guard")) for row in records),
                            "lookahead_complete": sum(row.get("event") == "lookahead" and bool(row.get("complete")) for row in records),
                            "lookahead_changed": sum(row.get("event") == "lookahead" and bool(row.get("changed")) for row in records),
                            "request_deferred": sum(row.get("event") == "request_deferred" for row in records),
                            "short_diagnostic_exposures": sum(row.get("event") == "diagnostic_short_exposure" for row in records),
                            "quality_calibration_exposures": sum(row.get("event") == "diagnostic_short_exposure" and row.get("purpose") == "uncensored_quality_calibration" for row in records),
                            "terminal_diagnostics": sum(row.get("event") == "terminal_diagnostic" for row in records)})
    keys = ["total", "sum_best_scores", "required_missing", "required_penalty", "uniformity_penalty",
            "report_settlement", "observation_request_reward", "wall_seconds", "model_attempts", "model_seconds"]
    medians = {card: {k: statistics.median(r[k] for r in rows) for k in keys if all(k in r for r in rows)}
               for card, rows in groups.items()}
    verified = all(r.get("source_sha256") == fingerprint and r.get("model_mode") == "configured-api"
                   and r.get("termination_reason") == "survey_complete" and not r.get("planner_errors", 0)
                   and not r.get("validation_errors", 0) for r in all_runs)
    return {**entry, "source_sha256": fingerprint, "model_configuration": config,
            "all_complete_same_source": verified, "samples_per_card": min(map(len, groups.values())),
            "cards": medians, "mean": statistics.mean(r["total"] for r in medians.values()),
            "minimum": min(r["total"] for r in medians.values()),
            "required_missing_total": sum(r["required_missing"] for r in medians.values()),
            "model_attempts_max": max(r.get("model_attempts", 0) for r in all_runs),
            "model_seconds_max": max(r.get("model_seconds", 0) for r in all_runs),
            "wall_seconds_max": max(r.get("wall_seconds", 0) for r in all_runs), "traces": trace_stats}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--whole-evaluations", action="store_true",
                        help="Also validate whole four-card evaluations and actual use of both model stages")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    descriptions = {e["name"]: describe(e) for e in manifest["experiments"]}
    if args.whole_evaluations:
        for entry in descriptions.values():
            paths = [ROOT / p for p in entry["runs"]]
            entry["whole_evaluations"] = evaluation_medians(paths)
            keys = ("card", "card_sha256", "card_source", "model_mode", "requested_wallclock_seconds", "engine_sha256", "engine_source_commit", "fair_clock",
                    "source_sha256", "total", "sum_best_scores", "required_missing", "required_penalty",
                    "uniformity_penalty", "report_settlement", "observation_request_reward", "wall_seconds",
                    "runner_seconds", "model_attempts", "model_failed_attempts", "model_seconds",
                    "model_stages_applied", "termination_reason", "planner_errors", "validation_errors")
            entry["card_runs"] = [{"evaluation": path.name, **{key: run[key] for key in keys if key in run}}
                                  for path in paths
                                  for run in json.loads((path / "summary.json").read_text(encoding="utf-8"))["runs"]]
    comparisons = {}
    for name, entry in descriptions.items():
        if entry.get("baseline"):
            baseline = descriptions[entry["baseline"]]
            if args.whole_evaluations:
                comparisons[name] = compare_evaluations([ROOT / p for p in baseline["runs"]], [ROOT / p for p in entry["runs"]])
            else:
                comparisons[name] = compare(load([ROOT / p for p in baseline["runs"]]), load([ROOT / p for p in entry["runs"]]))
    selected = descriptions[manifest["selected"]]
    if args.whole_evaluations and (not all(e["all_complete_same_source"] for e in descriptions.values())
                                  or not comparisons.get(selected["name"], {}).get("accepted")):
        raise ValueError("The selected candidate must pass both real-evaluation gates with verified frozen sources")
    report = {"note": "Real Kimi on the named card set; components are independently summarized medians. Practice and example cards do not establish formal or hidden performance.",
              "card_set": manifest.get("card_set", "not_declared"),
              "official_scored_acceptance": manifest.get("official_scored_acceptance", False),
              "selected": manifest["selected"], "experiments": descriptions, "comparisons": comparisons}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({n: {k: e[k] for k in ["mean", "minimum", "required_missing_total", "all_complete_same_source"]}
                      for n, e in descriptions.items()}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
