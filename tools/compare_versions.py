"""Compare paired real-model evaluations using per-card medians."""
import argparse
import json
import statistics
from pathlib import Path


def load(paths):
    groups = {}
    for path in paths:
        for run in json.loads((path / "summary.json").read_text(encoding="utf-8"))["runs"]:
            groups.setdefault(run["card"], []).append(run)
    if set(groups) != {"L1", "L2", "L3", "L4"}:
        raise ValueError("four cards are required")
    return groups


def compare(baseline, candidate):
    rows = []
    complete = all(r["termination_reason"] == "survey_complete" and r.get("model_mode") == "configured-api"
                   and not r.get("planner_errors", 0) and not r.get("validation_errors", 0)
                   for groups in (baseline, candidate) for runs in groups.values() for r in runs)
    complete = complete and all(len({r.get("source_sha256") for runs in groups.values() for r in runs}) == 1
                                for groups in (baseline, candidate))
    for card in ("L1", "L2", "L3", "L4"):
        a, b = baseline[card], candidate[card]
        sa, sb = (statistics.median(r["total"] for r in runs) for runs in (a, b))
        rows.append({"card": card, "baseline": sa, "candidate": sb, "delta": sb - sa,
                     "relative_delta": (sb - sa) / max(abs(sa), 1),
                     "baseline_required_missing": statistics.median(r["required_missing"] for r in a),
                     "candidate_required_missing": statistics.median(r["required_missing"] for r in b)})
    means = [statistics.mean(r[key] for r in rows) for key in ("baseline", "candidate")]
    minima = [min(r[key] for r in rows) for key in ("baseline", "candidate")]
    samples = min(len(v) for groups in (baseline, candidate) for v in groups.values())
    needs_repeat = samples < 3 and (any(r["delta"] < 0 for r in rows) or abs(means[1] - means[0]) / max(abs(means[0]), 1) < .02)
    accepted = (complete and not needs_repeat and means[1] >= means[0] and minima[1] >= minima[0]
                and sum(r["candidate_required_missing"] for r in rows) <= sum(r["baseline_required_missing"] for r in rows))
    return {"cards": rows, "baseline_mean": means[0], "candidate_mean": means[1],
            "baseline_min": minima[0], "candidate_min": minima[1], "samples_per_card": samples,
            "needs_paired_repeats": needs_repeat, "accepted": accepted, "all_complete": complete}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(load(args.baseline), load(args.candidate)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
