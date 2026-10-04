"""Compare paired real-model evaluations using per-card medians."""
import argparse
import json
import statistics
from pathlib import Path
from card_sets import card_order


def load(paths):
    groups = {}
    for path in paths:
        for run in json.loads((path / "summary.json").read_text(encoding="utf-8"))["runs"]:
            groups.setdefault(run["card"], []).append(run)
    card_order(groups)
    return groups


def compare(baseline, candidate):
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate must use the same card set")
    rows = []
    complete = all(r["termination_reason"] == "survey_complete" and r.get("model_mode") == "configured-api"
                   and not r.get("planner_errors", 0) and not r.get("validation_errors", 0)
                   for groups in (baseline, candidate) for runs in groups.values() for r in runs)
    complete = complete and all(len({r.get("source_sha256") for runs in groups.values() for r in runs}) == 1
                                for groups in (baseline, candidate))
    for card in card_order(baseline):
        a, b = baseline[card], candidate[card]
        identities = {r.get("card_sha256") for r in a + b}
        if len(identities) != 1:
            raise ValueError("baseline and candidate must use identical card files: " + card)
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


def evaluation_medians(paths):
    """Preserve each four-card evaluation before taking medians."""
    evaluations = []
    for path in paths:
        runs = json.loads((path / "summary.json").read_text(encoding="utf-8"))["runs"]
        if len(runs) != 4 or len({r["card"] for r in runs}) != 4:
            raise ValueError("each evaluation must contain exactly four distinct cards")
        card_order(r["card"] for r in runs)
        evaluations.append({"evaluation": path.name,
                            "mean": statistics.mean(r["total"] for r in runs),
                            "minimum": min(r["total"] for r in runs),
                            "required_missing_total": sum(r["required_missing"] for r in runs)})
    return {"samples": len(evaluations), "evaluations": evaluations,
            **{key: statistics.median(row[key] for row in evaluations)
               for key in ("mean", "minimum", "required_missing_total")}}


def compare_evaluations(baseline_paths, candidate_paths):
    baseline, candidate = load(baseline_paths), load(candidate_paths)
    per_card = compare(baseline, candidate)
    a, b = evaluation_medians(baseline_paths), evaluation_medians(candidate_paths)
    stages = {"message_understanding", "plan_adaptation"}
    real_stages = all(r.get("model_attempts", 0) > 0 and stages <= set(r.get("model_stages_applied", []))
                      for groups in (baseline, candidate) for rows in groups.values() for r in rows)
    complete = per_card["all_complete"] and real_stages
    repeat = per_card["needs_paired_repeats"] or (
        min(a["samples"], b["samples"]) < 3 and abs(b["mean"] - a["mean"]) / max(abs(a["mean"]), 1) < .02)
    whole_ok = (b["mean"] >= a["mean"] and b["minimum"] >= a["minimum"]
                and b["required_missing_total"] <= a["required_missing_total"])
    return {**per_card, "per_card_accepted": per_card["accepted"],
            "whole_evaluations": {"baseline": a, "candidate": b, "accepted": whole_ok},
            "two_model_stages_verified": real_stages, "all_complete": complete,
            "needs_paired_repeats": repeat,
            "accepted": per_card["accepted"] and whole_ok and complete and not repeat}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(compare_evaluations(args.baseline, args.candidate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
