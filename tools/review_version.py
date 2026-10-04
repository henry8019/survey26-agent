"""Review completed runs using public geometry and post-run scoring artifacts.

This is an offline audit tool. It never supplies results to the running agent.
Public ideal-sky estimates are diagnostic references, not actual-weather claims.
"""
import argparse
import collections
import csv
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_core.geometry import (Moon, SIDEREAL_DEG_PER_SECOND, local_sidereal_deg,
                                 lunar_factor, max_hour_angle_deg, parse_utc,
                                 radec_to_altaz, wrap180)
from agent_core.scoring import ScoringModel


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ideal_feasibility(target, site, scoring, exposure, nights):
    ra, dec, flux = (float(target[k]) for k in ("ra_deg", "dec_deg", "feature_flux"))
    h = max_hour_angle_deg(dec, site["latitude_deg"], site["minimum_altitude_deg"])
    best_cost, feasible_nights = None, 0
    for night in nights:
        start, end = (parse_utc(night[k]) for k in ("observing_start_utc", "observing_end_utc"))
        probe, feasible = start, False
        while probe < end:
            lst = local_sidereal_deg(probe, site["longitude_deg"])
            ha = wrap180(lst - ra)
            if -h <= ha <= h:
                alt, _ = radec_to_altaz(ra, dec, lst, site["latitude_deg"])
                moon = Moon(probe, lst, site["latitude_deg"])
                quality = scoring.quality_model(alt, lunar_factor(moon, ra, dec, scoring.lunar_model))
                cost = max(exposure["min_duration_seconds"],
                           math.ceil(scoring.required_threshold * scoring.f0t0 / max(1e-9, flux * quality)))
                setting = (h - ha) / SIDEREAL_DEG_PER_SECOND if h < 180 else 1e9
                if cost <= min(exposure["max_duration_seconds"], (end - probe).total_seconds(), setting):
                    feasible = True
                    best_cost = cost if best_cost is None else min(best_cost, cost)
            probe += timedelta(minutes=15)
        feasible_nights += feasible
    return best_cost, feasible_nights


def review_card(run, cards_root, card):
    folder = run / card / card
    public = cards_root / card
    scenario = json.loads((public / "config/v4_scenario.json").read_text(encoding="utf-8"))
    site = {**scenario["site"], "minimum_altitude_deg": scenario["minimum_altitude_deg"]}
    scoring = ScoringModel(json.loads((public / "config/v4_score_config.json").read_text(encoding="utf-8")), site)
    exposure = json.loads((public / "config/v4_fiber_config.json").read_text(encoding="utf-8"))["exposure"]
    targets = read_csv(public / "public/targets.csv")
    required = {t["target_id"]: t for t in targets if t["required"].lower() in {"true", "1", "yes"}}
    nights = read_csv(public / "public/v4_night_calendar.csv")
    observations = collections.defaultdict(list)
    for row in read_csv(folder / "observations.csv"):
        observations[row["target_id"]].append(row)
    actions = [json.loads(line) for line in (folder / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assigned = collections.Counter(t for a in actions for t in a.get("assignments", {}).values())
    categories, cost_categories = collections.Counter(), collections.Counter()
    for target_id, target in required.items():
        rows = observations[target_id]
        maximum = max((float(r["factor"]) for r in rows if r["valid"] == "true"), default=0)
        if maximum >= scoring.required_threshold:
            continue
        category = ("never_assigned" if not assigned[target_id] else
                    "assigned_without_hit" if not rows else "hit_below_threshold")
        categories[category] += 1
        if any(float(r["factor"]) >= scoring.required_threshold for r in rows):
            categories["lost_completed_exposure"] += 1
        cost, _ = ideal_feasibility(target, site, scoring, exposure, nights)
        cost_categories[category + ("_ideal_sample_feasible" if cost is not None else "_no_ideal_sample_found")] += 1
    report = json.loads((folder / "score_report.json").read_text(encoding="utf-8"))
    assert sum(categories[k] for k in ("never_assigned", "assigned_without_hit", "hit_below_threshold")) == report["counts"]["required_missing"]
    trace = [json.loads(line) for line in (folder / "model_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    decisions = read_csv(folder / "decisions.csv")
    return {"required_count": len(required), "missing_categories": dict(categories),
            "public_ideal_15minute_sampling": dict(cost_categories), "components": report["components"],
            "assigned_count": sum(int(r["assigned_count"]) for r in decisions if r["action"] == "observe"),
            "hit_count": sum(int(r["hit_count"]) for r in decisions if r["action"] == "observe"),
            "requests_completed": report["counts"]["observation_requests_completed"],
            "requests_issued": report["counts"]["observation_requests_issued"],
            "request_executions": sum(r.get("event") == "request_execution" for r in trace),
            "request_deferred": sum(r.get("event") == "request_deferred" for r in trace),
            "accepted_priorities": dict(collections.Counter(r["priority"] for r in trace
                if r.get("event") == "model_advice_applied" and r.get("stage") == "plan_adaptation" and r.get("accepted")))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cards-root", type=Path, default=Path(".local/local-cards"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"run": args.run.name,
              "note": "Offline post-run audit; ideal public estimates assume scale=1, use 15-minute samples, and exclude weather, events, pointing and fiber competition. No sample is not a proof of impossibility.",
              "cards": {card: review_card(args.run, args.cards_root, card) for card in ("L1", "L2", "L3", "L4")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
