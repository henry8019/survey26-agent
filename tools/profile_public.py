"""Geometry/latency smoke checks using published official inputs, without weather.

This is not a scored evaluation. No weather, forecasts or events are synthesized.
The runner's fibre classifier and whole-exposure altitude check verify actions.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from card_sets import CARD_SETS, ROOT, card_path


def initialize(card):
    root = card_path(card)
    config = json.loads((root / "config/v4_scenario.json").read_text(encoding="utf-8"))
    fiber = json.loads((root / "config/v4_fiber_config.json").read_text(encoding="utf-8"))
    score = json.loads((root / "config/v4_score_config.json").read_text(encoding="utf-8"))
    from challenge.v4_fiber_map import FiberGrid
    grid = FiberGrid.from_config(fiber)
    def rows(name):
        with (root / "public" / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    nights = rows("v4_night_calendar.csv")
    columns = ["target_id", "ra_deg", "dec_deg", "feature_flux", "science_weight", "required"]
    targets = [[r["target_id"], *[float(r[c]) for c in columns[1:5]], r["required"].lower() == "true"]
               for r in rows("targets.csv")]
    return {"site": {**config["site"], "minimum_altitude_deg": config["minimum_altitude_deg"]},
            "survey": {"start_utc": nights[0]["observing_start_utc"], "end_utc": nights[-1]["observing_end_utc"],
                       "slot_seconds": 900, "nights": nights},
            "instrument": {"n_fibers": grid.n_fibers, "grid_side": grid.n_side,
                           "glass_side_deg": round(grid.fiber_side_deg, 6), "pitch_deg": round(grid.pitch_deg, 6),
                           "fov_side_deg": round(grid.fov_side_deg, 6), "exposure": fiber["exposure"]},
            "scoring": score, "targets": {"columns": columns, "rows": targets}}, fiber


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / ".local/runner"))
    sys.path.insert(0, str(args.project.resolve()))
    from agent_core.planner import Planner
    from agent_core.state import SurveyState
    from agent_core.validation import validate_action
    from challenge.v4_fiber_map import FiberGrid, min_altitude_during
    results = []
    for card in CARD_SETS["official"]:
        init, fiber = initialize(card)
        state = SurveyState(init)
        planner = Planner(state)
        grid = FiberGrid.from_config(fiber)
        for night in (0, len(state.nights) // 2, len(state.nights) - 1):
            start, end = state.nights[night]
            now = start + timedelta(hours=3)
            before = time.perf_counter()
            action = planner.plan(now, end, night, (now - state.survey_start).total_seconds() / 3600)
            seconds = time.perf_counter() - before
            if action is None:
                results.append({"card": card, "night": night, "seconds": seconds, "assigned": 0})
                continue
            validate_action(action, state)
            errors = []
            for f, target in action["assignments"].items():
                i = state.index_of[target]
                location = grid.classify_target(state.ra[i], state.dec[i], now, action["pointing"]["alt_deg"],
                                                 action["pointing"]["az_deg"], state.lat, state.lon)
                alt = min_altitude_during(state.ra[i], state.dec[i], now,
                                         now + timedelta(seconds=action["duration_seconds"]), fiber)
                if location.fiber_id != int(f) or alt < state.min_alt:
                    errors.append(target)
            results.append({"card": card, "night": night, "seconds": seconds, "assigned": len(action["assignments"]),
                            "duration_seconds": action["duration_seconds"], "geometry_errors": errors})
            if errors:
                raise ValueError("public geometry check failed: " + card)
    project = args.project.resolve()
    files = [project / "agent.py", project / "observer.project.json", *sorted((project / "agent_core").glob("*.py"))]
    hashes = {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    inputs = {card: {p.relative_to(card_path(card)).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(card_path(card).rglob("*")) if p.is_file()}
              for card in CARD_SETS["official"]}
    report = {"kind": "public_geometry_only_no_score", "assumed_quality_scale": 1.0,
              "assumed_slot_seconds": 900, "project": str(args.project), "samples": results,
              "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
              "source_files": hashes, "public_input_files": inputs}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(results), "geometry_errors": 0, "max_seconds": max(r["seconds"] for r in results)}))


if __name__ == "__main__":
    main()
