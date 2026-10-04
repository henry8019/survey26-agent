"""Public-input survey planning with linear marginal score and two bounded LLM stages.
The model interprets sourced messages and adapts priorities; geometry, completion
state, candidate exposures and final protocol actions remain deterministic.
"""
from __future__ import annotations

from datetime import timedelta
import math

from agent_core.geometry import (
    Moon,
    SIDEREAL_DEG_PER_SECOND,
    altaz_to_radec,
    format_utc,
    local_sidereal_deg,
    lunar_factor,
    max_hour_angle_deg,
    parse_utc,
    radec_to_altaz,
    shift_altaz,
    tangent_offsets,
    wrap180,
)
from agent_core.llm_client import LLMClient
from agent_core.memory import TraceLog
from .state import PendingPrediction
from agent_core.advice import AdviceController
from .calibration import PointingCalibration
from agent_core.requests import RequestPlanner
from .diagnostics import verify_pair

PLAN_FACTOR_SAFETY = 0.9
EDGE_MARGIN_DEG = 0.08
DURATIONS = (300, 450, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600)
ANCHORS = 6
ANCHOR_POOL = 300
BLOCKING_KINDS = {"terrain_obstruction", "rocket_launch"}
DIRECTION_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0,
                "SW": 225.0, "W": 270.0, "NW": 315.0}


# Opportunistic duration tie-break; whole-request rewards are handled separately
# in RequestPlanner and are never added to individual target scores.
REQUEST_RATE_TOLERANCE = 0.9


def _az_distance(a: float, b: float) -> float:
    return abs(wrap180(a - b))


class Planner:
    def __init__(self, state, log=lambda text: None):
        self.state = state
        self.log = log
        self.grid = state.fiber_grid
        self.llm = LLMClient(log=log)
        self.trace = TraceLog(log=log)
        self.advice = AdviceController(state, self.llm, self.trace)
        self.calibration = PointingCalibration(self.grid)
        self.geometry_pending = None
        self.diagnostic_ceiling = None
        self.requests = RequestPlanner(state, self.trace)
        self._forced_request = None
        self.probe_first = None
        self.probe_mode = "DARK"
        self.verified_probe_fields = set()
        self.verified_probe_upper = None
        self.last_probe_hours = float("-inf")
        self.probe_night = None
        self.probes_this_night = 0

        self.observe_count = 0
        self.reports = 0
        self.last_report_hours = float("-inf")
        self.suspicion_hours: list[float] = []
        self.night_index_seen: int | None = None
        self.consecutive_reports = 0
        self._last_forecast_notices: list = []
        self.total_assigned = 0
        self.total_hit = 0
        self._current_action_index = None
        self._request_thresholds_now: dict = {}

        log(f"planner: {len(state.ids)} targets ({sum(state.required)} required), "
            f"{len(state.nights)} nights, llm model={self.llm.model} base_url={self.llm.base_url}")

    # -- top-level decision ----------------------------------------------------

    def decide(self, payload: dict) -> dict:
        state = self.state
        now = parse_utc(payload["now_utc"])
        hours = (now - state.survey_start).total_seconds() / 3600.0

        for message in payload.get("new_messages", []):
            if message.get("record_type") == "forecast":
                self._last_forecast_notices = message.get("notices", [])
            elif message.get("record_type") == "observation_request":
                self.log(f"planner: observation request {message.get('request_id')} issued, "
                         f"{len(message.get('target_ids', []))} targets by {message.get('deadline_utc')}")
            elif message.get("record_type") == "observation_request_result":
                self.log(f"planner: observation request {message.get('request_id')} "
                         f"{message.get('status')} (reward {message.get('score_delta', 0.0)})")
        result = payload.get("last_result") or {}
        if result.get("action") == "report":
            self.probe_first = None
            self.verified_probe_fields.clear()
            self.verified_probe_upper = None
        if result.get("action") == "observe" and self.probe_first:
            hits = {h["target_id"]: h["score"] for h in result.get("hits", [])}
            if self.probe_first.get("backup_sent"):
                evidence = state.fault_evidence()
                upper = verify_pair(state, self.probe_first, hits, state.pending, state.pending_duration,
                                    evidence.earlier_median if evidence else 0)
                if upper is not None:
                    self.verified_probe_fields.add(self.probe_first["field"])
                    self.verified_probe_upper = upper
                self.trace.write({"event": "diagnostic_pair", "efficiency_upper": upper,
                                  "program": self.probe_first["program"], "fields": len(self.verified_probe_fields),
                                  "earlier_lower": evidence.earlier_median if evidence else 0,
                                  "first_scores": self.probe_first["hits"], "second_scores": hits,
                                  "first_duration": self.probe_first["duration"], "second_duration": state.pending_duration})
                self.probe_mode = "BRIGHT" if self.probe_mode == "DARK" else "DARK"
                self.probe_first = None
            else:
                self.probe_first["hits"] = hits
        if result.get("action") == "observe" and self.geometry_pending:
            pointing, assignments, positions = self.geometry_pending
            changed = self.calibration.observe(pointing, assignments, {h["target_id"] for h in result.get("hits", [])}, positions)
            if changed:
                state.misses = [0] * len(state.ids)
                self.trace.write({"event": "pointing_calibration", "alt_bias": self.calibration.alt_bias, "az_bias": self.calibration.az_bias})
        self.geometry_pending = None
        state.on_result(result, hours)
        state.on_messages(payload.get("new_messages", []), payload.get("latest_bulletin"))
        last_result = payload.get("last_result")
        if last_result and last_result.get("action") == "observe":
            self.total_assigned += int(last_result.get("assigned_count", 0))
            self.total_hit += int(last_result.get("hit_count", 0))
        self._pace(payload, now)
        self._current_action_index = payload.get("observe_action_index")
        self._request_thresholds_now = self._request_thresholds(payload.get("active_requests") or [])
        self.requests.sync(payload.get("active_requests") or [])

        night = state.current_night(now)
        if night is None:
            nxt = state.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "no observing night left"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "daytime: sleep until the next night"}
        night_index, night_start, night_end = night

        self.advice.update(payload, night_index, now)

        if (night_end - now).total_seconds() < state.min_exposure:
            nxt = state.next_night_start(now)
            if nxt is None:
                return {"action": "wait", "until_utc": format_utc(state.survey_end), "reason": "last night ending"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "night ending"}

        if state.site_closed():
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "bulletin: rain/storm over the whole sky"}

        report = self._maybe_report(hours, payload)
        if report is not None:
            return report
        if self.calibration.searching:
            self.diagnostic_ceiling = max(state.min_exposure, min(state.max_exposure, 120))

        if self.probe_first and self.probe_first.get("hits"):
            backup = self._backup_probe(now, night_end, night_index)
            if backup is not None:
                self.observe_count += 1
                return backup
            self.probe_first = None

        action = self.plan(now, night_end, night_index, hours)
        if not self.diagnostic_ceiling and self.requests.requests:
            normal_rate = self._action_gain_rate(action, now) if action is not None else 0
            bundle = self.requests.choose(now, normal_rate, self.advice.priority == "requests")
            if bundle is not None and bundle["start"] <= now:
                self._forced_request = bundle
                try:
                    request_action = self.plan(now, min(night_end, bundle["deadline"]), night_index, hours)
                    if request_action is not None:
                        action = request_action
                        self.trace.write({"event": "request_execution", "request_id": bundle["request_id"],
                                          "target_id": state.ids[bundle["target"]], "now_utc": payload["now_utc"],
                                          "duration_seconds": action["duration_seconds"]})
                finally:
                    self._forced_request = None
        if action is None:
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "nothing useful is up"}
        self.observe_count += 1
        action["reason"] = f"{len(action['assignments'])} fibres, program {action['program']}"
        return action

    def on_finish(self, payload: dict) -> None:
        self.trace.write({"event": "finish", "model_attempts": self.llm.calls_made,
                          "model_seconds": self.llm.spent_seconds,
                          "model_stages_applied": sorted(self.advice.applied_stages), **payload})
        self.trace.close()
        self.llm.audit.close()
        self.log(f"planner: finished termination_reason={payload.get('termination_reason')} "
                 f"observes={self.observe_count} reports={self.reports} llm_calls={self.llm.calls_made}")

    def note_action(self, action: dict) -> None:
        """Called by agent.py right after an action is validated, so the consecutive-report
        counter (enforced by validation.py) stays correct even when a fallback replaced it."""
        self.consecutive_reports = self.consecutive_reports + 1 if action.get("action") == "report" else 0
        if action.get("action") == "observe" and self.state.pending_start:
            state = self.state
            lst = local_sidereal_deg(state.pending_start, state.lon)
            positions = {t: radec_to_altaz(state.ra[state.index_of[t]], state.dec[state.index_of[t]], lst, state.lat)
                         for t in action["assignments"].values()}
            self.geometry_pending = ((action["pointing"]["alt_deg"], action["pointing"]["az_deg"]),
                                     {t: int(f) for f, t in action["assignments"].items()}, positions)
            if self.diagnostic_ceiling and state.force_program in {"DARK", "BRIGHT"} and self.probe_first is None:
                if self.probe_night != state.pending_night:
                    self.probe_night, self.probes_this_night = state.pending_night, 0
                self.probes_this_night += 1
                self.last_probe_hours = (state.pending_start - state.survey_start).total_seconds() / 3600
                a, z = action["pointing"]["alt_deg"], action["pointing"]["az_deg"]
                self.probe_first = {"program": action["program"], "duration": action["duration_seconds"],
                                    "assignments": action["assignments"], "field": (int(z // 45), int(a // 20)),
                                    "radec": altaz_to_radec(a + self.calibration.alt_bias, (z + self.calibration.az_bias) % 360, lst, state.lat),
                                    "models": {t: v.model for t, v in state.pending.items()}}

    def _backup_probe(self, now, night_end, night_index):
        state, first = self.state, self.probe_first
        duration = first["duration"]
        if state.all_sky_notice() or (night_end - now).total_seconds() < duration:
            return None
        lst = local_sidereal_deg(now, state.lon)
        alt, az = radec_to_altaz(*first["radec"], lst, state.lat)
        moon = Moon(now, lst, state.lat)
        pending = {}
        for target in first["assignments"].values():
            i = state.index_of[target]
            a, z = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
            ha = wrap180(lst - state.ra[i])
            if state.hmax[i] < 180 and state.hmax[i] - ha < duration * SIDEREAL_DEG_PER_SECOND:
                return None
            model = state.scoring.quality_model(a, lunar_factor(moon, state.ra[i], state.dec[i], state.scoring.lunar_model))
            pending[target] = PendingPrediction(model, model / .95, a, z, True)
        state.pending = pending
        state.pending_start, state.pending_duration = now, duration
        state.pending_program, state.pending_night = "BACKUP", night_index
        state.pending_action_index = self._current_action_index
        first["backup_sent"] = True
        return {"action": "observe", "pointing": dict(zip(("alt_deg", "az_deg"), self.calibration.command(alt, az))),
                "assignments": first["assignments"], "program": "BACKUP", "duration_seconds": duration,
                "reason": "paired program diagnostic"}

    def _to_next_slot(self, now, night_start) -> int:
        slot = self.state.slot_seconds
        into = (now - night_start).total_seconds() % slot
        return int(max(self.state.min_exposure, min(self.state.max_exposure, slot - into if into else slot)))

    def _pace(self, payload: dict, now) -> None:
        """Do less work per decision when the wall clock is short for the nights still to come."""
        state = self.state
        remaining_wall = float((payload.get("wallclock") or {}).get("remaining_seconds", 1e9))
        night_seconds = sum(max(0.0, (end - max(start, now)).total_seconds()) for start, end in state.nights if end > now)
        decisions_left = max(1.0, night_seconds / 700.0)
        per_decision = remaining_wall / decisions_left
        level = 0 if per_decision > 0.12 else 1 if per_decision > 0.04 else 2
        if level != state.fast_level:
            self.log(f"planner: pace level {level} ({per_decision * 1000:.0f} ms per decision left)")
            state.fast_level = level

    # -- instrument evidence and bounded diagnostic probes ----------------------

    def _maybe_report(self, hours: float, payload: dict):
        state = self.state
        state.force_program = None
        self.diagnostic_ceiling = None
        if self.calibration.searching:
            return None
        evidence = state.fault_evidence()
        science_evidence = state.science_fault_evidence()
        # Retain the accepted, long-window science diagnostic as a second
        # evidence stream. Interval bounds alone can miss moderate degradation;
        # reports still require repeated verified band/field evidence below.
        if science_evidence is not None and (evidence is None or science_evidence.drop < evidence.drop):
            evidence = science_evidence
        if evidence is None or evidence.drop >= .62:
            self.suspicion_hours = []
            return None
        if hours - self.last_report_hours < 2 or self.consecutive_reports >= state.max_consecutive_reports:
            return None
        # Evidence was accumulated only from public clean, geometrically hit
        # targets. A warning elsewhere does not invalidate those earlier samples.
        band_confirmed = evidence.dark_checks >= 6 and evidence.dark_matched >= .5 * evidence.dark_checks
        if len(self.verified_probe_fields) < 2 and not band_confirmed:
            # Repeated saturated mismatches are evidence of a different sky band.
            # Keep normal science useful instead of endlessly forcing DARK.
            if evidence.dark_checks < 6:
                state.force_program = "DARK"
            else:
                self.suspicion_hours = []
            night = state.current_night(state.survey_start + timedelta(hours=hours))
            index = night[0] if night else None
            used = self.probes_this_night if self.probe_night == index else 0
            if used < 2 and hours - self.last_probe_hours >= 2:
                self.diagnostic_ceiling = max(state.min_exposure, min(state.max_exposure, 120))
                state.force_program = self.probe_mode
            return None
        if len(self.verified_probe_fields) < 2:
            # A single matched program result is insufficient: require repeated
            # clean evidence over different nights before the science-derived path.
            history = state.clean_history[-60:]
            if len(history) < 60 or len({r[1] for r in history}) < 2 or history[-1][0] - history[0][0] < 4:
                return None
            if self.suspicion_hours and hours - self.suspicion_hours[-1] < 6:
                return None
            self.suspicion_hours.append(hours)
            if len(self.suspicion_hours) < 3:
                return None
            self.suspicion_hours = []
        free = state.false_reports_since_repair < state.false_report_free_allowance
        confidence = .98
        net = confidence * state.report_reward - (1 - confidence) * (0 if free else state.false_report_penalty)
        if net <= 0:
            return None
        self.reports += 1
        self.last_report_hours = hours
        self.trace.write({"event": "instrument_report", "evidence": evidence._asdict(), "expected_report_gain": net, "free_false_report": free})
        return {"action": "report", "reason": "persistent clean geometric hits show an efficiency drop", "decision_source": "deterministic"}

    # -- planning value / achievability -----------------------------------------

    def _direction_factor(self, alt: float, az: float, *, use_model=True) -> float:
        state = self.state
        for direction in state.terrain:
            if direction in DIRECTION_AZ and alt < 50.0 and _az_distance(az, DIRECTION_AZ[direction]) <= 60.0:
                return 0.0
        risk = getattr(state, "model_altitude_risk", 0) if use_model else 0
        if use_model and not risk and any(key in {"haze|ALL", "cloud|ALL", "cold|ALL"} for key in state.notices):
            risk = .65  # Deterministic fallback for a currently published warning.
        factor = 1 - risk if alt < min(89, state.min_alt + 40) else 1.0
        for key in state.notices:
            kind, _, direction = key.partition("|")
            if direction not in DIRECTION_AZ:
                continue
            near = _az_distance(az, DIRECTION_AZ[direction]) <= 67.5
            if kind in BLOCKING_KINDS and near and alt < 62.0:
                return 0.0
            if near and alt < 75.0:
                factor = min(factor, 0.35)
        for direction in state.extra_avoid if use_model else ():
            if direction in DIRECTION_AZ and _az_distance(az, DIRECTION_AZ[direction]) <= 67.5 and alt < 70.0:
                factor = min(factor, getattr(state, "model_direction_factors", {}).get(direction, .35))
        for blocked_az, blocked_alt in state.blocked[-40:]:
            if _az_distance(az, blocked_az) <= 12.0 and alt <= blocked_alt + 3.0:
                factor = min(factor, 0.2)
        return factor

    def _request_thresholds(self, active_requests: list) -> dict:
        """target_index -> smallest still-needed completion_factor_threshold, for every
        target that is part of some still-open request (`remaining_count>0`), not
        already counted (`completed_target_ids`), and not already past that threshold.
        Used only as a read-only tie-break in _finish_plan's duration search -- it never
        feeds back into _value/achievable, so it cannot change which pointing gets
        chosen, only (within REQUEST_RATE_TOLERANCE) how long an already-chosen exposure
        runs once request targets happen to already be among its assigned fibres."""
        state = self.state
        thresholds: dict[int, float] = {}
        for request in active_requests:
            if int(request.get("remaining_count", 0)) <= 0:
                continue
            threshold = float(request.get("completion_factor_threshold", 1.0))
            completed = set(request.get("completed_target_ids") or [])
            for target_id in request.get("target_ids", []):
                if target_id in completed:
                    continue
                i = state.index_of.get(target_id)
                if i is None:
                    continue
                if i not in thresholds or threshold < thresholds[i]:
                    thresholds[i] = threshold
        return thresholds

    def _completion_goal(self):
        scoring = self.state.scoring
        multipliers = list(scoring.program_multipliers.values()) + [scoring.mismatch_multiplier]
        return min(1.0, scoring.required_threshold * max(multipliers) / min(m for m in multipliers if m > 0) + 0.00001)

    def _action_gain_rate(self, action, now):
        state = self.state
        lst = local_sidereal_deg(now, state.lon)
        moon = Moon(now, lst, state.lat)
        gain = 0
        for target_id in action["assignments"].values():
            i = state.index_of[target_id]
            alt, _ = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
            model = state.scoring.quality_model(alt, lunar_factor(moon, state.ra[i], state.dec[i], state.scoring.lunar_model))
            reached = state.scoring.completion_factor(state.flux[i], action["duration_seconds"], model * state.scale * PLAN_FACTOR_SAFETY)
            band = state.scoring.program_band(model * state.scale / .95)
            gain += self._gain(i, reached, action["program"], band)
        return gain / action["duration_seconds"]

    def _value(self, i):
        state = self.state
        gain = max(0.0, state.weight[i] * state.scoring.maximum_multiplier - state.best_score[i])
        if state.required[i] and state.factor[i] < state.scoring.required_threshold:
            gain += state.scoring.required_penalty
        return gain * (0.6 ** min(state.misses[i], 8))

    def _gain(self, i, reached, program=None, band=None):
        state = self.state
        if program is None:
            multiplier = state.scoring.maximum_multiplier
        else:
            multiplier = state.scoring.program_multiplier(program, band)
        gain = max(0.0, state.weight[i] * reached * multiplier - state.best_score[i])
        if state.required[i] and state.factor[i] < state.scoring.required_threshold and reached >= self._completion_goal():
            gain += state.scoring.required_penalty
        return gain

    def _required_window_boost(self, i, alt, moon, window, seconds_left, nights_left):
        state = self.state
        quality = state.scoring.quality_model(alt, lunar_factor(moon, state.ra[i], state.dec[i], state.scoring.lunar_model))
        cost = self._completion_goal() * state.scoring.f0t0 / max(1e-9, state.flux[i] * state.scale * PLAN_FACTOR_SAFETY * quality)
        available = min(state.max_exposure, window, seconds_left)
        # Urgency is useful only if one valid exposure can reach the threshold.
        # Repeated sub-threshold exposures do not accumulate; avoid chasing
        # impossible targets merely because their window is about to close.
        if cost > available or nights_left > 1 or window > 2 * cost:
            return 1.0
        return 1 + cost / max(state.min_exposure, window) / nights_left

    # -- main planning pass -------------------------------------------------------

    def plan(self, now, night_end, night_index: int, hours: float):
        state = self.state
        state.update_scale(hours)
        lst = local_sidereal_deg(now, state.lon)
        horizon = min(night_end, state.survey_end)
        seconds_left = (horizon - now).total_seconds()
        if seconds_left < state.min_exposure:
            return None
        min_visible = min(state.min_exposure, seconds_left) * SIDEREAL_DEG_PER_SECOND

        moon = Moon(now + timedelta(seconds=450), lst, state.lat)
        still_active = []
        candidates: list[tuple[float, int]] = []
        forced = self._forced_request["target"] if self._forced_request else None
        active = range(len(state.ids)) if self.diagnostic_ceiling else (state.active if forced is None or forced in state.active else [*state.active, forced])
        for i in active:
            v = self._value(i)
            if i == forced or self.diagnostic_ceiling:
                v = max(v, 1.0)
            if v <= 0.0:
                continue
            still_active.append(i)
            ha = wrap180(lst - state.ra[i])
            h = state.hmax[i]
            if -h <= ha <= h - min_visible:
                nights_left = max(1, state.last_night[i] - night_index + 1)
                setting = (1.0 + 0.5 * max(0.0, ha / h)) if h < 180 else 1.0
                if state.required[i] and state.factor[i] < state.scoring.required_threshold:
                    window = (h - ha) / SIDEREAL_DEG_PER_SECOND if h < 180 else seconds_left
                    alt, _ = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
                    setting *= self._required_window_boost(i, alt, moon, window, seconds_left, nights_left)
                candidates.append((v * (1.0 + 2.0 / nights_left) * setting, i))
        state.active = still_active
        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[1] != forced if forced is not None else False, -t[0]))

        altaz_cache: dict[int, tuple[float, float]] = {}

        def altaz(i: int) -> tuple[float, float]:
            cached = altaz_cache.get(i)
            if cached is None:
                cached = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
                altaz_cache[i] = cached
            return cached

        visible = {i for _, i in candidates}
        achievable_cache: dict[int, float] = {}
        scoring = state.scoring

        def achievable(i: int) -> float:
            cached = achievable_cache.get(i)
            if cached is not None:
                return cached
            alt, az = altaz(i)
            lunar = lunar_factor(moon, state.ra[i], state.dec[i], scoring.lunar_model)
            model = scoring.quality_model(alt, lunar) or 0.0
            k = (state.flux[i] * model * state.scale * PLAN_FACTOR_SAFETY) / scoring.f0t0
            ha = wrap180(lst - state.ra[i])
            up = (state.hmax[i] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[i] < 180 else 1e9
            reach = min(1.0, k * min(state.max_exposure, up, seconds_left))
            gain = self._gain(i, reach)
            if i == forced or self.diagnostic_ceiling:
                gain = max(gain, 1.0)
            # Ambiguous completion feedback must not progressively erase the
            # value of reaching the required threshold on a later exposure.
            unfinished = state.required[i] and state.factor[i] < scoring.required_threshold
            damp = (0.6 ** min(state.misses[i], 8)) * (1.0 if unfinished else 0.7 ** state.attempts[i])
            result = gain * damp * self._direction_factor(alt, az)
            achievable_cache[i] = result
            return result

        anchors: list[tuple[float, int]] = []
        for checked, (priority, i) in enumerate(candidates):
            if not self.diagnostic_ceiling and checked >= ANCHOR_POOL and len(anchors) >= 3 * ANCHORS:
                break
            weighted = achievable(i) * priority / max(1e-9, self._value(i))
            if state.required[i] and state.factor[i] < scoring.required_threshold and state.last_night[i] - night_index < 3:
                weighted *= self.advice.required_multiplier()
            if self.diagnostic_ceiling:
                a, z = altaz(i)
                weighted = scoring.quality_model(a, lunar_factor(moon, state.ra[i], state.dec[i], scoring.lunar_model)) * self._direction_factor(a, z)
                if (int(z // 45), int(a // 20)) in self.verified_probe_fields:
                    weighted *= .01
            if weighted > 0:
                anchors.append((weighted, i))
        if not anchors:
            return None
        anchors.sort(key=lambda t: -t[0])
        if forced is not None:
            anchors = [row for row in anchors if row[1] == forced]

        n_anchors = 1 if state.fast_level >= 1 else ANCHORS
        middle = (self.grid.side - 1) / 2
        central = sorted(range(self.grid.n), key=lambda f: sum(abs(v) for v in self.grid.fiber_center(f)))[:min(4, self.grid.n)]
        fibers = range(self.grid.n) if state.fast_level < 2 else central
        best = None  # (total, c_alt, c_az, chosen)
        tried = 0
        for _, anchor in anchors:
            if tried >= n_anchors and best is not None:
                break
            if tried >= n_anchors + 8:
                break
            tried += 1
            a_alt, a_az = altaz(anchor)
            near = [j for j in state.neighbours(state.ra[anchor], state.dec[anchor], math.degrees(math.atan(math.sqrt(2) * math.radians(self.grid.fov)))) if j in visible]
            near_values = {j: achievable(j) for j in near}
            for fiber in fibers:
                d_north, d_east = self.grid.fiber_center(fiber)
                c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
                if not (state.min_alt + 1.5 <= c_alt <= 89.0):
                    continue
                c_alt = round(c_alt, 4)
                c_az = round(c_az, 4) % 360.0
                chosen: dict[int, tuple[float, int, float]] = {}  # fiber -> (score, j, margin)
                for j, v in near_values.items():
                    if v <= 0.0:
                        continue
                    alt, az = altaz(j)
                    offsets = tangent_offsets(alt, az, c_alt, c_az)
                    if offsets is None:
                        continue
                    fib, margin = self.grid.classify(*offsets)
                    if fib is None:
                        continue
                    score = v * (1.0 if margin >= min(EDGE_MARGIN_DEG, self.grid.glass * 0.15) * (1 + 1.5 * state.misses[j]) else 0.4)
                    existing = chosen.get(fib)
                    if j == forced or (existing is None or (existing[1] != forced and score > existing[0])):
                        chosen[fib] = (score, j, margin)
                if not chosen:
                    continue
                if forced is not None and not any(j == forced for _, j, _ in chosen.values()):
                    continue
                total = sum(score for score, _, _ in chosen.values())
                if best is None or total > best[0]:
                    best = (total, c_alt, c_az, chosen)
        if best is None:
            return None
        _, c_alt, c_az, chosen = best
        return self._finish_plan(now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index)

    def _finish_plan(self, now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index):
        state = self.state
        scoring = state.scoring
        c_ra, c_dec = altaz_to_radec(c_alt, c_az, lst, state.lat)
        c_hmax = max_hour_angle_deg(c_dec, state.lat, state.min_alt + 0.3)
        c_ha = wrap180(lst - c_ra)

        info: dict[int, dict] = {}
        for fiber, (_, j, _margin) in chosen.items():
            alt, az = altaz(j)
            lunar = lunar_factor(moon, state.ra[j], state.dec[j], scoring.lunar_model)
            model = scoring.quality_model(alt, lunar) or 0.0
            ha = wrap180(lst - state.ra[j])
            up = (state.hmax[j] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[j] < 180 else 1e9
            k = (state.flux[j] * model * state.scale * PLAN_FACTOR_SAFETY) / scoring.f0t0
            info[fiber] = {"i": j, "alt": alt, "az": az, "model": model, "up": up, "k": k}
        center_up = (c_hmax - c_ha) / SIDEREAL_DEG_PER_SECOND if c_hmax < 180 else 1e9

        ceiling = int(min(state.max_exposure, seconds_left, center_up))
        if self.diagnostic_ceiling:
            ceiling = min(ceiling, self.diagnostic_ceiling)
        if ceiling < state.min_exposure:
            return None
        durations = {state.min_exposure, ceiling}
        for base in DURATIONS:
            durations.add(int(min(ceiling, max(state.min_exposure, round(base * state.duration_scale)))))
        for item in info.values():
            if item["k"] <= 0:
                continue
            for factor in (self._completion_goal(), 1.0):
                duration = math.ceil(factor / item["k"])
                if state.min_exposure <= duration <= ceiling:
                    durations.add(duration)
            if state.min_exposure <= item["up"] <= ceiling:
                durations.add(int(item["up"]))
            threshold = self._request_thresholds_now.get(item["i"])
            if self._forced_request and item["i"] == self._forced_request["target"]:
                threshold = self._forced_request["threshold"]
            if threshold is not None:
                duration = math.ceil(threshold / item["k"])
                if state.min_exposure <= duration <= ceiling:
                    durations.add(duration)
        best = None
        request_best = None
        # Preserve the accepted scientific DARK diagnostic: saturated matched
        # scores prove the program band when short paired probes are inconclusive.
        diagnostic_count = min(3, sum(item["k"] * min(ceiling, item["up"]) >= 1 for item in info.values())) if state.force_program == "DARK" and not self.diagnostic_ceiling else 0
        for duration in sorted(durations):
            valid = [item for item in info.values() if item["up"] >= duration]
            if self._forced_request and not any(item["i"] == self._forced_request["target"] and min(1.0, item["k"] * duration) >= self._forced_request["threshold"] for item in valid):
                continue
            if not valid:
                continue
            if diagnostic_count and sum(item["k"] * duration >= 1 for item in valid) < diagnostic_count:
                continue
            for program in (state.force_program,) if state.force_program else ("DARK", "BRIGHT", "BACKUP"):
                gain = 0.0
                completes_request = False
                for item in valid:
                    reached = min(1.0, item["k"] * duration)
                    band = scoring.program_band(item["model"] * state.scale / 0.95)
                    gain += self._gain(item["i"], reached, program, band)
                    threshold = self._request_thresholds_now.get(item["i"])
                    if threshold is not None and reached >= threshold:
                        completes_request = True
                candidate = (gain / duration, duration, program)
                if best is None or candidate[0] > best[0] * 1.000001 or (candidate[0] >= best[0] * .999999 and duration > best[1]):
                    best = candidate
                if completes_request and (request_best is None or candidate[0] > request_best[0]):
                    request_best = candidate
        if best is None or (best[0] <= 0 and self._forced_request is None and not self.diagnostic_ceiling):
            return None
        if request_best is not None and request_best[0] >= best[0] * REQUEST_RATE_TOLERANCE:
            best = request_best
        _, duration, program = best
        assignments = {str(fiber): state.ids[item["i"]] for fiber, item in info.items() if item["up"] >= duration}

        clean = not state.all_sky_notice()
        state.pending.clear()
        state.pending_action_index = self._current_action_index
        for fiber, item in info.items():
            if str(fiber) in assignments:
                state.pending[state.ids[item["i"]]] = PendingPrediction(
                    model=item["model"], band_model=item["model"] / 0.95, alt=item["alt"], az=item["az"],
                    clean=clean and self._direction_factor(item["alt"], item["az"], use_model=False) >= 1.0,
                )
        state.pending_program = program
        state.pending_duration = duration
        state.pending_start = now
        state.pending_night = night_index

        return {
            "action": "observe",
            "pointing": dict(zip(("alt_deg", "az_deg"), self.calibration.command(c_alt, c_az))),
            "assignments": assignments,
            "duration_seconds": duration,
            "program": program,
        }
