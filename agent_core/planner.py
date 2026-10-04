"""Public-input survey planning with linear marginal score and two bounded LLM stages.
The model interprets sourced messages and adapts priorities; geometry, completion
state, candidate exposures and final protocol actions remain deterministic.
"""
from __future__ import annotations

from datetime import timedelta
import math
import time
from typing import NamedTuple

from .geometry import (
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
from .llm_client import LLMClient
from .memory import TraceLog
from .state import PendingPrediction
from .advice import AdviceController
from .requests import RequestPlanner
from .exposure import Curve, optimize

REQUIRED_BONUS = 60.0
DONE_FACTOR = 0.95
PLAN_FACTOR_SAFETY = 0.9
EDGE_MARGIN_DEG = 0.08
DURATIONS = (300, 450, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600)
MIN_VISIBLE_SECONDS = 600
NEIGHBOUR_RADIUS_DEG = 2.1
ANCHORS = 6
ANCHOR_POOL = 300
POINTING_PLANS = 4
CLOSED_KINDS = {"rain", "storm"}
BLOCKING_KINDS = {"terrain_obstruction", "rocket_launch"}
DIRECTION_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0,
                "SW": 225.0, "W": 270.0, "NW": 315.0}

REPORT_DROP = 0.62
REPORT_CONFIRMATIONS = 3
REPORT_SPACING_HOURS = 6.0

# Request rewards are priced once per feasible bundle in RequestPlanner. Ordinary
# exposure optimization has no request bonus. A shared-exposure tie-break is
# enabled only after bundle pricing and is checked again before adoption.
# Dedicated actions are rechecked after sizing and preserve imminent required work.


def _az_distance(a: float, b: float) -> float:
    return abs(wrap180(a - b))


def _bulletin_text(notices: list) -> str:
    """A human-readable rendering of a bulletin's notices, for the LLM call that reads
    "live bulletin text" rather than structured JSON."""
    if not notices:
        return "clear (no active notices)"
    return "; ".join(f"{n.get('event_kind')} {n.get('direction')}" for n in notices)


class ExposurePlan(NamedTuple):
    action: dict
    predictions: dict
    gain_rate: float


class Planner:
    def __init__(self, state, log=lambda text: None):
        self.state = state
        self.log = log
        self.grid = state.fiber_grid
        self.llm = LLMClient(log=log)
        self.trace = TraceLog(log=log)
        self.advice = AdviceController(state, self.llm, self.trace)
        self.requests = RequestPlanner(state, self.trace)
        self._forced_request = None

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
        self._request_thresholds_now = {}
        self._short_probe_evidence = None
        self._short_probe_counts = {}
        self._last_short_probe_hours = float("-inf")
        self._terminal_report_attempted = False

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
        state.on_result(payload.get("last_result"), hours)
        state.on_messages(payload.get("new_messages", []), payload.get("latest_bulletin"))
        last_result = payload.get("last_result")
        if last_result and last_result.get("action") == "observe":
            self.total_assigned += int(last_result.get("assigned_count", 0))
            self.total_hit += int(last_result.get("hit_count", 0))
        self._pace(payload, now)
        self._current_action_index = payload.get("observe_action_index")
        self.requests.sync(payload.get("active_requests") or [])

        night = state.current_night(now)
        if night is None:
            nxt = state.next_night_start(now)
            if nxt is None:
                return self._terminal_action(payload, {"action": "finish", "reason": "no observing night left"})
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "daytime: sleep until the next night"}
        night_index, night_start, night_end = night

        self.advice.update(payload, night_index, now)

        # The runner may terminate immediately after the last exposure. Audit
        # before it, while a report still has zero simulated-time cost.
        if night_index == len(state.nights) - 1:
            final_report = self._terminal_action(payload, None)
            if final_report is not None:
                return final_report

        if (night_end - now).total_seconds() < state.min_exposure:
            nxt = state.next_night_start(now)
            if nxt is None:
                return self._terminal_action(payload, {"action": "wait", "until_utc": format_utc(state.survey_end), "reason": "last night ending"})
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "night ending"}

        if state.site_closed():
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "bulletin: rain/storm over the whole sky"}

        report = self._maybe_report(hours, payload)
        if report is not None:
            return report

        action = self.plan(now, night_end, night_index, hours)
        if self.requests.requests:
            normal_rate = self._action_gain_rate(action, now) if action is not None else 0
            bundle = self.requests.choose(now, normal_rate, self.advice.priority == "requests")
            if bundle is not None:
                needed = self._required_completion_predictions(action, now)
                normal_pending = {k: dict(getattr(state, k)) if k == "pending" else getattr(state, k)
                                  for k in ("pending", "pending_start", "pending_duration", "pending_program", "pending_night", "pending_action_index")}
                routes = ["shared"] + (["dedicated"] if bundle["start"] <= now else [])
                for route in routes:
                    self._forced_request = bundle if route == "dedicated" else None
                    self._request_thresholds_now = self._approved_request_thresholds(bundle) if route == "shared" else {}
                    adopted = False
                    try:
                        request_action = self.plan(now, min(night_end, bundle["deadline"]), night_index, hours)
                        provided = self._required_completion_predictions(request_action, now)
                        preserves_required = all(provided.get(i, 0) + 1e-9 >= factor for i, factor in needed.items())
                        settlement = (self.requests.assess_execution(bundle, request_action, now, normal_rate,
                                      self.advice.priority == "requests") if request_action is not None and preserves_required else None)
                        if request_action is not None and preserves_required and settlement is not None:
                            action = request_action
                            adopted = True
                            self.trace.write({"event": "request_execution", "request_id": bundle["request_id"],
                                              "mode": route, "normal_rate": normal_rate,
                                              "target_id": state.ids[bundle["target"]] if route == "dedicated" else None,
                                              "assigned_request_target_ids": [target for target in action["assignments"].values()
                                                                              if target in self.requests.requests[bundle["request_id"]]["target_ids"]],
                                              "now_utc": payload["now_utc"],
                                              "duration_seconds": action["duration_seconds"], **settlement})
                        elif request_action is not None:
                            self.trace.write({"event": "request_deferred", "request_id": bundle["request_id"], "mode": route,
                                              "reason": "preserve imminent required completion" if not preserves_required else "actual request schedule is infeasible or unprofitable",
                                              "required_targets": [state.ids[i] for i in needed]})
                    finally:
                        self._forced_request = None
                        self._request_thresholds_now = {}
                        if not adopted:
                            for key, value in normal_pending.items():
                                setattr(state, key, value)
                    if adopted:
                        break
        if action is None:
            return {"action": "wait", "duration_seconds": self._to_next_slot(now, night_start),
                    "reason": "nothing useful is up"}
        action = self._short_probe(action, now, night_end, night_index, hours)
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

    # -- instrument fault reporting (public evidence and configured settlement) --

    def _terminal_action(self, payload, fallback):
        """A bounded zero-penalty report uses no simulated observing time."""
        state = self.state
        result = payload.get("last_result") or {}
        if self._terminal_report_attempted and result.get("action") == "report" and result.get("correct") is True:
            self._terminal_report_attempted = False  # Check for another outstanding fault within the configured cap.
        free = state.false_reports_since_correct < state.false_report_free_allowance
        remaining = float((payload.get("wallclock") or {}).get("remaining_seconds", 1e9))
        if (self._terminal_report_attempted or remaining < 2
                or self.consecutive_reports >= state.max_consecutive_reports
                or state.scoring.correct_report_reward <= 0
                or (not free and state.scoring.false_report_penalty < 0)):
            return fallback
        self._terminal_report_attempted = True
        self.reports += 1
        self.trace.write({"event": "terminal_diagnostic", "now_utc": payload.get("now_utc"),
                          "free_allowance_remaining": max(0, state.false_report_free_allowance - state.false_reports_since_correct)})
        return {"action": "report", "reason": "final diagnostic with no configured false-report loss",
                "decision_source": "deterministic"}

    def _maybe_report(self, hours: float, payload: dict):
        state = self.state
        state.force_program = None
        self._short_probe_evidence = None
        if self.consecutive_reports >= state.max_consecutive_reports or state.all_sky_notice():
            return None
        evidence = state.fault_evidence()
        # One low night can justify collecting information, never a report.
        # Long exposures become saturated after the planner learns a low scale;
        # saturation supplies no upper bound on throughput.
        provisional = state.fault_evidence(recent_night_count=1, min_night_exposures=1)
        cost = (0 if state.false_reports_since_correct < state.false_report_free_allowance
                else max(0, -state.scoring.false_report_penalty))
        remaining_science = sum(max(0, w * state.scoring.maximum_multiplier - s)
                                for w, s in zip(state.weight, state.best_score))
        reward = max(0, state.scoring.correct_report_reward)
        # This is an optimistic benefit bound, not a calibrated fault probability.
        # A penalized report must first be economically possible; stronger drops
        # are required as the configured false-report cost grows.
        clue = evidence or provisional
        repair_bound = remaining_science * max(0, 1 - clue.drop) if clue else 0
        if clue:
            repair_bound += sum(state.scoring.required_penalty for r, f in zip(state.required, state.factor)
                                if r and f < state.scoring.required_threshold)
        if cost > reward + repair_bound:
            return None
        if reward + repair_bound <= 0:
            return None
        multipliers = [m for m in state.scoring.program_multipliers.values() if m > 0]
        if state.scoring.mismatch_multiplier > 0:
            multipliers.append(state.scoring.mismatch_multiplier)
        ambiguity = max(multipliers) / min(multipliers)
        threshold = (min(.9, REPORT_DROP * ambiguity) if cost == 0
                     else REPORT_DROP / (1 + cost / max(1, reward + repair_bound)))
        if (provisional is not None and provisional.drop < threshold and state.clean_intervals
                and hours - state.clean_intervals[-1][0] <= 36):
            self._short_probe_evidence = provisional
        if state.clean_history and hours - state.clean_history[-1][0] > 2:
            return None  # A report requires fresh clean evidence in this night.
        if evidence is None or evidence.drop >= threshold:
            self.suspicion_hours = []
            return None
        if cost > 0 and evidence.dark_checks < 6:
            state.force_program = "DARK"
            return None  # Collect band evidence before counting confirmations.
        elif cost > 0 and evidence.dark_matched < 0.5 * evidence.dark_checks:
            self.suspicion_hours = []
            return None
        if self.suspicion_hours and hours - self.suspicion_hours[-1] < REPORT_SPACING_HOURS:
            return None
        self.suspicion_hours.append(hours)
        # With a remaining free false-report allowance, the simulated action
        # consumes no survey time and its settlement cannot be negative. The
        # persistent two-night signal already justifies one exploratory report.
        if len(self.suspicion_hours) < (1 if cost == 0 else REPORT_CONFIRMATIONS):
            return None
        self.suspicion_hours = []
        self.reports += 1
        self.last_report_hours = hours
        state.forget_quality_history()
        self.log(f"planner: reporting instrument fault at {payload.get('now_utc')} evidence={evidence}")
        return {"action": "report", "reason": f"quality dropped to {evidence.drop:.0%} of the earlier level",
                "decision_source": "deterministic"}

    def _short_probe(self, action, now, night_end, night_index, hours):
        """Reuse a legal normal pointing for a bounded, unsaturated evidence exposure."""
        state = self.state
        evidence = self._short_probe_evidence
        calibration = state.calibration_due(hours, night_index)
        duration = state.min_exposure
        spacing = (max(7200, state.slot_seconds) if calibration and evidence is None else state.slot_seconds) / 3600
        if ((evidence is None and not calibration) or state.force_program or state.all_sky_notice() or self.requests.requests
                or action.get("action") != "observe" or action["duration_seconds"] <= duration
                or self._required_completion_predictions(action, now)
                or self._short_probe_counts.get(night_index, 0) >= 4
                or hours - self._last_short_probe_hours < spacing):
            return action
        informative = {hour for hour, night, _lower, _upper in state.clean_intervals if night == night_index}
        if len(informative) >= 4 and not calibration:
            return action
        # Even an unbroken instrument should leave at least three uncensored
        # targets. The pending predictions use only current public geometry.
        upper_scale = max(1, state.scale, state.prior_scale)
        targets = [target for target, p in state.pending.items()
                   if p.clean and state.flux[state.index_of[target]] > 0
                   and state.scoring.completion_factor(state.flux[state.index_of[target]], duration,
                                                       p.model * upper_scale) < .7]
        if len(targets) < 3:
            return action
        # Leave enough visibility to take the originally planned exposure next.
        # Never exchange a last-window required completion for a diagnostic.
        available = (min(night_end, state.survey_end) - now).total_seconds()
        if action["duration_seconds"] + duration > available:
            return action
        lst = local_sidereal_deg(now, state.lon)
        for target in action["assignments"].values():
            i = state.index_of[target]
            if state.required[i] and state.factor[i] < state.scoring.required_threshold:
                setting = ((state.hmax[i] - wrap180(lst - state.ra[i])) / SIDEREAL_DEG_PER_SECOND
                           if state.hmax[i] < 180 else float("inf"))
                if action["duration_seconds"] + duration > setting:
                    return action
        probe = {**action, "duration_seconds": duration, "program": "DARK"}
        state.pending_duration = duration
        state.pending_program = "DARK"
        self._short_probe_counts[night_index] = self._short_probe_counts.get(night_index, 0) + 1
        self._last_short_probe_hours = hours
        self.trace.write({"event": "diagnostic_short_exposure", "now_utc": format_utc(now),
                          "purpose": "persistent_drop" if evidence else "uncensored_quality_calibration",
                          "duration_seconds": duration, "informative_targets": targets,
                          "evidence": evidence._asdict() if evidence else None})
        return probe

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
        for patch in state.blocked:
            if _az_distance(az, patch.az) <= 12.0 and alt <= patch.alt + 3.0:
                factor = min(factor, 0.2)
        return factor

    def _completion_goal(self):
        scoring = self.state.scoring
        multipliers = list(scoring.program_multipliers.values()) + [scoring.mismatch_multiplier]
        return min(1.0, scoring.required_threshold * max(multipliers) / min(m for m in multipliers if m > 0) + 0.00001)

    def _approved_request_thresholds(self, bundle):
        request = self.requests.requests[bundle["request_id"]]
        done = self.requests.completed(request)
        return {self.state.index_of[target]: float(request["completion_factor_threshold"])
                for target in request["target_ids"] if target not in done and target in self.state.index_of}

    def _required_completion_predictions(self, action, now):
        if action is None:
            return {}
        state = self.state
        lst = local_sidereal_deg(now, state.lon)
        moon = Moon(now, lst, state.lat)
        completions = {}
        for target in action["assignments"].values():
            i = state.index_of[target]
            if not state.required[i] or state.factor[i] >= state.scoring.required_threshold:
                continue
            alt, _ = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
            quality = state.scoring.quality_model(alt, lunar_factor(moon, state.ra[i], state.dec[i], state.scoring.lunar_model))
            reached = state.scoring.completion_factor(state.flux[i], action["duration_seconds"], quality * state.scale * PLAN_FACTOR_SAFETY)
            if reached >= state.scoring.required_threshold:
                completions[i] = min(reached, self._completion_goal())
        return completions

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

        still_active = []
        candidates: list[tuple[float, int]] = []
        forced = self._forced_request["target"] if self._forced_request else None
        active = state.active if forced is None or forced in state.active else [*state.active, forced]
        for i in active:
            v = self._value(i)
            if i == forced:
                v = max(v, 1.0)
            if v <= 0.0:
                continue
            still_active.append(i)
            ha = wrap180(lst - state.ra[i])
            h = state.hmax[i]
            if -h <= ha <= h - min_visible:
                nights_left = max(1, state.last_night[i] - night_index + 1)
                setting = (1.0 + 0.5 * max(0.0, ha / h)) if h < 180 else 1.0
                candidates.append((v * (1.0 + 2.0 / nights_left) * setting, i))
        state.active = still_active
        if not candidates:
            return None
        candidates.sort(key=lambda t: -t[0])
        if forced is not None:
            candidates.sort(key=lambda row: row[1] != forced)

        moon = Moon(now + timedelta(seconds=450), lst, state.lat)
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
            if i == forced:
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
            if checked >= ANCHOR_POOL and len(anchors) >= 3 * ANCHORS:
                break
            weighted = achievable(i) * priority / max(1e-9, self._value(i))
            if state.required[i] and state.factor[i] < scoring.required_threshold and state.last_night[i] - night_index < 3:
                weighted *= self.advice.required_multiplier()
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
        pointings = []  # Best geometric placement for each searched anchor.
        tried = 0
        for _, anchor in anchors:
            if tried >= n_anchors and pointings:
                break
            if tried >= n_anchors + 8:
                break
            tried += 1
            a_alt, a_az = altaz(anchor)
            near = [j for j in state.neighbours(state.ra[anchor], state.dec[anchor], math.degrees(math.atan(math.sqrt(2) * math.radians(self.grid.fov)))) if j in visible]
            near_values = {j: achievable(j) for j in near}
            anchor_best = None
            for fiber in fibers:
                d_north, d_east = self.grid.fiber_center(fiber)
                c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
                if not (state.min_alt + 1.5 <= c_alt <= 89.0):
                    continue
                c_alt = round(c_alt, 4)
                c_az = round(c_az, 4) % 360.0
                chosen: dict[int, tuple[float, int, float]] = {}  # fiber -> (score, j, margin)
                alternatives = {}
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
                    alternatives.setdefault(fib, []).append((score, j, margin))
                    existing = chosen.get(fib)
                    if existing is None or (j == forced and existing[1] != forced) or (existing[1] != forced and score > existing[0]):
                        chosen[fib] = (score, j, margin)
                if not chosen:
                    continue
                total = sum(score for score, _, _ in chosen.values())
                if anchor_best is None or total > anchor_best[0]:
                    anchor_best = (total, c_alt, c_az, chosen, alternatives)
            if anchor_best is not None:
                pointings.append(anchor_best)
        if not pointings:
            return None
        evidence = state.fault_evidence()
        quality_guard = state.force_program is not None or (evidence is not None and evidence.drop < .65)
        search_deadline = time.monotonic() + (.35 if state.fast_level == 0 else .06)
        if not quality_guard and state.fast_level == 0 and forced is None:
            extras = self._extra_pointings(lst, night_index, seconds_left, moon, visible, altaz, achievable,
                                           {i for _, i in anchors[:tried]}, search_deadline)
            pointings.extend(extras)
        pointings.sort(key=lambda row: -row[0])
        limit = 1 if state.fast_level >= 2 else 2 if state.fast_level else 8
        # The same observations feed the quality estimator and fault detector.
        # During their existing anomaly phase, preserve the original sampling
        # policy instead of changing both scheduling and diagnostic evidence.
        if quality_guard:
            limit = 1
        seen = set()
        evaluated = []
        joint_trials = joint_adoptions = 0
        for total, c_alt, c_az, chosen, alternatives in pointings:
            signature = tuple((fiber, row[1]) for fiber, row in sorted(chosen.items()))
            if signature in seen:
                continue
            seen.add(signature)
            exposure = self._evaluate_pointing(now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz)
            if exposure is not None and not quality_guard and time.monotonic() < search_deadline:
                joint_trials += 1
                trial = self._evaluate_joint(now, lst, c_alt, c_az, alternatives, seconds_left, moon, altaz, search_deadline)
                if trial is not None and trial.gain_rate >= exposure.gain_rate:
                    exposure = trial
                    joint_adoptions += 1
            if exposure is not None:
                evaluated.append((total, exposure))
            if len(seen) >= limit:
                break
        if not evaluated:
            return None
        # Treat the original pointing's achievable required completions as hard
        # constraints. A higher local science rate must not discard them.
        needed = self._required_completion_predictions(evaluated[0][1].action, now)
        eligible = []
        for i, (_, trial) in enumerate(evaluated):
            provided = self._required_completion_predictions(trial.action, now)
            if all(provided.get(target, 0) + 1e-9 >= factor for target, factor in needed.items()):
                eligible.append(i)
        winner = max(eligible, key=lambda i: (evaluated[i][1].gain_rate, -i))
        if not quality_guard and forced is None and state.fast_level == 0:
            winner = self._lookahead(evaluated, eligible, winner, now, horizon, search_deadline)
        exposure = evaluated[winner][1]
        self.trace.write({"event": "pointing_selection", "now_utc": format_utc(now),
                          "forced_request": self._forced_request is not None,
                          "evaluated": len(evaluated), "selected_rank": winner,
                          "eligible": len(eligible), "protected_required": len(needed),
                          "quality_guard": quality_guard,
                          "generated_pointings": len(pointings), "joint_trials": joint_trials,
                          "joint_adoptions": joint_adoptions,
                          "rates": [row[1].gain_rate for row in evaluated],
                          "durations": [row[1].action["duration_seconds"] for row in evaluated]})
        return self._commit_exposure(exposure, now, night_index)

    def _extra_pointings(self, lst, night_index, seconds_left, moon, visible, altaz, achievable, previous, deadline):
        """Supplement legacy anchors with science-rate and density candidates.

        All geometry comes from this card's public layout. Placements rotate
        between anchor families; a 100-fibre field cannot consume the whole
        additional budget before the other anchors get a placement.
        """
        state, scoring = self.state, self.state.scoring
        science, required, density = [], [], {}
        width = max(1e-6, math.radians(self.grid.fov) * .7)
        for index, i in enumerate(sorted(visible)):
            if index % 64 == 0 and time.monotonic() >= deadline:
                break
            alt, az = altaz(i)
            model = scoring.quality_model(alt, lunar_factor(moon, state.ra[i], state.dec[i], scoring.lunar_model))
            k = state.flux[i] * model * state.scale * PLAN_FACTOR_SAFETY / scoring.f0t0
            if k <= 0:
                continue
            up = (state.hmax[i] - wrap180(lst - state.ra[i])) / SIDEREAL_DEG_PER_SECOND if state.hmax[i] < 180 else 1e9
            ceiling = min(state.max_exposure, up, seconds_left)
            if ceiling < state.min_exposure:
                continue
            times = (state.min_exposure, min(ceiling, max(state.min_exposure, 1 / k)), ceiling)
            rate = max(max(0, state.weight[i] * scoring.maximum_multiplier * min(1, k * t) - state.best_score[i]) / t for t in times)
            rate *= self._direction_factor(alt, az) * .6 ** min(state.misses[i], 8)
            science.append((rate, i))
            if state.required[i] and state.factor[i] < scoring.required_threshold and k * ceiling >= self._completion_goal():
                cost = max(state.min_exposure, self._completion_goal() / k)
                urgency = 1 + 2 / max(1, state.last_night[i] - night_index + 1)
                required.append(((scoring.required_penalty / cost + rate) * urgency, i))
            a, z = math.radians(alt), math.radians(az)
            cell = tuple(math.floor(v / width) for v in (math.cos(a) * math.cos(z), math.cos(a) * math.sin(z), math.sin(a)))
            total, best_rate, anchor = density.get(cell, (0, -1, i))
            density[cell] = (total + rate, max(best_rate, rate), i if rate > best_rate else anchor)
        groups = (sorted(science, reverse=True)[:6], sorted(required, reverse=True)[:6],
                  [(total, anchor) for total, _, anchor in sorted(density.values(), reverse=True)[:6]])
        extra_anchors = []
        used = set(previous)
        for rank in range(6):
            for group in groups:
                if rank < len(group) and group[rank][0] > 0 and group[rank][1] not in used:
                    extra_anchors.append(group[rank][1])
                    used.add(group[rank][1])
        neighborhoods = {i: [j for j in state.neighbours(state.ra[i], state.dec[i],
                           math.degrees(math.atan(math.sqrt(2) * math.radians(self.grid.fov)))) if j in visible]
                         for i in extra_anchors}
        fibers = sorted(range(self.grid.n), key=lambda f: sum(abs(v) for v in self.grid.fiber_center(f)))
        placements, best = 0, {}
        for fiber in fibers:
            for anchor in extra_anchors:
                if placements >= 384 or time.monotonic() >= deadline:
                    return list(best.values())
                placements += 1
                alt, az = altaz(anchor)
                dn, de = self.grid.fiber_center(fiber)
                c_alt, c_az = shift_altaz(alt, az, -dn, -de)
                if not state.min_alt + 1.5 <= c_alt <= 89:
                    continue
                c_alt, c_az = round(c_alt, 4), round(c_az, 4) % 360
                chosen, alternatives = {}, {}
                for j in neighborhoods[anchor]:
                    value = achievable(j)
                    if value <= 0:
                        continue
                    offsets = tangent_offsets(*altaz(j), c_alt, c_az)
                    if offsets is None:
                        continue
                    slot, margin = self.grid.classify(*offsets)
                    if slot is None:
                        continue
                    score = value * (1 if margin >= min(EDGE_MARGIN_DEG, self.grid.glass * .15) * (1 + 1.5 * state.misses[j]) else .4)
                    row = score, j, margin
                    alternatives.setdefault(slot, []).append(row)
                    if slot not in chosen or score > chosen[slot][0]:
                        chosen[slot] = row
                total = sum(row[0] for row in chosen.values())
                if chosen and (anchor not in best or total > best[anchor][0]):
                    best[anchor] = total, c_alt, c_az, chosen, alternatives
        return list(best.values())

    def _finish_plan(self, now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index):
        exposure = self._evaluate_pointing(now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz)
        return self._commit_exposure(exposure, now, night_index) if exposure is not None else None

    def _evaluate_pointing(self, now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz):
        """Size an exposure without changing feedback predictions or progress."""
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
        if ceiling < state.min_exposure:
            return None
        durations = {state.min_exposure, ceiling}
        for base in DURATIONS:
            durations.add(int(min(ceiling, max(state.min_exposure, round(base * state.duration_scale)))))
        multipliers = [m for m in [*scoring.program_multipliers.values(), scoring.mismatch_multiplier] if m > 0]
        diagnostic_goal = max(multipliers) / min(multipliers) if state.force_program == "DARK" else 1
        for item in info.values():
            if item["k"] <= 0:
                continue
            for factor in (self._completion_goal(), 1.0, diagnostic_goal):
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
        # Keep the existing DARK diagnostic check meaningful under a linear
        # objective, whose otherwise optimal exposures may never saturate.
        diagnostic_count = min(3, sum(item["k"] * min(ceiling, item["up"]) >= diagnostic_goal for item in info.values())) if state.force_program == "DARK" else 0
        for duration in sorted(durations):
            valid = [item for item in info.values() if item["up"] >= duration]
            if self._forced_request and not any(item["i"] == self._forced_request["target"] and min(1.0, item["k"] * duration) >= self._forced_request["threshold"] for item in valid):
                continue
            if not valid:
                continue
            if diagnostic_count and sum(item["k"] * duration >= diagnostic_goal for item in valid) < diagnostic_count:
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
                tie = best is not None and candidate[0] >= best[0] * .999999
                preferred_duration = best is not None and (duration < best[1] if self._forced_request else duration > best[1])
                if best is None or candidate[0] > best[0] * 1.000001 or (tie and preferred_duration):
                    best = candidate
                if completes_request and (request_best is None or candidate[0] > request_best[0]):
                    request_best = candidate
        if best is None or (best[0] <= 0 and self._forced_request is None):
            return None
        if request_best is not None and request_best[0] >= best[0] * .9:
            best = request_best
        _, duration, program = best
        assignments = {str(fiber): state.ids[item["i"]] for fiber, item in info.items() if item["up"] >= duration}

        clean = not state.all_sky_notice()
        predictions = {}
        for fiber, item in info.items():
            if str(fiber) in assignments:
                predictions[state.ids[item["i"]]] = PendingPrediction(
                    model=item["model"], band_model=item["model"] / 0.95, alt=item["alt"], az=item["az"],
                    clean=clean and self._direction_factor(item["alt"], item["az"], use_model=False) >= 1.0,
                )
        action = {
            "action": "observe",
            "pointing": {"alt_deg": c_alt, "az_deg": c_az},
            "assignments": assignments,
            "duration_seconds": duration,
            "program": program,
        }
        return ExposurePlan(action, predictions, best[0])

    def _evaluate_joint(self, now, lst, c_alt, c_az, alternatives, seconds_left, moon, altaz, deadline=None,
                        *, progress=None, quality_scale=1.0, future=False, prefix_gain=0.0, prefix_seconds=0):
        state = self.state
        scoring = state.scoring
        progress = progress or {}
        scale = state.scale * quality_scale
        c_ra, c_dec = altaz_to_radec(c_alt, c_az, lst, state.lat)
        hmax = max_hour_angle_deg(c_dec, state.lat, state.min_alt + .3)
        up = (hmax - wrap180(lst - c_ra)) / SIDEREAL_DEG_PER_SECOND if hmax < 180 else 1e9
        ceiling = int(min(state.max_exposure, seconds_left, up))
        if ceiling < state.min_exposure:
            return None
        cells = {}
        for fiber, targets in alternatives.items():
            curves = []
            def priority(row):
                i = row[1]
                previous, factor = progress.get(i, (state.best_score[i], state.factor[i]))
                return (max(0, state.weight[i] * scoring.maximum_multiplier - previous)
                        + (scoring.required_penalty if state.required[i] and factor < scoring.required_threshold else 0)) if future else row[0]
            shortlist = sorted(targets, key=priority, reverse=True)[:8]
            if self._forced_request:
                forced_target = self._forced_request["target"]
                shortlist.extend(row for row in targets if row[1] == forced_target and not any(r[1] == forced_target for r in shortlist))
            for _, i, margin in shortlist:
                alt, az = altaz(i)
                model = scoring.quality_model(alt, lunar_factor(moon, state.ra[i], state.dec[i], scoring.lunar_model))
                k = state.flux[i] * model * scale * PLAN_FACTOR_SAFETY / scoring.f0t0
                target_up = (state.hmax[i] - wrap180(lst - state.ra[i])) / SIDEREAL_DEG_PER_SECOND if state.hmax[i] < 180 else 1e9
                previous, factor = progress.get(i, (state.best_score[i], state.factor[i]))
                penalty = scoring.required_penalty if state.required[i] and factor < scoring.required_threshold else 0
                # Geometry uncertainty already affects the geometric shortlist;
                # compare actual marginal scores here, consistently with v2.
                curves.append(Curve(i, k, target_up, state.weight[i], previous, penalty,
                                    self._completion_goal(), model, alt, az))
            if curves:
                cells[fiber] = curves
        forced = (self._forced_request["target"], self._forced_request["threshold"]) if self._forced_request and not future else None
        if future and not any(c.gain(min(ceiling, c.up), scoring.maximum_multiplier) > 0 for curves in cells.values() for c in curves):
            return None
        result = optimize(cells, ("DARK", "BRIGHT", "BACKUP"), scoring, scale,
                          state.min_exposure, ceiling, thresholds={} if future else self._request_thresholds_now,
                          forced=forced, deadline=deadline, prefix_gain=prefix_gain, prefix_seconds=prefix_seconds,
                          discount=.9 if future else 1)
        if result is None or (result[0] <= 0 and forced is None):
            return None
        rate, duration, program, selected = result
        assignments = {str(f): state.ids[c.i] for f, c in selected.items()}
        clean = not state.all_sky_notice()
        predictions = {state.ids[c.i]: PendingPrediction(c.model, c.model / .95, c.alt, c.az,
                       clean and self._direction_factor(c.alt, c.az, use_model=False) >= 1) for c in selected.values()}
        return ExposurePlan({"action": "observe", "pointing": {"alt_deg": c_alt, "az_deg": c_az},
                             "assignments": assignments, "duration_seconds": duration, "program": program}, predictions, rate)

    def _shadow_progress(self, exposure, quality_scale):
        """Small overlay for a hypothetical exposure; no real ledger writes."""
        state, scoring = self.state, self.state.scoring
        progress, gain = {}, 0.0
        duration, program = exposure.action["duration_seconds"], exposure.action["program"]
        for target, prediction in exposure.predictions.items():
            i = state.index_of[target]
            scale = state.scale * quality_scale
            factor = scoring.completion_factor(state.flux[i], duration, prediction.model * scale * PLAN_FACTOR_SAFETY)
            band = scoring.program_band(prediction.model * scale / .95)
            score = state.weight[i] * factor * scoring.program_multiplier(program, band)
            lower, _ = state.factor_bounds(i, score, program)
            progress[i] = max(state.best_score[i], score), max(state.factor[i], lower)
            gain += self._gain(i, factor, program, band)
        return progress, gain

    def _lookahead(self, evaluated, eligible, fallback, now, horizon, deadline):
        state = self.state
        if len(eligible) < 2 or time.monotonic() >= deadline:
            return fallback
        firsts = sorted(eligible, key=lambda i: -evaluated[i][1].gain_rate)[:3]
        values = {}
        for first_index in firsts:
            first = evaluated[first_index][1]
            duration = first.action["duration_seconds"]
            next_time = now + timedelta(seconds=duration)
            end = min(horizon, now + timedelta(hours=2))
            lst = local_sidereal_deg(next_time, state.lon)
            moon = Moon(next_time, lst, state.lat)
            cache = {}
            def altaz(i):
                if i not in cache:
                    cache[i] = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
                return cache[i]
            # Same field plus one alternative from the live candidate set. No
            # unpublished forecast, message, or simulated model call is used.
            second_indices = [first_index] + [i for i in firsts if i != first_index][:1]
            fields = []
            for index in second_indices:
                action = evaluated[index][1].action
                center = action["pointing"]
                ra, dec = altaz_to_radec(center["alt_deg"], center["az_deg"], local_sidereal_deg(now, state.lon), state.lat)
                ca, cz = radec_to_altaz(ra, dec, lst, state.lat)
                if not state.min_alt + 1.5 <= ca <= 89:
                    continue
                alternatives = {}
                radius = math.degrees(math.atan(math.sqrt(2) * math.radians(self.grid.fov)))
                for i in state.neighbours(ra, dec, radius):
                    ha = wrap180(lst - state.ra[i])
                    if not -state.hmax[i] <= ha <= state.hmax[i] - state.min_exposure * SIDEREAL_DEG_PER_SECOND:
                        continue
                    offsets = tangent_offsets(*altaz(i), ca, cz)
                    if offsets is None:
                        continue
                    fiber, margin = self.grid.classify(*offsets)
                    if fiber is not None:
                        alternatives.setdefault(fiber, []).append((self._value(i), i, margin))
                fields.append((ca, cz, alternatives))
            scenarios = []
            for quality in (.8, 1.0, 1.2):
                if time.monotonic() >= deadline:
                    self.trace.write({"event": "lookahead", "complete": False, "reason": "search budget", "evaluated_firsts": len(values)})
                    return fallback
                progress, gain = self._shadow_progress(first, quality)
                best = gain / duration
                if (end - next_time).total_seconds() >= state.min_exposure:
                    future_rates = []
                    for ca, cz, alternatives in fields:
                        trial = self._evaluate_joint(next_time, lst, ca, cz, alternatives, (end - next_time).total_seconds(),
                                                     moon, altaz, deadline, progress=progress, quality_scale=quality,
                                                     future=True, prefix_gain=gain, prefix_seconds=duration)
                        if time.monotonic() >= deadline:
                            self.trace.write({"event": "lookahead", "complete": False, "reason": "search budget", "evaluated_firsts": len(values)})
                            return fallback
                        if trial:
                            future_rates.append(trial.gain_rate)
                    if future_rates:
                        best = max(future_rates)
                scenarios.append(best)
            values[first_index] = .7 * scenarios[1] + .3 * min(scenarios)
        winner = max(values, key=lambda i: (values[i], evaluated[i][1].gain_rate))
        self.trace.write({"event": "lookahead", "complete": True, "first_candidates": firsts,
                          "values": values, "selected_rank": winner, "changed": winner != fallback})
        return winner

    def _commit_exposure(self, exposure, now, night_index):
        state = self.state
        state.pending = dict(exposure.predictions)
        state.pending_action_index = self._current_action_index
        state.pending_program = exposure.action["program"]
        state.pending_duration = exposure.action["duration_seconds"]
        state.pending_start = now
        state.pending_night = night_index
        return exposure.action
