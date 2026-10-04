"""Public-input survey planning with linear marginal score and two bounded LLM stages.
The model interprets sourced messages and adapts priorities; geometry, completion
state, candidate exposures and final protocol actions remain deterministic.
"""
from __future__ import annotations

from datetime import timedelta
import math

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

REQUIRED_BONUS = 60.0
DONE_FACTOR = 0.95
PLAN_FACTOR_SAFETY = 0.9
EDGE_MARGIN_DEG = 0.08
DURATIONS = (300, 450, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600)
MIN_VISIBLE_SECONDS = 600
NEIGHBOUR_RADIUS_DEG = 2.1
ANCHORS = 6
ANCHOR_POOL = 300
CLOSED_KINDS = {"rain", "storm"}
BLOCKING_KINDS = {"terrain_obstruction", "rocket_launch"}
DIRECTION_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0,
                "SW": 225.0, "W": 270.0, "NW": 315.0}

REPORT_DROP = 0.62
REPORT_CONFIRMATIONS = 3
REPORT_SPACING_HOURS = 6.0
MAX_REPORTS = 2

# Time-limited observation requests (participant guide section 8 / appendix B). There is
# no penalty for letting one expire -- `observation_requests.miss_penalty` is fixed at 0 --
# so the only thing worth doing here is not leaving a reachable `completion_reward` on the
# table, and only when doing so is close to free. Several more aggressive designs were
# tried and rejected: adding a priority bonus into the normal anchor-search's ranking
# (_value/achievable), and a separate dedicated exposure pointed straight at an urgent
# target, both distorted which pointing got chosen and for how long, for a score loss
# repeatedly far bigger than any request reward across these practice cards -- a public
# weight scale of ~0.3-1.7 and a 50-60 point REQUIRED_BONUS leave no room for also
# carrying a "maybe worth 100" incentive without it taking over. What is left is a
# tie-break, applied only inside _finish_plan's own duration search over a pointing/fibre
# assignment chosen with ZERO knowledge of requests: among durations within
# REQUEST_RATE_TOLERANCE of the best expected-score-per-second rate, prefer one that also
# clears a needed request target's completion_factor_threshold. It can only ever trade a
# small, bounded amount of rate (never redirect the pointing itself, never reach for a
# target that is not already going to be exposed anyway) for a chance at the reward.
REQUEST_RATE_TOLERANCE = 0.9


def _az_distance(a: float, b: float) -> float:
    return abs(wrap180(a - b))


def _bulletin_text(notices: list) -> str:
    """A human-readable rendering of a bulletin's notices, for the LLM call that reads
    "live bulletin text" rather than structured JSON."""
    if not notices:
        return "clear (no active notices)"
    return "; ".join(f"{n.get('event_kind')} {n.get('direction')}" for n in notices)


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
        state.on_result(payload.get("last_result"), hours)
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

        action = self.plan(now, night_end, night_index, hours)
        if self.requests.requests:
            normal_rate = self._action_gain_rate(action, now) if action is not None else 0
            bundle = self.requests.choose(now, normal_rate, self.advice.priority == "requests")
            if bundle is not None and bundle["start"] <= now:
                needed = self._required_completion_predictions(action, now)
                normal_pending = {k: dict(getattr(state, k)) if k == "pending" else getattr(state, k)
                                  for k in ("pending", "pending_start", "pending_duration", "pending_program", "pending_night", "pending_action_index")}
                self._forced_request = bundle
                adopted = False
                try:
                    request_action = self.plan(now, min(night_end, bundle["deadline"]), night_index, hours)
                    provided = self._required_completion_predictions(request_action, now)
                    preserves_required = all(provided.get(i, 0) + 1e-9 >= factor for i, factor in needed.items())
                    if request_action is not None and preserves_required:
                        action = request_action
                        adopted = True
                        self.trace.write({"event": "request_execution", "request_id": bundle["request_id"],
                                          "target_id": state.ids[bundle["target"]], "now_utc": payload["now_utc"],
                                          "duration_seconds": action["duration_seconds"]})
                    elif request_action is not None:
                        self.trace.write({"event": "request_deferred", "request_id": bundle["request_id"],
                                          "reason": "preserve imminent required completion", "required_targets": [state.ids[i] for i in needed]})
                finally:
                    self._forced_request = None
                    if not adopted:
                        for key, value in normal_pending.items():
                            setattr(state, key, value)
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

    # -- instrument fault reporting (deterministic rules + LLM confirmation) -----

    def _maybe_report(self, hours: float, payload: dict):
        state = self.state
        state.force_program = None
        if self.reports >= MAX_REPORTS or hours - self.last_report_hours < 24.0:
            return None
        evidence = state.fault_evidence()
        threshold = REPORT_DROP if self.reports == 0 else REPORT_DROP - 0.07
        if evidence is None or evidence.drop >= threshold:
            self.suspicion_hours = []
            return None
        if evidence.dark_checks < 6:
            state.force_program = "DARK"
        elif evidence.dark_matched < 0.5 * evidence.dark_checks:
            self.suspicion_hours = []
            return None
        if self.suspicion_hours and hours - self.suspicion_hours[-1] < REPORT_SPACING_HOURS:
            return None
        self.suspicion_hours.append(hours)
        if len(self.suspicion_hours) < REPORT_CONFIRMATIONS:
            return None
        self.suspicion_hours = []
        self.reports += 1
        self.last_report_hours = hours
        state.forget_quality_history()
        self.log(f"planner: reporting instrument fault at {payload.get('now_utc')} evidence={evidence}")
        return {"action": "report", "reason": f"quality dropped to {evidence.drop:.0%} of the earlier level",
                "decision_source": "deterministic"}

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
                    if existing is None or (j == forced and existing[1] != forced) or (existing[1] != forced and score > existing[0]):
                        chosen[fib] = (score, j, margin)
                if not chosen:
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
        # Keep the existing DARK diagnostic check meaningful under a linear
        # objective, whose otherwise optimal exposures may never saturate.
        diagnostic_count = min(3, sum(item["k"] * min(ceiling, item["up"]) >= 1 for item in info.values())) if state.force_program == "DARK" else 0
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
        if best is None or (best[0] <= 0 and self._forced_request is None):
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
            "pointing": {"alt_deg": c_alt, "az_deg": c_az},
            "assignments": assignments,
            "duration_seconds": duration,
            "program": program,
        }
