"""Two bounded LLM stages: interpret source messages, then adapt priorities.

Only public protocol messages and our own observations are passed to the model.
Source directions/times are never invented by model output; actions stay in Planner.
"""
import hashlib
import json
import time

from .geometry import parse_utc


class AdviceController:
    def __init__(self, state, client, trace):
        self.state, self.client, self.trace = state, client, trace
        self.last_signature = None
        self.interpreted = {}
        self.priority = "required"
        self.priority_night = None
        self.attempts = {"message_understanding": 0, "plan_adaptation": 0}
        self.last_replan = None
        self.last_trigger = None
        self.last_night = None
        self.missed_exposures = 0
        self.applied_stages = set()
        self.urgent_requests = set()

    def _ask(self, phase, prompt, data, remaining):
        allowance = 24 if phase == "message_understanding" else 16
        if self.attempts[phase] >= allowance:
            return None
        before = self.client.calls_made
        retries = self.client.max_retries
        self.client.max_retries = min(retries, allowance - self.attempts[phase])
        try:
            return self.client.ask_json(prompt, data, remaining, stage=phase)
        finally:
            self.attempts[phase] += self.client.calls_made - before
            self.client.max_retries = retries

    def update(self, payload, night_index, now):
        started = time.monotonic()
        state = self.state
        bulletin = payload.get("latest_bulletin") or {}
        forecast = payload.get("latest_forecast") or {}
        night_date = state.night_dates[night_index]
        sources = []
        for scope, message in (("bulletin", bulletin), ("forecast", forecast)):
            for notice in message.get("notices") or []:
                if scope == "forecast" and night_date not in (notice.get("nights") or []):
                    continue
                sources.append({"scope": scope, "source_id": message.get("slot_id") if scope == "bulletin" else message.get("issued_at_utc"),
                                "event_kind": notice.get("event_kind"), "direction": notice.get("direction"),
                                "nights": notice.get("nights", [])})
        sources = sources[:8]
        # Bulletin advice applies only to its current slot. A cached semantic
        # classification of an unchanged event is rebound to the new public slot.
        signature_data = [{k: v for k, v in s.items() if k != "source_id"} for s in sources]
        requests = payload.get("active_requests") or []
        request_ids = {r["request_id"] for r in requests}
        signature = hashlib.sha256(json.dumps([signature_data, sorted(request_ids)], sort_keys=True).encode()).hexdigest()
        remaining = float((payload.get("wallclock") or {}).get("remaining_seconds", 0))
        if self.last_signature is None or signature != self.last_signature:
            answer = self._ask("message_understanding",
                               'Interpret the supplied telescope notices. Output only {"events":[{"source_index":0,"severity":"closed|degraded|obstruction|normal"}],"urgent_requests":["known request ID"]}. '
                               'Use only supplied source indices and request IDs. Rain/storm means closed; haze/cloud/cold means degraded; terrain/rocket means obstruction. Do not generate actions or directions.',
                               {"sources": sources, "requests": [{k: r.get(k) for k in ("request_id", "reason", "deadline_utc", "remaining_count")} for r in requests],
                                "night": night_date}, remaining)
            valid = isinstance(answer, dict) and set(answer) <= {"events", "urgent_requests"} and isinstance(answer.get("events"), list)
            parsed = {}
            if valid:
                for row in answer["events"]:
                    if not isinstance(row, dict) or set(row) != {"source_index", "severity"} or type(row.get("source_index")) is not int or not isinstance(row.get("severity"), str) or row.get("severity") not in {"closed", "degraded", "obstruction", "normal"}:
                        valid = False
                        break
                    index = row["source_index"]
                    if not 0 <= index < len(sources):
                        valid = False
                        break
                    parsed[index] = row["severity"]
                known = {r["request_id"] for r in requests}
                if not isinstance(answer.get("urgent_requests", []), list) or any(not isinstance(r, str) or r not in known for r in answer.get("urgent_requests", [])):
                    valid = False
            self.interpreted = parsed if valid else {}
            self.urgent_requests = set(answer.get("urgent_requests", [])) if valid else set()
            self.last_signature = signature
            if valid:
                self.applied_stages.add("message_understanding")
            self.trace.write({"event": "model_advice_applied", "stage": "message_understanding", "accepted": valid,
                              "sources": sources, "interpretations": self.interpreted})
        if not sources:
            self.interpreted = {}
            self.last_signature = signature
        self.urgent_requests &= request_ids
        state.extra_avoid = {s["direction"] for i, s in enumerate(sources)
                             if self.interpreted.get(i) in {"closed", "obstruction"} and s["direction"] in {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}}
        state.model_direction_factors = {}
        for i, source in enumerate(sources):
            if source["direction"] in state.extra_avoid and self.interpreted.get(i) in {"closed", "obstruction"}:
                factor = .8 if source["scope"] == "forecast" else .35
                direction = source["direction"]
                state.model_direction_factors[direction] = min(factor, state.model_direction_factors.get(direction, 1))
        state.model_altitude_risk = max((.65
                                        for i, source in enumerate(sources)
                                        if source["direction"] == "ALL" and self.interpreted.get(i) == "degraded"), default=0)
        state.duration_scale = 1.0
        result = payload.get("last_result") or {}
        if result.get("action") == "observe":
            self.missed_exposures = self.missed_exposures + 1 if result.get("assigned_count", 0) and result.get("hit_count", 0) == 0 else 0
        kinds = {m.get("record_type") for m in payload.get("new_messages", [])}
        evidence = state.fault_evidence()
        trigger = ("initial" if self.last_night is None else
                   "state_resync" if "state_resync" in kinds else
                   "request" if "observation_request" in kinds or "observation_request_result" in kinds else
                   "geometry_misses" if self.missed_exposures >= 3 else
                   "quality_drop" if evidence is not None and evidence.drop < .65 else None)
        cooldown = self.last_replan is None or (now - self.last_replan).total_seconds() >= 2 * state.slot_seconds
        if trigger is not None and cooldown and (trigger != self.last_trigger or self.last_night != night_index):
            remaining = max(0.0, remaining - (time.monotonic() - started))
            answer = self._ask("plan_adaptation",
                               'Adapt a telescope survey plan using the public progress and evidence. Output only {"priority":"required|requests|survey|diagnostic","reason":"brief evidence"}. '
                               'Required targets carry large penalties. Requests are worthwhile only if feasible. Diagnostic is for repeated geometric misses or unexplained quality drops. Never output an action or overwrite progress.',
                               {"trigger": trigger, "required_remaining": sum(r and f < state.scoring.required_threshold for r, f in zip(state.required, state.factor)),
                                "requests": [{k: r.get(k) for k in ("request_id", "remaining_count", "minimum_completed", "deadline_utc", "completion_reward")} for r in requests],
                                "interpreted_urgent_requests": sorted(self.urgent_requests),
                                "fault_evidence": evidence._asdict() if evidence else None,
                                "consecutive_missed_exposures": self.missed_exposures,
                                "remaining_wallclock_seconds": remaining}, remaining)
            valid = (isinstance(answer, dict) and set(answer) <= {"priority", "reason"}
                     and isinstance(answer.get("priority"), str) and isinstance(answer.get("reason", ""), str)
                     and answer.get("priority") in {"required", "requests", "survey", "diagnostic"})
            if valid and answer["priority"] == "requests" and not requests:
                valid = False
            if valid and answer["priority"] == "diagnostic" and trigger not in {"geometry_misses", "quality_drop", "state_resync"}:
                valid = False
            if valid:
                self.priority = answer["priority"]
                self.priority_night = night_index
                self.applied_stages.add("plan_adaptation")
            self.last_replan, self.last_trigger = now, trigger
            self.trace.write({"event": "model_advice_applied", "stage": "plan_adaptation", "accepted": valid,
                              "trigger": trigger, "priority": self.priority})
        if self.priority_night != night_index or (self.priority == "requests" and not requests):
            self.priority = "required"
        self.last_night = night_index

    def required_multiplier(self):
        return 1.15 if self.priority == "required" and self.priority_night == self.last_night and "plan_adaptation" in self.applied_stages else 1.0
