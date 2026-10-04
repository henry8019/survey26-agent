"""Whole-request opportunity-cost planning from public windows and valid feedback."""
from datetime import timedelta
from itertools import combinations
import math
from .geometry import (Moon, SIDEREAL_DEG_PER_SECOND, local_sidereal_deg,
                       lunar_factor, parse_utc, radec_to_altaz, wrap180)


class RequestPlanner:
    def __init__(self, state, trace):
        self.state, self.trace = state, trace
        self.requests = {}
        self.selected = None

    def sync(self, active):
        # The current snapshot is authoritative, including corrections. Never
        # union with the previous snapshot's completed IDs after data loss.
        self.requests = {r["request_id"]: dict(r) for r in active}

    def completed(self, request):
        done = set(request.get("completed_target_ids") or [])
        issued, deadline = parse_utc(request["issued_at_utc"]), parse_utc(request["deadline_utc"])
        threshold = float(request["completion_factor_threshold"])
        targets = set(request["target_ids"])
        for record in self.state.ledger:
            if (record.target_id in targets and record.start_utc and record.end_utc
                    and issued <= parse_utc(record.start_utc)
                    and parse_utc(record.end_utc) <= deadline and record.lower >= threshold):
                done.add(record.target_id)
        return done & targets

    def _slot(self, i, earliest, deadline, threshold):
        state = self.state
        for start, end in state.nights:
            start, end = max(start, earliest), min(end, deadline)
            if start >= end:
                continue
            # Public geometry supplies the rising/setting interval. Probe at
            # ten-minute spacing to avoid a dependency on a numerical solver.
            probe = start
            while probe < end:
                lst = local_sidereal_deg(probe, state.lon)
                ha, h = wrap180(lst - state.ra[i]), state.hmax[i]
                if -h <= ha <= h:
                    alt, az = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
                    moon = Moon(probe, lst, state.lat)
                    q = state.scoring.quality_model(alt, lunar_factor(moon, state.ra[i], state.dec[i], state.scoring.lunar_model))
                    duration = max(state.min_exposure, math.ceil(threshold * state.scoring.f0t0 / max(1e-9, state.flux[i] * q * state.scale * .8)))
                    setting = (h - ha) / SIDEREAL_DEG_PER_SECOND if h < 180 else 1e9
                    if duration <= min(state.max_exposure, (end - probe).total_seconds(), setting):
                        return probe, duration, min(end, probe + timedelta(seconds=setting))
                probe += timedelta(seconds=max(state.min_exposure, 600))
        return None

    def _observing_seconds(self, start, end):
        return sum(max(0.0, (min(night_end, end) - max(night_start, start)).total_seconds())
                   for night_start, night_end in self.state.nights)

    def _target_gain(self, i, factor):
        """Conservative ordinary-science and required-completion benefit.

        No guessed program bonus or uniformity gain; count each target once.
        """
        state = self.state
        science = max(0.0, state.weight[i] * min(1.0, factor)
                      * state.scoring.mismatch_multiplier - state.best_score[i])
        required = (state.scoring.required_penalty if state.required[i]
                    and state.factor[i] < state.scoring.required_threshold <= factor else 0.0)
        return science + required

    def assess_execution(self, bundle, action, now, normal_rate, prefer_requests=False):
        """Price the actual first exposure and recheck the remainder without writing progress."""
        request = self.requests.get(bundle["request_id"])
        if request is None:
            return None
        state = self.state
        issued, deadline = parse_utc(request["issued_at_utc"]), parse_utc(request["deadline_utc"])
        cursor = now + timedelta(seconds=action["duration_seconds"])
        if now < issued or cursor > deadline:
            return None
        threshold = float(request["completion_factor_threshold"])
        predicted = set()
        gains = {}
        targets = set(request["target_ids"])
        for target in action["assignments"].values():
            prediction = state.pending.get(target)
            if prediction is not None:
                i = state.index_of[target]
                factor = state.scoring.completion_factor(state.flux[i], action["duration_seconds"],
                                                        prediction.model * state.scale * .9)
                gains[i] = self._target_gain(i, factor)
                if target in targets and factor >= threshold:
                    predicted.add(target)
        done = self.completed(request)
        if not predicted - done:
            return None
        remaining = max(0, int(request["minimum_completed"]) - len(done | predicted))
        pool = [i for i in bundle.get("bundle", []) if state.ids[i] not in done | predicted]
        if len(pool) < remaining:
            return None
        for i in pool[:remaining]:
            slot = self._slot(i, cursor, deadline, threshold)
            if slot is None:
                return None
            cursor = slot[0] + timedelta(seconds=slot[1])
            gains[i] = max(gains.get(i, 0.0), self._target_gain(i, threshold))
        cost = self._observing_seconds(now, cursor)
        margin = .10 if prefer_requests else .20
        benefit = sum(gains.values())
        net = float(request["completion_reward"]) + benefit - normal_rate * cost * (1 + margin)
        return {"actual_bundle_cost_seconds": cost, "actual_bundle_net": net,
                "actual_bundle_observation_gain": benefit} if net > 0 else None

    def choose(self, now, normal_rate, prefer_requests=False):
        state = self.state
        best = None
        for request in self.requests.values():
            deadline = parse_utc(request["deadline_utc"])
            if deadline <= now:
                continue
            remaining = max(0, int(request["minimum_completed"]) - len(self.completed(request)))
            if not remaining:
                continue
            threshold = float(request["completion_factor_threshold"])
            earliest = max(now, parse_utc(request["issued_at_utc"]))
            slots = []
            done = self.completed(request)
            for target in request["target_ids"]:
                if target in done or target not in state.index_of:
                    continue
                i = state.index_of[target]
                slot = self._slot(i, earliest, deadline, threshold)
                if slot:
                    slots.append((i, *slot))
            if len(slots) < remaining:
                continue
            slots.sort(key=lambda row: row[2])
            pool = slots[:max(remaining, min(12, len(slots)))]
            for number, subset in enumerate(combinations(pool, remaining)):
                if number >= 64:
                    break
                ordered = sorted(subset, key=lambda row: row[3])
                cursor, feasible = earliest, True
                for i, start, duration, setting in ordered:
                    slot = self._slot(i, max(cursor, start), deadline, threshold)
                    if slot is None:
                        feasible = False
                        break
                    cursor = slot[0] + timedelta(seconds=slot[1])
                if not feasible:
                    continue
                # Reward is valued once. Charge observable waiting gaps too;
                # daytime between nights consumes no scientific opportunity.
                cost = self._observing_seconds(now, cursor)
                opportunity = normal_rate * cost
                margin = .10 if prefer_requests else .20
                benefit = sum(self._target_gain(i, threshold) for i, *_ in ordered)
                net = float(request["completion_reward"]) + benefit - opportunity * (1 + margin)
                if net <= 0:
                    continue
                first = ordered[0]
                candidate = {"request_id": request["request_id"], "target": first[0], "start": first[1],
                             "threshold": threshold, "deadline": deadline, "net": net, "cost": cost,
                             "bundle": [row[0] for row in ordered]}
                if best is None or net > best["net"]:
                    best = candidate
        self.selected = best
        if best:
            self.trace.write({"event": "request_bundle", **{k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in best.items()}})
        return best
