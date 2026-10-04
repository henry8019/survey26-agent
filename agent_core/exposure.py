"""Joint fibre filling and integer-second exposure sizing from public estimates.

For one fixed pointing/program, each target has a piecewise linear marginal gain
curve. Fibre envelopes can switch targets at intersections, not just saturation.
Their sum divided by time is monotone between breakpoints, so adjacent integer
seconds suffice for the local model. This is not a claim about future weather.
"""
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class Curve:
    i: int
    k: float
    up: float
    weight: float
    previous: float
    penalty: float
    goal: float
    model: float
    alt: float
    az: float
    confidence: float = 1.0

    def gain(self, duration, multiplier):
        if duration > self.up or self.k <= 0:
            return 0.0
        factor = min(1.0, self.k * duration)
        gain = max(0.0, self.weight * multiplier * factor - self.previous)
        if self.penalty and factor >= self.goal:
            gain += self.penalty
        return gain * self.confidence

    def breaks(self, multiplier):
        if self.k <= 0:
            return [self.up]
        points = [self.up, 1 / self.k, self.goal / self.k]
        slope = self.weight * multiplier * self.k
        if slope > 0:
            points.append(self.previous / slope)
        return points

    def line(self, duration, multiplier):
        if duration > self.up or self.k <= 0:
            return 0.0, 0.0
        factor = min(1.0, self.k * duration)
        score = self.weight * multiplier * factor
        if score <= self.previous:
            a, b = 0.0, 0.0
        elif factor >= 1:
            a, b = 0.0, score - self.previous
        else:
            a, b = self.weight * multiplier * self.k, -self.previous
        if self.penalty and factor >= self.goal:
            b += self.penalty
        return a * self.confidence, b * self.confidence


def critical_seconds(cells, multipliers, lower, upper):
    points = {float(lower), float(upper)}
    for fiber, curves in cells.items():
        values = multipliers[fiber]
        for curve, mult in zip(curves, values):
            points.update(p for p in curve.breaks(mult) if lower <= p <= upper)
        for index, (first, m1) in enumerate(zip(curves, values)):
            for second, m2 in zip(curves[index + 1:], values[index + 1:]):
                edges = sorted({float(lower), float(upper),
                                *(p for p in first.breaks(m1) + second.breaks(m2) if lower < p < upper)})
                for left, right in zip(edges, edges[1:]):
                    middle = (left + right) / 2
                    a1, b1 = first.line(middle, m1)
                    a2, b2 = second.line(middle, m2)
                    if abs(a1 - a2) > 1e-15:
                        crossing = (b2 - b1) / (a1 - a2)
                        if left < crossing < right:
                            points.add(crossing)
    return {t for point in points for t in (math.floor(point) - 1, math.floor(point), math.ceil(point), math.ceil(point) + 1)
            if lower <= t <= upper}


def choose(cells, multipliers, duration, forced=None):
    selected, gain = {}, 0.0
    for fiber, curves in cells.items():
        available = [(c.gain(duration, m), c) for c, m in zip(curves, multipliers[fiber]) if c.up >= duration]
        if forced is not None and any(c.i == forced[0] for c in curves):
            available = [(v, c) for v, c in available if c.i == forced[0] and c.k * duration >= forced[1]]
        if not available:
            continue
        value, winner = max(available, key=lambda row: row[0])
        selected[fiber] = winner
        gain += value
    if forced is not None and not any(c.i == forced[0] and c.k * duration >= forced[1] for c in selected.values()):
        return None
    return gain, selected


def optimize(cells, programs, scoring, scale, lower, upper, *, thresholds=None, forced=None, deadline=None,
             prefix_gain=0.0, prefix_seconds=0, discount=1.0):
    best = request_best = None
    thresholds = thresholds or {}
    for program in programs:
        multipliers = {f: [scoring.program_multiplier(program, scoring.program_band(c.model * scale / .95)) for c in curves]
                       for f, curves in cells.items()}
        durations = critical_seconds(cells, multipliers, lower, upper)
        for curves in cells.values():
            for curve in curves:
                threshold = forced[1] if forced and curve.i == forced[0] else thresholds.get(curve.i)
                if threshold is not None and curve.k > 0:
                    point = math.ceil(threshold / curve.k)
                    if lower <= point <= upper:
                        durations.add(point)
        for index, duration in enumerate(sorted(durations)):
            if deadline is not None and index % 8 == 0 and time.monotonic() >= deadline:
                break
            result = choose(cells, multipliers, duration, forced)
            if result is None:
                continue
            gain, selected = result
            trial = ((prefix_gain + discount * gain) / (prefix_seconds + duration), duration, program, selected)
            tie = best is not None and trial[0] >= best[0] * .999999
            preferred_duration = best is not None and (duration < best[1] if forced else duration > best[1])
            if best is None or trial[0] > best[0] * 1.000001 or (tie and preferred_duration):
                best = trial
            completes = any(c.i in thresholds and c.k * duration >= thresholds[c.i] for c in selected.values())
            if completes and (request_best is None or trial[0] > request_best[0]):
                request_best = trial
    if best and request_best and request_best[0] >= best[0] * .9:
        best = request_best
    return best
