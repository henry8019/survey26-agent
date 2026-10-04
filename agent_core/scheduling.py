"""Bounded maximum-coverage scheduling over public, fixed exposure columns.

Science uses per-target maxima and required completion pays once. This solver
knows no catalogue, weather, model client or card identity and never writes state.
"""
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class Column:
    start: int
    end: int
    scores: dict
    required: frozenset
    root: int = -1


def extend(scores, required, column, penalty):
    merged = dict(scores)
    gain = 0.0
    for target, score in column.scores.items():
        old = merged.get(target, 0.0)
        merged[target] = max(old, score)
        gain += max(0.0, score-old)
    return merged, required | column.required, gain + penalty*len(column.required-required)


def schedule(roots, future, penalty, fallback, deadline, width=32):
    """Compare all first actions under a common horizon and column inventory.

    Always retain a feasible greedy schedule starting with fallback. The beam
    may prune, so its result is a feasible incumbent, never a claim of optimality.
    If the fallback schedule cannot be completed within budget, do not switch.
    """
    initial = []
    for root in roots:
        scores, required, value = extend({}, frozenset(), root, penalty)
        initial.append((root.end, scores, required, value, root.root))
    reference = next((row for row in initial if row[4] == fallback), None)
    if reference is None:
        return fallback, {'complete': False, 'reason': 'missing incumbent'}
    end, scores, required, baseline, _ = reference
    while True:
        if time.monotonic() >= deadline:
            return fallback, {'complete': False, 'reason': 'baseline budget'}
        options = []
        for column in future:
            if column.start >= end:
                s, r, gain = extend(scores, required, column, penalty)
                if gain > 0:
                    options.append((gain/(column.end-end), column, s, r, gain))
        if not options:
            break
        _, column, scores, required, gain = max(options, key=lambda row: row[0])
        end, baseline = column.end, baseline+gain
    best_value, winner, beam = baseline, fallback, initial
    pruned = False
    while beam:
        next_beam = []
        for end, scores, required, value, root in beam:
            if value > best_value+1e-9:
                best_value, winner = value, root
            for column in future:
                if time.monotonic() >= deadline:
                    return fallback, {'complete': False, 'reason': 'search budget', 'baseline': baseline}
                if column.start < end:
                    continue
                s, r, gain = extend(scores, required, column, penalty)
                if gain > 1e-9:
                    next_beam.append((column.end, s, r, value+gain, root))
        next_beam.sort(key=lambda row: (-row[3], row[0], row[4] != fallback))
        pruned |= len(next_beam) > width
        beam = next_beam[:width]
    return winner, {'complete': True, 'pruned': pruned, 'baseline': baseline,
                    'selected_gain': best_value, 'changed': winner != fallback}
