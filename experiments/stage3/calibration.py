"""Fit pointing offsets to geometric hit/miss feedback, never to weather scores."""
from collections import deque
from agent_core.geometry import tangent_offsets


class PointingCalibration:
    def __init__(self, grid):
        self.grid = grid
        self.history = deque(maxlen=192)
        self.alt_bias = self.az_bias = 0.0
        self.exposures = 0
        self.searching = False
        self.search_index = 0
        step = grid.pitch / 4
        self.search_offsets = sorted([(a * step, z * step) for a in range(-4, 5) for z in range(-4, 5)],
                                     key=lambda p: p[0] ** 2 + p[1] ** 2)

    def observe(self, pointing, assignments, hits, public_altaz):
        self.exposures += 1
        for target_id, fiber in assignments.items():
            if target_id in public_altaz:
                self.history.append((pointing, public_altaz[target_id], fiber, target_id in hits, self.exposures))
        if len(self.history) < 24 or len({r[4] for r in self.history}) < 2:
            return False
        positive = sum(r[3] for r in self.history)
        if positive < 3:
            self.searching = self.exposures >= 3
            if self.searching:
                self.search_index = (self.search_index + 1) % len(self.search_offsets)
            return False
        self.searching = False
        # Frequent refitting is unnecessary when current pointing explains hits.
        if self.exposures % 5 and sum(r[3] for r in self.history) / len(self.history) > .85:
            return False
        def accuracy(a, z):
            correct = 0
            for (ca, cz), (alt, az), fiber, hit, _ in self.history:
                offsets = tangent_offsets(alt, az, ca + a, (cz + z) % 360)
                expected = offsets is not None and self.grid.classify(*offsets)[0] == fiber
                correct += expected == hit
            return correct / len(self.history)
        current = accuracy(self.alt_bias, self.az_bias)
        if current >= .90:
            return False
        step = self.grid.pitch / 4
        candidates = [(accuracy(a * step, z * step), a * step, z * step) for a in range(-4, 5) for z in range(-4, 5)]
        best = max(row[0] for row in candidates)
        if best < .90 or best - current < .05:
            return False
        tied = [(a, z) for score, a, z in candidates if score >= best - .005]
        self.alt_bias = sum(a for a, _ in tied) / len(tied)
        self.az_bias = sum(z for _, z in tied) / len(tied)
        return True

    def command(self, alt, az):
        a, z = self.search_offsets[self.search_index] if self.searching else (self.alt_bias, self.az_bias)
        return min(90.0, max(0.0, alt - a)), (az - z) % 360
