import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stage3.calibration import PointingCalibration
from agent_core.geometry import FiberGrid, shift_altaz, tangent_offsets
from test_strategy import payload


class CalibrationTests(unittest.TestCase):
    def test_fixed_bias_is_inferred_from_hits_only(self):
        grid = FiberGrid(payload()["instrument"])
        fit = PointingCalibration(grid)
        bias = grid.pitch / 2
        for trial in range(5):
            pointing = (50 + trial, 110 + trial * 20)
            positions, fibers, hits = {}, {}, set()
            for fiber in range(grid.n):
                n, e = grid.fiber_center(fiber)
                positions[str(fiber)] = shift_altaz(*pointing, n + (trial % 3 - 1) * .15, e)
                fibers[str(fiber)] = fiber
                offsets = tangent_offsets(*positions[str(fiber)], pointing[0] + bias, pointing[1])
                if offsets and grid.classify(*offsets)[0] == fiber:
                    hits.add(str(fiber))
            fit.observe(pointing, fibers, hits, positions)
        self.assertAlmostEqual(fit.alt_bias, bias, delta=.18)
        self.assertAlmostEqual(fit.az_bias, 0, delta=.18)

    def test_scoring_zero_does_not_make_geometric_misses(self):
        grid = FiberGrid(payload()["instrument"])
        fit = PointingCalibration(grid)
        for _ in range(3):
            positions = {str(f): shift_altaz(50, 100, *grid.fiber_center(f)) for f in range(grid.n)}
            fit.observe((50, 100), {str(f): f for f in range(grid.n)}, set(positions), positions)
        self.assertEqual((fit.alt_bias, fit.az_bias), (0, 0))

    def test_all_misses_use_short_scan_until_offset_is_identifiable(self):
        grid = FiberGrid(payload()["instrument"])
        fit = PointingCalibration(grid)
        desired = (50, 110)
        positions = {str(f): shift_altaz(*desired, *grid.fiber_center(f)) for f in range(grid.n)}
        assignments = {str(f): f for f in range(grid.n)}
        true_bias = grid.pitch
        recovered = False
        for _ in range(100):
            command = fit.command(*desired)
            actual = (command[0] + true_bias, command[1])
            hits = {t for t, pos in positions.items() if grid.classify(*tangent_offsets(*pos, *actual))[0] == assignments[t]}
            fit.observe(command, assignments, hits, positions)
            if len(hits) >= .8 * grid.n and not fit.searching:
                recovered = True
                break
        self.assertTrue(recovered)
        self.assertAlmostEqual(fit.alt_bias, true_bias, delta=grid.pitch / 3)

if __name__ == "__main__":
    unittest.main()
