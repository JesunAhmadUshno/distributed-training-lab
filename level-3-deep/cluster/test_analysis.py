"""Known-answer tests for analysis.py (standard library only). Run: python3 -m unittest test_analysis -v"""

import math
import unittest

import analysis as A


class T(unittest.TestCase):
    def test_ols_recovers_a_line(self):
        a, b, r2 = A.ols([0, 1, 2, 3], [1, 3, 5, 7])
        self.assertAlmostEqual(a, 1)
        self.assertAlmostEqual(b, 2)
        self.assertAlmostEqual(r2, 1)

    def test_link_fit_recovers_alpha_beta(self):
        rows = [
            {"exp": "E2", "op": "all_reduce", "bytes": s, "seconds": [0.001 + s / 100e6] * 3}
            for s in (4096, 2**20, 2**24, 2**28)
        ]
        f = A.link_fit(rows)
        self.assertAlmostEqual(f["alpha_s"], 0.001, places=9)
        self.assertAlmostEqual(f["beta_MBps"], 100, places=6)

    def test_comm_time_counts_buckets(self):
        fit = {"alpha_s": 0.01, "beta_MBps": 100}
        p = 10 * 2**20  # 40 MiB of fp32 gradients -> 2 buckets
        self.assertAlmostEqual(A.comm_time(p, fit), 2 * 0.01 + 40 * 2**20 / 100e6)

    def test_percentile(self):
        self.assertEqual(A.pct([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(A.pct([0, 10], 0.25), 2.5)

    def test_cell_stat_mean_of_medians(self):
        runs = [{"rep": 0, "step_seconds": [1, 2, 9]}, {"rep": 1, "step_seconds": [3, 4, 5]}]
        v, ci, _ = A.cell_stat(runs)
        self.assertEqual(v, 3)
        self.assertTrue(ci[0] <= v <= ci[1])

    def test_predict_matches_hand_calculation(self):
        rows = [
            {"exp": "E2", "op": "all_reduce", "bytes": s, "seconds": [0.002 + s / 110e6] * 2} for s in (4096, 2**26)
        ]
        rows += [
            {
                "exp": "E3",
                "mode": "single",
                "size": "L",
                "accum": 1,
                "rep": 0,
                "params": 2**24,
                "step_seconds": [0.4] * 5,
            }
        ]
        c = A.predict(rows)["cells"]["L-1"]
        comm = math.ceil(4 * 2**24 / A.BUCKET) * 0.002 + 4 * 2**24 / 110e6
        self.assertAlmostEqual(c["pred_t_ddp"], 0.2 + comm, places=6)
        self.assertAlmostEqual(c["pred_speedup"], 0.4 / (0.2 + comm), places=6)

    def test_same_tolerates_last_digit_but_not_real_change(self):
        a = {"x": [0.00035965241784430556, 1], "ok": True, "s": "L"}
        self.assertTrue(A.same(a, {"x": [0.00035965241784436107, 1], "ok": True, "s": "L"}))
        self.assertFalse(A.same(a, {"x": [0.00036, 1], "ok": True, "s": "L"}))
        self.assertFalse(A.same(a, {"x": [0.00035965241784430556, 1], "ok": False, "s": "L"}))
        self.assertFalse(A.same(a, {"x": [0.00035965241784430556], "ok": True, "s": "L"}))


if __name__ == "__main__":
    unittest.main()
