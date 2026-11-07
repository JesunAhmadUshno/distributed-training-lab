"""Unit tests for the Learn rank sweep and shard-split sweep.

Expectations come from the data-parallel algebra, not from the
recorded file: equal shards make the naive all-reduce average equal
the full-batch gradient up to float reduction order, unequal shards
bias it, and the size-weighted average is always unbiased.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
from data_parallel import build_dataset  # noqa: E402
from sweeps import imbalance_sweep, rank_sweep  # noqa: E402

FLOAT_ORDER_TOL = 1e-12


class RankSweepTests(unittest.TestCase):
    def setUp(self):
        self.rows = rank_sweep(build_dataset())

    def test_covers_one_to_eight_ranks(self):
        self.assertEqual([r["num_ranks"] for r in self.rows], list(range(1, 9)))

    def test_equal_shards_match_single_process_up_to_float_order(self):
        for r in self.rows:
            if r["equal_shards"]:
                self.assertLess(r["max_grad_diff_vs_single_process"], FLOAT_ORDER_TOL)
                self.assertLess(r["final_params_max_diff_vs_one_rank"], FLOAT_ORDER_TOL)

    def test_unequal_shards_bias_the_naive_average(self):
        unequal = [r for r in self.rows if not r["equal_shards"]]
        self.assertEqual([r["num_ranks"] for r in unequal], [3, 5, 6, 7])
        for r in unequal:
            self.assertGreater(r["max_grad_diff_vs_single_process"], 0.5)

    def test_one_rank_is_exact(self):
        self.assertEqual(self.rows[0]["max_grad_diff_vs_single_process"], 0.0)

    def test_accounting_scales_with_ranks(self):
        for r in self.rows:
            self.assertEqual(r["all_reduce_bytes_per_step_all_ranks"], 3 * 4 * r["num_ranks"])


class ImbalanceSweepTests(unittest.TestCase):
    def setUp(self):
        self.rows = imbalance_sweep(build_dataset())

    def test_all_seven_splits(self):
        self.assertEqual([r["shard_sizes"] for r in self.rows], [[k, 8 - k] for k in range(1, 8)])

    def test_weighted_average_is_always_unbiased(self):
        for r in self.rows:
            self.assertLess(r["weighted_max_abs_diff_vs_full"], FLOAT_ORDER_TOL)

    def test_naive_average_is_exact_only_for_the_even_split(self):
        for r in self.rows:
            if r["shard_sizes"] == [4, 4]:
                self.assertLess(r["naive_max_abs_diff_vs_full"], FLOAT_ORDER_TOL)
            else:
                self.assertGreater(r["naive_max_abs_diff_vs_full"], 1.0)

    def test_six_two_split_matches_the_recorded_experiment(self):
        row = self.rows[5]
        self.assertAlmostEqual(row["naive_max_abs_diff_vs_full"], 2.833333333333334, places=12)


if __name__ == "__main__":
    unittest.main()
