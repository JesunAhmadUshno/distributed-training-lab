"""Unit tests for the ring all-reduce simulation and the stage-aware
memory accounting in level-2-build/collectives.py.

Hand-checkable expectations: a ring all-reduce over N ranks leaves
every rank with the same vector, equal to the element-wise sum up to
float reduction order, and each rank sends 2 * (N - 1) chunks. Memory
counts for 31 parameters on 2 ranks (largest shard 16): DDP 3 * 31 =
93, ZeRO-1 2 * 31 + 16 = 78, ZeRO-2 31 + 2 * 16 = 63, ZeRO-3 3 * 16
= 48 persistent with a 31 + 48 = 79 peak.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
from collectives import ring_all_reduce, ring_study, stage_memory_accounting  # noqa: E402
from sharding import ShardingError, build_dataset  # noqa: E402


class RingAllReduceTests(unittest.TestCase):
    def test_small_integer_sum_is_exact_and_identical_on_every_rank(self):
        vecs = [[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0], [100.0, 0.0, -1.0, 2.0, 7.0]]
        out, sent = ring_all_reduce(vecs)
        for v in out:
            self.assertEqual(v, [111.0, 22.0, 32.0, 46.0, 62.0])
        # 5 elements over 3 ranks: chunks of 2, 2, 1; each rank sends
        # every chunk but its own final one, twice (reduce-scatter and all-gather).
        self.assertEqual(sum(sent), 2 * (3 - 1) * 5)

    def test_one_rank_sends_nothing(self):
        out, sent = ring_all_reduce([[1.0, 2.0]])
        self.assertEqual(out, [[1.0, 2.0]])
        self.assertEqual(sent, [0])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ShardingError):
            ring_all_reduce([[1.0, 2.0], [1.0]])

    def test_study_matches_full_batch_up_to_float_order(self):
        rows = ring_study(build_dataset())
        self.assertEqual([r["num_ranks"] for r in rows], [1, 2, 4, 8, 16])
        for r in rows:
            self.assertEqual(r["rank_copies_max_diff"], 0.0)
            self.assertLess(r["ring_max_abs_diff_vs_full"], 1e-12)
            self.assertLess(r["sequential_max_abs_diff_vs_full"], 1e-12)
            n = r["num_ranks"]
            self.assertEqual(sum(r["elements_sent_per_rank"]), 2 * (n - 1) * 31)
        self.assertEqual(rows[0]["ring_max_abs_diff_vs_full"], 0.0)


class StageMemoryTests(unittest.TestCase):
    def test_two_rank_counts(self):
        m = stage_memory_accounting(31, 2)
        self.assertEqual(m["ddp"], 93)
        self.assertEqual(m["zero1"], 78)
        self.assertEqual(m["zero2"], 63)
        self.assertEqual(m["zero3_fsdp_persistent"], 48)
        self.assertEqual(m["zero3_fsdp_peak"], 79)

    def test_one_rank_has_no_savings(self):
        m = stage_memory_accounting(31, 1)
        self.assertEqual(m["ddp"], m["zero1"])
        self.assertEqual(m["zero1"], m["zero2"])
        self.assertEqual(m["zero2"], m["zero3_fsdp_persistent"])


if __name__ == "__main__":
    unittest.main()
