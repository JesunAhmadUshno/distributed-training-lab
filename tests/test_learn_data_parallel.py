"""Unit tests for Level 1 Learn simulated data-parallel training.

Hand-computed expectations. At the initial parameters [0, 0, 0]
every prediction is 0, so each error is -y.

Full batch (8 examples): sum of e is -20, sum of e*x1 is -42,
sum of e*x2 is -12, so the gradient is (2/8) * (-42, -12, -20)
= (-10.5, -3.0, -5.0) and the loss is (9+0+4+25+1+16+1+36)/8
= 92/8 = 11.5.

Shard 0 (first 4): gradient (2/4) * (-15, -2, -10)
= (-7.5, -1.0, -5.0), loss (9+0+4+25)/4 = 9.5.
Shard 1 (last 4): gradient (2/4) * (-27, -10, -10)
= (-13.5, -5.0, -5.0), loss (1+16+1+36)/4 = 13.5.
Average of the shard gradients: (-10.5, -3.0, -5.0), equal to the
full-batch gradient, as equal shard sizes require.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
from data_parallel import (  # noqa: E402
    DataParallelError, all_reduce_average, build_dataset,
    communication_accounting, compute_gradient, compute_loss,
    imbalance_experiment, max_abs_diff, predict, shard_dataset,
    train_data_parallel, weighted_average,
)

TOL = 1e-9


def assert_vec(case, got, want):
    case.assertEqual(len(got), len(want))
    for g, w in zip(got, want):
        case.assertAlmostEqual(g, w, delta=TOL)


class TestModel(unittest.TestCase):
    def test_predict_and_true_params(self):
        # True parameters [2, -1, 1] reproduce every target exactly.
        data = build_dataset()
        for ex in data:
            self.assertAlmostEqual(predict([2.0, -1.0, 1.0], ex["x"]), ex["y"], delta=TOL)
        self.assertAlmostEqual(compute_loss([2.0, -1.0, 1.0], data), 0.0, delta=TOL)

    def test_initial_loss_hand_computed(self):
        self.assertAlmostEqual(compute_loss([0.0, 0.0, 0.0], build_dataset()), 11.5, delta=TOL)

    def test_full_gradient_hand_computed(self):
        assert_vec(self, compute_gradient([0.0, 0.0, 0.0], build_dataset()),
                   [-10.5, -3.0, -5.0])

    def test_bad_params_rejected(self):
        with self.assertRaises(DataParallelError):
            compute_gradient([0.0, 0.0], build_dataset())
        with self.assertRaises(DataParallelError):
            compute_loss([0.0, 0.0, 0.0], [])


class TestShardsAndAllReduce(unittest.TestCase):
    def setUp(self):
        self.data = build_dataset()
        self.shards = shard_dataset(self.data, 2)

    def test_balanced_shard_sizes_and_order(self):
        self.assertEqual([len(s) for s in self.shards], [4, 4])
        self.assertEqual(self.shards[0] + self.shards[1], self.data)

    def test_uneven_split_sizes(self):
        shards = shard_dataset(self.data, 3)
        self.assertEqual([len(s) for s in shards], [3, 3, 2])

    def test_bad_rank_counts_rejected(self):
        with self.assertRaises(DataParallelError):
            shard_dataset(self.data, 0)
        with self.assertRaises(DataParallelError):
            shard_dataset(self.data, 9)

    def test_shard_gradients_hand_computed(self):
        assert_vec(self, compute_gradient([0.0, 0.0, 0.0], self.shards[0]),
                   [-7.5, -1.0, -5.0])
        assert_vec(self, compute_gradient([0.0, 0.0, 0.0], self.shards[1]),
                   [-13.5, -5.0, -5.0])

    def test_shard_losses_hand_computed(self):
        self.assertAlmostEqual(compute_loss([0.0, 0.0, 0.0], self.shards[0]), 9.5, delta=TOL)
        self.assertAlmostEqual(compute_loss([0.0, 0.0, 0.0], self.shards[1]), 13.5, delta=TOL)

    def test_all_reduce_equals_single_process(self):
        grads = [compute_gradient([0.0, 0.0, 0.0], s) for s in self.shards]
        averaged = all_reduce_average(grads)
        assert_vec(self, averaged, [-10.5, -3.0, -5.0])
        full = compute_gradient([0.0, 0.0, 0.0], self.data)
        self.assertLessEqual(max_abs_diff(averaged, full), TOL)

    def test_all_reduce_empty_rejected(self):
        with self.assertRaises(DataParallelError):
            all_reduce_average([])


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.result = train_data_parallel(build_dataset(), num_ranks=2, steps=20, lr=0.05)
        self.summary = self.result["summary"]

    def test_first_step_params_hand_computed(self):
        # Step 1 update: 0 - 0.05 * (-10.5, -3.0, -5.0) = (0.525, 0.15, 0.25).
        assert_vec(self, self.result["log"][0]["params_after_update"], [0.525, 0.15, 0.25])
        assert_vec(self, self.result["log"][0]["averaged_gradient"], [-10.5, -3.0, -5.0])

    def test_grad_matches_single_process_every_step(self):
        for entry in self.result["log"]:
            self.assertLessEqual(entry["grad_max_abs_diff"], 1e-9)
        self.assertLessEqual(self.summary["max_grad_diff_vs_single_process"], 1e-9)

    def test_replicas_stay_identical(self):
        for entry in self.result["log"]:
            self.assertEqual(entry["replica_max_abs_diff_after_update"], 0.0)
        self.assertEqual(self.summary["max_replica_diff"], 0.0)
        self.assertEqual(self.result["replicas_final"][0], self.result["replicas_final"][1])

    def test_loss_falls_and_log_shape(self):
        self.assertEqual(len(self.result["log"]), 20)
        self.assertAlmostEqual(self.summary["initial_loss"], 11.5, delta=TOL)
        self.assertLess(self.summary["final_loss"], self.summary["initial_loss"])
        losses = [e["combined_loss_before_update"] for e in self.result["log"]]
        self.assertEqual(losses, sorted(losses, reverse=True))

    def test_summary_fields(self):
        s = self.summary
        self.assertEqual(s["num_ranks"], 2)
        self.assertEqual(s["num_examples"], 8)
        self.assertEqual(s["shard_sizes"], [4, 4])
        self.assertEqual(s["steps"], 20)
        assert_vec(self, s["final_params"], self.result["replicas_final"][0])

    def test_determinism(self):
        again = train_data_parallel(build_dataset(), num_ranks=2, steps=20, lr=0.05)
        self.assertEqual(again, self.result)

    def test_bad_training_inputs_rejected(self):
        with self.assertRaises(DataParallelError):
            train_data_parallel(build_dataset(), steps=0)
        with self.assertRaises(DataParallelError):
            train_data_parallel(build_dataset(), lr=0)


class TestCommunicationAccounting(unittest.TestCase):
    def test_accounting_hand_computed(self):
        # 3 params at 4 bytes = 12 bytes per rank per step; 2 ranks
        # give 24 bytes per step; 20 steps give 480 bytes in total
        # and 240 bytes per rank. Accounting only, no transfer.
        acc = communication_accounting(3, 2, 20)
        self.assertEqual(acc["payload_bytes_per_rank_per_step"], 12)
        self.assertEqual(acc["bytes_per_step_all_ranks"], 24)
        self.assertEqual(acc["total_bytes_all_ranks"], 480)
        self.assertEqual(acc["total_bytes_per_rank"], 240)
        with self.assertRaises(DataParallelError):
            communication_accounting(0, 2, 20)


class TestImbalance(unittest.TestCase):
    def test_naive_average_deviates_weighted_matches(self):
        # 6+2 split at initial params: shard gradients are
        # (2/6)*(-23, -4, -13) and (2/2)*(-19, -8, -7). The naive
        # average is (-13.333..., -4.666..., -5.666...), which
        # differs from the full gradient (-10.5, -3.0, -5.0) by
        # 2.8333... at most. The size-weighted average matches.
        exp = imbalance_experiment(build_dataset())
        self.assertEqual(exp["shard_sizes"], [6, 2])
        assert_vec(self, exp["rank_gradients"][1], [-19.0, -8.0, -7.0])
        self.assertAlmostEqual(exp["naive_max_abs_diff_vs_full"], 2.833333333333333, delta=1e-9)
        self.assertLessEqual(exp["weighted_max_abs_diff_vs_full"], TOL)
        assert_vec(self, exp["weighted_average"], [-10.5, -3.0, -5.0])
        assert_vec(self, weighted_average(exp["rank_gradients"], [6, 2]),
                   [-10.5, -3.0, -5.0])


if __name__ == "__main__":
    unittest.main()
