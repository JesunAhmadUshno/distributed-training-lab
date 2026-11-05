"""Unit tests for Level 2 Build sharded training emulation.

Hand-checkable arithmetic used below:
- The model has 31 parameters: W1 4 by 5 = 20, b1 = 5, W2 = 5,
  b2 = 1. Partitioned across 2 ranks contiguously, the shards are
  16 and 15, covering every index exactly once.
- Communication per step: all-gather received volume is
  (31 - 16) * 4 + (31 - 15) * 4 = 60 + 64 = 124 bytes, and
  reduce-scatter received volume is 16 * 4 + 15 * 4 = 124 bytes,
  so 248 bytes per step and 248 * 30 = 7,440 bytes over 30 steps.
- Memory: persistent per rank is 2 * shard = [32, 30] elements,
  peak is 31 + 3 * shard = [79, 76], and the full single-process
  baseline is 3 * 31 = 93 elements.
- Both strategies do the same arithmetic per element as the
  single-process reference, so final differences are exactly 0.0.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
from sharding import (  # noqa: E402
    NUM_PARAMS, ShardedTrainer, ShardingError, all_gather,
    build_dataset, communication_accounting, compute_gradient,
    compute_loss, initial_params, memory_accounting,
    partition_indices, run_strategy, shard_vector,
    train_single_process,
)

TOL = 1e-9


class TestPartition(unittest.TestCase):
    def test_partition_covers_all_params_exactly_once(self):
        parts = partition_indices(NUM_PARAMS, 2)
        flat = [i for p in parts for i in p]
        self.assertEqual(sorted(flat), list(range(NUM_PARAMS)))
        self.assertEqual(len(flat), len(set(flat)))

    def test_partition_sizes_16_and_15(self):
        self.assertEqual([len(p) for p in partition_indices(31, 2)], [16, 15])

    def test_partition_uneven_and_single_rank(self):
        self.assertEqual([len(p) for p in partition_indices(31, 3)], [11, 10, 10])
        self.assertEqual(partition_indices(31, 1), [list(range(31))])

    def test_partition_bad_inputs_rejected(self):
        for bad in (0, -1, 32):
            with self.assertRaises(ShardingError):
                partition_indices(31, bad)
        with self.assertRaises(ShardingError):
            partition_indices(0, 2)

    def test_all_gather_reconstructs_vector(self):
        vec = initial_params()
        parts = partition_indices(NUM_PARAMS, 2)
        shards = shard_vector(vec, parts)
        self.assertEqual([len(s) for s in shards], [16, 15])
        self.assertEqual(all_gather(shards, parts, NUM_PARAMS), vec)

    def test_all_gather_bad_shard_rejected(self):
        parts = partition_indices(NUM_PARAMS, 2)
        shards = shard_vector(initial_params(), parts)
        shards[0] = shards[0][:-1]
        with self.assertRaises(ShardingError):
            all_gather(shards, parts, NUM_PARAMS)
        with self.assertRaises(ShardingError):
            all_gather(shards[:1], parts, NUM_PARAMS)


class TestModelAndGradient(unittest.TestCase):
    def test_model_has_31_params_and_dataset_16(self):
        self.assertEqual(NUM_PARAMS, 31)
        self.assertEqual(len(build_dataset()), 16)
        self.assertEqual(len(initial_params()), 31)

    def test_gradient_matches_numeric_gradient(self):
        data = build_dataset()
        params = initial_params()
        analytic = compute_gradient(params, data)
        eps = 1e-6
        for k in (0, 13, 20, 25, 30):
            plus = list(params)
            minus = list(params)
            plus[k] += eps
            minus[k] -= eps
            numeric = (compute_loss(plus, data) - compute_loss(minus, data)) / (2 * eps)
            self.assertAlmostEqual(analytic[k], numeric, delta=1e-5)

    def test_loss_falls_in_single_process_reference(self):
        ref = train_single_process(build_dataset(), steps=30, lr=0.05, momentum=0.5)
        self.assertAlmostEqual(ref["initial_loss"], 3.9524402310764573, delta=1e-9)
        self.assertLess(ref["final_loss"], 0.05)
        self.assertLess(ref["final_loss"], ref["initial_loss"])


class TestStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = build_dataset()
        cls.results = {}
        for s in ("fsdp", "zero"):
            cls.results[s] = run_strategy(
                s, cls.data, os.path.join(cls.tmp.name, f"ck_{s}.json"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_step_one_matches_reference_both_strategies(self):
        ref = train_single_process(self.data, steps=1, lr=0.05, momentum=0.5)
        for s in ("fsdp", "zero"):
            t = ShardedTrainer(s, self.data)
            entry = t.run_step()
            for got, want in zip(entry["params_after_update"], ref["log"][0]["params_after_update"]):
                self.assertAlmostEqual(got, want, delta=TOL)

    def test_final_matches_single_process_exactly_both(self):
        for s in ("fsdp", "zero"):
            self.assertEqual(self.results[s]["max_diff_vs_single_process"], 0.0)

    def test_strategies_match_each_other_and_loss_falls(self):
        f = self.results["fsdp"]
        z = self.results["zero"]
        self.assertEqual(f["final_params"], z["final_params"])
        for r in (f, z):
            self.assertAlmostEqual(r["initial_loss"], 3.9524402310764573, delta=1e-9)
            self.assertAlmostEqual(r["final_loss"], 0.037717029906809106, delta=1e-9)
            self.assertEqual(len(r["log"]), 30)

    def test_determinism_both_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            for s in ("fsdp", "zero"):
                again = run_strategy(s, self.data, os.path.join(tmp, f"ck_{s}.json"))
                self.assertEqual(again["final_params"], self.results[s]["final_params"])
                self.assertEqual(again["log"], self.results[s]["log"])


class TestCheckpointRestart(unittest.TestCase):
    def setUp(self):
        self.data = build_dataset()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "ckpt.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoint_round_trip_exact_both_strategies(self):
        for s in ("fsdp", "zero"):
            t = ShardedTrainer(s, self.data)
            t.run_to(12)
            t.save_checkpoint(self.path)
            fresh = ShardedTrainer.from_checkpoint(self.path, self.data)
            self.assertEqual(fresh.step, 12)
            self.assertEqual(fresh.param_shards, t.param_shards)
            self.assertEqual(fresh.velocity_shards, t.velocity_shards)
            with open(self.path, encoding="utf-8") as f:
                ckpt = json.load(f)
            self.assertEqual(ckpt["step"], 12)
            self.assertEqual([len(x) for x in ckpt["param_shards"]], [16, 15])
            self.assertEqual(len(ckpt["config_hash"]), 64)

    def test_restarted_run_equals_uninterrupted_both(self):
        for s in ("fsdp", "zero"):
            r = run_strategy(s, self.data, self.path)
            self.assertEqual(r["restart_max_diff_vs_uninterrupted"], 0.0)
            self.assertEqual(r["restarted_final_params"], r["final_params"])

    def test_checkpoint_config_hash_tamper_rejected(self):
        t = ShardedTrainer("fsdp", self.data)
        t.run_to(12)
        t.save_checkpoint(self.path)
        with open(self.path, encoding="utf-8") as f:
            ckpt = json.load(f)
        ckpt["config"]["learning_rate"] = 0.5
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f)
        with self.assertRaises(ShardingError):
            ShardedTrainer.from_checkpoint(self.path, self.data)

    def test_checkpoint_wrong_step_rejected(self):
        t = ShardedTrainer("zero", self.data)
        t.run_to(5)
        # Manually write a checkpoint at a non-checkpoint step.
        ckpt = {"strategy": "zero", "step": t.step,
                "param_shards": t.param_shards,
                "velocity_shards": t.velocity_shards,
                "config": t.config}
        from sharding import config_hash
        ckpt["config_hash"] = config_hash(t.config)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f)
        with self.assertRaises(ShardingError):
            ShardedTrainer.from_checkpoint(self.path, self.data)


class TestAccounting(unittest.TestCase):
    def test_communication_arithmetic_hand_computed(self):
        acc = communication_accounting(31, [16, 15], 30)
        self.assertEqual(acc["all_gather_bytes_per_step_all_ranks"], 124)
        self.assertEqual(acc["reduce_scatter_bytes_per_step_all_ranks"], 124)
        self.assertEqual(acc["bytes_per_step_all_ranks"], 248)
        self.assertEqual(acc["total_bytes_all_ranks"], 7440)
        self.assertEqual(acc["total_bytes_per_rank_average"], 3720)

    def test_memory_arithmetic_hand_computed(self):
        mem = memory_accounting(31, [16, 15])
        self.assertEqual(mem["persistent_elements_per_rank"], [32, 30])
        self.assertEqual(mem["peak_elements_per_rank"], [79, 76])
        self.assertEqual(mem["baseline_full_elements_per_rank"], 93)
        self.assertLess(mem["peak_elements_max_rank"], 93)

    def test_accounting_bad_inputs_rejected(self):
        with self.assertRaises(ShardingError):
            communication_accounting(0, [16, 15], 30)
        with self.assertRaises(ShardingError):
            memory_accounting(31, [])


class TestValidation(unittest.TestCase):
    def test_bad_strategy_rejected(self):
        with self.assertRaises(ShardingError):
            ShardedTrainer("ddp", build_dataset())

    def test_bad_training_inputs_rejected(self):
        data = build_dataset()
        with self.assertRaises(ShardingError):
            ShardedTrainer("fsdp", data, steps=0)
        with self.assertRaises(ShardingError):
            ShardedTrainer("fsdp", data, lr=0)
        with self.assertRaises(ShardingError):
            ShardedTrainer("fsdp", data, momentum=1.0)
        with self.assertRaises(ShardingError):
            ShardedTrainer("fsdp", data, checkpoint_step=30)
        with self.assertRaises(ShardingError):
            ShardedTrainer("fsdp", [])

    def test_trainer_holds_only_shards(self):
        t = ShardedTrainer("zero", build_dataset())
        self.assertEqual([len(s) for s in t.param_shards], [16, 15])
        self.assertEqual([len(s) for s in t.velocity_shards], [16, 15])
        self.assertEqual(t.full_params(), initial_params())


if __name__ == "__main__":
    unittest.main()
