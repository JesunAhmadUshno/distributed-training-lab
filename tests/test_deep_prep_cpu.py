"""Tests for the Deep prep CPU equivalence check."""
import importlib.util
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    path = os.path.join(ROOT, "level-3-deep", "cpu_equivalence.py")
    spec = importlib.util.spec_from_file_location("p19_deep_prep_cpu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDeepPrepCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()
        cls.report = cls.mod.build_report()

    def test_all_pass(self):
        self.assertTrue(self.report["all_pass"])

    def test_no_gpu_numbers_claimed(self):
        self.assertFalse(self.report["gpu_numbers_measured"])

    def test_learn_all_reduce_equivalence(self):
        g = self.report["gradient_all_reduce_equivalence"]
        self.assertLess(g["learn_max_grad_diff_vs_single_process"], 1e-12)
        self.assertEqual(g["learn_replica_max_diff"], 0.0)
        self.assertEqual(g["build_fsdp_max_diff_vs_single_process"], 0.0)
        self.assertEqual(g["build_zero_max_diff_vs_single_process"], 0.0)

    def test_shard_partition_sums(self):
        p = self.report["shard_partition_sums"]
        self.assertTrue(p["covers_each_index_exactly_once"])
        self.assertEqual(p["shard_sizes"], [16, 15])
        self.assertLess(p["shard_sum_abs_diff_vs_full"], 1e-12)
        self.assertEqual(p["all_gather_reconstruction_max_diff"], 0.0)

    def test_checkpoint_restart_equality(self):
        c = self.report["checkpoint_restart_equality"]
        self.assertEqual(c["fsdp_restart_max_diff_vs_uninterrupted"], 0.0)
        self.assertEqual(c["zero_restart_max_diff_vs_uninterrupted"], 0.0)
        self.assertEqual(c["checkpoint_step"], 12)

    def test_deterministic_report(self):
        again = self.mod.build_report()
        self.assertEqual(json.dumps(self.report, sort_keys=True), json.dumps(again, sort_keys=True))

    def test_kaggle_scripts_compile_as_text(self):
        import py_compile
        for name in ("common.py", "ex1_ddp_correctness.py", "ex2_scaling_1v2.py", "ex3_fsdp_zero.py", "ex4_comm_timing.py"):
            path = os.path.join(ROOT, "level-3-deep", "kaggle", name)
            self.assertTrue(os.path.exists(path), name)
            py_compile.compile(path, doraise=True)


if __name__ == "__main__":
    unittest.main()
