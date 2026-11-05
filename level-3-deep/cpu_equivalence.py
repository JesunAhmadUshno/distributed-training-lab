"""CPU equivalence check for the Deep prep package, stdlib only.

Reuses the Build sharding code and the Learn data-parallel code by
import (loaded from their level folders, not copied). It verifies
the Deep exercise logic at CPU scale, in one process, with no GPU,
no torch, no processes, and no network:

1. Gradient all-reduce equivalence: the Learn 2-rank all-reduce
   gradient and final parameters match the single-process reference,
   and the Build sharded trainers match their single-process
   reference for both strategies.
2. Shard partition sums: the Build partition covers every parameter
   index exactly once, shard element sums add to the full vector
   sum, and all-gather reconstructs the full vector exactly.
3. Checkpoint restart equality: for both Build strategies, a run
   restarted from its step 12 checkpoint finishes with parameters
   identical to the uninterrupted run.

All figures are CPU simulation figures in one process. No GPU
number is produced or claimed here.
"""

import importlib.util
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, relpath):
    path = os.path.join(ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_build():
    return _load("p19_build_sharding", os.path.join("level-2-build", "sharding.py"))


def load_learn():
    return _load("p19_learn_dp", os.path.join("level-1-learn", "data_parallel.py"))


def run_checks():
    build = load_build()
    learn = load_learn()

    # 1a. Learn gradient all-reduce equivalence.
    learn_data = learn.build_dataset()
    learn_result = learn.train_data_parallel(learn_data, num_ranks=2, steps=20, lr=0.05)
    learn_summary = learn_result["summary"]

    # 1b and 3. Build sharded vs single-process, and restart equality.
    build_data = build.build_dataset()
    strategy_results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for strategy in ("fsdp", "zero"):
            ckpt = os.path.join(tmp, f"ckpt_{strategy}.json")
            r = build.run_strategy(strategy, build_data, ckpt)
            strategy_results[strategy] = {
                "shard_sizes": r["shard_sizes"],
                "initial_loss": r["initial_loss"],
                "final_loss": r["final_loss"],
                "max_diff_vs_single_process": r["max_diff_vs_single_process"],
                "restart_max_diff_vs_uninterrupted": r["restart_max_diff_vs_uninterrupted"],
                "final_params": r["final_params"],
            }

    # 2. Shard partition sums and reconstruction.
    parts = build.partition_indices(build.NUM_PARAMS, 2)
    flat = [i for part in parts for i in part]
    params = build.initial_params()
    shards = build.shard_vector(params, parts)
    reconstructed = build.all_gather(shards, parts, build.NUM_PARAMS)
    shard_sums = [sum(s) for s in shards]

    checks = {
        "learn_all_reduce_max_grad_diff_vs_single_process": learn_summary["max_grad_diff_vs_single_process"],
        "learn_replica_max_diff": learn_summary["max_replica_diff"],
        "learn_initial_loss": learn_summary["initial_loss"],
        "learn_final_loss": learn_summary["final_loss"],
        "learn_final_params": learn_summary["final_params"],
        "build_strategies": strategy_results,
        "partition": {
            "num_params": build.NUM_PARAMS,
            "parts": parts,
            "covers_each_index_exactly_once": (
                sorted(flat) == list(range(build.NUM_PARAMS)) and len(flat) == len(set(flat))
            ),
            "shard_sizes": [len(p) for p in parts],
            "shard_element_sums": shard_sums,
            "shard_sums_total": sum(shard_sums),
            "full_vector_sum": sum(params),
            "shard_sum_abs_diff_vs_full": abs(sum(shard_sums) - sum(params)),
            "all_gather_reconstruction_max_diff": build.max_abs_diff(reconstructed, params),
        },
    }
    checks["all_pass"] = (
        checks["learn_all_reduce_max_grad_diff_vs_single_process"] < 1e-12
        and checks["learn_replica_max_diff"] == 0.0
        and all(v["max_diff_vs_single_process"] == 0.0 for v in strategy_results.values())
        and all(v["restart_max_diff_vs_uninterrupted"] == 0.0 for v in strategy_results.values())
        and checks["partition"]["covers_each_index_exactly_once"]
        and checks["partition"]["shard_sum_abs_diff_vs_full"] < 1e-12
        and checks["partition"]["all_gather_reconstruction_max_diff"] == 0.0
    )
    return checks


def build_report():
    c = run_checks()
    return {
        "scenario": (
            "Deep prep CPU equivalence check, stdlib only, one process, no GPU, no torch, "
            "reusing Build sharding and Learn data-parallel code by import"
        ),
        "level": "Deep PREP only, Deep NOT closed",
        "all_pass": c["all_pass"],
        "gradient_all_reduce_equivalence": {
            "learn_max_grad_diff_vs_single_process": c["learn_all_reduce_max_grad_diff_vs_single_process"],
            "learn_replica_max_diff": c["learn_replica_max_diff"],
            "learn_initial_loss": c["learn_initial_loss"],
            "learn_final_loss": c["learn_final_loss"],
            "learn_final_params": c["learn_final_params"],
            "build_fsdp_max_diff_vs_single_process": c["build_strategies"]["fsdp"]["max_diff_vs_single_process"],
            "build_zero_max_diff_vs_single_process": c["build_strategies"]["zero"]["max_diff_vs_single_process"],
        },
        "shard_partition_sums": c["partition"],
        "checkpoint_restart_equality": {
            "fsdp_restart_max_diff_vs_uninterrupted": (
                c["build_strategies"]["fsdp"]["restart_max_diff_vs_uninterrupted"]
            ),
            "zero_restart_max_diff_vs_uninterrupted": (
                c["build_strategies"]["zero"]["restart_max_diff_vs_uninterrupted"]
            ),
            "checkpoint_step": 12,
            "total_steps": 30,
        },
        "build_strategies": c["build_strategies"],
        "gpu_numbers_measured": False,
        "gpu_note": (
            "No GPU number was measured in this check. Scaling, FSDP on GPU, and communication "
            "timing on GPU remain unmeasured until the Kaggle run in KAGGLE-RUN.md is executed."
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(build_report(), indent=2, sort_keys=True))
