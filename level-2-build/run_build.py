"""Run the Build scenario and write deterministic results.

Run: python level-2-build/run_build.py
Writes level-2-build/results_build.json and results_scaling.json
(the ring all-reduce study and stage-aware memory counts from
collectives.py). No timestamps, no
randomness, no wall clock values, so two runs produce an
identical file. The checkpoint file is a generated intermediate
written to a temp directory during the run and not committed.
Simulated ranks in one process: no GPU, no real FSDP or
DeepSpeed libraries, no network transfer is used or claimed.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collectives import ring_study, stage_memory_accounting  # noqa: E402
from sharding import NUM_PARAMS, build_dataset, run_strategy  # noqa: E402

STEPS = 30
CHECKPOINT_STEP = 12
LR = 0.05
MOMENTUM = 0.5


def main():
    data = build_dataset()
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for strategy in ("fsdp", "zero"):
            ckpt_path = os.path.join(tmp, f"checkpoint_{strategy}.json")
            r = run_strategy(strategy, data, ckpt_path, steps=STEPS,
                             checkpoint_step=CHECKPOINT_STEP, lr=LR,
                             momentum=MOMENTUM)
            results[strategy] = r
            print(f"{strategy}: shards={r['shard_sizes']} steps={r['steps']} "
                  f"loss {round(r['initial_loss'], 6)} -> {round(r['final_loss'], 6)}, "
                  f"diff_vs_single={r['max_diff_vs_single_process']}, "
                  f"restart_diff={r['restart_max_diff_vs_uninterrupted']}, "
                  f"total_bytes={r['communication']['total_bytes_all_ranks']}, "
                  f"peak_elements={r['memory']['peak_elements_per_rank']}")
    out = {
        "scenario": "simulated parameter-sharded training, 2 simulated ranks in one "
                    "process, 2-layer MLP with 31 parameters, 16 synthetic examples, "
                    "FSDP-class and ZeRO-class strategies, checkpoint save and restart "
                    "at step 12 of 30, CPU, no GPU, no real FSDP or DeepSpeed libraries",
        "strategies": results,
    }
    path = os.path.join(HERE, "results_build.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print("Wrote", path)
    scaling = {
        "scenario": "chunked ring all-reduce of per-rank MLP gradients at 1, 2, 4, 8, 16 "
                    "simulated ranks, and per-rank model-state element counts for DDP and "
                    "ZeRO stages 1 to 3, one process, CPU, logical accounting only",
        "ring_all_reduce": ring_study(data),
        "stage_memory": [stage_memory_accounting(NUM_PARAMS, n) for n in (1, 2, 4, 8, 16)],
    }
    for r in scaling["ring_all_reduce"]:
        print(f"ring {r['num_ranks']} ranks: max elements sent {r['max_elements_sent_by_one_rank']}, "
              f"diff vs full {r['ring_max_abs_diff_vs_full']}")
    scaling_path = os.path.join(HERE, "results_scaling.json")
    with open(scaling_path, "w", encoding="utf-8") as f:
        json.dump(scaling, f, indent=2, sort_keys=True)
        f.write("\n")
    print("Wrote", scaling_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
