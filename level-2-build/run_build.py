"""Run the Build scenario and write deterministic results.

Run: python level-2-build/run_build.py
Writes level-2-build/results_build.json. No timestamps, no
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
from sharding import build_dataset, run_strategy  # noqa: E402

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
