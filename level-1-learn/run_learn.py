"""Run the Learn scenario and write deterministic results.

Run: python level-1-learn/run_learn.py
Writes level-1-learn/results_learn.json. No timestamps, no
randomness, no wall clock values, so two runs produce an
identical file. Simulated ranks in one process: no GPU, no real
DDP, no network transfer is used or claimed.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_parallel import (  # noqa: E402
    build_dataset,
    imbalance_experiment,
    train_data_parallel,
)


def main():
    data = build_dataset()
    result = train_data_parallel(data, num_ranks=2, steps=20, lr=0.05)
    imbalance = imbalance_experiment(data)
    s = result["summary"]
    for entry in result["log"]:
        print(f"step {entry['step']}: rank_losses="
              f"{[round(v, 6) for v in entry['rank_losses']]} "
              f"combined_loss={round(entry['combined_loss_before_update'], 6)} "
              f"grad_diff={entry['grad_max_abs_diff']}")
    print(f"Summary: {s['num_ranks']} simulated ranks, {s['num_examples']} examples, "
          f"shards {s['shard_sizes']}, {s['steps']} steps, "
          f"loss {s['initial_loss']} -> {round(s['final_loss'], 6)}, "
          f"max_grad_diff={s['max_grad_diff_vs_single_process']}, "
          f"max_replica_diff={s['max_replica_diff']}, "
          f"total_all_reduce_bytes={s['communication']['total_bytes_all_ranks']}")
    print(f"Imbalance 6+2: naive_diff={imbalance['naive_max_abs_diff_vs_full']}, "
          f"weighted_diff={imbalance['weighted_max_abs_diff_vs_full']}")
    out = {"scenario": "simulated data-parallel training, 2 ranks in one process, "
                       "3-parameter linear model, 8 synthetic examples, no GPU",
           "training": result, "imbalance_experiment": imbalance}
    path = os.path.join(HERE, "results_learn.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
