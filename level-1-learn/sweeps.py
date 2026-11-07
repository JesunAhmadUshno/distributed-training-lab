"""Rank sweep and shard-split sweep for the Learn level.

Two cheap CPU studies on the same 8-example dataset, standard
library only, deterministic:

- rank_sweep: the Learn run (20 steps, lr 0.05) at 1 to 8 simulated
  ranks. With equal shard sizes (1, 2, 4, 8 ranks) the naive
  all-reduce average equals the full-batch gradient except for
  floating-point reduction order: summing shard means and then
  averaging rounds differently from one sum over all examples, so
  the difference is a few units in the last place, not a bug. With
  unequal shards (3, 5, 6, 7 ranks, from contiguous sharding) the
  naive average gives every rank equal weight and is biased.
- imbalance_sweep: every 2-rank split k | 8-k at the initial
  parameters, comparing the naive average, the size-weighted
  average, and the full-batch gradient.

Ranks are simulated in one process. Byte counts are logical
accounting, nothing is transferred and no time is measured.
"""
from data_parallel import (
    all_reduce_average,
    compute_gradient,
    max_abs_diff,
    train_data_parallel,
    weighted_average,
)


def rank_sweep(examples, steps=20, lr=0.05, max_ranks=8):
    """Run the Learn trainer at 1..max_ranks simulated ranks."""
    rows = []
    reference = None
    for n in range(1, max_ranks + 1):
        s = train_data_parallel(examples, num_ranks=n, steps=steps, lr=lr)["summary"]
        if reference is None:
            reference = s["final_params"]
        rows.append({
            "num_ranks": n,
            "shard_sizes": s["shard_sizes"],
            "equal_shards": len(set(s["shard_sizes"])) == 1,
            "max_grad_diff_vs_single_process": s["max_grad_diff_vs_single_process"],
            "final_loss": s["final_loss"],
            "final_params": s["final_params"],
            "final_params_max_diff_vs_one_rank": max_abs_diff(s["final_params"], reference),
            "all_reduce_bytes_per_step_all_ranks": s["communication"]["bytes_per_step_all_ranks"],
        })
    return rows


def imbalance_sweep(examples, initial_params=None):
    """Every contiguous 2-rank split of the examples at one point."""
    params0 = list(initial_params) if initial_params is not None else [0.0, 0.0, 0.0]
    full = compute_gradient(params0, examples)
    rows = []
    for k in range(1, len(examples)):
        shards = [examples[:k], examples[k:]]
        grads = [compute_gradient(params0, s) for s in shards]
        naive = all_reduce_average(grads)
        weighted = weighted_average(grads, [len(s) for s in shards])
        rows.append({
            "shard_sizes": [k, len(examples) - k],
            "naive_average": naive,
            "weighted_average": weighted,
            "full_batch_gradient": full,
            "naive_max_abs_diff_vs_full": max_abs_diff(naive, full),
            "weighted_max_abs_diff_vs_full": max_abs_diff(weighted, full),
        })
    return rows
