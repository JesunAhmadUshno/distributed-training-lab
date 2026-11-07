"""Ring all-reduce simulation and stage-aware memory accounting.

Two additions to the Build level, standard library only, one
process, deterministic:

- ring_all_reduce: the chunked ring algorithm (a reduce-scatter
  pass, then an all-gather pass) across N simulated ranks. Each rank
  sends 2 * (N - 1) chunks of about P / N elements, so the volume
  per rank tends to 2 * (N - 1) / N * P elements instead of the
  (N - 1) * P a naive all-to-all exchange would need. The ring adds
  partial sums in a different order from a plain left-to-right sum,
  so its result can differ in the last float bits; ring_study
  records that difference against the full-batch gradient.
- stage_memory_accounting: per-rank model-state element counts for
  replicated data parallel (DDP) and the three ZeRO stages, for this
  repo's SGD-with-momentum optimizer (parameters, gradients, one
  velocity buffer). Stage definitions follow Rajbhandari et al.,
  "ZeRO: Memory Optimizations Toward Training Trillion Parameter
  Models", SC 2020: stage 1 partitions optimizer state, stage 2 also
  gradients, stage 3 also parameters (the FSDP layout).

Nothing is transferred and no memory is measured: every count is
logical accounting from the parameter count and the partition.
"""
from sharding import (
    ShardingError,
    compute_gradient,
    initial_params,
    max_abs_diff,
    partition_indices,
)


def ring_all_reduce(vectors):
    """Sum one vector per rank with the chunked ring algorithm.

    Returns (per-rank result vectors, elements sent per rank).
    """
    if not vectors or any(len(v) != len(vectors[0]) for v in vectors):
        raise ShardingError("ring all-reduce needs one vector per rank, all the same length")
    n = len(vectors)
    parts = partition_indices(len(vectors[0]), n)
    buf = [list(v) for v in vectors]
    sent = [0] * n
    # Reduce-scatter: after n - 1 steps rank r holds the full sum of chunk (r + 1) % n.
    for t in range(n - 1):
        msgs = []
        for r in range(n):
            c = (r - t) % n
            msgs.append(((r + 1) % n, c, [buf[r][i] for i in parts[c]]))
            sent[r] += len(parts[c])
        for dst, c, vals in msgs:
            for i, v in zip(parts[c], vals):
                buf[dst][i] += v
    # All-gather: pass each finished chunk around the ring.
    for t in range(n - 1):
        msgs = []
        for r in range(n):
            c = (r + 1 - t) % n
            msgs.append(((r + 1) % n, c, [buf[r][i] for i in parts[c]]))
            sent[r] += len(parts[c])
        for dst, c, vals in msgs:
            for i, v in zip(parts[c], vals):
                buf[dst][i] = v
    return buf, sent


def ring_study(examples, ranks=(1, 2, 4, 8, 16)):
    """Average per-rank MLP gradients by ring and by plain sum.

    Each rank holds an equal contiguous shard of the examples and
    computes its shard-mean gradient at the initial parameters. The
    mean of the shard means is compared with the full-batch gradient.
    """
    params = initial_params()
    full = compute_gradient(params, examples)
    num_params = len(full)
    rows = []
    for n in ranks:
        if len(examples) % n:
            raise ShardingError("ring study ranks must divide the example count")
        size = len(examples) // n
        grads = [compute_gradient(params, examples[r * size:(r + 1) * size]) for r in range(n)]
        out, sent = ring_all_reduce(grads)
        ring = [v / n for v in out[0]]
        seq = [0.0] * num_params
        for g in grads:
            seq = [a + b for a, b in zip(seq, g)]
        seq = [v / n for v in seq]
        rows.append({
            "num_ranks": n,
            "elements_sent_per_rank": sent,
            "max_elements_sent_by_one_rank": max(sent),
            "bytes_sent_all_ranks": sum(sent) * 4,
            "ideal_elements_per_rank": 2 * (n - 1) / n * num_params,
            "rank_copies_max_diff": max(max_abs_diff(out[0], o) for o in out),
            "ring_max_abs_diff_vs_full": max_abs_diff(ring, full),
            "sequential_max_abs_diff_vs_full": max_abs_diff(seq, full),
            "ring_max_abs_diff_vs_sequential": max_abs_diff(ring, seq),
        })
    return rows


def stage_memory_accounting(num_params, num_ranks):
    """Model-state elements on the largest rank, by strategy.

    DDP replicates everything: 3 * P. ZeRO-1 shards the velocity:
    2 * P + s. ZeRO-2 also shards gradients: P + 2 * s. ZeRO-3 / FSDP
    also shards parameters: 3 * s persistent, and P + 3 * s at peak
    while a gathered full parameter copy exists (the same peak as
    sharding.memory_accounting). s is the largest shard size.
    Activations are not counted.
    """
    s = max(len(p) for p in partition_indices(num_params, num_ranks))
    p = num_params
    return {
        "num_params": p,
        "num_ranks": num_ranks,
        "largest_shard": s,
        "ddp": 3 * p,
        "zero1": 2 * p + s,
        "zero2": p + 2 * s,
        "zero3_fsdp_persistent": 3 * s,
        "zero3_fsdp_peak": p + 3 * s,
    }
